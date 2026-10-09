"""Tensor form of FastGRPO's sample-once, token-matching tree verifier.

Used only by ACTIVE Reflex. It never samples, reweights probabilities, invokes a
model or reads GPU scalars on the host. The legacy OFF verifier stays untouched.
"""

from dataclasses import dataclass

import torch


@dataclass
class PackedTree:
    parents: torch.Tensor  # [B, packed_nodes+1], root=-1; indices include root=0
    tokens: torch.Tensor
    feedback_contexts: torch.Tensor  # -1 for nodes without a draft-head context
    max_depth: int

    def attention_mask(self, past_length, dtype, padding_positions=None, *, kernels=None, workspace=None):
        batch, rows = self.parents.shape
        shape = (batch, 1, rows, past_length + rows)
        elements = batch * rows * (past_length + rows)
        if workspace is not None and workspace.numel() >= elements:
            mask = workspace[:elements].view(shape)
        else:
            mask = torch.empty(shape, device=self.parents.device, dtype=dtype)
        if kernels is not None:
            kernels.tree_mask(self, past_length, mask)
            if isinstance(padding_positions, torch.Tensor):
                mask[padding_positions[:, 0], 0, :, padding_positions[:, 1]] = torch.finfo(dtype).min
            return mask
        mask.zero_()
        mask[..., past_length + 1:] = torch.finfo(dtype).min
        row_ids = torch.arange(rows, device=self.parents.device).expand(batch, -1)
        cursor = row_ids
        for _ in range(self.max_depth + 1):
            # Invalid/root parents write only the already-visible root column.
            mask[:, 0].scatter_(2, (cursor.clamp_min(0) + past_length).unsqueeze(-1), 0.0)
            cursor = self.parents.gather(1, cursor.clamp_min(0))
        if isinstance(padding_positions, torch.Tensor):
            mask[padding_positions[:, 0], 0, :, padding_positions[:, 1]] = torch.finfo(dtype).min
        return mask


def pack_tree(full_parents, full_contexts, chosen, all_tokens, max_depth, *, workspace=None):
    """Keep the original sorted confidence-selected packing and child order."""
    batch, full_count = full_parents.shape
    packed_count = chosen.shape[1]
    device = chosen.device
    inverse = (torch.empty((batch, full_count+1),device=device,dtype=torch.long) if workspace is None
               else workspace[0][:batch*(full_count+1)].view(batch,full_count+1))
    inverse.fill_(-1)
    inverse[:, 0] = 0  # full parent=-1 is the root
    packed_ids = torch.arange(1, packed_count + 1, device=device).expand(batch, -1)
    inverse.scatter_(1, chosen + 1, packed_ids)
    parents = inverse.gather(1, full_parents.gather(1, chosen) + 1)
    # Upstream tree construction also requires every selected ancestor to
    # exist. Diagnose invalid ancestry without changing native TopK pruning.
    torch._assert_async(((parents >= 0) & (parents < packed_ids)).all(),
                        "confidence-selected draft tree is not parent-closed")
    if workspace is not None:
        outputs=[w[:batch*(packed_count+1)].view(batch,packed_count+1) for w in workspace[1:]]
        outputs[0][:,0]=-1;outputs[0][:,1:].copy_(parents)
        outputs[1][:,0]=-1;torch.gather(all_tokens,1,chosen,out=outputs[1][:,1:])
        outputs[2][:,0]=0;torch.gather(full_contexts,1,chosen,out=outputs[2][:,1:])
        return PackedTree(*outputs,int(max_depth))
    return PackedTree(
        torch.cat((torch.full((batch, 1), -1, device=device, dtype=torch.long), parents), 1),
        torch.cat((torch.full((batch, 1), -1, device=device, dtype=torch.long), all_tokens.gather(1, chosen)), 1),
        torch.cat((torch.zeros((batch, 1), device=device, dtype=torch.long), full_contexts.gather(1, chosen)), 1),
        int(max_depth),
    )


@dataclass
class VerifiedPath:
    tokens: torch.Tensor  # -1 padding; emitted tokens, including target bonus
    packed_indices: torch.Tensor  # target-logit rows for those tokens
    feedback_contexts: torch.Tensor
    lengths: torch.Tensor

    def padded_gpu(self, past_length: int, width: int, eos_token_id: int, *, kernels=None, workspace=None):
        """Pad committed paths on device, preserving FastGRPO's gap-first rule.

        Only ``width=max(lengths)`` is host metadata. Accepted tokens and KV
        indices never make a CPU round trip. The mask marks synthetic EOS slots.
        """
        batch, capacity = self.tokens.shape
        if not 1 <= width <= capacity:
            raise ValueError("invalid committed path width")
        if kernels is not None:
            return kernels.pad_verified_path(self, past_length, width, eos_token_id, workspace)
        device = self.tokens.device
        slots = torch.arange(capacity, device=device)[None, :]
        valid = slots < self.lengths[:, None]
        # Each accepted tree row occupies its sorted absolute KV position until
        # the per-response padding budget is exhausted. Remaining slots are
        # synthetic EOS, indexed exactly as the original gap/tail insertion.
        budget = width - self.lengths[:, None]
        gaps = (self.packed_indices - slots).clamp_min(0)
        inserted_before = torch.minimum(gaps, budget.clamp_min(0))
        destination = slots + inserted_before
        positions = torch.arange(width, device=device)[None, :, None]
        accepted = (positions == destination[:, None, :]) & valid[:, None, :]
        accepted_mask = accepted.any(-1)
        source_slot = accepted.to(torch.int32).argmax(-1).long()
        output_tokens = torch.where(accepted_mask,
                                    self.tokens.gather(1, source_slot), int(eos_token_id))
        output_indices = torch.where(accepted_mask,
                                     self.packed_indices.gather(1, source_slot) + int(past_length),
                                     torch.arange(width, device=device)[None, :] + int(past_length))
        last_valid = torch.where(valid, destination, -1).amax(-1).long()[:, None]
        return output_tokens, output_indices, ~accepted_mask, last_valid

    def host_bookkeeping(self, past_length):
        """One small D2H packet, only AFTER GPU feedback has consumed this path.

        Legacy KV compaction/padding/finish handling still needs Python lists.
        No full tree/candidate token list or feedback data travels to the host.
        """
        packet = torch.stack((self.tokens, self.packed_indices), dim=1).cpu().tolist()
        tokens, chosen, lengths = [], [], []
        for row in packet:
            length = sum(index >= 0 for index in row[1])
            tokens.append(row[0][:length])
            chosen.append([past_length + index for index in row[1][:length]])
            lengths.append(length)
        return lengths, chosen, tokens


def trace_verified_path(tree, sampled_tokens, eos_token_id, *, kernels=None, workspace=None):
    """Traverse only matched children, first in original packed order; stop EOS.

    Torch vectorizes over batch/children, looping over bounded depth only.
    Triton fuses the whole extraction into one launch. Both consume exactly the
    same already-sampled target tokens; counterfactual branches never enter it.
    """
    batch, rows = sampled_tokens.shape
    if sampled_tokens.shape != tree.parents.shape:
        raise ValueError("sampled tokens do not match the packed tree")
    width = tree.max_depth + 1
    if workspace is None:
        buffers = [torch.empty((batch, width), device=sampled_tokens.device, dtype=torch.long) for _ in range(3)]
        lengths = torch.empty(batch, device=sampled_tokens.device, dtype=torch.long)
    else:
        buffers = [tensor[:batch, :width] for tensor in workspace[:3]]
        lengths = workspace[3][:batch]
    emitted, indices, contexts = buffers
    if kernels is not None:
        kernels.trace_path(tree, sampled_tokens, eos_token_id, emitted, indices, contexts, lengths)
        return VerifiedPath(emitted, indices, contexts, lengths)
    current = torch.zeros(batch, device=sampled_tokens.device, dtype=torch.long)
    live = torch.ones(batch, device=sampled_tokens.device, dtype=torch.bool)
    candidates = torch.arange(rows, device=sampled_tokens.device).expand(batch, -1)
    lengths.zero_()
    for depth in range(width):
        token = sampled_tokens.gather(1, current[:, None]).squeeze(1)
        emitted[:, depth] = torch.where(live, token, -1)
        indices[:, depth] = torch.where(live, current, -1)
        contexts[:, depth] = torch.where(live, tree.feedback_contexts.gather(1, current[:, None]).squeeze(1), -1)
        lengths.add_(live.long())
        matches = ((tree.parents == current[:, None]) & (tree.tokens == token[:, None])
                   & (candidates > 0) & live[:, None] & (token[:, None] != int(eos_token_id)))
        found = torch.where(matches, candidates, rows).amin(dim=1)
        live = live & (found < rows) & (token != int(eos_token_id))
        current = found.clamp_max(rows - 1)
    return VerifiedPath(emitted, indices, contexts, lengths)
