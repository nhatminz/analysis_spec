"""Original FastGRPO sampler; consume its intermediates before releasing them."""
import torch
import warnings
import torch.nn.functional as F

def sampling(
    logits, 
    top_k=None, 
    top_p=None, 
    temperature=0.6, 
    eos_token_id=2, metadata_builder=None, return_probs=True
):
    """
    Perform combined top-k and top-p (nucleus) sampling on logits.
    
    Args:
        logits (torch.Tensor): Logits from the model output (shape: [batch_size, seq_len, vocab_size]).
        top_k (int or None): Number of highest probability tokens to consider for top-k sampling.
        top_p (float or None): Cumulative probability threshold for top-p sampling.
        temperature (float): Temperature to adjust the sharpness of the distribution.
        eos_token_id (int): The ID of the end-of-sequence token (used as fallback when logits are invalid).
    
    Returns:
        torch.Tensor: Sampled token indices (shape: [batch_size, seq_len]).
    """
    assert logits.dim() == 3, f"Expected logits to have shape [bsz, seq, vocab], got {logits.shape}"
    bsz, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.view(-1, vocab_size)
    metadata = None

    if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
        logits_flat = torch.where(
            torch.isnan(logits_flat) | torch.isinf(logits_flat),
            torch.tensor(float('-inf'), dtype=logits_flat.dtype, device=logits_flat.device),
            logits_flat
        )

    valid_mask = ~torch.isinf(logits_flat).all(dim=-1)  
    if not valid_mask.any():
        warnings.warn("All sequences in the batch are invalid. Returning EOS token IDs as fallback.")
        tokens = torch.full((bsz, seq_len), eos_token_id, dtype=torch.long, device=logits.device)
        probs = torch.zeros_like(logits); probs[..., eos_token_id] = 1.
        small = metadata_builder(tokens, probs, None) if metadata_builder else None
        return tokens, probs if return_probs else None, small

    sampled_tokens_flat = torch.full(
        (logits_flat.shape[0], ), 
        fill_value=eos_token_id, 
        dtype=torch.long, 
        device=logits.device
    )

    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() > 0:
        valid_logits = logits_flat[valid_indices] / temperature
        probs = F.softmax(valid_logits, dim=-1)
        
        if top_p:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative_probs > top_p
            mask = torch.roll(mask, shifts=1, dims=-1)
            mask[..., 0] = False  

            sorted_probs.masked_fill_(mask, 0.0)
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            metadata = (sorted_probs, sorted_indices)
        
        if top_k:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
            metadata = (top_k_probs, top_k_indices)

        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled_tokens_flat[valid_indices] = sampled_indices

    sampled_tokens = sampled_tokens_flat.view(bsz, seq_len)
    # Normal finite-logit path retains the exact upstream probabilities/sort.
    # Invalid rows are EOS deltas and consume no multinomial draws, as upstream.
    if valid_indices.numel() != bsz * seq_len:
        full = torch.zeros_like(logits_flat); full[:, eos_token_id] = 1.
        full[valid_indices] = probs
        probs = full
        if metadata is not None:
            values, ids = metadata
            all_values = values.new_zeros((bsz * seq_len, values.shape[-1]))
            all_ids = ids.new_full(all_values.shape, eos_token_id)
            all_values[:, 0] = 1.
            all_values[valid_indices], all_ids[valid_indices] = values, ids
            metadata = (all_values, all_ids)
    probs = probs.view(bsz, seq_len, vocab_size)
    small = metadata_builder(sampled_tokens, probs, metadata) if metadata_builder else None
    return sampled_tokens, probs if return_probs else None, small

def sample_target_with_metadata(logits, *, do_sample, temperature, top_p, top_k,
                                eos_token_id, metadata_builder=None, return_probs=True):
    if do_sample:
        return sampling(logits, top_k, top_p, temperature, eos_token_id,
                        metadata_builder, return_probs)
    tokens = logits.softmax(-1).argmax(-1)
    small = metadata_builder(tokens, None, None) if metadata_builder else None
    return tokens, None, small
