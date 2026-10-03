"""Align FastGRPO's already-shifted EAGLE-3 tokens with SpecForge teachers."""

from __future__ import annotations


def supervision_bounds(sequence_length: int, prompt_length: int) -> tuple[int, int]:
    """Rows with a response token and a *following* captured teacher row.

    Decoder row p contains token x[p+1] and auxiliary features for x[p].
    TargetHead.preprocess right-shifts final hidden and the original response
    mask, so the first supervised row is prompt_length-1. The final row has no
    next hidden and must never contribute to the loss.
    """
    if sequence_length < 0 or prompt_length < 0:
        raise ValueError("sequence and prompt lengths must be nonnegative")
    return min(max(prompt_length - 1, 0), sequence_length), max(sequence_length - 1, 0)


def aligned_eagle3_row(features, target_hidden, shifted_input_ids, prompt_length, token_budget=None):
    """Prepare one unpadded rollout row for OnlineEagle3Model.forward.

    ``shifted_input_ids`` came from the decoder; it is intentionally not shifted
    again. Auxiliary features likewise stay at their original positions. Only
    final target hidden is shifted as in TargetHead.preprocess.
    """
    import torch

    length = int(shifted_input_ids.shape[0])
    if features.shape[0] != length or target_hidden.shape[0] != length:
        raise ValueError("EAGLE-3 token, auxiliary feature and teacher lengths disagree")
    if token_budget is not None and int(token_budget) < 0:
        raise ValueError("token_budget must be nonnegative")
    shifted_teacher = torch.cat(
        (target_hidden[1:], torch.zeros_like(target_hidden[:1])), dim=0
    ) if length else target_hidden
    mask = torch.zeros(length, dtype=torch.float32, device=target_hidden.device)
    start, stop = supervision_bounds(length, int(prompt_length))
    if start < stop:
        mask[start:stop] = 1.0
    # A missing/non-finite captured hidden is not a teacher target. Zero it as
    # well so the compact projection cannot propagate NaNs from masked rows.
    if length:
        finite = torch.isfinite(shifted_teacher).all(dim=-1)
        mask *= finite.to(mask.dtype)
        shifted_teacher = torch.where(finite[:, None], shifted_teacher, 0)
    if token_budget is not None:
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        if selected.numel() > int(token_budget):
            mask.zero_()
            mask[selected[:int(token_budget)]] = 1.0
    return shifted_input_ids, features, shifted_teacher, mask[:, None]


def count_eagle3_supervision(token_rows, target_rows, prompt_mask, responses_per_prompt):
    """Count the same valid mask rows that branch training will actually use."""
    import torch

    if len(token_rows) != len(target_rows):
        raise ValueError("EAGLE-3 token/teacher row counts disagree")
    total = 0
    for index, (tokens, teacher) in enumerate(zip(token_rows, target_rows)):
        length = int(tokens.shape[0])
        if int(teacher.shape[0]) != length:
            raise ValueError("EAGLE-3 token/teacher lengths disagree")
        prompt_length = int(prompt_mask[index // responses_per_prompt].sum().item())
        start, stop = supervision_bounds(length, prompt_length)
        if start < stop:
            total += int(torch.isfinite(teacher[start + 1:stop + 1]).all(dim=-1).sum().item())
    return total
