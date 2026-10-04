"""Sampling probabilities shared by target sampling and Reflex supervision."""

from __future__ import annotations

import torch


def build_sampling_probs(logits, temperature=1.0, top_p=None, top_k=None, eos_token_id=2):
    """Return the normalized distribution actually used for target sampling."""
    if logits.ndim != 3:
        raise ValueError(f"expected [batch, sequence, vocabulary], got {tuple(logits.shape)}")
    if float(temperature) <= 0.0:
        raise ValueError("sampling temperature must be positive")
    vocabulary = logits.shape[-1]
    flat = logits.reshape(-1, vocabulary).float() / float(temperature)
    flat = torch.where(torch.isfinite(flat), flat, torch.full_like(flat, -torch.inf))
    invalid = torch.isneginf(flat).all(dim=-1)
    fallback = torch.full_like(flat, -torch.inf)
    fallback[:, int(eos_token_id)] = 0.0
    flat = torch.where(invalid.unsqueeze(-1), fallback, flat)
    probs = flat.softmax(dim=-1)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
        remove = torch.roll(remove, shifts=1, dims=-1)
        remove[..., 0] = False
        sorted_probs.masked_fill_(remove, 0.0)
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20))
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

    if top_k is not None and int(top_k) > 0 and int(top_k) < vocabulary:
        values, indices = torch.topk(probs, k=int(top_k), dim=-1)
        values.div_(values.sum(dim=-1, keepdim=True).clamp_min(1.0e-20))
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)

    return probs.reshape(*logits.shape[:-1], vocabulary)


def sample_from_probs(probs):
    vocabulary = probs.shape[-1]
    return torch.multinomial(probs.reshape(-1, vocabulary), 1).reshape(*probs.shape[:-1])


def sample_target_from_logits(
    logits,
    *,
    do_sample,
    temperature,
    top_p,
    top_k,
    eos_token_id,
    reflex=None,
    compact_to_target=None,
):
    """Sample once from existing logits and reuse the same probs for Reflex.

    This function deliberately accepts logits rather than a model, so Reflex
    supervision cannot trigger another target forward.
    """
    if do_sample == True:
        probs = build_sampling_probs(
            logits, temperature, top_p, top_k, eos_token_id
        )
        tokens = sample_from_probs(probs)
    elif do_sample == False:
        tokens = logits.argmax(-1)
        probs = None
    else:
        raise ValueError('"do_sample" must be True or False')
    if reflex is not None:
        if compact_to_target is None:
            raise ValueError("compact_to_target is required for Reflex supervision")
        if probs is None:
            reflex.update_from_target_tokens(tokens[:, 0], compact_to_target)
        else:
            reflex.update_from_target_probs(probs[:, 0, :], compact_to_target)
    return tokens, probs
