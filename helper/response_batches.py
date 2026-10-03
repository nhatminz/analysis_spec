"""Stable response permutation and target-GRPO microbatch boundaries."""

from __future__ import annotations


def response_permutation(lengths):
    return sorted(range(len(lengths)), key=lambda index: lengths[index])


def reorder_response_fields(lengths, fields):
    """Apply one stable length permutation to every response-level field."""
    order = response_permutation(lengths)
    for name, values in fields.items():
        if len(values) != len(order):
            raise ValueError(f"response field {name} has {len(values)} rows, expected {len(order)}")
    return order, {
        name: [values[index] for index in order]
        for name, values in fields.items()
    }


def microbatch_index_groups(lengths, max_training_token, max_padding_gap):
    """Mirror the original packing rule, including the final partial group."""
    group = []
    current_max = 0
    for index, length in enumerate(lengths):
        fits = (
            max(current_max, length) * (len(group) + 1) <= max_training_token
            and (length - current_max) * len(group) <= max_padding_gap
        )
        if group and not fits:
            yield group
            group = []
            current_max = 0
        group.append(index)
        current_max = max(current_max, length)
    if group:
        yield group
