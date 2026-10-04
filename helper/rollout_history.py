"""Owned per-response rollout history with amortized, append-only GPU copies.

No response holds a view into an old active batch. Only the newly committed
chunk is copied on ordinary rounds; storage grows geometrically when required
by verification padding. Finished buffers are released independently.
"""
from __future__ import annotations

import torch


class RolloutHistory:
    def __init__(self, initial, *, repeats=1, max_length, reserve_tokens=256):
        if repeats < 1 or max_length < 1 or reserve_tokens < 1:
            raise ValueError("history repeats, max_length and reserve_tokens must be positive")
        batches = {tensor.shape[0] for tensor in initial.values()}
        if len(batches) != 1:
            raise ValueError("history fields must have the same batch size")
        self.response_count = next(iter(batches)) * repeats
        self.lengths = {}
        self.capacities = {}
        self.buffers = {}
        self.specs = {}
        self.max_length = int(max_length)
        self.finished = set()
        for name, tensor in initial.items():
            length = tensor.shape[1]
            capacity = max(length, min(self.max_length, length + reserve_tokens))
            self.lengths[name] = length
            self.specs[name] = (tensor.dtype, tensor.device, tensor.shape[2:])
            self.capacities[name] = capacity
            rows = [tensor.new_empty((capacity, *tensor.shape[2:]))
                    for _ in range(self.response_count)]
            torch._foreach_copy_([row[:length] for row in rows],
                                [tensor[i // repeats] for i in range(self.response_count)])
            self.buffers[name] = rows

    def append(self, original_rows, chunks):
        """Append a padded accepted chunk in the unchanged active row order."""
        if set(chunks) != set(self.buffers):
            raise ValueError("history append fields differ from initialization")
        for name, chunk in chunks.items():
            if (chunk.dtype, chunk.device, chunk.shape[2:]) != self.specs[name]:
                raise ValueError("history chunk dtype/device/feature shape changed")
            if chunk.shape[0] != len(original_rows):
                raise ValueError("history chunk is misaligned with active trajectories")
            start = self.lengths[name]
            end = start + chunk.shape[1]
            rows = self.buffers[name]
            if end > self.capacities[name]:
                capacity = max(end, 2 * self.capacities[name])
                # max_length bounds real tokens, not verification padding. Grow
                # rather than truncate when padded history exceeds that bound.
                grown = [chunk.new_empty((capacity, *chunk.shape[2:]))
                         for _ in original_rows]
                torch._foreach_copy_([row[:start] for row in grown],
                                    [rows[i][:start] for i in original_rows])
                for i, row in zip(original_rows, grown):
                    rows[i] = row
                self.capacities[name] = capacity
            torch._foreach_copy_([rows[i][start:end] for i in original_rows],
                                list(chunk.unbind(0)))
            self.lengths[name] = end

    def finish(self, original_row):
        """Clone only the used row data; keep no old backing storage alive."""
        if original_row in self.finished:
            raise ValueError("response history was already finalized")
        result = {}
        for name, rows in self.buffers.items():
            result[name] = rows[original_row][:self.lengths[name]].clone()
            rows[original_row] = None
        self.finished.add(original_row)
        return result
