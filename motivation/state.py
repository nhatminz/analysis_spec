"""Owned CPU snapshots, RNG isolation and atomic experiment journals."""
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch
from helper.checkpointing import capture_rng_state, restore_rng_state


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    import copy
    return copy.deepcopy(value)


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().resolve_conj().resolve_neg().contiguous()
            h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, Mapping):
            for k in sorted(v, key=str):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
        else:
            h.update(repr(v).encode()); h.update(b'\0')
    visit(value)
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def generation_seed(seed, step, prompt_id, sample=0):
    key = f'{seed}:{step}:{prompt_id}:{sample}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)


@contextmanager
def isolated_rng(seed=None, state=None):
    saved = capture_rng_state()
    try:
        if state is not None: restore_rng_state(state)
        elif seed is not None: seed_all(seed)
        yield
    finally:
        restore_rng_state(saved)


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def gradients(module):
    return {n: cpu_copy(p.grad) for n, p in module.named_parameters() if p.grad is not None}


def restore_gradients(module, state):
    for n, p in module.named_parameters():
        p.grad = None if n not in state else state[n].to(device=p.device, dtype=p.dtype).clone()
