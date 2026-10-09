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


def compare_training_values(actual,expected):
    """Check placebo updates at storage precision; freeze/resume use digests.

    Long CUDA SDPA backward has reduction-order noise. Require both maximum
    absolute error and relative tensor L2 error within one dtype epsilon
    (3e-6 for FP32), rather than treating any changed gradient hash as failure.
    Counters, structure, dtypes and nonfloating values still match exactly.
    """
    import math
    report=dict(passed=True,bitwise_equal=True,max_abs_error=0.,max_relative_l2_error=0.,
                max_tolerance_fraction=0.,different_tensors=0,failed_paths=[])
    def fail(path):
        report['passed']=False;report['bitwise_equal']=False
        if len(report['failed_paths'])<16:report['failed_paths'].append(path)
    def visit(a,b,path):
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape!=b.shape or a.dtype!=b.dtype:fail(path);return
            a=a.detach().cpu();b=b.detach().cpu()
            if torch.equal(a,b):return
            report['bitwise_equal']=False;report['different_tensors']+=1
            if not a.is_floating_point() or not torch.isfinite(a).all() or not torch.isfinite(b).all():fail(path);return
            delta=(a.float()-b.float());maximum=float(delta.abs().max())
            tolerance=max(3e-6,torch.finfo(a.dtype).eps)
            reference_norm=float(b.float().norm());error_norm=float(delta.norm())
            scale=float(b.float().abs().max());absolute_bound=1e-12+tolerance*scale
            norm_bound=1e-12+tolerance*reference_norm
            fraction=max(maximum/absolute_bound,error_norm/norm_bound)
            report['max_abs_error']=max(report['max_abs_error'],maximum)
            report['max_relative_l2_error']=max(report['max_relative_l2_error'],error_norm/max(reference_norm,1e-30))
            report['max_tolerance_fraction']=max(report['max_tolerance_fraction'],fraction)
            if not math.isfinite(fraction) or fraction>1:fail(path)
        elif torch.is_tensor(a) or torch.is_tensor(b):fail(path)
        elif isinstance(a,Mapping) and isinstance(b,Mapping):
            if a.keys()!=b.keys():fail(path);return
            for key in a:visit(a[key],b[key],path+'.'+str(key))
        elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
            if len(a)!=len(b):fail(path);return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'.'+str(i))
        elif isinstance(a,float) and isinstance(b,float):
            if a!=b:report['bitwise_equal']=False
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=3e-6,abs_tol=1e-12):fail(path)
        elif a!=b:report['bitwise_equal']=False;fail(path)
    visit(actual,expected,'root')
    return report


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
