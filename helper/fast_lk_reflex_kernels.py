"""Lazy-loaded CUDA/Triton kernels for FP32 trajectory-local LK Reflex.

No model forward, autograd, RNG, target softmax, optimizer or host synchronization.
Reductions are tiled to support both compact and full Qwen vocabularies.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _feature_kernel(H, R, PSI, HS0, HS1,
                    HS2, CONTEXTS: tl.constexpr,
                    HIDDEN: tl.constexpr, DIM: tl.constexpr,
                    BH: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    hidden = tl.load(H + (row // CONTEXTS) * HS0 + (row % CONTEXTS) * HS1 + h * HS2,
                     h < HIDDEN, other=0).to(tl.float32)
    projection = tl.load(R + h[:, None] * DIM + d[None, :],
                         (h[:, None] < HIDDEN) & (d[None, :] < DIM), other=0)
    projected = tl.sum(hidden[:, None] * projection, axis=0)
    norm = tl.maximum(tl.sqrt(tl.sum(projected * projected, axis=0)), 1.0e-6)
    tl.store(PSI + row * DIM + d, tl.div_rn(projected, norm), d < DIM)


@triton.jit
def _correction_kernel(Z, PSI, A, OUT,
                       ZS0, ZS1, ZS2,
                       VOCAB: tl.constexpr, DIM: tl.constexpr, CONTEXTS: tl.constexpr,
                       BV: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    batch = batch.to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    d = tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    # One A tile, reused across ALL small contexts. No context grid dimension.
    for context in range(CONTEXTS):
        psi = tl.load(PSI + (batch * CONTEXTS + context) * DIM + d, d < DIM, other=0)
        z = tl.load(Z + batch * ZS0 + context * ZS1 + v * ZS2, v < VOCAB, other=0).to(tl.float32)
        corrected = z + tl.sum(a * psi[None, :], axis=1)
        tl.store(OUT + (batch * CONTEXTS + context) * VOCAB + v, corrected, v < VOCAB)


@triton.jit
def _correction_parallel(Z, PSI, A, OUT,
                         ZS0, ZS1, ZS2, VOCAB: tl.constexpr, DIM: tl.constexpr,
                         CONTEXTS: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr):
    tile, context, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    v, d = tile * BV + tl.arange(0, BV), tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    psi = tl.load(PSI + (batch * CONTEXTS + context) * DIM + d, d < DIM, other=0)
    z = tl.load(Z + batch * ZS0 + context * ZS1 + v * ZS2, v < VOCAB, other=0).to(tl.float32)
    tl.store(OUT + (batch * CONTEXTS + context) * VOCAB + v,
             z + tl.sum(a * psi[None, :], axis=1), v < VOCAB)


@triton.jit
def _correction_tiled(Z, PSI, A, OUT,
                      ZS0, ZS1, ZS2, VOCAB: tl.constexpr, DIM: tl.constexpr,
                      CONTEXTS: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
                      BD: tl.constexpr):
    tile, context_tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    v, c, d = tile * BV + tl.arange(0, BV), context_tile * BC + tl.arange(0, BC), tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    psi = tl.load(PSI + (batch * CONTEXTS + c[:, None]) * DIM + d[None, :],
                  (c[:, None] < CONTEXTS) & (d[None, :] < DIM), other=0)
    correction = tl.sum(a[:, None, :] * psi[None, :, :], axis=2)
    z = tl.load(Z + batch * ZS0 + c[None, :] * ZS1 + v[:, None] * ZS2,
                (v[:, None] < VOCAB) & (c[None, :] < CONTEXTS), other=0).to(tl.float32)
    tl.store(OUT + (batch * CONTEXTS + c[None, :]) * VOCAB + v[:, None],
             z + correction, (v[:, None] < VOCAB) & (c[None, :] < CONTEXTS))


@triton.jit
def _teacher_values(T, MAP, batch, v, VOCAB: tl.constexpr,
                    TS0, TS1, MS0,
                    GREEDY: tl.constexpr):
    ids = tl.load(MAP + v * MS0, v < VOCAB, other=0)
    if GREEDY:
        token = tl.load(T + batch * TS0)
        return ((ids == token) & (v < VOCAB)).to(tl.float32)
    return tl.load(T + batch * TS0 + ids * TS1, v < VOCAB, other=0).to(tl.float32)


@triton.jit
def _teacher_mass_kernel(T, MAP, MASS,
                         VOCAB: tl.constexpr, TS0, TS1,
                         MS0, GREEDY: tl.constexpr,
                         TILES: tl.constexpr, BV: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    batch = batch.to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
    tl.store(MASS + batch * TILES + tile, tl.sum(p, axis=0))


@triton.jit
def _lk_stats_kernel(Q, T, MAP, MASS, STATS,
                     QS0, QS1,
                     VOCAB: tl.constexpr, TS0, TS1,
                     MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                     TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    batch = batch.to(tl.int64)
    t = tl.arange(0, BT)
    denominator = tl.sum(tl.load(MASS + batch * TILES + t, t < TILES, other=0), axis=0) + EPS
    v = tile * BV + tl.arange(0, BV)
    p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
    p = tl.div_rn(p, denominator)
    q = tl.load(Q + batch * QS0 + v * QS1, v < VOCAB, other=0).to(tl.float32)
    alpha = tl.sum(tl.minimum(p, q), axis=0)
    selected = tl.sum(tl.where(q < p, q, 0.0), axis=0)
    tl.store(STATS + (batch * TILES + tile) * 2, alpha)
    tl.store(STATS + (batch * TILES + tile) * 2 + 1, selected)


@triton.jit
def _state_update_kernel(A, Q, PSI, T, MAP, MASS, STATS, ALPHA,
                         QS0, QS1,
                         PS0, PS1,
                         VOCAB: tl.constexpr, DIM: tl.constexpr,
                         TS0, TS1, MS0,
                         GREEDY: tl.constexpr, EPS: tl.constexpr,
                         LR: tl.constexpr, DECAY: tl.constexpr,
                         TILES: tl.constexpr, BT: tl.constexpr,
                         BV: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1)
    # Root probabilities are a strided view of [B, verification, full_vocab].
    # B*stride can exceed 2**31 on B200 even when the compact adapter is small.
    batch = batch.to(tl.int64)
    t = tl.arange(0, BT)
    mass = tl.sum(tl.load(MASS + batch * TILES + t, t < TILES, other=0), axis=0) + EPS
    alpha = tl.sum(tl.load(STATS + (batch * TILES + t) * 2, t < TILES, other=0), axis=0)
    selected = tl.sum(tl.load(STATS + (batch * TILES + t) * 2 + 1, t < TILES, other=0), axis=0)
    if tile == 0:
        tl.store(ALPHA + batch, alpha)
    if LR != 0.0 or DECAY != 1.0:
        v = tile * BV + tl.arange(0, BV)
        d = tl.arange(0, BD)
        p = _teacher_values(T, MAP, batch, v, VOCAB, TS0, TS1, MS0, GREEDY)
        p = tl.div_rn(p, mass)
        q = tl.load(Q + batch * QS0 + v * QS1, v < VOCAB, other=0).to(tl.float32)
        gradient = tl.div_rn(q * (selected - (q < p).to(tl.float32)), alpha + EPS)
        psi = tl.load(PSI + batch * PS0 + d * PS1, d < DIM, other=0).to(tl.float32)
        ptr = A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :]
        mask = (v[:, None] < VOCAB) & (d[None, :] < DIM)
        old = tl.load(ptr, mask, other=0)
        new = old * DECAY - LR * gradient[:, None] * psi[None, :]
        tl.store(ptr, new, mask)


def feature(hidden, projection, *, strategy="auto"):
    """Fuse FP32 conversion, seeded projection and feature normalization."""
    batch, contexts, hidden_size = hidden.shape
    dim = projection.shape[1]
    # Large unusual feature matrices are better left to cuBLAS; correction and
    # update remain fused. This is a shape-only choice, never a device sync.
    if strategy not in {"auto", "triton", "torch"}:
        raise ValueError("feature strategy must be auto, triton or torch")
    if strategy == "torch" or (strategy == "auto" and
                                triton.next_power_of_2(hidden_size) * triton.next_power_of_2(dim) > 32768):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return torch.nn.functional.normalize(hidden.float().matmul(projection), dim=-1, eps=1e-6)
    output = torch.empty((batch, contexts, dim), device=hidden.device, dtype=torch.float32)
    _feature_kernel[(batch * contexts,)](
        hidden, projection, output, *hidden.stride(), contexts, hidden_size, dim,
        triton.next_power_of_2(hidden_size), triton.next_power_of_2(dim),
        num_warps=8, enable_fp_fusion=False,
    )
    return output


def correct_logits(logits, psi, state, *, strategy="serial", block_vocab=256,
                   context_tile=2, num_warps=4, out=None):
    """Materialized diagnostic/reference API, not used by fused proposals."""
    batch, contexts, vocab = logits.shape
    dim = state.shape[-1]
    if strategy not in {"serial", "parallel", "tiled"}:
        raise ValueError("correction strategy must be serial, parallel or tiled")
    if block_vocab not in {64, 128, 256} or context_tile not in {1, 2, 4, 8} or num_warps not in {2, 4, 8}:
        raise ValueError("invalid correction launch configuration")
    output = out if out is not None else torch.empty((batch, contexts, vocab), device=logits.device, dtype=torch.float32)
    if output.shape != (batch, contexts, vocab) or output.dtype != torch.float32:
        raise ValueError("invalid correction output workspace")
    bd = triton.next_power_of_2(dim)
    if strategy == "serial":
        _correction_kernel[(triton.cdiv(vocab, block_vocab), batch)](
            logits, psi, state, output, *logits.stride(), vocab, dim, contexts,
            block_vocab, bd, num_warps=num_warps, enable_fp_fusion=False)
    elif strategy == "parallel":
        _correction_parallel[(triton.cdiv(vocab, block_vocab), contexts, batch)](
            logits, psi, state, output, *logits.stride(), vocab, dim, contexts,
            block_vocab, bd, num_warps=num_warps, enable_fp_fusion=False)
    else:
        _correction_tiled[(triton.cdiv(vocab, block_vocab), triton.cdiv(contexts, context_tile), batch)](
            logits, psi, state, output, *logits.stride(), vocab, dim, contexts,
            block_vocab, context_tile, bd, num_warps=num_warps, enable_fp_fusion=False)
    return output


def correct(logits, psi, state):
    return correct_logits(logits, psi, state).softmax(dim=-1)


def update(state, root_q, root_psi, target, mapping, *, greedy, eps, learning_rate, decay):
    """Three tiled launches, no dense p/mask/gradient/outer-product temporaries."""
    batch, vocab, dim = state.shape
    tiles = triton.cdiv(vocab, 2048)
    mass = torch.empty((batch, tiles), device=state.device, dtype=torch.float32)
    stats = torch.empty((batch, tiles, 2), device=state.device, dtype=torch.float32)
    alpha = torch.empty((batch,), device=state.device, dtype=torch.float32)
    ts0 = target.stride(0)
    ts1 = 0 if greedy else target.stride(1)
    grid = (tiles, batch)
    _teacher_mass_kernel[grid](
        target, mapping, mass, vocab, ts0, ts1, mapping.stride(0), greedy, tiles, 2048,
        num_warps=4, enable_fp_fusion=False,
    )
    _lk_stats_kernel[grid](
        root_q, target, mapping, mass, stats, *root_q.stride(), vocab, ts0, ts1,
        mapping.stride(0), greedy, eps, tiles, triton.next_power_of_2(tiles), 2048,
        num_warps=4, enable_fp_fusion=False,
    )
    _state_update_kernel[(triton.cdiv(vocab, 128), batch)](
        state, root_q, root_psi, target, mapping, mass, stats, alpha,
        *root_q.stride(), *root_psi.stride(), vocab, dim, ts0, ts1, mapping.stride(0),
        greedy, eps, learning_rate, decay, tiles, triton.next_power_of_2(tiles),
        128, triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False,
    )
    return alpha


@triton.jit
def _proposal_tiles(Z, PSI, A, MAX, SUM, VALUES, IDS,
                    ZS0, ZS1, ZS2, VOCAB: tl.constexpr, DIM: tl.constexpr,
                    CONTEXTS: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                    BV: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    d = tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    for c in range(CONTEXTS):
        psi = tl.load(PSI + (batch * CONTEXTS + c) * DIM + d, d < DIM, other=0)
        raw = tl.load(Z + batch * ZS0 + c * ZS1 + v * ZS2, v < VOCAB, other=0).to(tl.float32)
        z = tl.where(v < VOCAB, raw + tl.sum(a * psi[None, :], axis=1), -float('inf'))
        maximum = tl.max(z, axis=0)
        total = tl.sum(tl.exp(z - maximum), axis=0)
        offset = (batch * CONTEXTS + c) * TILES + tile
        tl.store(MAX + offset, maximum)
        tl.store(SUM + offset, total)
        for k in range(K):
            value = tl.max(z, axis=0)
            # Ties have deterministic compact-id order (Torch ties unspecified).
            index = tl.min(tl.where((z == value) & (v < VOCAB), v, VOCAB), axis=0)
            tl.store(VALUES + offset * K + k, value)
            tl.store(IDS + offset * K + k, index)
            z = tl.where(v == index, -float('inf'), z)


@triton.jit
def _proposal_merge(MAX, SUM, VALUES, IDS, PROBS, TOP_IDS, NORM,
                    CONTEXTS: tl.constexpr, VOCAB: tl.constexpr, K: tl.constexpr,
                    TILES: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BT)
    maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float('inf'))
    maximum = tl.max(maxima, axis=0)
    sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
    total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
    tl.store(NORM + row * 2, maximum)
    tl.store(NORM + row * 2 + 1, total)
    candidates = tl.arange(0, BK)
    values = tl.load(VALUES + row * TILES * K + candidates, candidates < TILES * K, other=-float('inf'))
    ids = tl.load(IDS + row * TILES * K + candidates, candidates < TILES * K, other=VOCAB)
    for k in range(K):
        value = tl.max(values, axis=0)
        index = tl.min(tl.where(values == value, ids, VOCAB), axis=0)
        tl.store(PROBS + row * K + k, tl.div_rn(tl.exp(value - maximum), total))
        tl.store(TOP_IDS + row * K + k, index)
        values = tl.where(ids == index, -float('inf'), values)


@triton.jit
def _rank_key(score, index):
    """Monotone FP32 score plus deterministic lower-ID tie break."""
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, ~bits, bits ^ (-2147483648)).to(tl.uint32)
    return (ordered.to(tl.uint64) << 32) | ((0xffffffff - index).to(tl.uint32).to(tl.uint64))


@triton.jit
def _unpack_rank_key(key):
    ordered = (key >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
    value = bits.to(tl.float32, bitcast=True)
    index = (0xffffffff - key.to(tl.uint32)).to(tl.int32)
    return value, index


@triton.jit
def _proposal_tiles_sort(Z, PSI, A, MAX, SUM, VALUES, IDS,
                         ZS0, ZS1, ZS2, VOCAB: tl.constexpr, DIM: tl.constexpr,
                         CONTEXTS: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                         BV: tl.constexpr, BD: tl.constexpr):
    tile, context, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    v, d = tile * BV + tl.arange(0, BV), tl.arange(0, BD)
    a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    psi = tl.load(PSI + (batch * CONTEXTS + context) * DIM + d, d < DIM, other=0)
    raw = tl.load(Z + batch * ZS0 + context * ZS1 + v * ZS2, v < VOCAB, other=0).to(tl.float32)
    z = tl.where(v < VOCAB, raw + tl.sum(a * psi[None, :], axis=1), -float('inf'))
    maximum = tl.max(z, axis=0)
    total = tl.sum(tl.exp(z - maximum), axis=0)
    offset = (batch * CONTEXTS + context) * TILES + tile
    tl.store(MAX + offset, maximum)
    tl.store(SUM + offset, total)
    ranked = tl.sort(_rank_key(z, v), descending=True)
    value, index = _unpack_rank_key(ranked)
    local = tl.arange(0, BV)
    tl.store(VALUES + offset * K + local, value, local < K)
    tl.store(IDS + offset * K + local, index, local < K)


@triton.jit
def _proposal_merge_sort(MAX, SUM, VALUES, IDS, PROBS, TOP_IDS, NORM,
                         VOCAB: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                         BT: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BT)
    maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float('inf'))
    maximum = tl.max(maxima, axis=0)
    sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
    total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
    tl.store(NORM + row * 2, maximum)
    tl.store(NORM + row * 2 + 1, total)
    candidates = tl.arange(0, BK)
    values = tl.load(VALUES + row * TILES * K + candidates,
                     candidates < TILES * K, other=-float('inf'))
    ids = tl.load(IDS + row * TILES * K + candidates,
                  candidates < TILES * K, other=VOCAB)
    ranked = tl.sort(_rank_key(values, ids), descending=True)
    value, index = _unpack_rank_key(ranked)
    tl.store(PROBS + row * K + candidates, tl.div_rn(tl.exp(value - maximum), total), candidates < K)
    tl.store(TOP_IDS + row * K + candidates, index, candidates < K)


def propose(logits, psi, state, k, workspace=None, *, strategy="fused", block_vocab=256, num_warps=4):
    """Two launches; only tile summaries/top-k leave registers, never full q."""
    batch, contexts, vocab = logits.shape
    if strategy not in {"fused", "sort"} or block_vocab not in {64, 128, 256} or num_warps not in {2, 4, 8}:
        raise ValueError("invalid proposal strategy or launch configuration")
    dim, tiles = state.shape[-1], triton.cdiv(vocab, block_vocab)
    sizes = [(batch, contexts, tiles), (batch, contexts, tiles),
             (batch, contexts, tiles, k), (batch, contexts, tiles, k)]
    # Flat pools avoid non-contiguous slices when active batch/C/K shrink.
    if workspace is None:
        buffers = [torch.empty(size, device=logits.device, dtype=torch.long if i == 3 else torch.float32)
                   for i, size in enumerate(sizes)]
    else:
        buffers = [pool[:math.prod(size)].view(size) for pool, size in zip(workspace, sizes)]
    maxima, sums, values, ids = buffers
    probabilities = torch.empty((batch, contexts, k), device=logits.device, dtype=torch.float32)
    selected = torch.empty((batch, contexts, k), device=logits.device, dtype=torch.long)
    norm = torch.empty((batch, contexts, 2), device=logits.device, dtype=torch.float32)
    if strategy == "fused":
        _proposal_tiles[(tiles, batch)](logits, psi, state, maxima, sums, values, ids,
            *logits.stride(), vocab, dim, contexts, k, tiles, block_vocab, triton.next_power_of_2(dim),
            num_warps=num_warps, enable_fp_fusion=False)
        _proposal_merge[(batch * contexts,)](maxima, sums, values, ids, probabilities, selected, norm,
            contexts, vocab, k, tiles, triton.next_power_of_2(tiles), triton.next_power_of_2(tiles * k),
            num_warps=num_warps, enable_fp_fusion=False)
    else:
        _proposal_tiles_sort[(tiles, contexts, batch)](logits, psi, state, maxima, sums, values, ids,
            *logits.stride(), vocab, dim, contexts, k, tiles, block_vocab, triton.next_power_of_2(dim),
            num_warps=num_warps, enable_fp_fusion=False)
        _proposal_merge_sort[(batch * contexts,)](maxima, sums, values, ids, probabilities, selected, norm,
            vocab, k, tiles, triton.next_power_of_2(tiles), triton.next_power_of_2(tiles * k),
            num_warps=num_warps, enable_fp_fusion=False)
    return probabilities, selected, norm


@triton.jit
def _trace_path(PARENTS, TOKENS, CONTEXTS, SAMPLES, OUT_T, OUT_I, OUT_C, LENGTHS,
                ROWS: tl.constexpr, WIDTH: tl.constexpr, EOS: tl.constexpr,
                OS0, OS1, BR: tl.constexpr):
    batch = tl.program_id(0).to(tl.int64)
    candidates = tl.arange(0, BR)
    parents = tl.load(PARENTS + batch * ROWS + candidates, candidates < ROWS, other=-2)
    tokens = tl.load(TOKENS + batch * ROWS + candidates, candidates < ROWS, other=-1)
    current = tl.full((), 0, tl.int32)
    live, length = current == 0, current
    for j in range(WIDTH):
        token = tl.load(SAMPLES + batch * ROWS + current)
        context = tl.load(CONTEXTS + batch * ROWS + current)
        tl.store(OUT_T + batch * OS0 + j * OS1, tl.where(live, token, -1))
        tl.store(OUT_I + batch * OS0 + j * OS1, tl.where(live, current, -1))
        tl.store(OUT_C + batch * OS0 + j * OS1, tl.where(live, context, -1))
        length = length + live.to(tl.int32)
        matches = (parents == current) & (tokens == token) & (candidates > 0) & (candidates < ROWS)
        found = tl.min(tl.where(matches & live & (token != EOS), candidates, ROWS), axis=0)
        live = live & (found < ROWS) & (token != EOS)
        current = tl.minimum(found, ROWS - 1)
    tl.store(LENGTHS + batch, length)


def trace_path(tree, samples, eos, tokens, indices, contexts, lengths):
    batch, rows = samples.shape
    _trace_path[(batch,)](tree.parents, tree.tokens, tree.feedback_contexts, samples,
        tokens, indices, contexts, lengths, rows, tokens.shape[1], int(eos), *tokens.stride(),
        triton.next_power_of_2(rows), num_warps=4)


@triton.jit
def _pad_verified_path(TOKENS, INDICES, LENGTHS, OUT_TOKENS, OUT_INDICES, OUT_MASK, LAST,
                       CAPACITY: tl.constexpr, WIDTH: tl.constexpr, PAST: tl.constexpr,
                       EOS: tl.constexpr, TS0, TS1, IS0, IS1, OS0, OS1, BW: tl.constexpr):
    batch = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BW)
    length = tl.load(LENGTHS + batch)
    index = tl.load(INDICES + batch * IS0 + slot * IS1, slot < CAPACITY, other=-1)
    valid = slot < length
    budget = WIDTH - length
    before = tl.minimum(tl.maximum(index - slot, 0), budget)
    destination = slot + before
    candidates = tl.where((destination[None, :] == slot[:, None]) & valid[None, :],
                          tl.broadcast_to(slot[None, :], (BW, BW)), BW)
    source = tl.min(candidates, axis=1)
    accepted = source < BW
    safe_source = tl.minimum(source, CAPACITY - 1)
    token = tl.load(TOKENS + batch * TS0 + safe_source * TS1, accepted & (slot < WIDTH), other=EOS)
    chosen = tl.load(INDICES + batch * IS0 + safe_source * IS1, accepted & (slot < WIDTH), other=0)
    tl.store(OUT_TOKENS + batch * OS0 + slot * OS1, tl.where(accepted, token, EOS), slot < WIDTH)
    tl.store(OUT_INDICES + batch * OS0 + slot * OS1,
             tl.where(accepted, chosen + PAST, slot + PAST), slot < WIDTH)
    tl.store(OUT_MASK + batch * OS0 + slot * OS1, ~accepted, slot < WIDTH)
    tl.store(LAST + batch, tl.max(tl.where(valid, destination, -1), axis=0))


def pad_verified_path(path, past_length, width, eos_token_id, workspace=None):
    batch, capacity = path.tokens.shape
    if workspace is None:
        tokens = torch.empty((batch, width), device=path.tokens.device, dtype=torch.long)
        indices = torch.empty_like(tokens)
        mask = torch.empty((batch, width), device=path.tokens.device, dtype=torch.bool)
        last = torch.empty((batch, 1), device=path.tokens.device, dtype=torch.long)
    else:
        tokens, indices, mask = [buffer[:batch, :width] for buffer in workspace[:3]]
        last = workspace[3][:batch, :1]
    _pad_verified_path[(batch,)](path.tokens, path.packed_indices, path.lengths,
        tokens, indices, mask, last, capacity, width, int(past_length), int(eos_token_id),
        *path.tokens.stride(), *path.packed_indices.stride(), *tokens.stride(),
        triton.next_power_of_2(capacity),
        num_warps=4)
    return tokens, indices, mask, last


@triton.jit
def _tree_mask(PARENTS, MASK, ROWS: tl.constexpr, PAST: tl.constexpr,
               WIDTH: tl.constexpr, MINIMUM: tl.constexpr, BK: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    columns = tl.arange(0, BK)
    visible = columns <= PAST  # prefix and root are shared by every query
    current = row.to(tl.int64)  # parent loads are int64; stable loop-carried dtype
    for depth in range(WIDTH):
        visible = visible | ((current >= 0) & (columns == PAST + current))
        current = tl.load(PARENTS + batch * ROWS + tl.maximum(current, 0))
    tl.store(MASK + (batch * ROWS + row) * (PAST + ROWS) + columns,
             tl.where(visible, 0., MINIMUM), columns < PAST + ROWS)


def tree_mask(tree, past_length, mask):
    batch, rows = tree.parents.shape
    _tree_mask[(rows, batch)](tree.parents, mask, rows, past_length, tree.max_depth + 1,
                             torch.finfo(mask.dtype).min, triton.next_power_of_2(past_length + rows), num_warps=4)


@triton.jit
def _path_values(RAW, PSI, NORM, a, TARGET, MAP, PATH, CONTEXT,
                 batch, slot, v, d,
                 VOCAB: tl.constexpr, DIM: tl.constexpr, CACHE: tl.constexpr,
                 WIDTH: tl.constexpr, RS0, RS1, RS2, TS0, TS1, TS2,
                 IS0, IS1, CS0, CS1, MS0, GREEDY: tl.constexpr):
    context = tl.load(CONTEXT + batch * CS0 + slot * CS1)
    row = tl.load(PATH + batch * IS0 + slot * IS1)
    valid = (context >= 0) & (row >= 0)
    safe = tl.maximum(context, 0)
    psi = tl.load(PSI + (batch * CACHE + safe) * DIM + d, (d < DIM) & valid, other=0)
    z = tl.load(RAW + batch * RS0 + safe * RS1 + v * RS2, (v < VOCAB) & valid, other=0).to(tl.float32)
    maximum = tl.load(NORM + (batch * CACHE + safe) * 2, valid, other=0)
    denominator = tl.load(NORM + (batch * CACHE + safe) * 2 + 1, valid, other=1)
    q = tl.where(valid & (v < VOCAB), tl.div_rn(tl.exp(z + tl.sum(a * psi[None, :], axis=1) - maximum), denominator), 0.)
    ids = tl.load(MAP + v * MS0, v < VOCAB, other=0)
    if GREEDY:
        token = tl.load(TARGET + batch * TS0 + tl.maximum(row, 0) * TS1, valid, other=-1)
        p = ((ids == token) & valid & (v < VOCAB)).to(tl.float32)
    else:
        p = tl.load(TARGET + batch * TS0 + tl.maximum(row, 0) * TS1 + ids * TS2,
                    valid & (v < VOCAB), other=0).to(tl.float32)
    return q, p, psi, valid


@triton.jit
def _path_stats(RAW, PSI, NORM, A, TARGET, MAP, PATH, CONTEXT, MASS, STATS,
                VOCAB: tl.constexpr, DIM: tl.constexpr, CACHE: tl.constexpr,
                WIDTH: tl.constexpr, RS0, RS1, RS2, TS0, TS1, TS2,
                IS0, IS1, CS0, CS1, MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                MASS_ONLY: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    v, d = tile * BV + tl.arange(0, BV), tl.arange(0, BD)
    if MASS_ONLY:
        a = tl.full((BV, BD), 0., tl.float32)  # dead along with q; no A read
    else:
        a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                    (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    for slot in range(WIDTH):
        q, p, psi, valid = _path_values(RAW, PSI, NORM, a, TARGET, MAP, PATH, CONTEXT,
            batch, slot, v, d, VOCAB, DIM, CACHE, WIDTH, RS0, RS1, RS2, TS0, TS1, TS2,
            IS0, IS1, CS0, CS1, MS0, GREEDY)
        offset = (batch * WIDTH + slot) * TILES
        if MASS_ONLY:
            tl.store(MASS + offset + tile, tl.sum(p, axis=0))
        else:
            t = tl.arange(0, BT)
            mass = tl.sum(tl.load(MASS + offset + t, t < TILES, other=0), axis=0) + EPS
            p = tl.div_rn(p, mass)
            tl.store(STATS + (offset + tile) * 2, tl.sum(tl.minimum(q, p), axis=0))
            tl.store(STATS + (offset + tile) * 2 + 1, tl.sum(tl.where(q < p, q, 0.), axis=0))


@triton.jit
def _path_stats_parallel(RAW, PSI, NORM, A, TARGET, MAP, PATH, CONTEXT, MASS, STATS,
                         VOCAB: tl.constexpr, DIM: tl.constexpr, CACHE: tl.constexpr,
                         WIDTH: tl.constexpr, RS0, RS1, RS2, TS0, TS1, TS2,
                         IS0, IS1, CS0, CS1, MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                         TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                         MASS_ONLY: tl.constexpr):
    tile, slot, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    v, d = tile * BV + tl.arange(0, BV), tl.arange(0, BD)
    if MASS_ONLY:
        a = tl.full((BV, BD), 0., tl.float32)
    else:
        a = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                    (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    q, p, psi, valid = _path_values(RAW, PSI, NORM, a, TARGET, MAP, PATH, CONTEXT,
        batch, slot, v, d, VOCAB, DIM, CACHE, WIDTH, RS0, RS1, RS2, TS0, TS1, TS2,
        IS0, IS1, CS0, CS1, MS0, GREEDY)
    offset = (batch * WIDTH + slot) * TILES
    if MASS_ONLY:
        tl.store(MASS + offset + tile, tl.sum(p, axis=0))
    else:
        t = tl.arange(0, BT)
        mass = tl.sum(tl.load(MASS + offset + t, t < TILES, other=0), axis=0) + EPS
        p = tl.div_rn(p, mass)
        tl.store(STATS + (offset + tile) * 2, tl.sum(tl.minimum(q, p), axis=0))
        tl.store(STATS + (offset + tile) * 2 + 1, tl.sum(tl.where(q < p, q, 0.), axis=0))


@triton.jit
def _path_update(RAW, PSI, NORM, A, TARGET, MAP, PATH, CONTEXT, MASS, STATS, ALPHA,
                 VOCAB: tl.constexpr, DIM: tl.constexpr, CACHE: tl.constexpr,
                 WIDTH: tl.constexpr, RS0, RS1, RS2, TS0, TS1, TS2,
                 IS0, IS1, CS0, CS1, MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                 LR: tl.constexpr, DECAY: tl.constexpr,
                 TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    v, d, t = tile * BV + tl.arange(0, BV), tl.arange(0, BD), tl.arange(0, BT)
    old = tl.load(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :],
                  (v[:, None] < VOCAB) & (d[None, :] < DIM), other=0)
    delta = tl.full((BV, BD), 0., tl.float32)
    count = 0
    for slot in range(WIDTH):
        # All reads of A are this program's own tile, before its single write.
        q, p, psi, valid = _path_values(RAW, PSI, NORM, old, TARGET, MAP, PATH, CONTEXT,
            batch, slot, v, d, VOCAB, DIM, CACHE, WIDTH, RS0, RS1, RS2, TS0, TS1, TS2,
            IS0, IS1, CS0, CS1, MS0, GREEDY)
        offset = (batch * WIDTH + slot) * TILES
        mass = tl.sum(tl.load(MASS + offset + t, t < TILES, other=0), axis=0) + EPS
        alpha = tl.sum(tl.load(STATS + (offset + t) * 2, t < TILES, other=0), axis=0)
        selected = tl.sum(tl.load(STATS + (offset + t) * 2 + 1, t < TILES, other=0), axis=0)
        p = tl.div_rn(p, mass)
        gradient = tl.div_rn(q * (selected - (q < p).to(tl.float32)), alpha + EPS)
        delta += gradient[:, None] * psi[None, :]
        count += valid.to(tl.int32)
        if tile == 0:
            tl.store(ALPHA + batch * WIDTH + slot, alpha)
    # div_rn accepts FP32 operands only (unlike '/' it does not promote ints).
    # Cast the scalar count inside this same kernel; keep the fused mean update.
    denominator = tl.maximum(count, 1).to(tl.float32)
    new = tl.where(count > 0, old * DECAY - LR * tl.div_rn(delta, denominator), old)
    tl.store(A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :], new,
             (v[:, None] < VOCAB) & (d[None, :] < DIM))


@triton.jit
def _path_update_parallel(RAW, PSI, NORM, A, TARGET, MAP, PATH, CONTEXT, MASS, STATS, ALPHA,
                          VOCAB: tl.constexpr, DIM: tl.constexpr, CACHE: tl.constexpr,
                          WIDTH: tl.constexpr, RS0, RS1, RS2, TS0, TS1, TS2,
                          IS0, IS1, CS0, CS1, MS0, GREEDY: tl.constexpr, EPS: tl.constexpr,
                          LR: tl.constexpr, DECAY: tl.constexpr,
                          TILES: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
                          BP: tl.constexpr, BD: tl.constexpr):
    tile, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    v, s, d, t = (tile * BV + tl.arange(0, BV), tl.arange(0, BP),
                  tl.arange(0, BD), tl.arange(0, BT))
    ptr = A + batch * VOCAB * DIM + v[:, None] * DIM + d[None, :]
    mask = (v[:, None] < VOCAB) & (d[None, :] < DIM)
    old = tl.load(ptr, mask, other=0)
    context = tl.load(CONTEXT + batch * CS0 + s * CS1, s < WIDTH, other=-1)
    row = tl.load(PATH + batch * IS0 + s * IS1, s < WIDTH, other=-1)
    valid = (s < WIDTH) & (context >= 0) & (row >= 0)
    safe_context, safe_row = tl.maximum(context, 0), tl.maximum(row, 0)
    feature = tl.load(PSI + (batch * CACHE + safe_context[:, None]) * DIM + d[None, :],
                      valid[:, None] & (d[None, :] < DIM), other=0)
    z = tl.load(RAW + batch * RS0 + safe_context[None, :] * RS1 + v[:, None] * RS2,
                (v[:, None] < VOCAB) & valid[None, :], other=0).to(tl.float32)
    correction = tl.sum(old[:, None, :] * feature[None, :, :], axis=2)
    maximum = tl.load(NORM + (batch * CACHE + safe_context) * 2, valid, other=0)
    denominator = tl.load(NORM + (batch * CACHE + safe_context) * 2 + 1, valid, other=1)
    q = tl.where(valid[None, :] & (v[:, None] < VOCAB),
                 tl.div_rn(tl.exp(z + correction - maximum[None, :]), denominator[None, :]), 0.)
    ids = tl.load(MAP + v * MS0, v < VOCAB, other=0)
    if GREEDY:
        token = tl.load(TARGET + batch * TS0 + safe_row * TS1, valid, other=-1)
        p = ((ids[:, None] == token[None, :]) & valid[None, :] & (v[:, None] < VOCAB)).to(tl.float32)
    else:
        p = tl.load(TARGET + batch * TS0 + safe_row[None, :] * TS1 + ids[:, None] * TS2,
                    valid[None, :] & (v[:, None] < VOCAB), other=0).to(tl.float32)
    offsets = (batch * WIDTH + s[:, None]) * TILES + t[None, :]
    mass = tl.sum(tl.load(MASS + offsets, (s[:, None] < WIDTH) & (t[None, :] < TILES), other=0), axis=1) + EPS
    stat_offsets = offsets * 2
    alpha = tl.sum(tl.load(STATS + stat_offsets,
                           (s[:, None] < WIDTH) & (t[None, :] < TILES), other=0), axis=1)
    selected = tl.sum(tl.load(STATS + stat_offsets + 1,
                              (s[:, None] < WIDTH) & (t[None, :] < TILES), other=0), axis=1)
    if tile == 0:
        tl.store(ALPHA + batch * WIDTH + s, alpha, s < WIDTH)
    p = tl.div_rn(p, mass[None, :])
    gradient = tl.div_rn(q * (selected[None, :] - (q < p).to(tl.float32)), alpha[None, :] + EPS)
    delta = tl.sum(gradient[:, :, None] * feature[None, :, :], axis=1)
    count = tl.maximum(tl.sum(valid.to(tl.float32), axis=0), 1.)
    updated = tl.where(tl.sum(valid.to(tl.int32), axis=0) > 0,
                       old * DECAY - LR * tl.div_rn(delta, count), old)
    tl.store(ptr, updated, mask)


def update_path(state, raw, psi, norm, target, mapping, indices, contexts, *, greedy,
                eps, learning_rate, decay, workspace=None, strategy="serial",
                block_vocab=256, num_warps=4):
    """Gather/reconstruct only visited contexts; THREE launches per round.

    Statistics are computed against the pre-update A. A single write kernel
    aggregates the mean of all valid contexts and applies decay exactly once.
    """
    if strategy not in {"serial", "parallel"}:
        raise ValueError("feedback strategy must be serial or parallel")
    if block_vocab not in {64, 128, 256} or num_warps not in {2, 4, 8}:
        raise ValueError("invalid feedback launch configuration")
    batch, vocab, dim = state.shape
    width, cache, tiles = indices.shape[1], psi.shape[1], triton.cdiv(vocab, block_vocab)
    if workspace is None:
        mass = torch.empty((batch * width * tiles,), device=state.device)
        stats = torch.empty((batch * width * tiles * 2,), device=state.device)
        alpha = torch.empty((batch, width), device=state.device, dtype=torch.float32)
    else:
        mass, stats = workspace[:2]
        alpha = (workspace[2][:batch * width].view(batch, width) if len(workspace) > 2
                 else torch.empty((batch, width), device=state.device, dtype=torch.float32))
    strides = (*raw.stride(), target.stride(0), target.stride(1), 0 if greedy else target.stride(2),
               *indices.stride(), *contexts.stride(), mapping.stride(0))
    args = (raw, psi, norm, state, target, mapping, indices, contexts, mass, stats)
    constants = (vocab, dim, cache, width, *strides, greedy, eps)
    stats_kernel = _path_stats if strategy == "serial" else _path_stats_parallel
    stats_grid = (tiles, batch) if strategy == "serial" else (tiles, width, batch)
    for mass_only in (True, False):
        stats_kernel[stats_grid](*args, *constants, tiles, triton.next_power_of_2(tiles),
            block_vocab, triton.next_power_of_2(dim), mass_only,
            num_warps=num_warps, enable_fp_fusion=False)
    if strategy == "serial":
        _path_update[(tiles, batch)](*args, alpha, *constants, learning_rate, decay,
            tiles, triton.next_power_of_2(tiles), block_vocab, triton.next_power_of_2(dim),
            num_warps=num_warps, enable_fp_fusion=False)
    else:
        _path_update_parallel[(tiles, batch)](*args, alpha, *constants, learning_rate, decay,
            tiles, triton.next_power_of_2(tiles), block_vocab, triton.next_power_of_2(width),
            triton.next_power_of_2(dim), num_warps=num_warps, enable_fp_fusion=False)
    return alpha
