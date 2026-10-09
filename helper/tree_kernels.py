"""Shared unchanged GPU tree verifier/padding and exact top-k merge."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B","CAP","PAST","TS0","TS1","IS0","IS1","OS0","OS1","PS0","PS1"])
def _pad_schedule(TOKENS,INDICES,LENGTHS,OT,OI,OM,LAST,PACKET,SNAPSHOT,
                   B,CAP,PAST,EOS:tl.constexpr,
                   TS0,TS1,IS0,IS1,OS0,OS1,PS0,PS1,
                   BW:tl.constexpr,BB:tl.constexpr,HAS_SNAPSHOT:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64);slot=tl.arange(0,BW);rows=tl.arange(0,BB)
    lengths=tl.load(LENGTHS+rows,rows<B,other=0);width=tl.max(lengths,axis=0)
    length=tl.load(LENGTHS+batch)
    index=tl.load(INDICES+batch*IS0+slot*IS1,slot<CAP,other=-1)
    valid=slot<length;budget=width-length
    before=tl.minimum(tl.maximum(index-slot,0),budget);destination=slot+before
    candidates=tl.where((destination[None,:]==slot[:,None])&valid[None,:],
                         tl.broadcast_to(slot[None,:],(BW,BW)),BW)
    source=tl.min(candidates,axis=1);accepted=source<BW;safe=tl.minimum(source,CAP-1)
    token=tl.load(TOKENS+batch*TS0+safe*TS1,accepted&(slot<width),other=EOS)
    chosen=tl.load(INDICES+batch*IS0+safe*IS1,accepted&(slot<width),other=0)
    output_ids=tl.where(accepted,chosen+PAST,slot+PAST)
    last=tl.max(tl.where(valid,destination,-1),axis=0)
    tl.store(OT+batch*OS0+slot*OS1,tl.where(accepted,token,EOS),slot<width)
    tl.store(OI+batch*OS0+slot*OS1,output_ids,slot<width)
    tl.store(OM+batch*OS0+slot*OS1,~accepted,slot<width);tl.store(LAST+batch,last)
    eos_token=tl.load(TOKENS+batch*TS0+(length-1)*TS1)
    extension=tl.min(tl.where((slot<width)&(output_ids!=slot+PAST),slot,width),axis=0)
    tl.store(PACKET+batch*PS0,length)
    tl.store(PACKET+batch*PS0+PS1,(eos_token==EOS).to(tl.int64))
    tl.store(PACKET+batch*PS0+2*PS1,extension)
    if HAS_SNAPSHOT:tl.store(PACKET+batch*PS0+(CAP+3)*PS1,tl.load(SNAPSHOT))
    tl.store(PACKET+batch*PS0+(slot+3)*PS1,tl.where((slot<width)&~accepted,output_ids,-1),slot<CAP)


def pad_schedule(path,past,eos,tokens,indices,mask,last,packet,snapshot=None):
    b,cap=path.tokens.shape
    _pad_schedule[(b,)](path.tokens,path.packed_indices,path.lengths,tokens,indices,mask,last,packet,snapshot,
        b,cap,past,eos,*path.tokens.stride(),*path.packed_indices.stride(),*tokens.stride(),*packet.stride(),
        triton.next_power_of_2(cap),triton.next_power_of_2(b),snapshot is not None,num_warps=4)

@triton.jit(do_not_specialize=["CONTEXTS"])
def _proposal_merge(MAX, SUM, VALUES, IDS, PROBS, TOP_IDS, NORM,
                    CONTEXTS, VOCAB: tl.constexpr, K: tl.constexpr,
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
    live = (candidates < TILES * K) & (ids < VOCAB)
    for k in range(K):
        value = tl.max(tl.where(live, values, -float('inf')), axis=0)
        index = tl.min(tl.where(live & (values == value), ids, VOCAB), axis=0)
        tl.store(PROBS + row * K + k, tl.div_rn(tl.exp(value - maximum), total))
        tl.store(TOP_IDS + row * K + k, index)
        # -inf is also a legitimate masked score. Keep selection eligibility
        # separately, or exhausted finite candidates are emitted again at q=0.
        live = live & (ids != index)

@triton.jit(do_not_specialize=["CONTEXTS","ROWS","WIDTH","OS0","OS1"])
def _trace_path(PARENTS, TOKENS, CONTEXTS, SAMPLES, OUT_T, OUT_I, OUT_C, LENGTHS,
                ROWS, WIDTH, EOS: tl.constexpr,
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

@triton.jit(do_not_specialize=["CAPACITY","WIDTH","PAST","TS0","TS1","IS0","IS1","OS0","OS1"])
def _pad_verified_path(TOKENS, INDICES, LENGTHS, OUT_TOKENS, OUT_INDICES, OUT_MASK, LAST,
                       CAPACITY, WIDTH, PAST,
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

@triton.jit(do_not_specialize=["ROWS","PAST","WIDTH"])
def _tree_mask(PARENTS, MASK, ROWS, PAST,
               WIDTH, MINIMUM: tl.constexpr, BK: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1).to(tl.int64)
    columns = tl.program_id(2) * BK + tl.arange(0, BK)
    visible = columns <= PAST  # prefix and root are shared by every query
    current = row.to(tl.int64)  # parent loads are int64; stable loop-carried dtype
    for depth in range(WIDTH):
        visible = visible | ((current >= 0) & (columns == PAST + current))
        current = tl.load(PARENTS + batch * ROWS + tl.maximum(current, 0))
    tl.store(MASK + (batch * ROWS + row) * (PAST + ROWS) + columns,
             tl.where(visible, 0., MINIMUM), columns < PAST + ROWS)

def tree_mask(tree, past_length, mask):
    batch, rows = tree.parents.shape
    _tree_mask[(rows, batch, triton.cdiv(past_length + rows, 256))](tree.parents, mask, rows, past_length, tree.max_depth + 1,
                             torch.finfo(mask.dtype).min, 256, num_warps=4)
