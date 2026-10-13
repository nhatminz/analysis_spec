"""Prompt-indexed kernels ported from read-only SpecNaacl, evaluation only.

Only group-sensitive proposal/reconstruction/update kernels are specialized.
Teacher extraction, union/tail objective (including numerical validation),
proposal normalization, and tree verification reuse analysis_spec helpers.
"""
import torch
import triton
import triton.language as tl
from helper.tree_kernels import _proposal_merge
from helper.opd_reflex_kernels import (
    _feature, _union, _reduce, cache_proposal, prepare_compact_teacher,
    prepare_teacher,
)


def feedback(state,tree,path,target,greedy=False,sampling_metadata=None):
    if not state.grouped or state.train_projector:
        raise ValueError('Only frozen-A prompt-group feedback is supported')
    b,q=tree.parents.shape;n=b*q;k=state.topk;rank=state.rank
    weights=state.selected_weights[:n].view(b,q);kind=state.selected_kind[:n].view(b,q)
    compact=sampling_metadata is not None and len(sampling_metadata)==4
    row_map=state.feedback_row_map
    # Ownership tensors may be replaced by main-stream KV compaction while
    # queued feedback still reads this immutable previous-round mapping.
    state.row_groups.record_stream(torch.cuda.current_stream(state.mapping.device))
    groups=dict(GROUP_IDS=state.row_groups,GROUP_VOCAB=state.vocab,GROUPED=True)
    if sampling_metadata is not None and len(sampling_metadata) in (3,4):
        probs,ids,mass=sampling_metadata[:3]
    else:probs,ids,mass=prepare_teacher(state,tree,path,target,greedy,sampling_metadata)
    head=state.head.weight;bias=state.head.bias if state.head.bias is not None else head
    _selected_head[(k,min(n,32))](state.selected_ids,state.selected_count,state.head_cache,head,bias,
        state.u_cache,state.B_fast,state.ids_cache,state.q_cache,ids,state.norm_cache,weights,mass,
        tree.feedback_contexts,state.teacher_q,row_map,
        tl.bfloat16 if state.logits_dtype==torch.bfloat16 else (tl.float16 if state.logits_dtype==torch.float16 else tl.float32),
        q,state.cache_contexts,head.shape[1],rank,k,state.head.bias is not None,triton.next_power_of_2(head.shape[1]),
        triton.next_power_of_2(rank),triton.next_power_of_2(k),row_map is not None,
        **groups,num_warps=4,enable_fp_fusion=False)
    state.state_stats[:n*10].zero_()
    strides=(0,0,0) if compact else (*target.stride()[:2],0 if greedy else target.stride(2))
    draft_p=sampling_metadata[3] if compact else state.teacher_draft_p
    _union[(min(n,32),)](state.selected_ids,state.selected_count,draft_p if compact else target,
        state.mapping,weights,kind,tree.feedback_contexts,state.ids_cache,state.q_cache,ids,probs,
        state.teacher_q,mass,state.u_cache,state.union_ids,state.union_g,state.state_stats,draft_p,row_map,
        *strides,q,state.cache_contexts,state.vocab,k,rank,10,triton.next_power_of_2(2*k),greedy,
        compact,row_map is not None,num_warps=4,enable_fp_fusion=False)
    _reduce[(10,)](state.state_stats,state.round_weight,state.counters,n,10,triton.next_power_of_2(n),num_warps=4)
    _group_weights[(state.num_groups,)](state.state_stats,row_map,state.row_groups,state.group_weight,
        n,q,triton.next_power_of_2(n),row_map is not None,num_warps=4)
    state.group_feedback_weight.add_(state.group_weight)
    if state.fast_lr>0:
        _update[(min(n,32),)](state.selected_ids,state.selected_count,state.union_ids,state.union_g,
            state.u_cache,tree.feedback_contexts,state.group_weight,state.B_fast,state.bitmap,
            state.active_ids,state.active_count,row_map,q,state.cache_contexts,k,rank,state.fast_lr,
            triton.next_power_of_2(2*k),triton.next_power_of_2(rank),row_map is not None,
            **groups,num_warps=4,enable_fp_fusion=False)
        state.group_updates.add_((state.group_weight>0).long())
    _group_round_end[(1,)](state.active_count,state.group_weight,state.dispatch_count,state.counters,
        state.num_groups,state.fast_lr,triton.next_power_of_2(state.num_groups),num_warps=4)


def feature(hidden,projector,out,head_inputs,state,offset):
    b,c,h=hidden.shape;r=projector.shape[1]
    _feature[(b*c,)](hidden,projector,out,head_inputs,state.head_cache,state.u_cache,
        state.dispatch_count,state.dispatch_snapshot,
        *hidden.stride(),*head_inputs.stride(),c,h,r,state.cache_contexts,offset,
        triton.next_power_of_2(h),triton.next_power_of_2(r),num_warps=8,enable_fp_fusion=False)

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","THRESHOLD","CAPACITY"])
def _sparse_scores(Z,U,B,ACTIVE,COUNT,SCORES,SLOTS,CAPACITY,COUNTERS,ZS0,ZS1,ZS2,
                   C,V:tl.constexpr,R:tl.constexpr,THRESHOLD,
                   MODE:tl.constexpr,ROOT:tl.constexpr,BS:tl.constexpr,
                   GROUP_IDS=None,GROUP_FIRST=None,GROUPED:tl.constexpr=False):
    row=tl.program_id(0).to(tl.int64)
    group=tl.load(GROUP_IDS+row//C).to(tl.int64) if GROUPED else 0
    count=tl.load(COUNT+group)
    use_sparse=(MODE==0)|((MODE==2)&(count<THRESHOLD))
    if use_sparse:
        s=tl.arange(0,BS)
        for start in range(0,count,BS):
            ids=tl.load(ACTIVE+group*V+start+s,start+s<count,other=0).to(tl.int64)
            dot=tl.full((BS,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+(group*V+ids)*R+r,start+s<count,other=0)
                u=tl.load(U+row*R+r)
                dot=dot+w*u
            raw=tl.load(Z+row//C*ZS0+row%C*ZS1+ids*ZS2,start+s<count,other=0).to(tl.float32)
            tl.store(SCORES+row*CAPACITY+start+s,raw+dot,start+s<count)
            first=tl.load(GROUP_FIRST+group)*C if GROUPED else 0
            if row==first:tl.store(SLOTS+group*V+ids,start+s,start+s<count)
        if ROOT:
            if row==0:tl.atomic_add(COUNTERS+13,1.)

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","N","C","THRESHOLD"])
def _dense_gemm(Z,U,B,COUNT,SCORES,COUNTERS,ZS0,ZS1,ZS2,
                N,C,V:tl.constexpr,R:tl.constexpr,
                THRESHOLD,MODE:tl.constexpr,ROOT:tl.constexpr,
                BV:tl.constexpr,BC:tl.constexpr,GROUP_IDS=None,GROUPED:tl.constexpr=False):
    # Tiled rank GEMM U @ B.T; reuse each B tile across BC contexts.
    # The explicit ordered FP32 multiply/add is IDENTICAL to sparse dot. No
    # TF32/FMA/BLAS association drift is allowed merely to switch strategy.
    tile,ct=tl.program_id(0),tl.program_id(1)
    count=tl.load(COUNT)
    use_dense=(MODE==1)|((MODE==2)&(count>=THRESHOLD))
    if use_dense:
        row=ct*BC+tl.arange(0,BC);v=tile*BV+tl.arange(0,BV)
        dot=tl.full((BC,BV),0.,tl.float32)
        for r in tl.static_range(R):
            u=tl.load(U+row*R+r,row<N,other=0)
            if GROUPED:
                group=tl.load(GROUP_IDS+row//C,row<N,other=0).to(tl.int64)
                w=tl.load(B+(group[:,None]*V+v[None,:])*R+r,(row[:,None]<N)&(v[None,:]<V),other=0)
                dot=dot+u[:,None]*w
            else:
                w=tl.load(B+v*R+r,v<V,other=0)
                dot=dot+u[:,None]*w[None,:]
        raw=tl.load(Z+(row//C)[:,None]*ZS0+(row%C)[:,None]*ZS1+v[None,:]*ZS2,
                    (row[:,None]<N)&(v[None,:]<V),other=0).to(tl.float32)
        tl.store(SCORES+row[:,None]*V+v[None,:],raw+dot,(row[:,None]<N)&(v[None,:]<V))
        if ROOT:
            if (tile==0)&(ct==0):tl.atomic_add(COUNTERS+14,1.)

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","ENABLED","CAPACITY"])
def _corrected_scan(Z,SCORES,BITS,SLOTS,CAPACITY,MAX,SUM,VALUES,IDS,U,B,COUNTERS,ZS0,ZS1,ZS2,
                     C,V:tl.constexpr,K:tl.constexpr,TILES:tl.constexpr,
                     BV:tl.constexpr,ENABLED,R:tl.constexpr,DENSE:tl.constexpr,ROOT:tl.constexpr,SPARSE:tl.constexpr,GROUP_IDS=None,GROUPED:tl.constexpr=False):
    tile,context,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2).to(tl.int64)
    group=tl.load(GROUP_IDS+batch).to(tl.int64) if GROUPED else 0
    v=tile*BV+tl.arange(0,BV)
    raw=tl.load(Z+batch*ZS0+context*ZS1+v*ZS2,v<V,other=0).to(tl.float32)
    if ENABLED:
        if DENSE:
            dot=tl.full((BV,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+(group*V+v)*R+r,v<V,other=0)
                u=tl.load(U+(batch*C+context)*R+r)
                dot=dot+w*u
            raw=raw+dot
            if ROOT:
                if (tile==0)&(context==0)&(batch==0):tl.atomic_add(COUNTERS+14,1.)
        else:
            bits=tl.load(BITS+group*triton.cdiv(V,32)+v//32,v<V,other=0)
            active=(v<V)&(((bits>>(v%32))&1)!=0)
            if SPARSE:
                slots=tl.load(SLOTS+group*V+v,active,other=0)
                corrected=tl.load(SCORES+(batch*C+context)*CAPACITY+slots,active,other=0)
            else:corrected=tl.load(SCORES+(batch*C+context)*V+v,active,other=0)
            raw=tl.where(active,corrected,raw)
    z=tl.where(v<V,raw,-float('inf'));live=v<V
    maximum=tl.max(z,axis=0)
    # An all-masked tile has mass zero, not exp(-inf - -inf)=NaN.
    # Finite tiles keep exactly the previous subtraction/reduction semantics.
    shift=tl.where(maximum==-float('inf'),0.,maximum)
    # Fixed pairwise FP32 tree: tl.sum alone changes association with Triton's
    # elements-per-thread layout (strided B loads vs contiguous score loads).
    # Pair reductions make strategy switching bitwise stable without another
    # vocabulary pass or global workspace (also supported by Triton 3.1).
    mass=tl.where(live,tl.exp(z-shift),0.)
    tl.static_assert(BV==256)
    for level in tl.static_range(8): # BV=256 in both proposal implementations
        mass=tl.sum(tl.reshape(mass,(BV//(1<<(level+1)),2)),axis=1)
    total=tl.sum(mass,axis=0)
    offset=(batch*C+context)*TILES+tile
    tl.store(MAX+offset,maximum);tl.store(SUM+offset,total)
    for j in range(K):
        score=tl.max(z,axis=0);token=tl.min(tl.where((z==score)&live,v,V),axis=0)
        tl.store(VALUES+offset*K+j,score);tl.store(IDS+offset*K+j,token)
        z=tl.where(v==token,-float('inf'),z);live=live&(v!=token)

def prepare_scores(logits,u,state,root=False,selected=None):
    b,c,v=logits.shape
    groups=dict(GROUP_IDS=state.row_groups,GROUPED=True) if state.grouped else {}
    mode=0 if (selected or state.selected_proposal_backend(b,c))=='sparse' else 1
    if mode==0:
        _sparse_scores[(b*c,)](logits,u,state.B_fast,state.active_ids,state.active_count,state.sparse_scores,state.active_slots,state.sparse_capacity,
            state.counters,*logits.stride(),c,v,state.rank,0,mode,root,128,
            **groups,**(dict(GROUP_FIRST=state.group_first) if state.grouped else {}),num_warps=4,enable_fp_fusion=False)
    else:
        _dense_gemm[(triton.cdiv(v,128),triton.cdiv(b*c,4))](logits,u,state.B_fast,state.active_count,
            state.score_workspace,state.counters,*logits.stride(),b*c,c,v,state.rank,0,mode,root,128,4,
            **groups,num_warps=4,enable_fp_fusion=False)

def propose(logits,u,state,k,workspace,outputs,enabled,root=False):
    b,c,v=logits.shape;tiles=triton.cdiv(v,256)
    maxima,sums,values,ids=[p[:b*c*tiles*(k if i>=2 else 1)] for i,p in enumerate(workspace)]
    out,indices,norm=outputs
    selected='sparse' if state.selected_proposal_backend(b,c)=='sparse' else state.selected_dense_implementation(b,c)
    sparse=selected=='sparse'
    dense=enabled and selected=='fused'
    if root and enabled:
        # Selection is already known on host. No extra GPU atomics, kernel,
        # scalar read or synchronization merely for backend-specific logging.
        if selected=='fused':state.host_fused_rounds+=1
        elif selected=='gemm':state.host_gemm_rounds+=1
    if enabled:
        state.prepare_proposal_workspace(b,c,selected)
        ticket=state.begin('opd_proposal_extra_ms')
        if not dense:prepare_scores(logits,u,state,root,selected)
        state.end(ticket)
    elif root and state.enabled:
        # Cold raw path has zero B. No correction preparation at all.
        state.counters[13:14].add_(1)
    _corrected_scan[(tiles,c,b)](logits,state.sparse_scores if sparse else state.score_workspace,state.bitmap,state.active_slots,state.sparse_capacity,maxima,sums,values,ids,
        u,state.B_fast,state.counters if state.enabled else None,
        *logits.stride(),c,v,k,tiles,256,enabled,state.rank,dense,root,sparse,
        **(dict(GROUP_IDS=state.row_groups,GROUPED=True) if state.grouped else {}),num_warps=4,enable_fp_fusion=False)
    _proposal_merge[(b*c,)](maxima,sums,values,ids,out,indices,norm,c,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),num_warps=4,enable_fp_fusion=False)

@triton.jit(do_not_specialize=["ROWS","CACHE"])
def _selected_head(SELECTED,SELECTED_COUNT,H,HEAD,BIAS,U,B,D_IDS,D_Q,T_IDS,NORM,W,MASS,CONTEXTS,OUT,ROW_MAP,
                   HEAD_DTYPE:tl.constexpr,ROWS,CACHE,
                   HIDDEN:tl.constexpr,R:tl.constexpr,K:tl.constexpr,
                   HAS_BIAS:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr,BK:tl.constexpr,HAS_ROW_MAP:tl.constexpr,
                    GROUP_IDS=None,GROUP_VOCAB:tl.constexpr=0,GROUPED:tl.constexpr=False):
    j=tl.program_id(0)
    worker=tl.program_id(1);stride=tl.num_programs(1)
    count=tl.load(SELECTED_COUNT)
    for ordinal in range(worker,count,stride):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        batch=tl.load(ROW_MAP+state//ROWS).to(tl.int64) if HAS_ROW_MAP else state//ROWS
        weight=tl.load(W+state);context=tl.load(CONTEXTS+state)
        token=tl.load(T_IDS+state*K+j)
        k=tl.arange(0,BK);cached_ids=tl.load(D_IDS+(batch*CACHE+tl.maximum(context,0))*K+k,k<K,other=-1)
        matches=cached_ids==token;found=tl.sum(matches.to(tl.int32),axis=0)>0
        mass=tl.load(MASS+state)
        if (weight>0)&(context>=0)&(mass>0)&(mass<float('inf'))&(token>=0):
            if found:
                q=tl.sum(tl.where(matches,tl.load(D_Q+(batch*CACHE+context)*K+k,k<K,other=0),0.),axis=0)
            else:
                h,r=tl.arange(0,BH),tl.arange(0,BR)
                x=tl.load(H+(batch*CACHE+context)*HIDDEN+h,h<HIDDEN,other=0).to(HEAD_DTYPE).to(tl.float32)
                row=tl.load(HEAD+token*HIDDEN+h,h<HIDDEN,other=0).to(HEAD_DTYPE).to(tl.float32)
                raw=tl.sum(x*row,axis=0)
                if HAS_BIAS:raw=raw+tl.load(BIAS+token).to(HEAD_DTYPE).to(tl.float32)
                # Shared FastGRPO head returns the model dtype before FP32 softmax.
                raw=raw.to(HEAD_DTYPE).to(tl.float32)
                u=tl.load(U+(batch*CACHE+context)*R+r,r<R,other=0)
                group=tl.load(GROUP_IDS+batch).to(tl.int64) if GROUPED else 0
                adapter=tl.load(B+(group*GROUP_VOCAB+token)*R+r,r<R,other=0)
                z=raw+tl.sum(adapter*u,axis=0)
                maximum=tl.load(NORM+(batch*CACHE+context)*2)
                total=tl.load(NORM+(batch*CACHE+context)*2+1)
                q=tl.div_rn(tl.exp(z-maximum),total)
        else:q=0.
        tl.store(OUT+state*K+j,q)

@triton.jit(do_not_specialize=["ROWS","CACHE"])
def _update(SELECTED,SELECTED_COUNT,IDS,G,U,CONTEXTS,ROUND_WEIGHT,B,BITS,ACTIVE,COUNT,
             ROW_MAP,
             ROWS,CACHE,K:tl.constexpr,R:tl.constexpr,
             LR:tl.constexpr,BU:tl.constexpr,BR:tl.constexpr,HAS_ROW_MAP:tl.constexpr,
             GROUP_IDS=None,GROUP_VOCAB:tl.constexpr=0,GROUPED:tl.constexpr=False):
    worker=tl.program_id(0);stride=tl.num_programs(0)
    count=tl.load(SELECTED_COUNT)
    for ordinal in range(worker,count,stride):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        batch=tl.load(ROW_MAP+state//ROWS).to(tl.int64) if HAS_ROW_MAP else state//ROWS
        j,r=tl.arange(0,BU),tl.arange(0,BR)
        token=tl.load(IDS+state*2*K+j,j<2*K,other=-1)
        g=tl.load(G+state*2*K+j,j<2*K,other=0)
        context=tl.maximum(tl.load(CONTEXTS+state),0)
        u=tl.load(U+(batch*CACHE+context)*R+r,r<R,other=0)
        group=tl.load(GROUP_IDS+batch).to(tl.int64) if GROUPED else 0
        denominator=tl.load(ROUND_WEIGHT+group)
        delta=-LR*tl.div_rn(g[:,None]*u[None,:],tl.where(denominator>0,denominator,1.))
        valid=(j<2*K)&(token>=0)&(denominator>0)
        tl.atomic_add(B+(group*GROUP_VOCAB+tl.maximum(token[:,None],0))*R+r[None,:],delta,valid[:,None]&(r[None,:]<R))
        changed=valid&(tl.sum((delta!=0).to(tl.int32),axis=1)>0)
        bit=1<<(token%32)
        old=tl.atomic_or(BITS+group*triton.cdiv(GROUP_VOCAB,32)+tl.maximum(token,0)//32,bit,changed)
        first=changed&((old&bit)==0)
        slot=tl.atomic_add(COUNT+group+tl.zeros((BU,),tl.int32),1,first)
        tl.store(ACTIVE+group*GROUP_VOCAB+slot,token,first)

@triton.jit(do_not_specialize=['N','ROWS'])
def _group_weights(STATS,ROW_MAP,GROUP_IDS,WEIGHT,N,ROWS,BN:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
    group=tl.program_id(0);n=tl.arange(0,BN)
    physical=tl.load(ROW_MAP+n//ROWS,n<N,other=0).to(tl.int64) if HAS_ROW_MAP else n//ROWS
    owner=tl.load(GROUP_IDS+physical,n<N,other=-1)
    w=tl.load(STATS+n*10,n<N,other=0.)
    tl.store(WEIGHT+group,tl.sum(tl.where(owner==group,w,0.),axis=0))

@triton.jit
def _group_round_end(COUNT,WEIGHT,DISPATCH,COUNTERS,GROUPS:tl.constexpr,LR:tl.constexpr,BG:tl.constexpr):
    g=tl.arange(0,BG)
    count=tl.load(COUNT+g,g<GROUPS,other=0)
    weight=tl.load(WEIGHT+g,g<GROUPS,other=0.)
    updates=tl.sum(((weight>0)&(LR>0)).to(tl.float64),axis=0)
    total=tl.sum(count,axis=0).to(tl.float64)
    tl.store(DISPATCH,tl.max(count,axis=0))
    tl.atomic_add(COUNTERS+9,updates)
    tl.atomic_add(COUNTERS+10,total)
    tl.atomic_add(COUNTERS+11,1.)
    tl.atomic_max(COUNTERS+15,total)
