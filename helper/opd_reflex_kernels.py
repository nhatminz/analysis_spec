"""Inference-only OPD kernels. Shared B, full target vocabulary, no model/autograd/RNG.

Proposal scans inactive raw logits and ONLY S active adapter rows, then merges
exact global top-k. No [contexts,V,r] correction or probability cache exists.
Feedback reads the existing post-sampling teacher probabilities once in the
selected states; never another target softmax/sort or transformer forward.
"""
import torch
import triton
import triton.language as tl
from helper.tree_kernels import _proposal_merge


@triton.jit(do_not_specialize=["HS0","HS1","HS2","NS0","NS1","NS2","C","CACHE","OFFSET"])
def _feature(H,A,U,HEAD_IN,HC,UC,COUNT,SNAPSHOT,HS0,HS1,HS2,NS0,NS1,NS2,C,HIDDEN:tl.constexpr,R:tl.constexpr,
             CACHE,OFFSET,BH:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    if (row==0)&(OFFSET==0):tl.store(SNAPSHOT,tl.load(COUNT))
    h,r=tl.arange(0,BH),tl.arange(0,BR)
    x=tl.load(H+row//C*HS0+row%C*HS1+h*HS2,h<HIDDEN,other=0).to(tl.float32)
    a=tl.load(A+h[:,None]*R+r[None,:],(h[:,None]<HIDDEN)&(r[None,:]<R),other=0)
    projected=tl.sum(x[:,None]*a,axis=0)
    cached=row//C*CACHE+OFFSET+row%C
    tl.store(U+row*R+r,projected,r<R);tl.store(UC+cached*R+r,projected,r<R)
    inputs=tl.load(HEAD_IN+row//C*NS0+row%C*NS1+h*NS2,h<HIDDEN,other=0)
    tl.store(HC+cached*HIDDEN+h,inputs,h<HIDDEN)


def feature(hidden,projector,out,head_inputs,state,offset):
    b,c,h=hidden.shape;r=projector.shape[1]
    _feature[(b*c,)](hidden,projector,out,head_inputs,state.head_cache,state.u_cache,
        state.active_count,state.dispatch_snapshot,
        *hidden.stride(),*head_inputs.stride(),c,h,r,state.cache_contexts,offset,
        triton.next_power_of_2(h),triton.next_power_of_2(r),num_warps=8,enable_fp_fusion=False)



@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","THRESHOLD","CAPACITY"])
def _sparse_scores(Z,U,B,ACTIVE,COUNT,SCORES,SLOTS,CAPACITY,COUNTERS,ZS0,ZS1,ZS2,
                   C,V:tl.constexpr,R:tl.constexpr,THRESHOLD,
                   MODE:tl.constexpr,ROOT:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);count=tl.load(COUNT)
    use_sparse=(MODE==0)|((MODE==2)&(count<THRESHOLD))
    if use_sparse:
        s=tl.arange(0,BS)
        for start in range(0,count,BS):
            ids=tl.load(ACTIVE+start+s,start+s<count,other=0).to(tl.int64)
            dot=tl.full((BS,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+ids*R+r,start+s<count,other=0)
                u=tl.load(U+row*R+r)
                dot=dot+w*u
            raw=tl.load(Z+row//C*ZS0+row%C*ZS1+ids*ZS2,start+s<count,other=0).to(tl.float32)
            tl.store(SCORES+row*CAPACITY+start+s,raw+dot,start+s<count)
            if row==0:tl.store(SLOTS+ids,start+s,start+s<count)
        if ROOT:
            if row==0:tl.atomic_add(COUNTERS+13,1.)

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","N","C","THRESHOLD"])
def _dense_gemm(Z,U,B,COUNT,SCORES,COUNTERS,ZS0,ZS1,ZS2,
                N,C,V:tl.constexpr,R:tl.constexpr,
                THRESHOLD,MODE:tl.constexpr,ROOT:tl.constexpr,
                BV:tl.constexpr,BC:tl.constexpr):
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
                     BV:tl.constexpr,ENABLED,R:tl.constexpr,DENSE:tl.constexpr,ROOT:tl.constexpr,SPARSE:tl.constexpr):
    tile,context,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2).to(tl.int64)
    v=tile*BV+tl.arange(0,BV)
    raw=tl.load(Z+batch*ZS0+context*ZS1+v*ZS2,v<V,other=0).to(tl.float32)
    if ENABLED:
        if DENSE:
            dot=tl.full((BV,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+v*R+r,v<V,other=0)
                u=tl.load(U+(batch*C+context)*R+r)
                dot=dot+w*u
            raw=raw+dot
            if ROOT:
                if (tile==0)&(context==0)&(batch==0):tl.atomic_add(COUNTERS+14,1.)
        else:
            bits=tl.load(BITS+v//32,v<V,other=0)
            active=(v<V)&(((bits>>(v%32))&1)!=0)
            if SPARSE:
                slots=tl.load(SLOTS+v,active,other=0)
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
    mode=0 if (selected or state.selected_proposal_backend(b,c))=='sparse' else 1
    if mode==0:
        _sparse_scores[(b*c,)](logits,u,state.B_fast,state.active_ids,state.active_count,state.sparse_scores,state.active_slots,state.sparse_capacity,
            state.counters,*logits.stride(),c,v,state.rank,0,mode,root,128,
            num_warps=4,enable_fp_fusion=False)
    else:
        _dense_gemm[(triton.cdiv(v,128),triton.cdiv(b*c,4))](logits,u,state.B_fast,state.active_count,
            state.score_workspace,state.counters,*logits.stride(),b*c,c,v,state.rank,0,mode,root,128,4,
            num_warps=4,enable_fp_fusion=False)

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
        *logits.stride(),c,v,k,tiles,256,enabled,state.rank,dense,root,sparse,num_warps=4,enable_fp_fusion=False)
    _proposal_merge[(b*c,)](maxima,sums,values,ids,out,indices,norm,c,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),num_warps=4,enable_fp_fusion=False)

@triton.jit(do_not_specialize=["C","CACHE","OFFSET"])
def _cache_proposal(Q,IDS,NORM,QC,IC,NC,C,CACHE,
                    OFFSET,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);k=tl.arange(0,BK);n=tl.arange(0,2)
    cached=row//C*CACHE+OFFSET+row%C
    tl.store(QC+cached*K+k,tl.load(Q+row*K+k,k<K,other=0),k<K)
    tl.store(IC+cached*K+k,tl.load(IDS+row*K+k,k<K,other=0),k<K)
    tl.store(NC+cached*2+n,tl.load(NORM+row*2+n))


def cache_proposal(q,ids,norm,state,offset):
    b,c,k=q.shape
    _cache_proposal[(b*c,)](q,ids,norm,state.q_cache,state.ids_cache,state.norm_cache,
        c,state.cache_contexts,offset,k,triton.next_power_of_2(k),num_warps=4)


@triton.jit(do_not_specialize=["ROWS","WIDTH","PS0","PS1"])
def _select(PARENTS,CONTEXTS,PATH,WEIGHTS,KIND,ROWS,WIDTH,
            PS0,PS1,VW:tl.constexpr,FW:tl.constexpr,BR:tl.constexpr,BW:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64)
    row,pathslot=tl.arange(0,BR),tl.arange(0,BW)
    visited=tl.load(PATH+batch*PS0+pathslot*PS1,pathslot<WIDTH,other=-2)
    parent=tl.load(PARENTS+batch*ROWS+row,row<ROWS,other=-2)
    context=tl.load(CONTEXTS+batch*ROWS+row,row<ROWS,other=-1)
    on_path=tl.sum((row[:,None]==visited[None,:]).to(tl.int32),axis=1)>0
    parent_visited=tl.sum((parent[:,None]==visited[None,:]).to(tl.int32),axis=1)>0
    frontier=~on_path&parent_visited&(row>0)
    kind=tl.where(on_path,1,tl.where(frontier,2,0))
    weight=tl.where(on_path,VW,tl.where(frontier,FW,0.))
    valid=(row<ROWS)&(context>=0)&(weight>0)
    tl.store(WEIGHTS+batch*ROWS+row,tl.where(valid,weight,0.),row<ROWS)
    tl.store(KIND+batch*ROWS+row,tl.where(valid,kind,0),row<ROWS)


def select_states(tree,path,weights,kind,visited_weight,frontier_weight):
    b,q=tree.parents.shape
    _select[(b,)](tree.parents,tree.feedback_contexts,path.packed_indices,weights,kind,q,path.packed_indices.shape[1],
        *path.packed_indices.stride(),visited_weight,frontier_weight,triton.next_power_of_2(q),
        triton.next_power_of_2(path.packed_indices.shape[1]),num_warps=4)


@triton.jit(do_not_specialize=['N'])
def _compact_selected(W,SELECTED,COUNT,N,BN:tl.constexpr):
    row=tl.arange(0,BN)
    live=tl.load(W+row,row<N,other=0)>0
    offsets=tl.cumsum(live.to(tl.int32))-1
    tl.store(SELECTED+tl.maximum(offsets,0),row,live)
    tl.store(COUNT,tl.sum(live.to(tl.int32)))


@triton.jit(do_not_specialize=["TS0","TS1","TS2","ROWS"])
def _teacher_scan(TARGET,MAP,SELECTED,COUNT,MAX,SUM,VALUES,IDS,TS0,TS1,TS2,
                  ROWS,V:tl.constexpr,K:tl.constexpr,TILES:tl.constexpr,
                  BV:tl.constexpr,GREEDY:tl.constexpr):
    tile=tl.program_id(0)
    v=tile*BV+tl.arange(0,BV)
    count=tl.load(COUNT)
    for ordinal in range(tl.program_id(1),count,tl.num_programs(1)):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        target_id=tl.load(MAP+v,v<V,other=0)
        if GREEDY:
            p=((target_id==tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1))&(v<V)).to(tl.float32)
        else:
            p=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1+target_id*TS2,v<V,other=0).to(tl.float32)
        offset=state*TILES+tile
        tl.store(MAX+offset,0.);tl.store(SUM+offset,tl.sum(p,axis=0))
        live=(v<V)&(p>0)
        score=tl.where(live,p,-float('inf'))
        for j in range(K):
            value=tl.max(score,axis=0);token=tl.min(tl.where((score==value)&live,v,V),axis=0)
            tl.store(VALUES+offset*K+j,value);tl.store(IDS+offset*K+j,token)
            score=tl.where(v==token,-float('inf'),score);live=live&(v!=token)


def teacher(target,mapping,weights,topk,pools,outputs,greedy=False,selection=None,capture=None):
    b,q=weights.shape;v=mapping.numel();tiles=triton.cdiv(v,256);n=b*q;k=min(topk,v)
    maxima,sums,values,ids=[p[:n*tiles*(k if i>=2 else 1)] for i,p in enumerate(pools)]
    probs,indices,norm=outputs
    if selection is None: # standalone tests; production always reuses its pool
        selected=torch.empty(n,device=weights.device,dtype=torch.int32)
        count=torch.empty(1,device=weights.device,dtype=torch.int32)
    else:selected,count=selection
    _compact_selected[(1,)](weights,selected,count,n,triton.next_power_of_2(n),num_warps=4)
    _teacher_scan[(tiles,min(n,32))](target,mapping,selected,count,maxima,sums,values,ids,
        target.stride(0),target.stride(1),0 if greedy else target.stride(2),q,v,k,tiles,256,greedy,num_warps=4)
    # _proposal_merge exponentiates score values; teacher top-k instead needs
    # literal probability ranking and division by compact mass.
    _teacher_merge[(n,)](weights,sums,values,ids,probs,indices,norm,n,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),**(capture or {}),num_warps=4)


@triton.jit(do_not_specialize=["N","D_ROWS","D_CACHE","D_TS0","D_TS1","D_TS2"])
def _teacher_merge(W,SUM,VALUES,IDS,P,OUT_IDS,MASS,N,V:tl.constexpr,K:tl.constexpr,
                   TILES:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr,
                   CAPTURE:tl.constexpr=False,D_TARGET=None,D_MAP=None,D_IDS=None,D_CONTEXTS=None,D_ROW_MAP=None,D_OUT=None,
                   D_ROWS=0,D_CACHE=0,D_TS0=0,D_TS1=0,D_TS2=0,D_GREEDY:tl.constexpr=False,D_HAS_ROW_MAP:tl.constexpr=False):
    state=tl.program_id(0).to(tl.int64);t=tl.arange(0,BT);candidate=tl.arange(0,BK)
    selected=tl.load(W+state)>0
    if selected:
        mass=tl.sum(tl.load(SUM+state*TILES+t,(t<TILES)&selected,other=0),axis=0)
        valid_mass=(mass>0)&(mass<float('inf'))
        tl.store(MASS+state,mass)
        if CAPTURE:
            _capture_draft_coordinates(state,D_TARGET,D_MAP,D_IDS,D_CONTEXTS,D_ROW_MAP,D_OUT,
                D_ROWS,D_CACHE,D_TS0,D_TS1,D_TS2,K,triton.next_power_of_2(K),D_GREEDY,D_HAS_ROW_MAP)
        value=tl.load(VALUES+state*TILES*K+candidate,(candidate<TILES*K)&selected,other=-float('inf'))
        ids=tl.load(IDS+state*TILES*K+candidate,(candidate<TILES*K)&selected,other=V)
        for j in range(K):
            best=tl.max(value,axis=0);token=tl.min(tl.where(value==best,ids,V),axis=0)
            positive=valid_mass&(best>0)&(token<V)
            tl.store(P+state*K+j,tl.where(positive,tl.div_rn(best,tl.where(valid_mass,mass,1.)),0.))
            tl.store(OUT_IDS+state*K+j,tl.where(positive,token,-1))
            value=tl.where(ids==token,-float('inf'),value)
    else:
        unused=tl.arange(0,triton.next_power_of_2(K))
        tl.store(MASS+state,0.)
        tl.store(P+state*K+unused,0.,unused<K)
        tl.store(OUT_IDS+state*K+unused,-1,unused<K)


@triton.jit(do_not_specialize=["ROWS","CACHE"])
def _selected_head(SELECTED,SELECTED_COUNT,H,HEAD,BIAS,U,B,D_IDS,D_Q,T_IDS,NORM,W,MASS,CONTEXTS,OUT,ROW_MAP,
                   HEAD_DTYPE:tl.constexpr,ROWS,CACHE,
                   HIDDEN:tl.constexpr,R:tl.constexpr,K:tl.constexpr,
                   HAS_BIAS:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr,BK:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
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
                adapter=tl.load(B+token*R+r,r<R,other=0)
                z=raw+tl.sum(adapter*u,axis=0)
                maximum=tl.load(NORM+(batch*CACHE+context)*2)
                total=tl.load(NORM+(batch*CACHE+context)*2+1)
                q=tl.div_rn(tl.exp(z-maximum),total)
        else:q=0.
        tl.store(OUT+state*K+j,q)


@triton.jit(do_not_specialize=["TS0","TS1","TS2","ROWS","CACHE"])
def _union(SELECTED,SELECTED_COUNT,TARGET,MAP,W,KIND,CONTEXTS,D_IDS,D_Q,T_IDS,T_P,T_Q,MASS,
            OUT_IDS,OUT_G,STATS,DRAFT_P,ROW_MAP,TS0,TS1,TS2,ROWS,CACHE,
            V:tl.constexpr,K:tl.constexpr,FIELDS:tl.constexpr,BU:tl.constexpr,GREEDY:tl.constexpr,
            COMPACT:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
    worker=tl.program_id(0);stride=tl.num_programs(0)
    count=tl.load(SELECTED_COUNT)
    for ordinal in range(worker,count,stride):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        batch=tl.load(ROW_MAP+state//ROWS).to(tl.int64) if HAS_ROW_MAP else state//ROWS
        j=tl.arange(0,BU);first=j<K
        context=tl.maximum(tl.load(CONTEXTS+state),0);cache=batch*CACHE+context
        kk=j%K
        d_ids=tl.load(D_IDS+cache*K+kk,j<2*K,other=-1)
        token=tl.where(first,d_ids,tl.load(T_IDS+state*K+kk,j<2*K,other=-1))
        # Each top-k is unique. Only target additions require cross-set dedup.
        dk=tl.arange(0,triton.next_power_of_2(K))
        draft_ids=tl.load(D_IDS+cache*K+dk,dk<K,other=-2)
        overlap=tl.sum((token[:,None]==draft_ids[None,:]).to(tl.int32),axis=1)>0
        mass=tl.load(MASS+state);weight=tl.load(W+state)
        good=(mass>0)&(mass<float('inf'))
        target_positive=(token>=0)&(tl.load(T_P+state*K+kk,j<2*K,other=0)>0)
        valid=(j<2*K)&(first|(~overlap&target_positive))&(weight>0)&good
        if COMPACT:
            raw=tl.load(DRAFT_P+state*K+kk,first&valid,other=0.)
            normalized=tl.div_rn(raw,tl.where(good,mass,1.))
            p=tl.where(first,normalized,tl.load(T_P+state*K+kk,valid,other=0.))
            p=tl.where(valid,p,0.)
        else:
            target_id=tl.load(MAP+tl.maximum(token,0),j<2*K,other=0)
            if GREEDY:
                p=(target_id==tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1)).to(tl.float32)
            else:p=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1+target_id*TS2,valid,other=0).to(tl.float32)
            p=tl.where(valid,tl.div_rn(p,tl.where(good,mass,1.)),0.)
        q=tl.where(first,tl.load(D_Q+cache*K+kk,j<2*K,other=0),tl.load(T_Q+state*K+kk,j<2*K,other=0))
        q=tl.where(valid,q,0.)
        p_tail=tl.maximum(1.-tl.sum(p,axis=0),0.);q_tail=tl.maximum(1.-tl.sum(q,axis=0),0.)
        empty_tail=tl.sum(valid.to(tl.int32),axis=0)==V
        p_tail=tl.where(empty_tail,0.,p_tail);q_tail=tl.where(empty_tail,0.,q_tail)
        # Exact categorical KL, with 0*log(0/q)=0. Infinite KL is not hidden.
        kl=tl.sum(tl.where(p>0,p*(tl.log(p)-tl.log(q)),0.),axis=0)
        kl=kl+tl.where(p_tail>0,p_tail*(tl.log(p_tail)-tl.log(q_tail)),0.)
        w=tl.where(good,weight,0.)
        tl.store(OUT_IDS+state*2*K+j,tl.where(valid,token,-1),j<2*K)
        tl.store(OUT_G+state*2*K+j,tl.where(valid,w*(q-p),0.),j<2*K)
        kind=tl.load(KIND+state);selected=w>0
        # Per-state summaries; one batched reduction feeds metrics and SGD denominator.
        tl.store(STATS+state*FIELDS+0,w)
        tl.store(STATS+state*FIELDS+1,selected.to(tl.float32))
        tl.store(STATS+state*FIELDS+2,(selected&(kind==1)).to(tl.float32))
        tl.store(STATS+state*FIELDS+3,(selected&(kind==2)).to(tl.float32))
        finite=(kl==kl)&(tl.abs(kl)<float('inf'))
        tl.store(STATS+state*FIELDS+4,tl.where(selected&finite,w*kl,0.))
        tl.store(STATS+state*FIELDS+5,tl.sum(valid.to(tl.float32),axis=0))
        tl.store(STATS+state*FIELDS+6,tl.where(selected,mass,0.))
        tl.store(STATS+state*FIELDS+7,tl.sum(tl.where(first,p,0.),axis=0))
        tl.store(STATS+state*FIELDS+8,((weight>0)&~good).to(tl.float32))
        tl.store(STATS+state*FIELDS+9,(selected&~finite).to(tl.float32))


@triton.jit(do_not_specialize=["N"])
def _reduce(STATS,ROUND_WEIGHT,COUNTERS,N,FIELDS:tl.constexpr,BN:tl.constexpr):
    field=tl.program_id(0);n=tl.arange(0,BN)
    total=tl.sum(tl.load(STATS+n*FIELDS+field,n<N,other=0),axis=0)
    if field==0:tl.store(ROUND_WEIGHT,total)
    tl.atomic_add(COUNTERS+tl.where(field==9,12,field),total.to(tl.float64))


@triton.jit(do_not_specialize=["ROWS","CACHE"])
def _update(SELECTED,SELECTED_COUNT,IDS,G,U,CONTEXTS,ROUND_WEIGHT,B,BITS,ACTIVE,COUNT,
             ROW_MAP,
             ROWS,CACHE,K:tl.constexpr,R:tl.constexpr,
             LR:tl.constexpr,BU:tl.constexpr,BR:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
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
        denominator=tl.load(ROUND_WEIGHT)
        delta=-LR*tl.div_rn(g[:,None]*u[None,:],tl.where(denominator>0,denominator,1.))
        valid=(j<2*K)&(token>=0)&(denominator>0)
        tl.atomic_add(B+tl.maximum(token[:,None],0)*R+r[None,:],delta,valid[:,None]&(r[None,:]<R))
        changed=valid&(tl.sum((delta!=0).to(tl.int32),axis=1)>0)
        bit=1<<(token%32)
        old=tl.atomic_or(BITS+tl.maximum(token,0)//32,bit,changed)
        first=changed&((old&bit)==0)
        slot=tl.atomic_add(COUNT+tl.zeros((BU,),tl.int32),1,first)
        tl.store(ACTIVE+slot,token,first)


@triton.jit(do_not_specialize=[])
def _round_end(B,BITS,ACTIVE,COUNT,WEIGHT,COUNTERS,R:tl.constexpr,
               LR:tl.constexpr,BS:tl.constexpr,BR:tl.constexpr):
    weight=tl.load(WEIGHT);count=tl.load(COUNT)
    tl.atomic_add(COUNTERS+9,((weight>0)&(LR>0)).to(tl.float64))
    tl.atomic_add(COUNTERS+10,count.to(tl.float64))
    tl.atomic_add(COUNTERS+11,1.)
    tl.atomic_max(COUNTERS+15,count.to(tl.float64))


@triton.jit(do_not_specialize=["ROWS","CACHE"])
def _projector_terms(SELECTED,SELECTED_COUNT,IDS,G,HEAD,B,CONTEXTS,H_OUT,V_OUT,
                      ROW_MAP,
                      ROWS,CACHE,K:tl.constexpr,HIDDEN:tl.constexpr,
                      R:tl.constexpr,BU:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
    worker=tl.program_id(0);stride=tl.num_programs(0)
    count=tl.load(SELECTED_COUNT)
    for ordinal in range(worker,count,stride):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        batch=tl.load(ROW_MAP+state//ROWS).to(tl.int64) if HAS_ROW_MAP else state//ROWS
        j,r,h=tl.arange(0,BU),tl.arange(0,BR),tl.arange(0,BH)
        ids=tl.load(IDS+state*2*K+j,j<2*K,other=-1)
        g=tl.load(G+state*2*K+j,j<2*K,other=0)
        b=tl.load(B+tl.maximum(ids[:,None],0)*R+r[None,:],(ids[:,None]>=0)&(j[:,None]<2*K)&(r[None,:]<R),other=0)
        v=tl.sum(g[:,None]*b,axis=0)
        context=tl.maximum(tl.load(CONTEXTS+state),0)
        head=tl.load(HEAD+(batch*CACHE+context)*HIDDEN+h,h<HIDDEN,other=0).to(tl.float32)
        tl.store(H_OUT+ordinal*HIDDEN+h,head,h<HIDDEN)
        tl.store(V_OUT+ordinal*R+r,v,r<R)


@triton.jit
def _projector_reduce(H,V,COUNT,OUT,HIDDEN:tl.constexpr,R:tl.constexpr,BH:tl.constexpr,BS:tl.constexpr,BR:tl.constexpr):
    h=tl.program_id(0)*BH+tl.arange(0,BH);r=tl.arange(0,BR);s=tl.arange(0,BS)
    count=tl.load(COUNT);acc=tl.full((BH,BR),0.,tl.float32)
    for start in range(0,count,BS):
        x=tl.load(H+(start+s[:,None])*HIDDEN+h[None,:],(start+s[:,None]<count)&(h[None,:]<HIDDEN),other=0)
        v=tl.load(V+(start+s[:,None])*R+r[None,:],(start+s[:,None]<count)&(r[None,:]<R),other=0)
        acc+=tl.sum(x[:,:,None]*v[:,None,:],axis=0)
    tl.store(OUT+h[:,None]*R+r[None,:],acc,(h[:,None]<HIDDEN)&(r[None,:]<R))


@triton.jit(do_not_specialize=['WIDTH','D_ROWS','D_CACHE','D_TS0','D_TS1','D_TS2'])
def _teacher_sorted(P,IDS,INVERSE,SELECTED,COUNT,OUT_P,OUT_IDS,MASS,WIDTH,K:tl.constexpr,BLOCK:tl.constexpr,V:tl.constexpr,
                    CAPTURE:tl.constexpr=False,D_TARGET=None,D_MAP=None,D_IDS=None,D_CONTEXTS=None,D_ROW_MAP=None,D_OUT=None,
                    D_ROWS=0,D_CACHE=0,D_TS0=0,D_TS1=0,D_TS2=0,D_GREEDY:tl.constexpr=False,D_HAS_ROW_MAP:tl.constexpr=False):
    lane=tl.arange(0,BLOCK)
    for ordinal in range(tl.program_id(0),tl.load(COUNT),tl.num_programs(0)):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        tl.store(MASS+state,1.)
        if CAPTURE:
            _capture_draft_coordinates(state,D_TARGET,D_MAP,D_IDS,D_CONTEXTS,D_ROW_MAP,D_OUT,
                D_ROWS,D_CACHE,D_TS0,D_TS1,D_TS2,K,triton.next_power_of_2(K),D_GREEDY,D_HAS_ROW_MAP)
        # Read sampler prefix; extend ONLY through positive boundary ties.
        # Sort integer keys locally to preserve token-ID teacher tie semantics.
        boundary=tl.load(P+state*WIDTH+tl.minimum(K,WIDTH)-1).to(tl.float32)
        retained=tl.full((BLOCK,),0,tl.uint64)
        offset=tl.full((),0,tl.int32)
        more=tl.full((),True,tl.int1)
        while more:
            pos=offset+lane
            value=tl.load(P+state*WIDTH+pos,pos<WIDTH,other=0.).to(tl.float32)
            target=tl.load(IDS+state*WIDTH+pos,pos<WIDTH,other=0)
            compact=tl.load(INVERSE+target)
            bits=value.to(tl.uint32,bitcast=True).to(tl.uint64)
            key=tl.where((pos<WIDTH)&(value>0),(bits<<32)|(V-compact).to(tl.uint64),0)
            merged=tl.sort(tl.reshape(tl.join(retained,key),(2*BLOCK,)),descending=True)
            retained,discarded=tl.split(tl.trans(tl.reshape(merged,(2,BLOCK))))
            retained=tl.where(lane<K,retained,0)
            offset+=BLOCK
            next_value=tl.load(P+state*WIDTH+offset,offset<WIDTH,other=0.).to(tl.float32)
            more=(offset<WIDTH)&(next_value>0)&(next_value>=boundary)
        probability=(retained>>32).to(tl.uint32).to(tl.float32,bitcast=True)
        compact=V-(retained&4294967295).to(tl.int64)
        tl.store(OUT_P+state*K+lane,probability,lane<K)
        tl.store(OUT_IDS+state*K+lane,tl.where(probability>0,compact,-1),lane<K)


@triton.jit(do_not_specialize=['ROWS','TS0','TS1','D_ROWS','D_CACHE','D_TS0','D_TS1','D_TS2'])
def _teacher_full_greedy(TARGET,INVERSE,SELECTED,COUNT,P,IDS,MASS,ROWS,TS0,TS1,K:tl.constexpr,BK:tl.constexpr,
                         CAPTURE:tl.constexpr=False,D_TARGET=None,D_MAP=None,D_IDS=None,D_CONTEXTS=None,D_ROW_MAP=None,D_OUT=None,
                         D_ROWS=0,D_CACHE=0,D_TS0=0,D_TS1=0,D_TS2=0,D_GREEDY:tl.constexpr=False,D_HAS_ROW_MAP:tl.constexpr=False):
    lane=tl.arange(0,BK)
    for ordinal in range(tl.program_id(0),tl.load(COUNT),tl.num_programs(0)):
        state=tl.load(SELECTED+ordinal).to(tl.int64)
        token=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1)
        compact=tl.load(INVERSE+token)
        tl.store(P+state*K+lane,tl.where(lane==0,1.,0.),lane<K)
        tl.store(IDS+state*K+lane,tl.where(lane==0,compact,-1),lane<K)
        tl.store(MASS+state,1.)
        if CAPTURE:
            _capture_draft_coordinates(state,D_TARGET,D_MAP,D_IDS,D_CONTEXTS,D_ROW_MAP,D_OUT,
                D_ROWS,D_CACHE,D_TS0,D_TS1,D_TS2,K,triton.next_power_of_2(K),D_GREEDY,D_HAS_ROW_MAP)


def prepare_teacher(state,tree,path,target,greedy=False,sampling_metadata=None,capture=None):
    b,q=tree.parents.shape;n=b*q;k=state.topk;rank=state.rank
    weights=state.selected_weights[:n].view(b,q);kind=state.selected_kind[:n].view(b,q)
    ticket=state.begin('opd_state_select_ms')
    select_states(tree,path,weights,kind,state.visited_weight,state.frontier_weight)
    state.end(ticket);ticket=state.begin('opd_teacher_extract_ms')
    probs=state.teacher_p[:n*k].view(n,k);ids=state.teacher_ids[:n*k].view(n,k);mass=state.teacher_mass[:n]
    if greedy and state.full_vocab_inverse is not None:
        _compact_selected[(1,)](weights,state.selected_ids,state.selected_count,n,triton.next_power_of_2(n),num_warps=4)
        _teacher_full_greedy[(min(n,32),)](target,state.full_vocab_inverse,state.selected_ids,state.selected_count,
            probs,ids,mass,q,*target.stride(),k,triton.next_power_of_2(k),**(capture or {}),num_warps=4)
    elif sampling_metadata is not None and state.full_vocab_inverse is not None:
        sorted_p,sorted_ids=sampling_metadata
        _compact_selected[(1,)](weights,state.selected_ids,state.selected_count,n,triton.next_power_of_2(n),num_warps=4)
        _teacher_sorted[(min(n,32),)](sorted_p,sorted_ids,state.full_vocab_inverse,state.selected_ids,state.selected_count,
            probs,ids,mass,sorted_p.shape[-1],k,max(32,triton.next_power_of_2(k)),state.vocab,**(capture or {}),num_warps=4)
    else:
        teacher(target,state.mapping,weights,k,state.teacher_tiles,(probs,ids,mass),greedy,
                selection=(state.selected_ids,state.selected_count),capture=capture)
    state.end(ticket)
    return probs,ids,mass


@triton.jit(do_not_specialize=['ROWS','CACHE','TS0','TS1','TS2'])
def _capture_draft_coordinates(state,TARGET,MAP,D_IDS,CONTEXTS,ROW_MAP,OUT,
                        ROWS,CACHE,TS0,TS1,TS2,K:tl.constexpr,BK:tl.constexpr,
                        GREEDY:tl.constexpr,HAS_ROW_MAP:tl.constexpr):
    lane=tl.arange(0,BK)
    batch=tl.load(ROW_MAP+state//ROWS).to(tl.int64) if HAS_ROW_MAP else state//ROWS
    context=tl.maximum(tl.load(CONTEXTS+state),0)
    ids=tl.load(D_IDS+(batch*CACHE+context)*K+lane,lane<K,other=0)
    token=tl.load(MAP+ids,lane<K,other=0)
    if GREEDY:p=(token==tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1)).to(tl.float32)
    else:p=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1+token*TS2,lane<K,other=0)
    # Raw FP32 p is divided by the same mass in union as before.
    tl.store(OUT+state*K+lane,p,lane<K)


def prepare_compact_teacher(state,tree,path,target,greedy=False,sampling_metadata=None):
    b,q=tree.parents.shape;n=b*q;k=state.topk
    raw=state.teacher_draft_p[:n*k].view(n,k)
    row_map=state.feedback_row_map
    capture=dict(CAPTURE=True,D_TARGET=target,D_MAP=state.mapping,D_IDS=state.ids_cache,
        D_CONTEXTS=tree.feedback_contexts,D_ROW_MAP=row_map,D_OUT=raw,D_ROWS=q,D_CACHE=state.cache_contexts,
        D_TS0=target.stride(0),D_TS1=target.stride(1),D_TS2=0 if greedy else target.stride(2),
        D_GREEDY=greedy,D_HAS_ROW_MAP=row_map is not None)
    probs,ids,mass=prepare_teacher(state,tree,path,target,greedy,sampling_metadata,capture)
    return probs,ids,mass,raw


def feedback(state,tree,path,target,greedy=False,sampling_metadata=None):
    b,q=tree.parents.shape;n=b*q;k=state.topk;rank=state.rank
    weights=state.selected_weights[:n].view(b,q);kind=state.selected_kind[:n].view(b,q)
    compact=sampling_metadata is not None and len(sampling_metadata)==4
    row_map=state.feedback_row_map
    if sampling_metadata is not None and len(sampling_metadata) in (3,4):
        probs,ids,mass=sampling_metadata[:3]
    else:
        probs,ids,mass=prepare_teacher(state,tree,path,target,greedy,sampling_metadata)
    ticket=state.begin('opd_union_loss_ms')
    head=state.head.weight;bias=state.head.bias if state.head.bias is not None else head
    _selected_head[(k,min(n,32))](state.selected_ids,state.selected_count,state.head_cache,head,bias,state.u_cache,state.B_fast,state.ids_cache,state.q_cache,
        ids,state.norm_cache,weights,mass,tree.feedback_contexts,state.teacher_q,row_map,
        triton.language.bfloat16 if state.logits_dtype==torch.bfloat16 else (triton.language.float16 if state.logits_dtype==torch.float16 else triton.language.float32),
        q,state.cache_contexts,head.shape[1],rank,k,state.head.bias is not None,triton.next_power_of_2(head.shape[1]),
        triton.next_power_of_2(rank),triton.next_power_of_2(k),row_map is not None,num_warps=4,enable_fp_fusion=False)
    state.state_stats[:n*10].zero_()
    strides=(0,0,0) if compact else (*target.stride()[:2],0 if greedy else target.stride(2))
    draft_p=sampling_metadata[3] if compact else state.teacher_draft_p
    _union[(min(n,32),)](state.selected_ids,state.selected_count,draft_p if compact else target,state.mapping,weights,kind,tree.feedback_contexts,state.ids_cache,state.q_cache,ids,probs,
        state.teacher_q,mass,state.union_ids,state.union_g,state.state_stats,
        draft_p,row_map,*strides,q,state.cache_contexts,state.vocab,k,10,
        triton.next_power_of_2(2*k),greedy,compact,row_map is not None,num_warps=4,enable_fp_fusion=False)
    _reduce[(10,)](state.state_stats,state.round_weight,state.counters,n,10,triton.next_power_of_2(n),num_warps=4)
    if state.train_projector and state._ever_updated:
        # B is STILL frozen B_t. No per-round backward/optimizer/DDP sync.
        h=state.head.weight.shape[1]
        _projector_terms[(min(n,32),)](state.selected_ids,state.selected_count,state.union_ids,state.union_g,state.head_cache,state.B_fast,tree.feedback_contexts,
            state.projector_head,state.projector_v,row_map,q,state.cache_contexts,k,h,rank,
            triton.next_power_of_2(2*k),triton.next_power_of_2(h),triton.next_power_of_2(rank),row_map is not None,num_warps=4,enable_fp_fusion=False)
        _projector_reduce[(triton.cdiv(h,32),)](state.projector_head,state.projector_v,state.selected_count,
            state.projector_delta,h,rank,32,32,triton.next_power_of_2(rank),num_warps=4,enable_fp_fusion=False)
        state.model.opd_projector_grad_sum.add_(state.projector_delta)
    if state.train_projector:state.model.opd_projector_grad_weight.add_(state.round_weight)
    state.end(ticket);ticket=state.begin('opd_update_ms')
    if state.fast_lr>0:
        _update[(min(n,32),)](state.selected_ids,state.selected_count,state.union_ids,state.union_g,state.u_cache,tree.feedback_contexts,state.round_weight,
            state.B_fast,state.bitmap,state.active_ids,state.active_count,row_map,q,state.cache_contexts,k,rank,
            state.fast_lr,triton.next_power_of_2(2*k),triton.next_power_of_2(rank),row_map is not None,num_warps=4,enable_fp_fusion=False)
    _round_end[(1,)](state.B_fast,state.bitmap,state.active_ids,
        state.active_count,state.round_weight,state.counters,rank,state.fast_lr,128,triton.next_power_of_2(rank),num_warps=4)
    state.end(ticket)
