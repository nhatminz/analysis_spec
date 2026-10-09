"""Exact optimized proposal engine and rollout-local OPD Reflex.

A is a persistent learned projector owned/saved by the FastGRPO draft. Its
analytical gradients accumulate here, but only the existing draft optimizer
boundary updates A. B_fast is ONE shared full-vocabulary adapter, reset every
rollout. No optimizer/autograd or extra model forward runs here. CUDA production
requires Triton; Torch is CPU oracle.
"""
from contextlib import nullcontext
import importlib
import math
import json
import os
import warnings
from pathlib import Path
import torch
from helper.opd_attention import AttentionWorkspace
from helper.shared_rollout import allocate_tree_buffers
from helper.opd_profiles import execution_key,fingerprint,discover_profile,validate_profile

OPD_COUNTER_NAMES=(
    'opd_state_weight','opd_selected_states','opd_visited_states','opd_frontier_states',
    'opd_kl_sum','opd_union_size_sum','opd_compact_mass_sum','opd_draft_topk_target_mass_sum',
    'opd_invalid_states','opd_updates','opd_active_rows_sum','opd_rounds',
    'opd_nonfinite_kl_states',
    'opd_proposal_mode_sparse_rounds','opd_proposal_mode_dense_rounds',
    'opd_active_rows_max',
    'opd_proposal_mode_fused_rounds','opd_proposal_mode_gemm_rounds',
)
GENERATION_COUNTER_NAMES=('verification_batches','active_response_rounds','verified_tree_nodes')


def initialize_projector(hidden,rank,seed=42,head=None):
    if not 1<=rank<=64:raise ValueError('OPD rank must be in [1,64]')
    if head is not None:
        ids=torch.linspace(0,head.shape[0]-1,rank,device=head.device).long()
        rows=head.detach().index_select(0,ids).float().t().cpu()
        return torch.linalg.qr(rows,mode='reduced').Q.contiguous()
    # Oracle fixtures only. Production uses a deterministic head-aligned basis.
    return torch.randn(hidden,rank,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)/math.sqrt(hidden)


def select_states_reference(tree,path,visited_weight=1.,frontier_weight=1.):
    """Vectorized CPU oracle, also used for non-CUDA integration tests."""
    b,q=tree.parents.shape;rows=torch.arange(q,device=tree.parents.device)
    visited=(rows[None,:,None]==path.packed_indices[:,None,:]).any(-1)
    parent_visited=(tree.parents[:,:,None]==path.packed_indices[:,None,:]).any(-1)
    frontier=~visited&parent_visited&(rows[None,:]>0)
    expanded=tree.feedback_contexts>=0
    weights=torch.where(visited,float(visited_weight),torch.where(frontier,float(frontier_weight),0.))
    weights=torch.where(expanded,weights,0.)
    kind=torch.where((weights>0)&visited,1,torch.where((weights>0)&frontier,2,0))
    return weights,kind


def union_reference(p,q,target_ids,draft_ids):
    """Top-k union + ONE tail; not a renormalized shortlist loss."""
    k=draft_ids.shape[-1]
    ids=torch.cat((draft_ids,target_ids),-1)
    positive_target=(target_ids>=0)&(p.gather(-1,target_ids.clamp_min(0))>0)
    valid=torch.cat((torch.ones_like(draft_ids,dtype=torch.bool),
                     positive_target&~(target_ids[:,:,None]==draft_ids[:,None,:]).any(-1)),-1)
    pu=torch.where(valid,p.gather(-1,ids.clamp_min(0)),0.);qu=torch.where(valid,q.gather(-1,ids.clamp_min(0)),0.)
    empty_tail=valid.sum(-1)==p.shape[-1]
    pt=torch.where(empty_tail,0.,(1-pu.sum(-1)).clamp_min(0.))
    qt=torch.where(empty_tail,0.,(1-qu.sum(-1)).clamp_min(0.))
    kl=torch.where(pu>0,pu*(pu.log()-qu.log()),0.).sum(-1)
    kl+=torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
    return ids,valid,pu,qu,pt,qt,kl


class OPDReflex:
    def __init__(self,rank=8,topk=16,fast_lr=.01,visited_weight=1.,frontier_weight=1.,
                 profile=False,diagnostics=False,enabled=True,backend='auto',train_projector=False):
        if not 1<=rank<=64 or topk<1 or fast_lr<0 or min(visited_weight,frontier_weight)<0:
            raise ValueError('invalid OPD rank/topk/lr/state weights')
        if backend not in ('auto','torch','triton'):raise ValueError('invalid OPD backend')
        self.rank,self.requested_topk,self.fast_lr=int(rank),int(topk),float(fast_lr)
        self.visited_weight,self.frontier_weight=float(visited_weight),float(frontier_weight)
        self.profile,self.diagnostics,self.enabled=bool(profile),bool(diagnostics),bool(enabled)
        self.requested_backend=backend;self._events=[];self._layout=None
        self.proposal_mode=os.environ.get('OPD_PROPOSAL_MODE','auto')
        if self.proposal_mode not in ('sparse','dense','auto','adaptive'):raise ValueError('invalid OPD_PROPOSAL_MODE')
        self.dense_implementation=os.environ.get('OPD_DENSE_IMPLEMENTATION','auto')
        if self.dense_implementation not in ('auto','fused','gemm'):raise ValueError('invalid OPD_DENSE_IMPLEMENTATION')
        profile_path=os.environ.get('OPD_PROPOSAL_PROFILE','')
        self.tuning=json.loads(Path(profile_path).read_text()) if profile_path else None
        self.profile_path=profile_path
        self.explicit_profile_path=profile_path
        self.profile_selector=None
        self._profile_execution_shape=None
        self.train_projector=bool(train_projector)
        self._validated_tuning=False
        self._threshold_cache={}
        self._dense_choice_cache={}

    def proposal_threshold(self,b,c):
        key=(b*c,self.vocab,self.rank,str(self.logits_dtype))
        if key not in self._threshold_cache:self._threshold_cache[key]=self._interpolated_threshold(b,c)
        return self._threshold_cache[key]

    def _interpolated_threshold(self,b,c):
        # Dispatch depends on flattened context workload, not arbitrary batch
        # factorization. Log-space interpolation covers shrinking live batches.
        points={}
        for key,value in (self.tuning or {}).get('thresholds',{}).items():
            pb,pc,v,r,dtype=key.split(',')
            if (int(v),int(r),dtype)==(self.vocab,self.rank,str(self.logits_dtype)):
                points.setdefault(int(pb)*int(pc),[]).append(int(value))
        if not points:
            # Explicit uncalibrated policy, NOT a claimed B200 measurement.
            return max(1,self.vocab//8)
        points=sorted((n,sum(values)/len(values)) for n,values in points.items())
        work=b*c
        if work<=points[0][0]:return round(points[0][1])
        for (lo,a),(hi,z) in zip(points,points[1:]):
            if work<=hi:
                w=math.log(work/lo)/math.log(hi/lo)
                return round(a+(z-a)*w)
        return round(points[-1][1])

    def selected_proposal_backend(self,b,c):
        if self.proposal_mode in ('sparse','dense'):return self.proposal_mode
        if self.profile_selector is not None:
            return 'sparse' if self.profile_selector.choose(b*c,self.host_active_count)=='sparse' else 'dense'
        # Snapshot piggybacks on the existing HF scheduling packet. It is one
        # feedback round old when an update stream is used; this affects only
        # speed, never the correction, which reads the CURRENT GPU active set.
        return 'dense' if self.host_active_count>=self.proposal_threshold(b,c) else 'sparse'

    def selected_dense_implementation(self,b,c):
        if self.dense_implementation!='auto':return self.dense_implementation
        if self.profile_selector is not None:
            selected=self.profile_selector.choose(b*c,self.host_active_count)
            if selected!='sparse':return selected
            # Explicit MODE=dense can override a calibrated sparse choice;
            # still use measured dense costs, never a global dense heuristic.
            costs=self.profile_selector.costs(b*c,self.host_active_count)
            return 'fused' if costs[1]<=costs[2] else 'gemm'
        key=(b*c,self.vocab,self.rank,str(self.logits_dtype))
        if key not in self._dense_choice_cache:
            self._dense_choice_cache[key]=self._nearest_dense_implementation(b,c)
        return self._dense_choice_cache[key]

    def _nearest_dense_implementation(self,b,c):
        buckets=[]
        for key,value in (self.tuning or {}).get('dense_implementations',{}).items():
            pb,pc,v,r,dtype=key.split(',')
            if (int(v),int(r),dtype)==(self.vocab,self.rank,str(self.logits_dtype)):
                buckets.append((abs(math.log((b*c)/(int(pb)*int(pc)))),value))
        return min(buckets)[1] if buckets else 'fused'

    def start(self,model,batch,mapping,hidden_size,*,max_contexts,max_nodes,max_path,max_proposal_contexts):
        device=mapping.device;v=mapping.numel();k=min(v,self.requested_topk)
        if not hasattr(self,'attention_workspace') or self.attention_workspace.device!=device:
            self.attention_workspace=AttentionWorkspace(device)
        if self.requested_backend=='torch' and device.type=='cuda':
            raise ValueError('Torch OPD is CPU oracle only; production CUDA requires Triton')
        self.backend='triton' if device.type=='cuda' else 'torch'
        if self.requested_backend=='triton' and device.type!='cuda':raise ValueError('Triton OPD requires CUDA')
        self._kernels=importlib.import_module('helper.tree_kernels') if self.backend=='triton' else None
        self._opd_kernels=importlib.import_module('helper.opd_reflex_kernels') if self.backend=='triton' else None
        self.mapping,self.vocab,self.topk,self.max_batch=mapping,v,k,batch
        self.cache_contexts=max_contexts
        self.head=model.lm_head
        layout=(batch,v,hidden_size,max_contexts,max_nodes,max_path,max_proposal_contexts,k,str(device),self.enabled,self.head.weight.dtype)
        execution_shape=(str(device),v,self.rank,self.head.weight.dtype,k)
        if self._profile_execution_shape is not None and execution_shape!=self._profile_execution_shape:
            self._validated_tuning=False;self._profile_shape_checked=False
            self.profile_selector=None;self.tuning=None;self.profile_path=self.explicit_profile_path
            self._threshold_cache.clear();self._dense_choice_cache.clear()
        if self.backend=='triton' and not self._validated_tuning:
            key=execution_key(fingerprint(device),v,self.rank,self.head.weight.dtype,k)
            directory=os.environ.get('OPD_PROPOSAL_PROFILE_DIR',str(Path(__file__).resolve().parents[1]/'outputs/benchmarks/opd_proposals'))
            path,payload=discover_profile(directory,key,self.profile_path)
            if path:
                self.profile_path=str(path);self.tuning=payload
                self.profile_selector=validate_profile(payload,key)
            elif self.proposal_mode in ('auto','adaptive'):
                warnings.warn('No exact compatible OPD proposal profile; using UNCALIBRATED safe fallback. Run scripts/tune_opd_proposals.sh; no tuning in hot path.')
            print(f'OPD proposal mode: {self.proposal_mode}\nprofile: {self.profile_path or "NONE (uncalibrated fallback)"}\nGPU: {key["gpu"]} cc{key["compute_capability"]}\nV: {v}\nrank: {self.rank}\ndtype: {key["dtype"]}\nkernel hash: {key["kernel_sha256"]}',flush=True)
            self._validated_tuning=True
        self._profile_execution_shape=execution_shape
        self.full_vocab_inverse=mapping  # Full target vocabulary: token IDs are identity.
        self.model=model
        if self.enabled:
            self.projector=model.get_opd_projector(self.rank) if hasattr(model,'get_opd_projector') else None
            if self.projector is None:
                if not hasattr(model,'opd_projector'):model.opd_projector=initialize_projector(hidden_size,self.rank).to(device)
                self.projector=model.opd_projector
            if tuple(self.projector.shape)!=(hidden_size,self.rank):raise ValueError('persistent A rank/hidden mismatch')
            if self.train_projector and not hasattr(model,'opd_projector_grad_sum'):
                with torch.inference_mode(False):
                    model.opd_projector_grad_sum=torch.zeros_like(self.projector)
                    model.opd_projector_grad_weight=torch.zeros(1,device=device)
        if layout!=self._layout:
            self._layout=layout
            def alloc(shape,dtype=torch.float32):return torch.empty(shape,device=device,dtype=dtype)
            self.bitmap=alloc((v+31)//32,torch.int32)
            self.active_count=alloc(1,torch.int32)
            self.dispatch_snapshot=alloc(1,torch.int32)
            self.B_fast=alloc((v,self.rank)) if self.enabled else None
            self.active_ids=alloc(v,torch.int32) if self.enabled else None
            self.proposal_u=alloc((batch*max_proposal_contexts,self.rank)) if self.enabled else None
            self.proposal_q=alloc(batch*max_proposal_contexts*k)
            self.proposal_ids=alloc(batch*max_proposal_contexts*k,torch.long)
            self.proposal_norm=alloc(batch*max_proposal_contexts*2)
            # Reserved proposal-only workspace: sparse touches ONLY S token
            # scalars; dense GEMM writes all. Not a feedback probability cache.
            self.sparse_capacity=min(v,256)
            self.sparse_scores=alloc(batch*max_proposal_contexts*self.sparse_capacity if self.enabled else 1)
            self.active_slots=alloc(v,torch.int32)
            # Dense GEMM workspace is lazy, never allocated by sparse/fused runs.
            self.score_workspace=alloc(1)
            self.proposal_capacity=batch*max_proposal_contexts
            self.max_feedback_rows=max_nodes
            tiles=(v+255)//256+1
            self.proposal_tiles=[alloc(batch*max_proposal_contexts*tiles*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if self.backend=='triton' else []
            allocate_tree_buffers(self,alloc,batch,max_contexts,max_path,max_proposal_contexts)
            if self.enabled:
                self.head_cache=alloc((batch,max_contexts,hidden_size),self.head.weight.dtype)
                self.u_cache=alloc((batch,max_contexts,self.rank))
                self.ids_cache=alloc((batch,max_contexts,k),torch.long);self.q_cache=alloc((batch,max_contexts,k))
                self.norm_cache=alloc((batch,max_contexts,2))
                # max_nodes bounds TOTAL packed verification rows, not per
                # response. B*q is bounded by verification_capacity + B.
                n=max_nodes;t=(v+255)//256
                self.selected_weights=alloc(n);self.selected_kind=alloc(n,torch.int32)
                self.selected_ids=alloc(n,torch.int32);self.selected_count=alloc(1,torch.int32)
                self.teacher_p=alloc(n*k);self.teacher_ids=alloc(n*k,torch.long);self.teacher_mass=alloc(n)
                self.teacher_q=alloc(n*k);self.union_ids=alloc(n*2*k,torch.long);self.union_g=alloc(n*2*k)
                self.teacher_draft_p=alloc(n*k)
                self.state_stats=alloc(n*10);self.round_weight=alloc(1)
                self.teacher_tiles=[alloc(n*t*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if self.backend=='triton' else []
                self.counters=alloc(len(OPD_COUNTER_NAMES),torch.float64)
                if self.train_projector:
                    self.projector_head=alloc((n,hidden_size))
                    self.projector_v=alloc((n,self.rank))
                    self.projector_delta=alloc((hidden_size,self.rank))
        self.bitmap.zero_();self.active_count.zero_();self.dispatch_snapshot.zero_()
        self.host_active_count=0
        self.host_sync_count=0
        self.feedback_row_map=None
        self.host_fused_rounds=self.host_gemm_rounds=0
        if self.enabled:self.B_fast.zero_();self.counters.zero_()
        self._ever_updated=False;self._events.clear()

    def prepare_proposal_workspace(self,b,c,selected=None):
        selected=selected or ('sparse' if self.selected_proposal_backend(b,c)=='sparse' else self.selected_dense_implementation(b,c))
        if selected=='sparse':
            # Snapshot is one update behind; one union per selected state bounds
            # all unseen activations. No GPU read or allocation each round.
            unseen=2*self.max_feedback_rows*self.topk if self._ever_updated else 0
            required=min(self.vocab,self.host_active_count+unseen)
            if required>self.sparse_capacity:
                self.sparse_capacity=min(self.vocab,max(required,2*self.sparse_capacity))
                self.sparse_scores=torch.empty(self.proposal_capacity*self.sparse_capacity,device=self.mapping.device)
        elif selected=='gemm' and self.score_workspace.numel()<self.proposal_capacity*self.vocab:
            self.score_workspace=torch.empty(self.proposal_capacity*self.vocab,device=self.mapping.device)

    def begin(self,label):
        if self.profile and self.backend=='triton':
            event=torch.cuda.Event(enable_timing=True);event.record();return label,event
        return None

    def end(self,ticket):
        if ticket is not None:
            end=torch.cuda.Event(enable_timing=True);end.record();self._events.append((ticket[0],ticket[1],end))

    def prepare_sampler_teacher(self,tree,path,target,sorted_metadata):
        if self.backend=='triton' and self.full_vocab_inverse is not None and sorted_metadata is not None:
            return self._opd_kernels.prepare_teacher(self,tree,path,target,sampling_metadata=sorted_metadata)
        # Subsets use selected compact extraction in feedback; no sorted retention.
        return None

    def prepare_compact_teacher(self,tree,path,target,sorted_metadata=None,greedy=False):
        if self.backend=='triton':
            return self._opd_kernels.prepare_compact_teacher(self,tree,path,target,greedy,sorted_metadata)
        b,q=tree.parents.shape;k=self.topk
        batch=torch.arange(b)[:,None].expand(b,q)
        if self.feedback_row_map is not None:batch=self.feedback_row_map[batch]
        context=tree.feedback_contexts.clamp_min(0)
        di=self.ids_cache[batch,context]
        p=(self.mapping[None,None,:]==target[...,None]).float() if greedy else target[...,self.mapping]
        mass=p.sum(-1);good=torch.isfinite(mass)&(mass>0)
        normalized=p/torch.where(good,mass,1.)[...,None]
        ti=torch.argsort(normalized,descending=True,stable=True)[...,:k]
        tp=normalized.gather(-1,ti)
        ti=torch.where(tp>0,ti,-1)
        raw=p.gather(-1,di)
        return tp.reshape(-1,k),ti.reshape(-1,k),mass.flatten(),raw.reshape(-1,k)

    @torch.no_grad()
    def propose(self,logits,hidden,k,mapping,*,root=False,context_offset=0,head_inputs=None):
        """Return transient q/id views, valid only until the next propose().

        Context caches are independent. Consumers retaining returned values
        across proposal calls must copy just the retained slice to owned storage.
        Mapped target IDs are already independently materialized by indexing.
        """
        b,c,v=logits.shape;keep=self.topk
        self.logits_dtype=logits.dtype
        if self.tuning is not None and not getattr(self,'_profile_shape_checked',False):
            suffix=f',{v},{self.rank},{logits.dtype}'
            if not any(key.endswith(suffix) for key in self.tuning.get('thresholds',{})):
                raise ValueError('proposal profile does not cover actual full vocabulary/rank/dtype; retune for this target')
            self._profile_shape_checked=True
        if root and self.backend=='torch':self.dispatch_snapshot.copy_(self.active_count)
        if k>keep or v!=self.vocab:raise ValueError('proposal k/vocabulary mismatch')
        ticket=self.begin('opd_feature_ms')
        u=None
        if self.enabled:
            if head_inputs is None:head_inputs=hidden
            u=self.proposal_u[:b*c].view(b,c,self.rank)
            if self.backend=='triton':self._opd_kernels.feature(head_inputs,self.projector,u,head_inputs,self,context_offset)
            else:
                with torch.autocast(device_type='cpu',enabled=False):u.copy_(head_inputs.float().matmul(self.projector))
        self.end(ticket);ticket=self.begin('proposal_ms')
        values=self.proposal_q[:b*c*keep].view(b,c,keep);ids=self.proposal_ids[:b*c*keep].view(b,c,keep)
        norm=self.proposal_norm[:b*c*2].view(b,c,2)
        if self.backend=='triton':
            self._opd_kernels.propose(logits,u,self,keep,self.proposal_tiles,(values,ids,norm),self.enabled and self._ever_updated,root)
        else:
            z=logits.float().clone()
            if self.enabled and self._ever_updated:
                active=self.active_ids[:int(self.active_count)]
                z[...,active.long()]+=u.matmul(self.B_fast[active.long()].t())
            maximum=z.amax(-1);total=(z-maximum[...,None]).exp().sum(-1)
            norm[...,0].copy_(maximum);norm[...,1].copy_(total)
            # Deterministic low-ID tie convention for the OPD CPU oracle.
            order=torch.argsort(z,dim=-1,descending=True,stable=True)[...,:keep]
            probabilities=z.softmax(-1)
            ids.copy_(order);values.copy_(probabilities.gather(-1,order))
        if self.enabled:
            if head_inputs is None:
                if hasattr(self.head,'opd_inputs'):head_inputs=self.head.opd_inputs(hidden)
                else:head_inputs=hidden
            if self.backend=='triton':self._opd_kernels.cache_proposal(values,ids,norm,self,context_offset)
            else:
                self.head_cache[:b,context_offset:context_offset+c].copy_(head_inputs)
                self.u_cache[:b,context_offset:context_offset+c].copy_(u)
                self.ids_cache[:b,context_offset:context_offset+c].copy_(ids)
                self.q_cache[:b,context_offset:context_offset+c].copy_(values)
                self.norm_cache[:b,context_offset:context_offset+c].copy_(norm)
        self.end(ticket)
        return values[...,:k],ids[...,:k],mapping[ids[...,:k]]

    @torch.no_grad()
    def feedback(self,tree,path,target,*,greedy=False,sampling_metadata=None):
        if not self.enabled:return
        if self.backend=='triton':self._opd_kernels.feedback(self,tree,path,target,greedy,sampling_metadata)
        else:self._feedback_reference(tree,path,target,greedy,sampling_metadata)
        if self.fast_lr>0:self._ever_updated=True

    def _feedback_reference(self,tree,path,target,greedy,sampling_metadata=None):
        # CPU oracle only: dense reconstructed q and gradient permitted HERE.
        b,q=tree.parents.shape;k=self.topk
        weights,kind=select_states_reference(tree,path,self.visited_weight,self.frontier_weight)
        batch=torch.arange(b)[:,None].expand(b,q);context=tree.feedback_contexts.clamp_min(0)
        if self.feedback_row_map is not None:batch=self.feedback_row_map[batch]
        head_inputs=self.head_cache[batch,context];u=self.u_cache[batch,context]
        raw=torch.nn.functional.linear(head_inputs,self.head.weight,self.head.bias).float()
        corrected=raw+u.matmul(self.B_fast.t())
        norm=self.norm_cache[batch,context]
        draft=((corrected-norm[...,0,None]).exp()/norm[...,1,None])
        di=self.ids_cache[batch,context].reshape(-1,k)
        if sampling_metadata is not None and len(sampling_metadata)==4:
            tp,ti,mass,pd=sampling_metadata
            mass=mass.view(b,q);good=torch.isfinite(mass)&(mass>0)
            # Dense oracle ONLY: reconstruct union coordinates, not teacher
            # outside the union. Tail=1-sum(union) retains the same objective.
            teacher=torch.zeros_like(draft).reshape(-1,self.vocab)
            teacher.scatter_(1,di,pd/torch.where(good,mass,1.).reshape(-1,1))
            rows=torch.arange(b*q)[:,None].expand(-1,k)
            positive=ti>=0;teacher[rows[positive],ti[positive]]=tp[positive]
            teacher=teacher.view(b,q,self.vocab)
        else:
            if greedy:teacher=(self.mapping[None,None,:]==target[...,None]).float()
            else:teacher=target[...,self.mapping].float()
            mass=teacher.sum(-1);good=torch.isfinite(mass)&(mass>0)
            teacher=teacher/torch.where(good,mass,1.)[...,None]
            ti=torch.argsort(teacher,dim=-1,descending=True,stable=True)[...,:k].reshape(-1,k)
            ti=torch.where(teacher.reshape(-1,self.vocab).gather(-1,ti)>0,ti,-1)
        w=torch.where(good,weights,0.).reshape(-1)
        ids,valid,p,qq,pt,qt,kl=union_reference(teacher.reshape(-1,self.vocab),draft.reshape(-1,self.vocab),ti,di)
        # Cached top-k q avoids selected-row output-head reconstruction drift.
        qq[:,:k]=self.q_cache[batch,context].reshape(-1,k)
        qt=torch.where(valid.sum(-1)==self.vocab,0.,(1-qq.sum(-1)).clamp_min(0.))
        kl=torch.where(p>0,p*(p.log()-qq.log()),0.).sum(-1)+torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
        selected=w>0;g=torch.where(valid,(qq-p)*w[:,None],0.)
        total=w.sum()
        if self.train_projector and self._ever_updated:
            rows=self.B_fast[ids.clamp_min(0)]
            v=((g[...,None]*rows)*valid[...,None]).sum(1)
            self.model.opd_projector_grad_sum.add_(head_inputs.reshape(-1,head_inputs.shape[-1]).float().t().matmul(v))
        if self.train_projector:self.model.opd_projector_grad_weight.add_(total)
        if self.fast_lr>0 and total>0:
            delta=torch.zeros_like(self.B_fast)
            delta.index_add_(0,ids.clamp_min(0).flatten(),(g[...,None]*u.reshape(-1,self.rank)[:,None,:]).reshape(-1,self.rank))
            self.B_fast.add_(delta,alpha=-self.fast_lr/float(total))
            old=self.active_ids[:int(self.active_count)].long()
            touched=torch.nonzero(self.B_fast.abs().sum(-1)>0,as_tuple=False).flatten()
            active=torch.unique(torch.cat((old,touched)))
            self.active_count.fill_(active.numel());self.active_ids[:active.numel()].copy_(active)
        count=selected.sum();categories=kind.flatten()
        finite=torch.isfinite(kl)
        stats=torch.stack((total,count,(selected&(categories==1)).sum(),(selected&(categories==2)).sum(),
            torch.where(selected&finite,w*kl,0.).sum(),(valid&selected[:,None]).sum(),
            torch.where(selected,mass.flatten(),0.).sum(),torch.where(selected,p[:,:k].sum(-1),0.).sum(),
            ((weights>0)&~good).sum(),((total>0)&(self.fast_lr>0)).float(),self.active_count[0].float(),total.new_tensor(1.),
            (selected&~finite).sum()))
        self.counters[:13].add_(stats.to(torch.float64))
        self.counters[15]=torch.maximum(self.counters[15],self.active_count[0].double())

    def remove_finished(self,indices):
        # B is shared, not row-owned. Current contexts are overwritten next tree.
        pass

    def finish(self):
        if self.enabled:
            payload=self.counters
            if self.diagnostics:
                extra=torch.stack((self.B_fast.norm().double(),self.B_fast.abs().amax().double(),self.active_count[0].double()))
                payload=torch.cat((payload,extra))
            packet=payload.cpu().tolist()  # exactly ONE counter/diagnostic packet
        else:packet=[0.]*len(OPD_COUNTER_NAMES)
        result=dict(zip(OPD_COUNTER_NAMES,packet[:len(OPD_COUNTER_NAMES)]))
        result['opd_proposal_mode_fused_rounds']=float(self.host_fused_rounds)
        result['opd_proposal_mode_gemm_rounds']=float(self.host_gemm_rounds)
        if self.enabled and self.diagnostics:
            result.update(dict(zip(('opd_final_b_norm','opd_final_b_max_abs','opd_final_active_rows'),packet[len(OPD_COUNTER_NAMES):])))
        sections={}
        if self.profile:
            for label,start,end in self._events:
                if not end.query():end.synchronize()  # opt-in profiling after rollout only
                sections[label]=sections.get(label,0.)+start.elapsed_time(end)
            sections.setdefault('opd_proposal_extra_ms',0.)
        result['opd_profile_sections_ms']=sections if self.profile else None
        result['opd_backend']=self.backend if self.enabled else 'off'
        # Full proposal is an auxiliary inclusive timer, not an additional OPD
        # phase. Wait can overlap OPD-stream work. Neither is double-counted.
        result['opd_profile_time_ms']=sum(v for k,v in sections.items() if k not in ('opd_wait_ms','proposal_ms'))
        self._events.clear()
        return result

    def clear(self):
        # Runtime pools are deliberately retained by model cache. No per-round
        # teacher or batch hidden references survive the feedback call.
        self._events.clear()
        if self.enabled:self.B_fast.zero_();self.bitmap.zero_();self.active_count.zero_()
        self._ever_updated=False
