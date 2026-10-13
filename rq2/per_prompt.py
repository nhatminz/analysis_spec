"""Frozen-A, rollout-local prompt-group SPARK; no batch-wide fallback.

Group-specific arithmetic follows SpecNaacl's per_prompt_group implementation.
A group's denominator includes only selected feedback from its own responses.
"""
import importlib
import torch
from helper.opd_reflex import OPDReflex, select_states_reference, union_reference


class PerPromptReflex(OPDReflex):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.train_projector or not self.enabled:
            raise ValueError('Per-prompt evaluation requires enabled correction and frozen checkpoint A')
        self.grouped=True

    def start(self,model,batch,mapping,hidden_size,*,prompt_groups,responses_per_prompt,canonical,**kwargs):
        if getattr(model,'opd_projector',None) is None:
            raise ValueError('A must be loaded from the learned checkpoint; initialization is forbidden')
        self.num_groups=int(prompt_groups);self.responses_per_prompt=int(responses_per_prompt)
        if self.num_groups<1 or self.responses_per_prompt<1:
            raise ValueError('Invalid prompt-group layout')
        # Existing calibration profiles describe batch-wide B, not this engine.
        self._profile_execution_shape=None;self._validated_tuning=True
        self.profile_selector=None;self.tuning=None;self.profile_path=''
        super().start(model,batch,mapping,hidden_size,**kwargs)
        if self.backend=='triton':self._opd_kernels=importlib.import_module('rq2.per_prompt_kernels')
        device=mapping.device;v=self.vocab;g=self.num_groups
        key=(g,v,self.rank,str(device))
        if getattr(self,'_group_storage_key',None)!=key:
            self._group_storage_key=key
            self._group_storage=dict(
                B_fast=torch.empty((g,v,self.rank),device=device),
                bitmap=torch.empty((g,(v+31)//32),device=device,dtype=torch.int32),
                active_ids=torch.empty((g,v),device=device,dtype=torch.int32),
                active_count=torch.empty(g,device=device,dtype=torch.int32),
                active_slots=torch.empty((g,v),device=device,dtype=torch.int32),
                group_weight=torch.empty(g,device=device),
                group_feedback_weight=torch.empty(g,device=device),
                group_updates=torch.empty(g,device=device,dtype=torch.int64),
                dispatch_count=torch.empty(1,device=device,dtype=torch.int32))
        for name,value in self._group_storage.items():setattr(self,name,value)
        for name in ('B_fast','bitmap','active_ids','active_count','group_weight','group_feedback_weight','group_updates','dispatch_count'):
            getattr(self,name).zero_()
        self.initial_b_norm=float(self.B_fast.norm().cpu())
        self.set_response_order(canonical)

    def set_response_order(self,canonical):
        if len(set(canonical))!=len(canonical) or any(i<0 or i>=self.num_groups*self.responses_per_prompt for i in canonical):
            raise ValueError('Invalid canonical response ownership')
        owners=[i//self.responses_per_prompt for i in canonical]
        first=[0]*self.num_groups;seen=set()
        for physical,group in enumerate(owners):
            if group not in seen:first[group]=physical;seen.add(group)
        self.row_groups=torch.tensor(owners,device=self.mapping.device,dtype=torch.long)
        self.group_first=torch.tensor(first,device=self.mapping.device,dtype=torch.long)

    def finish(self):
        result=super().finish()
        result.update(opd_b_sharing_scope='per_prompt_group',opd_initial_b_norm=self.initial_b_norm,
            opd_group_b_norms=self.B_fast.norm(dim=(1,2)).cpu().tolist(),
            opd_group_updates=self.group_updates.cpu().tolist(),
            opd_group_feedback_weight=self.group_feedback_weight.cpu().tolist(),
            opd_final_active_rows=int(self.active_count.sum().cpu()))
        return result

    def clear(self):
        super().clear()
        self.group_weight.zero_();self.group_feedback_weight.zero_();self.group_updates.zero_();self.dispatch_count.zero_();self._ever_updated=False

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
        if root and self.backend=='torch':self.dispatch_snapshot.copy_(self.dispatch_count)
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
                z+=torch.bmm(u,self.B_fast[self.row_groups[:b]].transpose(1,2))
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

    def _feedback_reference(self,tree,path,target,greedy,sampling_metadata=None):
        # CPU oracle only: dense reconstructed q and gradient permitted HERE.
        b,q=tree.parents.shape;k=self.topk
        weights,kind=select_states_reference(tree,path,self.visited_weight,self.frontier_weight)
        batch=torch.arange(b)[:,None].expand(b,q);context=tree.feedback_contexts.clamp_min(0)
        if self.feedback_row_map is not None:batch=self.feedback_row_map[batch]
        head_inputs=self.head_cache[batch,context];u=self.u_cache[batch,context]
        raw=torch.nn.functional.linear(head_inputs,self.head.weight,self.head.bias).float()
        groups=self.row_groups[batch[:,0]]
        corrected=raw+torch.bmm(u,self.B_fast[groups].transpose(1,2))
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
        flat_groups=groups.repeat_interleave(q)
        group_weight=torch.zeros(self.num_groups,device=w.device).index_add_(0,flat_groups,w)
        self.group_weight.copy_(group_weight)
        actual_updates=((group_weight>0)&(self.fast_lr>0)).sum().float()
        if self.fast_lr>0:
            delta=torch.zeros_like(self.B_fast).view(-1,self.rank)
            coordinates=flat_groups[:,None]*self.vocab+ids.clamp_min(0)
            weighted=g[...,None]*u.reshape(-1,self.rank)[:,None,:]
            delta.index_add_(0,coordinates.flatten(),weighted.reshape(-1,self.rank))
            self.B_fast.add_(delta.view_as(self.B_fast)/group_weight.clamp_min(1e-30)[:,None,None],alpha=-self.fast_lr)
            for group in range(self.num_groups):  # CPU oracle only
                old=self.active_ids[group,:int(self.active_count[group])].long()
                touched=torch.nonzero(self.B_fast[group].abs().sum(-1)>0,as_tuple=False).flatten()
                active=torch.unique(torch.cat((old,touched)))
                self.active_count[group]=active.numel();self.active_ids[group,:active.numel()].copy_(active)
        active_rows=self.active_count.sum().float()
        self.dispatch_count.copy_(self.active_count.max().view(1))
        count=selected.sum();categories=kind.flatten()
        finite=torch.isfinite(kl)
        probabilities_valid=(~valid|(torch.isfinite(p)&(p>=0)&torch.isfinite(qq)&(qq>=0))).all(-1)
        features_valid=torch.isfinite(u.reshape(-1,self.rank)).all(-1)
        invalid_feedback=((weights>0)&~good).reshape(-1)|(selected&~(probabilities_valid&features_valid))
        self.group_updates.add_(((self.group_weight>0)&(self.fast_lr>0)).long())
        self.group_feedback_weight.add_(self.group_weight)
        stats=torch.stack((total,count,(selected&(categories==1)).sum(),(selected&(categories==2)).sum(),
            torch.where(selected&finite,w*kl,0.).sum(),(valid&selected[:,None]).sum(),
            torch.where(selected,mass.flatten(),0.).sum(),torch.where(selected,p[:,:k].sum(-1),0.).sum(),
            invalid_feedback.sum(),actual_updates,active_rows,total.new_tensor(1.),
            (selected&~finite).sum()))
        self.counters[:13].add_(stats.to(torch.float64))
        self.counters[15]=torch.maximum(self.counters[15],active_rows.double())
