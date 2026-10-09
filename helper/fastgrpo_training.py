"""Production FastGRPO objectives with explicit empty-mask and numerical fixes.

For valid examples the objective remains mean_example(2*SmoothL1 + .1*soft CE).
Empty examples contribute neither loss nor its normalization. Every packing
microbatch uses the same global normalization, including the final one.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def token_logps(logits, labels, chunk_size=128,mask=None):
    """Gather next-token log probabilities without a full FP32 [B,T,V] copy."""
    parts=[]
    def gather(values, ids, valid):
        values=values.float().masked_fill(~valid[...,None],0.)
        return values.log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1)
    if mask is None:mask=torch.ones_like(labels,dtype=torch.bool)
    for start in range(0,logits.shape[1],chunk_size):
        values=logits[:,start:start+chunk_size];ids=labels[:,start:start+chunk_size]
        valid=mask[:,start:start+chunk_size]
        parts.append(checkpoint(gather,values,ids,valid,use_reentrant=False,preserve_rng_state=False)
                     if values.requires_grad else gather(values,ids,valid))
    return torch.cat(parts,dim=1)


def compute_target_loss(logits,ref_logits,old_logits,labels,mask,reward,epsilon,beta,grpo_iteration):
    mask=mask[...,:-1].to(device=logits.device,dtype=torch.bool)
    labels=labels[...,1:].to(logits.device)
    logps=token_logps(logits[...,:-1,:],labels,mask=mask)
    if grpo_iteration==0:
        ref_logps=(token_logps(ref_logits[...,:-1,:],labels,mask=mask) if ref_logits.ndim==3 else ref_logits).detach()
        old_logps=logps.detach()
    else:
        ref_logps=ref_logits;old_logps=old_logits
    # Mask before exponentiation: ignored padding must never produce 0*Inf/NaN.
    logps=logps.masked_fill(~mask,0.)
    ref_logps=ref_logps.masked_fill(~mask,0.)
    old_logps=old_logps.masked_fill(~mask,0.)
    ratio=(logps-old_logps).exp()
    policy=torch.minimum(ratio*reward,ratio.clamp(1-epsilon,1+epsilon)*reward)
    log_ratio=ref_logps-logps
    kl=torch.expm1(log_ratio)-log_ratio
    denominator=mask.sum(-1).clamp_min(1)
    policy=(policy.masked_fill(~mask,0.)).sum(-1)/denominator
    kl=(kl.masked_fill(~mask,0.)).sum(-1)/denominator
    return (-policy+beta*kl).sum(),policy.abs().sum(),kl.sum(),old_logps,ref_logps


def draft_supervision_stats(outputs,prompt_mask,repeated_generate_nums):
    states=outputs['all_draft_input_states'];ids=outputs['all_draft_input_ids']
    if repeated_generate_nums<1 or len(states)!=len(ids) or len(ids)!=len(prompt_mask)*repeated_generate_nums:
        raise ValueError('Draft histories and prompt/response counts differ')
    valid=[];tokens=0
    for i,(h,d) in enumerate(zip(states,ids)):
        p=int(prompt_mask[i//repeated_generate_nums].sum())
        if d.ndim!=1 or h.ndim!=2 or len(d)!=len(h) or p<1 or len(d)<p:
            raise ValueError('Malformed shifted draft history/prompt boundary')
        n=max(0,len(d)-p-1)
        if n:valid.append(i);tokens+=n
    return dict(valid_examples=len(valid),empty_examples=len(ids)-len(valid),
                supervised_tokens=tokens,valid_indices=valid)


def training_draft_model(model,outputs,prompt_mask,*,repeated_generate_nums,
                         max_training_token,max_training_padding_gap,draft_accumulation_steps,
                         ce_chunk_size=128):
    if min(max_training_token,draft_accumulation_steps,ce_chunk_size)<1:
        raise ValueError('Positive packing, accumulation and CE chunk sizes required')
    stats=draft_supervision_stats(outputs,prompt_mask,repeated_generate_nums)
    if not stats['valid_examples']:return 0.,0.
    states=outputs['all_draft_input_states'];ids=outputs['all_draft_input_ids']
    rows=sorted([(ids[i],states[i],int(prompt_mask[i//repeated_generate_nums].sum()))
                 for i in stats['valid_indices']],key=lambda x:len(x[0]))
    device=model.target_model.device
    groups=[];group=[];maximum=0
    for row in rows:
        length=len(row[0])
        if group and (length*(len(group)+1)>2*max_training_token or
                      (length-maximum)*len(group)>max_training_padding_gap):
            groups.append(group);group=[]
        group.append(row);maximum=length
    if group:groups.append(group)
    total_feature=total_ce=0.
    for group in groups:
        features=torch.nn.utils.rnn.pad_sequence([r[1].to(device) for r in group],batch_first=True)
        inputs=torch.nn.utils.rnn.pad_sequence([r[0].to(device) for r in group],batch_first=True)
        length=inputs.shape[1];positions=torch.arange(length,device=device)[None,:]
        sizes=torch.tensor([len(r[0]) for r in group],device=device)[:,None]
        prompts=torch.tensor([r[2] for r in group],device=device)[:,None]
        attention=(positions<sizes).long()
        valid=(positions[:,:-1]>=prompts)&(positions[:,:-1]<sizes-1)
        counts=valid.sum(-1)
        with torch.autocast(device_type=torch.device(device).type,dtype=model.dtype,
                            enabled=torch.device(device).type=='cuda'):
            out=model(hidden_states=features,input_ids=inputs,attention_mask=attention,use_cache=False)
        predicted=out['next_feature_states'][:,:-1][valid].float()
        teacher=features[:,1:][valid].detach()
        weights=(valid/counts[:,None]).float()[valid]
        feature=2.*(F.smooth_l1_loss(predicted,teacher.float(),reduction='none').mean(-1)*weights).sum()
        actor=out['hidden_states'][:,:-1][valid].to(model.target_model.dtype)
        def soft_ce(a,t,w):
            with torch.no_grad():p=model.lm_head(t.to(model.target_model.dtype)).float().softmax(-1)
            logq=model.lm_head(a).float().log_softmax(-1)
            return -(p*logq).sum(-1).mul(w).sum()
        ce=actor.new_zeros((),dtype=torch.float32)
        for start in range(0,len(actor),ce_chunk_size):
            args=(actor[start:start+ce_chunk_size],teacher[start:start+ce_chunk_size],weights[start:start+ce_chunk_size])
            ce=ce+checkpoint(soft_ce,*args,use_reentrant=False,preserve_rng_state=False)
        ce=ce*.1
        loss=feature+ce
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite FastGRPO feature/CE loss; no optimizer step is allowed')
        total_feature+=float(feature.detach());total_ce+=float(ce.detach())
        (loss/(stats['valid_examples']*draft_accumulation_steps)).backward()
    return total_feature/stats['valid_examples'],total_ce/stats['valid_examples']
