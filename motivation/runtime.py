"""Production kernels plus the single-step scientific protocol adapter."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import gc
import time
import numpy as np
import torch
from motivation.state import cpu_copy,digest,gradients,restore_gradients,isolated_rng,generation_seed
from motivation.data import tokenize
from teacher_relabel import base_model,distribution


@contextmanager
def evaluation_state(models):
    """Suspend training RNG, modes, gradients and reusable inference workspaces."""
    unique=list({id(m):m for model in models for m in model.modules()}.values())
    modes=[m.training for m in unique]
    parameters=list({id(p):p for model in models for p in model.parameters()}.values())
    trainable=[p.requires_grad for p in parameters]
    saved_caches=[]
    for model in models:
        attrs={n:v for n,v in vars(model).items() if n.startswith('_opd_')}
        saved_caches.append(attrs)
        for name in attrs:delattr(model,name)
    with isolated_rng():
        try:
            for m in unique:m.training=False
            for p in parameters:p.requires_grad_(False)
            yield
        finally:
            if torch.cuda.is_available():torch.cuda.synchronize()
            for model,attrs in zip(models,saved_caches):
                for name in list(vars(model)):
                    if name.startswith('_opd_'):delattr(model,name)
                for name,value in attrs.items():setattr(model,name,value)
            for m,mode in zip(unique,modes):m.training=mode
            for p,flag in zip(parameters,trainable):p.requires_grad_(flag)


def build_models(config):
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import get_peft_model,LoraConfig,TaskType
    from helper.fastgrpo_model import FastGRPOModel
    from helper.opd_optimizer import draft_optimizer
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16}[config.dtype]
    target=AutoModelForCausalLM.from_pretrained(config.model,local_files_only=True,torch_dtype=dtype,
                                               attn_implementation=config.attention_implementation).cuda().eval()
    dc=AutoConfig.from_pretrained(config.model,local_files_only=True)
    dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=dtype
    # Construct BOTH before PEFT: shared target remains frozen, and a later
    # wrapper constructor must not switch off already-enabled target LoRA grads.
    r=FastGRPOModel(deepcopy(dc),target).cuda();s=FastGRPOModel(deepcopy(dc),target).cuda()
    for model in (r,s):
        model.load_model(config.draft_checkpoint)
        for p in model.draft_model.parameters():p.requires_grad_(True)
    if digest(r.draft_model.state_dict())!=digest(s.draft_model.state_dict()):raise RuntimeError('Draft initializations differ')
    r.enable_opd(config.opd_rank)
    for p in target.parameters():p.requires_grad_(False)
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=32,lora_dropout=0.,
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    target=get_peft_model(target,lora)
    if config.target_adapter:target.load_adapter(config.target_adapter,adapter_name='default',is_trainable=True)
    r.target_model=target;s.target_model=target
    if r.embed_tokens is not s.embed_tokens or r.lm_head is not s.lm_head:raise RuntimeError('Frozen head/embedding identity differs')
    if r.lm_head.weight.requires_grad:raise RuntimeError('Teacher head must stay frozen')
    if any(a.data_ptr()==b.data_ptr() for a,b in zip(r.draft_model.parameters(),s.draft_model.parameters())):
        raise RuntimeError('Shadow and Reflex share mutable weights')
    opt_t=torch.optim.AdamW(target.parameters(),lr=config.target_lr)
    opt_r=draft_optimizer(r.draft_model,config.draft_lr,config.opd_projector_lr)
    opt_s=draft_optimizer(s.draft_model,config.draft_lr)
    r.eval();s.eval()  # reference keeps target/draft dropout disabled while training gradients
    return r,s,opt_t,opt_r,opt_s


def generator_kwargs(config,*,method,feedback=True,train=False):
    return dict(method=method,do_sample=True,repeated_generate_nums=config.responses_per_prompt if train else 1,
                temperature=config.temperature,top_p=config.top_p,top_k=config.top_k,
                verification_capacity=config.verification_capacity,max_verification_num=config.max_verification_num,
                max_draft_token_length=config.max_draft_token_length,max_draft_k=config.max_draft_k,
                min_draft_token_length=config.min_draft_token_length,draft_token_length_c=config.draft_token_length_c,
                statistical_time=False,return_all_draft_input=train,
                opd_rank=config.opd_rank,opd_topk=config.opd_topk,opd_fast_lr=config.opd_fast_lr if feedback else 0.,
                opd_visited_weight=config.opd_visited_weight,opd_frontier_weight=config.opd_frontier_weight,
                opd_update_stream=config.opd_update_stream,opd_train_projector=train and method=='opd_reflex',
                opd_diagnostics=not train,opd_backend='triton',kv_gather_strategy='stacked')


def rollout(model,batch,tokenizer,config,method,seed,teacher_trace=None):
    from helper.specualtive_generate import speculative_generate
    with isolated_rng(seed),torch.inference_mode():
        out=speculative_generate(model=model,input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                                 tokenizer=tokenizer,max_length=config.max_length,
                                 **({'teacher_trace':teacher_trace} if teacher_trace is not None else {}),
                                 **generator_kwargs(config,method=method,train=True))
    torch.cuda.synchronize()
    return out


def update_draft(model,optimizer,outputs,prompt_mask,config,*,projector=False):
    from helper.fastgrpo_training import training_draft_model
    normal=dict(outputs)
    for key in ('all_draft_input_states','all_draft_input_ids'):
        normal[key]=[x.to(model.device).clone() for x in outputs[key]]
    optimizer.zero_grad(set_to_none=True)
    loss=training_draft_model(model,normal,prompt_mask,repeated_generate_nums=config.responses_per_prompt,
                             max_training_token=config.max_training_token,max_training_padding_gap=config.max_training_padding_gap,
                             draft_accumulation_steps=1)
    if not np.isfinite(loss).all():raise RuntimeError('Nonfinite production draft objective; refusing optimizer step')
    if projector:model.apply_opd_projector_gradient()
    grads=gradients(model.draft_model)
    if not grads or any(not torch.isfinite(g).all() for g in grads.values()):raise RuntimeError('Missing/nonfinite draft gradients')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return dict(feature_loss=loss[0],distribution_loss=loss[1],gradient_sha256=digest(grads))


def target_batch(rows,batch,outputs,tokenizer,config):
    """Only accepted Reflex rollout IDs enter GRPO. Decode ONLY for rewards."""
    from helper.rewards import accuracy_reward_func,format_reward_func
    records=[];all_rewards=[]
    for j,row in enumerate(rows):
        prompt=batch['input_ids'][j][batch['attention_mask'][j].bool()].tolist()
        responses=outputs['generated_token_ids'][j*config.responses_per_prompt:(j+1)*config.responses_per_prompt]
        texts=[tokenizer.decode(ids,skip_special_tokens=True) for ids in responses]
        rewards=np.array(accuracy_reward_func(texts,[row['answer']]*len(texts)))+.2*np.array(format_reward_func(texts))
        all_rewards.append(rewards.tolist())
        # Preserve the reference exclusion of zero-variance groups. If all are
        # excluded, execute a real zero-gradient target optimizer step and label
        # zero drift instead of changing the meaning of policy_step.
        if rewards.std()==0:continue
        advantages=(rewards-rewards.mean())/rewards.std()
        for ids,adv in zip(responses,advantages):
            records.append(dict(ids=prompt+ids,mask=[0]*(len(prompt)-1)+[1]*(len(ids)+1),
                                advantage=float(adv),prompt_id=row['id']))
    return records,all_rewards


def update_target(target,optimizer,records,config,*,zero_update=False):
    from helper.fastgrpo_training import compute_target_loss
    optimizer.zero_grad(set_to_none=True);total=0.
    ordered=sorted(records,key=lambda row:len(row['ids']))
    groups=[];group=[];maximum=0
    for row in ordered:
        length=len(row['ids'])
        if group and (max(maximum,length)*(len(group)+1)>config.max_training_token or
                      (length-maximum)*len(group)>config.max_training_padding_gap):
            groups.append(group);group=[];maximum=0
        group.append(row);maximum=max(maximum,length)
    if group:groups.append(group)
    for group in groups:
        length=max(len(r['ids']) for r in group);device=target.device
        ids=torch.tensor([r['ids']+[0]*(length-len(r['ids'])) for r in group],device=device)
        attn=torch.tensor([[1]*len(r['ids'])+[0]*(length-len(r['ids'])) for r in group],device=device)
        mask=torch.tensor([r['mask']+[0]*(length-len(r['ids'])) for r in group],device=device)
        reward=torch.tensor([r['advantage'] for r in group],device=device).unsqueeze(-1)
        with target.disable_adapter(),torch.no_grad():reference=target(input_ids=ids,attention_mask=attn,use_cache=False).logits
        logits=target(input_ids=ids,attention_mask=attn,use_cache=False).logits
        loss,*_=compute_target_loss(logits,reference,None,ids,mask,reward,config.epsilon,config.beta,0)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite GRPO loss')
        (loss/max(len(records),1)).backward();total+=float(loss.detach())
        del logits,reference,loss
    if not records or zero_update:
        # AdamW momentum/weight decay would change a zero-gradient parameter.
        # Temporarily set LR=0 to make this a true no-change placebo, while
        # calling exactly one real optimizer.step(). Optimizer moments advance.
        for p in target.parameters():
            if p.requires_grad:p.grad=torch.zeros_like(p)
        rates=[g['lr'] for g in optimizer.param_groups]
        for g in optimizer.param_groups:g['lr']=0.
        try:optimizer.step()
        finally:
            for g,lr in zip(optimizer.param_groups,rates):g['lr']=lr
    else:optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(target_loss=total/max(len(records),1),target_trajectories=len(records),target_optimizer_steps=1,
                zero_update_control=zero_update,all_reward_groups_excluded=not records)


def target_state(target):
    from peft import get_peft_model_state_dict
    return cpu_copy(get_peft_model_state_dict(target))


def load_target(target,state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(target,state)
    if digest(target_state(target))!=digest(state):raise RuntimeError('Target adapter restoration failed')


def learner_snapshot(model,optimizer):
    return dict(weights=cpu_copy(model.draft_model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                gradients=gradients(model.draft_model))


def load_learner(model,optimizer,snapshot):
    model.draft_model.load_state_dict(snapshot['weights']);optimizer.load_state_dict(deepcopy(snapshot['optimizer']))
    restore_gradients(model.draft_model,snapshot['gradients'])


@torch.no_grad()
def drift_features(target,eval_rows,tokenizer):
    features=[]
    for row in eval_rows:
        batch=tokenize([row],tokenizer);ids=batch['input_ids'].to(target.device)
        features.append(base_model(target).model(input_ids=ids,attention_mask=batch['attention_mask'].to(target.device),
                                                 use_cache=False).last_hidden_state[0,-1].cpu().clone())
    return torch.stack(features)


@torch.no_grad()
def measured_drift(target,old,new):
    head=base_model(target).lm_head;tv=[]
    for a,b in zip(old,new):tv.append(float(.5*(distribution(head,a[None])-distribution(head,b[None])).abs().sum()))
    return dict(teacher_policy_tv_full_softmax=float(np.mean(tv)),teacher_policy_tv_max=max(tv),
                teacher_policy_drift_contexts=len(tv),teacher_policy_drift_distribution='full softmax T=1 no top-p/top-k, fixed prompt-final positions')


def evaluate(model,rows,tokenizer,config,step,condition,target_old_id,target_new_id):
    from helper.specualtive_generate import speculative_generate
    method='opd_reflex' if condition.startswith('reflex_') else 'fastgrpo'
    feedback=condition.startswith('reflex_on')
    before=digest(model.draft_model.state_dict());head_id=digest(model.lm_head.state_dict())
    projector_id=digest(model.opd_projector) if method=='opd_reflex' else None
    records=[]
    with evaluation_state([model]):
        for row in rows:
            batch=tokenize([row],tokenizer);prompt_length=int(batch['attention_mask'].sum())
            seed=generation_seed(config.seed,step,row['id'])
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with isolated_rng(seed),torch.inference_mode():
                out=speculative_generate(model=model,input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],tokenizer=tokenizer,
                    # Optional cap truncates the final accepted path inside the
                    # verifier before feedback and counters; ordinary production is unchanged.
                    max_length=prompt_length+config.eval_max_new_tokens,max_new_tokens=config.eval_max_new_tokens,
                    **generator_kwargs(config,method=method,feedback=feedback,train=False))
            torch.cuda.synchronize()
            if len(out['generated_token_ids'][0])>config.eval_max_new_tokens:raise RuntimeError('Evaluation exceeded common response cap')
            if len(out['generated_token_ids'])!=1:raise RuntimeError('Eval produced multiple responses per test prompt')
            denom=int(out['response_verification_rounds'][0]);num=int(out['response_accepted_length_sum'][0])
            if denom<=0:raise RuntimeError('No verification rounds; cannot report AAL')
            if method=='opd_reflex' and not feedback and (out['opd_updates']!=0 or out.get('opd_final_b_norm',0)!=0):
                raise RuntimeError('A2 OFF changed B despite lr=0')
            records.append(dict(policy_step=step,condition=condition,prompt_id=row['id'],question_sha256=row['question_sha256'],
                                sample_index=0,generation_seed=seed,accepted_draft_length_sum=num,verification_rounds=denom,
                                generated_token_length=len(out['generated_token_ids'][0]),
                                generated_token_ids=out['generated_token_ids'][0],
                                proposed_draft_tokens=int(out['total_proposed_draft_tokens']),accepted_proposal_tokens=int(out['total_accepted_draft_tokens']),
                                target_old_sha256=target_old_id,target_new_sha256=target_new_id,
                                target_sha256=target_old_id if condition.endswith('/old') else target_new_id,
                                draft_sha256=before,learned_A_sha256=projector_id,teacher_head_sha256=head_id,
                                eval_max_new_tokens=config.eval_max_new_tokens,stopping='exact emitted-token cap; clamp verified path before feedback/counters',
                                response_count=1,temperature=config.temperature,top_p=config.top_p,top_k=config.top_k,
                                runtime_s=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                **{k:v for k,v in out.items() if k.startswith('opd_')}))
    if digest(model.draft_model.state_dict())!=before:raise RuntimeError('Evaluation changed frozen draft/A')
    return records
