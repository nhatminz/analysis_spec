"""Exact measured-boundary timeline; Fresh temporarily reuses shadow storage."""
from pathlib import Path
import json
import sys
import time
import torch
from tqdm.auto import tqdm
from helper.checkpointing import capture_rng_state,restore_rng_state
from motivation.state import file_hash,cpu_copy,digest,gradients,restore_gradients,isolated_rng,generation_seed,atomic_json,atomic_torch,compare_training_values
from motivation.data import tokenize
from motivation.runtime import (build_models,rollout,update_draft,target_batch,update_target,target_state,load_target,
                                learner_snapshot,load_learner,evaluate,drift_features,measured_drift,
                                prepare_target_update,commit_target_update)
from motivation.metrics import summarize,rebuild_exports
from teacher_relabel import TeacherTrace,capture,replay_old_gate,relabel,training_outputs

class TrainSampler:
    """Deterministic epoch shuffling, eight distinct IDs per iteration, resumable."""
    def __init__(self,rows,seed,batch_size):
        import random
        self.rows=rows;self.rng=random.Random(seed);self.batch_size=batch_size
        self.order=[];self.cursor=0;self.epoch=-1
    def next(self):
        if self.cursor+self.batch_size>len(self.order):
            self.order=list(range(len(self.rows)));self.rng.shuffle(self.order);self.cursor=0;self.epoch+=1
        chosen=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=self.batch_size
        return [self.rows[i] for i in chosen]
    def state_dict(self):return dict(order=self.order,cursor=self.cursor,epoch=self.epoch,rng=self.rng.getstate())
    def load_state_dict(self,state):
        self.order=state['order'];self.cursor=state['cursor'];self.epoch=state['epoch'];self.rng.setstate(state['rng'])


def compare_placebo(model,optimizer,stale,stale_loss,fresh_loss,stale_gradients,fresh_gradients):
    losses=lambda row:{k:v for k,v in row.items() if k!='gradient_sha256'}
    parts=dict(loss=compare_training_values(losses(fresh_loss),losses(stale_loss)),
               gradients=compare_training_values(fresh_gradients,stale_gradients),
               weights=compare_training_values(model.draft_model.state_dict(),stale['weights']),
               optimizer=compare_training_values(optimizer.state_dict(),stale['optimizer']))
    return dict(passed=all(p['passed'] for p in parts.values()),
                bitwise_equal=all(p['bitwise_equal'] for p in parts.values()),comparisons=parts)


def checkpoint_payload(r,s,ot,orr,os,sampler,step,completed,config_hash,manifest_hash,training_rows,attempt,attempts_at_step):
    return dict(format='simplelr_opd_policy_lag_a1a2_v2',policy_step=step,rollout_attempt=attempt,attempts_at_step=attempts_at_step,
                completed_analysis_steps=sorted(completed),
                config_sha256=config_hash,manifest_sha256=manifest_hash,target_lora=target_state(r.target_model),
                optimizer_target=cpu_copy(ot.state_dict()),reflex=learner_snapshot(r,orr),shadow=learner_snapshot(s,os),
                target_gradients=gradients(r.target_model),
                projector_pending_sum=cpu_copy(r.opd_projector_grad_sum),projector_pending_weight=cpu_copy(r.opd_projector_grad_weight),
                sampler=sampler.state_dict(),rng=cpu_copy(capture_rng_state()),training_rows=training_rows,
                # B_fast and mutable generation caches deliberately never saved.
                training_trajectory='OPD-driven')


def run(config,train,heldout,tokenizer,manifest,resume=''):
    from motivation.state import seed_all
    root=Path(config.output_dir);analysis=root/'analysis';checkpoint=root/'checkpoints/latest.pt'
    config_hash=digest(config.to_dict());manifest_hash=digest(manifest)
    seed_all(config.seed)
    if 'draft' in manifest and file_hash(config.draft_checkpoint)!=manifest['draft']['sha256']:
        raise ValueError('Draft checkpoint changed after validation')
    r,s,ot,orr,os=build_models(config)
    if not resume:
        atomic_json(root/'initialization.json',dict(target_sha256=digest(target_state(r.target_model)),
            draft_backbone_sha256=digest(s.draft_model.state_dict()),reflex_A_sha256=digest(r.opd_projector),
            projector_pretrained=manifest.get('draft',{}).get('projector_pretrained',False),
            checkpoint= config.draft_checkpoint,shadow_has_projector=False))
    if 'draft' in manifest and file_hash(config.draft_checkpoint)!=manifest['draft']['sha256']:
        raise ValueError('Draft checkpoint changed during model loading')
    sampler=TrainSampler(train,config.seed,config.batch_size)
    step=attempt=attempts_at_step=0;completed=set();training_rows=[]
    if resume:
        p=checkpoint if resume=='auto' else Path(resume)
        saved=torch.load(p,map_location='cpu',weights_only=False)
        if saved['format']!='simplelr_opd_policy_lag_a1a2_v2' or saved['config_sha256']!=config_hash or saved['manifest_sha256']!=manifest_hash:
            raise ValueError('Resume config/source/model/data identity differs; resume only an unchanged experiment')
        load_target(r.target_model,saved['target_lora']);ot.load_state_dict(saved['optimizer_target'])
        load_learner(r,orr,saved['reflex']);load_learner(s,os,saved['shadow'])
        restore_gradients(r.target_model,saved['target_gradients'])
        r.opd_projector_grad_sum.copy_(saved['projector_pending_sum'].to(r.device))
        r.opd_projector_grad_weight.copy_(saved['projector_pending_weight'].to(r.device))
        sampler.load_state_dict(saved['sampler']);restore_rng_state(saved['rng'])
        step=saved['policy_step'];attempt=saved['rollout_attempt'];attempts_at_step=saved['attempts_at_step']
        completed=set(saved['completed_analysis_steps']);training_rows=saved['training_rows']
        actual=dict(target_lora=target_state(r.target_model),optimizer_target=ot.state_dict(),
                    reflex=learner_snapshot(r,orr),shadow=learner_snapshot(s,os),
                    target_gradients=gradients(r.target_model),projector_pending_sum=r.opd_projector_grad_sum,
                    projector_pending_weight=r.opd_projector_grad_weight,sampler=sampler.state_dict(),rng=capture_rng_state())
        for key,value in actual.items():
            if digest(value)!=digest(saved[key]):raise RuntimeError(f'Resume did not restore exact {key}')
        atomic_json(root/'resume_audit.json',dict(state_identity_passed=True,policy_step=step,
                    rollout_attempt=attempt,completed_analysis_steps=sorted(completed),verified_fields=sorted(actual)))
        del saved,actual
    elif checkpoint.exists():raise ValueError('Output already contains a checkpoint; use --resume auto or a new output directory')
    rebuild_exports(analysis,completed)
    # A crash may leave uncommitted journals. Replay from the last durable
    # checkpoint overwrites them; never infer completion from file existence.
    for marker in (analysis/'boundaries').glob('step_*/complete.json'):
        if int(marker.parent.name.split('_')[-1]) not in completed:marker.unlink()
    with tqdm(total=config.train_steps,initial=step,desc="Policy training",unit="step",dynamic_ncols=True) as progress:
        while step<config.train_steps:
            next_step=step+1;measured=next_step in config.eval_steps;started=time.perf_counter()
            if attempts_at_step>=config.max_attempts_per_step:
                raise RuntimeError(f'No nonzero GRPO gradient after {attempts_at_step} rollout attempts at policy_step={step}. '
                                   'See training_metrics.json for real rewards and skip reasons; no dummy optimizer steps or drift were recorded.')
            attempt+=1;attempts_at_step+=1
            progress.set_postfix(stage="main rollout",attempt=attempt,skipped=attempt-step-1)
            torch.cuda.reset_peak_memory_stats()
            rows=sampler.next();batch=tokenize(rows,tokenizer)
            lengths=batch['attention_mask'].sum(-1)
            if int(lengths.max())>=config.max_length or int(lengths.max())>config.max_prompt_length:
                raise ValueError('Train prompt exceeds configured length. Choose a sufficient max_length/max_prompt_length; no silent truncation')
            if any(row['split']!='train' for row in rows):raise RuntimeError('Held-out prompt reached optimizer input')
            if len({row['id'] for row in rows})!=config.batch_size:raise RuntimeError('Training prompts are not distinct')
            prompt_ids=[row['id'] for row in rows]
            old_policy=target_state(r.target_model);old_id=digest(old_policy)
            out_r=rollout(r,batch,tokenizer,config,'opd_reflex',generation_seed(config.seed,attempt,'main_training'))
            records,rewards=target_batch(rows,batch,out_r,tokenizer,config)
            tqdm.write(json.dumps(dict(event='main_rollout_complete',rollout_attempt=attempt,
                responses=len(out_r['generated_token_ids']),generated_tokens=sum(map(len,out_r['generated_token_ids'])),
                rewards=rewards,target_trajectories=len(records))))
            sys.stdout.flush()
            measured=measured and bool(records)
            pre_shadow=learner_snapshot(s,os) if measured else None
            teacher_trace=TeacherTrace(config.responses_per_prompt) if measured else None
            progress.set_postfix(stage="shadow rollout",attempt=attempt,skipped=attempt-step-1)
            out_s=rollout(s,batch,tokenizer,config,'fastgrpo',generation_seed(config.seed,attempt,'shadow_training'),teacher_trace=teacher_trace)
            tqdm.write(json.dumps(dict(event='shadow_rollout_complete',rollout_attempt=attempt,
                responses=len(out_s['generated_token_ids']),generated_tokens=sum(map(len,out_s['generated_token_ids'])))))
            sys.stdout.flush()
            # Capture all exact original contexts before any learner/target update.
            captured=capture(out_s,batch['input_ids'],batch['attention_mask'],prompt_ids,config.responses_per_prompt) if measured else None
            grpo_id=digest(records)
            progress.set_postfix(stage="draft update",attempt=attempt,skipped=attempt-step-1)
            loss_r=update_draft(r,orr,out_r,batch['attention_mask'],config,projector=True)
            shadow_update_rng=cpu_copy(capture_rng_state())
            shadow_gradients={} if measured else None
            loss_s=update_draft(s,os,out_s,batch['attention_mask'],config,gradient_snapshot=shadow_gradients)
            rollout_counters={}
            for name,out in (('reflex',out_r),('shadow',out_s)):
                rollout_counters[name]=dict(responses=len(out['generated_token_ids']),
                    generated_tokens=sum(map(len,out['generated_token_ids'])),accepted_length_sum=sum(out['response_accepted_length_sum']),
                    verification_rounds=sum(out['response_verification_rounds']))
            # Train outputs can be large. Fresh is represented by CPU captured
            # features/context and a CPU optimizer snapshot, never a second target.
            del out_r,out_s
            prepared=prepare_target_update(r.target_model,ot,records,config)
            training_peak=torch.cuda.max_memory_allocated()
            tqdm.write(json.dumps(dict(event='gradients_prepared',rollout_attempt=attempt,
                peak_training_cuda_allocated_bytes=training_peak,**prepared)))
            sys.stdout.flush()
            if prepared['target_skip_reason'] is not None:
                progress.set_postfix(stage=prepared['target_skip_reason'],attempt=attempt,skipped=attempt-step)
                training_rows.append(dict(policy_step=step,rollout_attempt=attempt,prompt_ids=prompt_ids,
                    train_response_count_per_learner=len(rows)*config.responses_per_prompt,rewards=rewards,
                    reflex_loss=loss_r,shadow_loss=loss_s,main_grpo_buffer_sha256=grpo_id,
                    target_old_sha256=old_id,target_new_sha256=old_id,zero_drift=True,
                    transition_kind='skipped_no_target_step',**prepared))
                training_rows[-1].update(rollout_counters=rollout_counters,peak_training_cuda_allocated_bytes=training_peak)
                state=checkpoint_payload(r,s,ot,orr,os,sampler,step,completed,config_hash,manifest_hash,training_rows,attempt,attempts_at_step)
                atomic_torch(checkpoint,state);del state
                atomic_json(root/'training_metrics.json',training_rows)
                tqdm.write(json.dumps(dict(policy_step=step,rollout_attempt=attempt,skipped=prepared['target_skip_reason'])))
                sys.stdout.flush()
                del pre_shadow,captured,teacher_trace,shadow_gradients
                continue
            eval_rows=heldout[:config.n_eval(next_step)] if measured else []
            eval_records=[];old_drift=None;gate=None
            frozen_r=digest(r.draft_model.state_dict())
            if measured:
                progress.set_postfix(stage="A1/A2 old",prompts=len(eval_rows),attempt=attempt)
                old_guard=digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model),
                                      reflex_optimizer=orr.state_dict(),shadow_optimizer=os.state_dict()))
                boundary=analysis/'boundaries'/f'step_{next_step}';boundary.mkdir(parents=True,exist_ok=True)
                with isolated_rng():
                    gate=replay_old_gate(r.target_model,captured,config,teacher_trace=teacher_trace)
                    old_drift=drift_features(r.target_model,eval_rows,tokenizer)
                tqdm.write(json.dumps(dict(event='old_replay_gate_passed',at_policy_step=next_step,
                    hidden_max_abs=max(x['hidden_max_abs'] for x in gate['examples']),
                    distribution_tv_max=max(x['distribution_tv_max'] for x in gate['examples']))))
                sys.stdout.flush()
                if next_step in config.zero_update_steps:
                    # Isolated Stale/Fresh placebo. No target optimizer call, no
                    # replacement of the real transition at this measurement.
                    control_stale=learner_snapshot(s,os);control_rng=cpu_copy(capture_rng_state())
                    try:
                        load_learner(s,os,pre_shadow)
                        control_gradients={}
                        with isolated_rng(state=shadow_update_rng):
                            control_loss=update_draft(s,os,training_outputs(captured,s.device),batch['attention_mask'],config,
                                                     gradient_snapshot=control_gradients)
                        control=compare_placebo(s,os,control_stale,loss_s,control_loss,shadow_gradients,control_gradients)
                        atomic_json(boundary/'zero_drift_control.json',dict(**control,role='isolated_shadow_placebo',
                            at_policy_step=next_step,target_optimizer_steps=0,target_sha256=old_id,
                            loss_stale=loss_s,loss_fresh=control_loss))
                        if not control['passed']:raise RuntimeError(f'Isolated zero-drift control failed numerical checks: {control}')
                    finally:
                        load_learner(s,os,control_stale);restore_rng_state(control_rng)
                    del control_stale,control_gradients
                for condition in ('reflex_off/old','reflex_on/old'):
                    eval_records.extend(evaluate(r,eval_rows,tokenizer,config,next_step,condition,old_id,None))
                if digest(r.draft_model.state_dict())!=frozen_r:raise RuntimeError('A2 old eval altered trained Reflex state')
                if old_guard!=digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model),
                                          reflex_optimizer=orr.state_dict(),shadow_optimizer=os.state_dict())):
                    raise RuntimeError('Old-target replay/evaluation modified training state')
            if digest(records)!=grpo_id:raise RuntimeError('Shadow/Fresh/eval modified the main GRPO buffer')
            progress.set_postfix(stage="target update",attempt=attempt)
            target_result=commit_target_update(ot,prepared)
            step+=1  # ONLY after exactly one completed optimizer.step()
            attempts_at_step=0
            new_id=digest(target_state(r.target_model));zero_drift=old_id==new_id
            if measured:
                progress.set_postfix(stage="A1/A2 new",prompts=len(eval_rows),attempt=attempt)
                target_eval_guard=digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model)))
                stale=learner_snapshot(s,os)
                training_rng=cpu_copy(capture_rng_state())
                with isolated_rng():
                    fresh_sequences=relabel(r.target_model,captured,zero_drift=zero_drift,teacher_trace=teacher_trace)
                    drift=measured_drift(r.target_model,old_drift,drift_features(r.target_model,eval_rows,tokenizer))
                eval_records.extend(evaluate(s,eval_rows,tokenizer,config,step,'stale/new',old_id,new_id))
                try:
                    # Disposable Fresh fork reuses GPU shadow parameter storage;
                    # pre/stale weights and optimizer moments are OWNED CPU copies.
                    load_learner(s,os,pre_shadow)
                    fresh_gradients={} if zero_drift else None
                    with isolated_rng(state=shadow_update_rng):
                        loss_f=update_draft(s,os,training_outputs(fresh_sequences,s.device),batch['attention_mask'],config,
                                           gradient_snapshot=fresh_gradients)
                    if zero_drift:
                        placebo=compare_placebo(s,os,stale,loss_s,loss_f,shadow_gradients,fresh_gradients)
                        atomic_json(boundary/'observed_zero_drift_comparison.json',placebo)
                        if not placebo['passed']:raise RuntimeError(f'A1 zero-drift placebo failed numerical checks: {placebo}')
                    del fresh_gradients
                    eval_records.extend(evaluate(s,eval_rows,tokenizer,config,step,'fresh/new',old_id,new_id))
                finally:
                    load_learner(s,os,stale);restore_rng_state(training_rng)
                for condition in ('reflex_off/new','reflex_on/new'):
                    eval_records.extend(evaluate(r,eval_rows,tokenizer,config,step,condition,old_id,new_id))
                if digest(r.draft_model.state_dict())!=frozen_r:raise RuntimeError('A2 must freeze SAME R and A across all four conditions')
                if digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model)))!=target_eval_guard:
                    raise RuntimeError('Fresh/eval changed target, target optimizer or gradients')
                if digest(s.draft_model.state_dict())!=digest(stale['weights']) or digest(os.state_dict())!=digest(stale['optimizer']):
                    raise RuntimeError('Persistent shadow was not restored after Fresh')
                for item in eval_records:item['target_new_sha256']=new_id
                metrics=summarize(eval_records,step,len(eval_rows),seed=generation_seed(config.seed,step,'bootstrap'),samples=config.bootstrap_samples)
                metrics.update(drift,target_old_sha256=old_id,target_new_sha256=new_id,zero_drift=zero_drift,
                               zero_drift_placebo_passed=zero_drift,training_trajectory='OPD-driven',research_result=not(config.smoke or config.validation_run),
                               eval_max_new_tokens=config.eval_max_new_tokens,aal_length_capped=True,
                               shadow_stale_feature_loss=loss_s['feature_loss'],shadow_stale_distribution_loss=loss_s['distribution_loss'],
                               shadow_fresh_feature_loss=loss_f['feature_loss'],shadow_fresh_distribution_loss=loss_f['distribution_loss'],
                               old_replay_hidden_max_abs=max(x['hidden_max_abs'] for x in gate['examples']),
                               old_replay_distribution_tv_max=max(x['distribution_tv_max'] for x in gate['examples']),
                               transition_kind='observed_zero_drift' if zero_drift else 'genuine_target_update',
                               isolated_zero_drift_control_passed=step in config.zero_update_steps,
                               target_parameters_changed=not zero_drift,policy_drift_observed=drift['teacher_policy_tv_full_softmax']>0,
                               peak_training_cuda_allocated_bytes=training_peak,
                               boundary_peak_cuda_allocated_bytes=max(training_peak,max(r['peak_cuda_allocated_bytes'] for r in eval_records)),
                               boundary_wall_s=time.perf_counter()-started)
                atomic_json(boundary/'alignment_gate.json',gate)
                # Compact trace is auditable without persisting huge full-V arrays.
                atomic_torch(boundary/'shadow_sequences.pt',dict(sequences=captured,teacher_head_sha256=gate['teacher_head_sha256'],
                             teacher_trace=teacher_trace,optimizer_pre_sha256=digest(pre_shadow['optimizer']),draft_pre_sha256=digest(pre_shadow['weights']),
                             rng_pre_update=shadow_update_rng,old_target_lora=old_policy,loss_stale=loss_s,loss_fresh=loss_f,
                             relabeled_teacher_features=[q.teacher_features for q in fresh_sequences]))
                atomic_json(boundary/'results.json',dict(per_response=eval_records,metrics=metrics))
                completed.add(step)
                del pre_shadow,stale,captured,fresh_sequences,eval_records,shadow_gradients
            training_rows.append(dict(policy_step=step,rollout_attempt=attempt,prompt_ids=prompt_ids,train_response_count_per_learner=len(rows)*config.responses_per_prompt,
                                      rewards=rewards,reflex_loss=loss_r,shadow_loss=loss_s,main_grpo_buffer_sha256=grpo_id,
                                      target_old_sha256=old_id,target_new_sha256=new_id,zero_drift=zero_drift,
                                      transition_kind='observed_zero_drift' if zero_drift else 'genuine_target_update',**target_result))
            training_rows[-1].update(rollout_counters=rollout_counters,peak_training_cuda_allocated_bytes=training_peak)
            if measured or step%config.save_every==0 or step==config.train_steps:
                progress.set_postfix(stage="save checkpoint",attempt=attempt)
                state=checkpoint_payload(r,s,ot,orr,os,sampler,step,completed,config_hash,manifest_hash,training_rows,attempt,attempts_at_step)
                atomic_torch(checkpoint,state);del state
                rebuild_exports(analysis,completed)
                atomic_json(root/'training_metrics.json',training_rows)
            tqdm.write(json.dumps(dict(policy_step=step,measured=measured,zero_drift=zero_drift,target_trajectories=len(records),
                                  wall_s=time.perf_counter()-started)))
            sys.stdout.flush()
            progress.set_postfix(stage="complete",attempt=attempt,skipped=attempt-step)
            progress.update(1)
    return dict(policy_step=step,completed_analysis_steps=sorted(completed),checkpoint=str(checkpoint))
