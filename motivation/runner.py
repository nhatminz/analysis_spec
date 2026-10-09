"""Exact measured-boundary timeline; Fresh temporarily reuses shadow storage."""
from pathlib import Path
import json
import time
import torch
from helper.checkpointing import capture_rng_state,restore_rng_state
from motivation.state import file_hash,cpu_copy,digest,gradients,restore_gradients,isolated_rng,generation_seed,atomic_json,atomic_torch
from motivation.data import tokenize
from motivation.runtime import (build_models,rollout,update_draft,target_batch,update_target,target_state,load_target,
                                learner_snapshot,load_learner,evaluate,drift_features,measured_drift)
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


def checkpoint_payload(r,s,ot,orr,os,sampler,step,completed,config_hash,manifest_hash,training_rows):
    return dict(format='simplelr_opd_policy_lag_a1a2_v1',policy_step=step,completed_analysis_steps=sorted(completed),
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
    if 'draft' in manifest and file_hash(config.draft_checkpoint)!=manifest['draft']['sha256']:
        raise ValueError('Draft checkpoint changed during model loading')
    sampler=TrainSampler(train,config.seed,config.batch_size)
    step=0;completed=set();training_rows=[]
    if resume:
        p=checkpoint if resume=='auto' else Path(resume)
        saved=torch.load(p,map_location='cpu',weights_only=False)
        if saved['format']!='simplelr_opd_policy_lag_a1a2_v1' or saved['config_sha256']!=config_hash or saved['manifest_sha256']!=manifest_hash:
            raise ValueError('Resume config/source/model/data identity differs; resume only an unchanged experiment')
        load_target(r.target_model,saved['target_lora']);ot.load_state_dict(saved['optimizer_target'])
        load_learner(r,orr,saved['reflex']);load_learner(s,os,saved['shadow'])
        restore_gradients(r.target_model,saved['target_gradients'])
        r.opd_projector_grad_sum.copy_(saved['projector_pending_sum'].to(r.device))
        r.opd_projector_grad_weight.copy_(saved['projector_pending_weight'].to(r.device))
        sampler.load_state_dict(saved['sampler']);restore_rng_state(saved['rng'])
        step=saved['policy_step'];completed=set(saved['completed_analysis_steps']);training_rows=saved['training_rows'];del saved
    elif checkpoint.exists():raise ValueError('Output already contains a checkpoint; use --resume auto or a new output directory')
    rebuild_exports(analysis,completed)
    # A crash may leave uncommitted journals. Replay from the last durable
    # checkpoint overwrites them; never infer completion from file existence.
    for marker in (analysis/'boundaries').glob('step_*/complete.json'):
        if int(marker.parent.name.split('_')[-1]) not in completed:marker.unlink()
    while step<config.train_steps:
        next_step=step+1;measured=next_step in config.eval_steps;started=time.perf_counter()
        rows=sampler.next();batch=tokenize(rows,tokenizer)
        lengths=batch['attention_mask'].sum(-1)
        if int(lengths.max())>=config.max_length or int(lengths.max())>config.max_prompt_length:
            raise ValueError('Train prompt exceeds configured length. Choose a sufficient max_length/max_prompt_length; no silent truncation')
        if any(row['split']!='train' for row in rows):raise RuntimeError('Held-out prompt reached optimizer input')
        if len({row['id'] for row in rows})!=config.batch_size:raise RuntimeError('Training prompts are not distinct')
        prompt_ids=[row['id'] for row in rows]
        old_policy=target_state(r.target_model);old_id=digest(old_policy)
        pre_shadow=learner_snapshot(s,os) if measured else None
        out_r=rollout(r,batch,tokenizer,config,'opd_reflex',generation_seed(config.seed,next_step,'main_training'))
        teacher_trace=TeacherTrace(config.responses_per_prompt) if measured else None
        out_s=rollout(s,batch,tokenizer,config,'fastgrpo',generation_seed(config.seed,next_step,'shadow_training'),teacher_trace=teacher_trace)
        # Capture all exact original contexts before any learner/target update.
        captured=capture(out_s,batch['input_ids'],batch['attention_mask'],prompt_ids,config.responses_per_prompt) if measured else None
        records,rewards=target_batch(rows,batch,out_r,tokenizer,config)
        grpo_id=digest(records)
        loss_r=update_draft(r,orr,out_r,batch['attention_mask'],config,projector=True)
        shadow_update_rng=cpu_copy(capture_rng_state())
        loss_s=update_draft(s,os,out_s,batch['attention_mask'],config)
        # Train outputs can be large. Fresh is represented by CPU captured
        # features/context and a CPU optimizer snapshot, never a second target.
        del out_r,out_s
        eval_rows=heldout[:config.n_eval(next_step)] if measured else []
        eval_records=[];old_drift=None;gate=None
        frozen_r=digest(r.draft_model.state_dict())
        if measured:
            old_guard=digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model),
                                  reflex_optimizer=orr.state_dict(),shadow_optimizer=os.state_dict()))
            boundary=analysis/'boundaries'/f'step_{next_step}';boundary.mkdir(parents=True,exist_ok=True)
            with isolated_rng():
                gate=replay_old_gate(r.target_model,captured,config,teacher_trace=teacher_trace)
                old_drift=drift_features(r.target_model,eval_rows,tokenizer)
            for condition in ('reflex_off/old','reflex_on/old'):
                eval_records.extend(evaluate(r,eval_rows,tokenizer,config,next_step,condition,old_id,None))
            if digest(r.draft_model.state_dict())!=frozen_r:raise RuntimeError('A2 old eval altered trained Reflex state')
            if old_guard!=digest(dict(target=target_state(r.target_model),optimizer=ot.state_dict(),gradients=gradients(r.target_model),
                                      reflex_optimizer=orr.state_dict(),shadow_optimizer=os.state_dict())):
                raise RuntimeError('Old-target replay/evaluation modified training state')
        if digest(records)!=grpo_id:raise RuntimeError('Shadow/Fresh/eval modified the main GRPO buffer')
        target_result=update_target(r.target_model,ot,records,config,zero_update=next_step in config.zero_update_steps)
        step+=1  # ONLY after exactly one completed optimizer.step()
        new_id=digest(target_state(r.target_model));zero_drift=old_id==new_id
        if measured:
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
                with isolated_rng(state=shadow_update_rng):
                    loss_f=update_draft(s,os,training_outputs(fresh_sequences,s.device),batch['attention_mask'],config)
                if zero_drift:
                    if loss_s!=loss_f or digest(s.draft_model.state_dict())!=digest(stale['weights']) or digest(os.state_dict())!=digest(stale['optimizer']):
                        raise RuntimeError('A1 zero-drift placebo failed: stale/fresh loss, gradients, weights or optimizer differ')
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
                           zero_drift_placebo_passed=zero_drift,training_trajectory='OPD-driven',research_result=not config.smoke,
                           eval_max_new_tokens=config.eval_max_new_tokens,aal_length_capped=True,
                           shadow_stale_feature_loss=loss_s['feature_loss'],shadow_stale_distribution_loss=loss_s['distribution_loss'],
                           shadow_fresh_feature_loss=loss_f['feature_loss'],shadow_fresh_distribution_loss=loss_f['distribution_loss'],
                           old_replay_hidden_max_abs=max(x['hidden_max_abs'] for x in gate['examples']),
                           old_replay_distribution_tv_max=max(x['distribution_tv_max'] for x in gate['examples']),
                           boundary_wall_s=time.perf_counter()-started)
            atomic_json(boundary/'alignment_gate.json',gate)
            # Compact trace is auditable without persisting huge full-V arrays.
            atomic_torch(boundary/'shadow_sequences.pt',dict(sequences=captured,teacher_head_sha256=gate['teacher_head_sha256'],
                         teacher_trace=teacher_trace,optimizer_pre_sha256=digest(pre_shadow['optimizer']),draft_pre_sha256=digest(pre_shadow['weights']),
                         rng_pre_update=shadow_update_rng,old_target_lora=old_policy,loss_stale=loss_s,loss_fresh=loss_f,
                         relabeled_teacher_features=[q.teacher_features for q in fresh_sequences]))
            atomic_json(boundary/'results.json',dict(per_response=eval_records,metrics=metrics))
            completed.add(step)
            del pre_shadow,stale,captured,fresh_sequences,eval_records
        training_rows.append(dict(policy_step=step,prompt_ids=prompt_ids,train_response_count_per_learner=len(rows)*config.responses_per_prompt,
                                  rewards=rewards,reflex_loss=loss_r,shadow_loss=loss_s,main_grpo_buffer_sha256=grpo_id,
                                  target_old_sha256=old_id,target_new_sha256=new_id,zero_drift=zero_drift,**target_result))
        if measured or step%config.save_every==0 or step==config.train_steps:
            state=checkpoint_payload(r,s,ot,orr,os,sampler,step,completed,config_hash,manifest_hash,training_rows)
            atomic_torch(checkpoint,state);del state
            rebuild_exports(analysis,completed)
            atomic_json(root/'training_metrics.json',training_rows)
        print(json.dumps(dict(policy_step=step,measured=measured,zero_drift=zero_drift,target_trajectories=len(records),
                              wall_s=time.perf_counter()-started)),flush=True)
    return dict(policy_step=step,completed_analysis_steps=sorted(completed),checkpoint=str(checkpoint))
