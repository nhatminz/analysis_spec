"""Standalone reference FastGRPO online draft + real target optimizer steps."""
from contextlib import contextmanager
from pathlib import Path
import json
import signal
import time
import torch
from tqdm.auto import tqdm
from helper.checkpointing import capture_rng_state,restore_rng_state
from helper.opd_optimizer import draft_optimizer
from motivation.runner import TrainSampler
from motivation.runtime import update_draft,prepare_target_update,commit_target_update,target_state,load_target
from motivation.data import tokenize
from motivation.state import atomic_json,atomic_bytes,cpu_copy,digest,seed_all
from e2e6.config import generation_kwargs
from e2e6.runtime import build_model,reference_target_batch
from e2e6.checkpoints import save,verify,latest,repair_milestone,MILESTONES,FORMAT


@contextmanager
def stop_signals():
    flag={'requested':False,'signal':None}
    def handler(number,frame):flag.update(requested=True,signal=number)
    old={s:signal.signal(s,handler) for s in (signal.SIGINT,signal.SIGTERM)}
    try:yield flag
    finally:
        for s,handler in old.items():signal.signal(s,handler)


def append_log(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def train(c,rows,tokenizer,manifest,resume=False,model_factory=build_model,model_holder=None):
    from helper.fastgrpo_generate import speculative_generate
    root=Path(c.output_dir);directory=root/'checkpoints';log=root/'logs/train.jsonl'
    seed_all(c.seed)
    model=model_factory(c,training=True)
    ot=torch.optim.AdamW(model.target_model.parameters(),lr=c.target_lr)
    od=draft_optimizer(model.draft_model,c.draft_lr)
    sampler=TrainSampler(rows,c.seed,c.batch_size)
    step=attempt=draft_steps=0;elapsed=0.
    identity=digest(manifest)
    if resume:
        resume_path=latest(directory);repair_milestone(directory,resume_path)
        state=verify(resume_path)
        if state['manifest_sha256']!=identity:raise ValueError('Resume data/model/config/source identity differs')
        load_target(model.target_model,state['target_lora']);model.draft_model.load_state_dict(state['draft_weights'])
        ot.load_state_dict(state['optimizer_target']);od.load_state_dict(state['optimizer_draft'])
        sampler.load_state_dict(state['sampler']);restore_rng_state(state['rng'])
        step=state['policy_step'];attempt=state['attempt'];draft_steps=state['draft_steps'];elapsed=state['elapsed_s']
        actual=dict(target_lora=target_state(model.target_model),draft_weights=model.draft_model.state_dict(),
                    optimizer_target=ot.state_dict(),optimizer_draft=od.state_dict(),sampler=sampler.state_dict(),rng=capture_rng_state())
        for key,value in actual.items():
            if digest(value)!=digest(state[key]):raise ValueError(f'Exact resume restoration failed: {key}')
        atomic_json(root/'resume_audit.json',dict(passed=True,policy_step=step,attempt=attempt,verified_fields=list(actual)))
        if log.exists():
            committed=[line for line in log.read_text().splitlines() if json.loads(line).get('attempt',0)<=attempt]
            atomic_bytes(log,(''.join(x+'\n' for x in committed)).encode())
        del state,actual
    elif (directory/'latest.json').exists():raise ValueError('Training output already exists; use resume')
    started=time.perf_counter()
    def payload():
        # Every save is at a completed rollout/update boundary; no pending grads.
        return dict(format=FORMAT,policy_step=step,attempt=attempt,draft_steps=draft_steps,
            target_lora=target_state(model.target_model),draft_weights=cpu_copy(model.draft_model.state_dict()),
            optimizer_target=cpu_copy(ot.state_dict()),optimizer_draft=cpu_copy(od.state_dict()),
            rng=cpu_copy(capture_rng_state()),sampler=sampler.state_dict(),manifest_sha256=identity,
            elapsed_s=elapsed+time.perf_counter()-started,config=c.to_dict(),pending_target_gradients=False,
            pending_draft_gradients=False,training_algorithm='pure_fastgrpo')
    with stop_signals() as stop,tqdm(total=c.train_steps,initial=step,desc='Pure FastGRPO',unit='target step',dynamic_ncols=True) as bar:
        while step<c.train_steps and not stop['requested']:
            if attempt>=c.max_rollout_attempts:raise RuntimeError('Maximum rollout attempts reached; inspect real reward/skip logs')
            attempt+=1;begin=time.perf_counter();rows_batch=sampler.next()
            if any(row['split']!='train' for row in rows_batch):raise ValueError('Test prompt reached training')
            batch=tokenize(rows_batch,tokenizer)
            if int(batch['attention_mask'].sum(-1).max())>=c.max_length or int(batch['attention_mask'].sum(-1).max())>c.max_prompt_length:
                raise ValueError('Train prompt exceeds max_length; no silent truncation')
            bar.set_postfix(attempt=attempt,phase='rollout')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();rollout_start=time.perf_counter()
            with torch.inference_mode():
                out=speculative_generate(model=model,tokenizer=tokenizer,input_ids=batch['input_ids'],
                                         attention_mask=batch['attention_mask'],**generation_kwargs(c,training=True))
            torch.cuda.synchronize();rollout_s=time.perf_counter()-rollout_start
            records,rewards=reference_target_batch(rows_batch,out,tokenizer,c)
            counters={key:out[key] for key in ('total_acc_length','total_decoded_token_num',
                'total_proposed_draft_tokens','total_accepted_draft_tokens','draft_acceptance_rate')}
            generated=sum(map(len,out['generated_token_ids']))
            bar.set_postfix(attempt=attempt,phase='draft update')
            draft_loss=update_draft(model,od,out,batch['attention_mask'],c)
            draft_steps+=draft_loss['optimizer_steps'];del out
            bar.set_postfix(attempt=attempt,phase='target update')
            prepared=prepare_target_update(model.target_model,ot,records,c)
            result=commit_target_update(ot,prepared)
            if result['target_optimizer_steps']:
                step+=1;bar.update(1)
            diagnostics=dict(policy_step=step,attempt=attempt,draft_steps=draft_steps,
                prompt_ids=[r['id'] for r in rows_batch],rewards=rewards,
                reward_mean=sum(map(sum,rewards))/(c.batch_size*c.responses_per_prompt),
                draft_loss=draft_loss,**result,**counters,generated_tokens=generated,
                aal=counters['total_acc_length']/max(counters['total_decoded_token_num'],1),
                rollout_wall_s=rollout_s,iteration_wall_s=time.perf_counter()-begin,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_path=None,failure_reason=result['target_skip_reason'])
            should_save=(result['target_optimizer_steps'] and (step%c.save_every==0 or step in MILESTONES or step==c.train_steps)) or stop['requested']
            if should_save:
                bar.set_postfix(attempt=attempt,phase='checkpoint')
                path=save(directory,payload(),milestone=step in MILESTONES)
                diagnostics['checkpoint_path']=str(path)
            append_log(log,diagnostics)
            atomic_json(root/'training_status.json',dict(policy_step=step,attempt=attempt,draft_steps=draft_steps,
                elapsed_s=elapsed+time.perf_counter()-started,stopped=stop['requested'],signal=stop['signal']))
            bar.set_postfix(attempt=attempt,reward=f'{diagnostics["reward_mean"]:.3f}',aal=f'{diagnostics["aal"]:.2f}',
                            skip=result['target_skip_reason'] or '',phase='complete')
        # Covers a signal arriving between iterations and a fully resumed run.
        save(directory,payload(),milestone=step in MILESTONES)
    if model_holder is not None:model_holder['model']=model
    return dict(policy_step=step,attempt=attempt,draft_steps=draft_steps,stopped=stop['requested'],
                runtime_s=time.perf_counter()-started)
