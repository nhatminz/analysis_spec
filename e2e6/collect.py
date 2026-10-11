"""One shared event log per checkpoint; a separate event-disabled timing replay."""
from pathlib import Path
import json
import os
import random
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from motivation.data import tokenize
from motivation.state import atomic_json,digest,file_hash,generation_seed,isolated_rng
from motivation.runtime import evaluation_state,target_state,load_target
from e2e6.checkpoints import select,verify
from e2e6.config import Config,generation_kwargs
from e2e6.events import VerificationLog,ReplayTiming,validate_reconstruction
from e2e6.runtime import build_model
from e2e6.train import stop_signals,append_log

EVENT_SCHEMA=pa.schema([(name,dtype) for name,dtype in [
    ('checkpoint_step',pa.int32()),('rollout_id',pa.int32()),('response_id',pa.int32()),
    ('prompt_id',pa.string()),('group_id',pa.string()),('sample_index',pa.int16()),
    ('token_position',pa.int32()),('target_token',pa.int32()),('draft_top1',pa.int32()),
    ('draft_candidates',pa.list_(pa.int32())),('retained_children',pa.list_(pa.int32())),
    ('context2',pa.list_(pa.int32())),('context4',pa.list_(pa.int32())),('context8',pa.list_(pa.int32())),
    ('full_context_sha256',pa.string()),('event_order',pa.int32()),('round_index',pa.int32()),
    ('tree_depth',pa.int16()),('reason',pa.string()),('is_miss',pa.bool_()),
    ('active_batch_size',pa.int16()),('draft_k',pa.int16()),('tree_budget',pa.int32()),
    ('tree_nodes',pa.int32()),('accepted_length',pa.int16())]])


def atomic_parquet(path,rows,schema=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.parquet.tmp')
    table=pa.Table.from_pylist(rows,schema=schema)
    pq.write_table(table,temporary,compression='zstd')
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path)


def validate_rollout(directory,identity=None):
    directory=Path(directory);marker=directory/'complete.json'
    if not marker.exists():return False
    commit=json.loads(marker.read_text())
    step_manifest=directory.parent/'manifest.json'
    if step_manifest.exists():
        step_info=json.loads(step_manifest.read_text())
        plan=json.loads((directory.parent.parent/'manifest.json').read_text())
        if step_info['shared_plan_sha256']!=digest(plan):raise ValueError('Step/shared collection manifest identity differs')
        if identity is None:identity=step_info['identity_sha256']
        if commit['checkpoint_step']!=step_info['checkpoint']['policy_step']:raise ValueError('Rollout checkpoint step differs')
    if identity is not None and commit['identity_sha256']!=identity:raise ValueError('Finalized rollout configuration differs')
    for name,expected in commit['files'].items():
        if file_hash(directory/name)!=expected:raise ValueError(f'Finalized rollout checksum failed: {directory/name}')
    output=json.loads((directory/'responses.json').read_text())
    validate_reconstruction(pq.read_table(directory/'events.parquet').to_pylist(),output['tokens'],output['eos_token_id'])
    return True


def counters(out):
    return {key:out[key] for key in ('response_accepted_length_sum','response_verification_rounds',
        'total_acc_length','total_decoded_token_num','total_proposed_draft_tokens','total_accepted_draft_tokens')}


def collect(c,test_rows,tokenizer,manifest,steps,model_factory=build_model,model=None):
    from helper.fastgrpo_generate import speculative_generate
    root=Path(c.output_dir);paths,missing=select(root/'checkpoints',steps)
    if not paths:return dict(completed=[],missing_checkpoints=missing,status='no_available_checkpoints')
    shuffled=list(test_rows);random.Random(c.eval_seed).shuffle(shuffled)
    needed=c.eval_rollouts*c.batch_size
    if needed>len(shuffled):raise ValueError('Evaluation requires disjoint test prompts across rollouts; reduce eval-rollouts')
    chosen=shuffled[:needed]
    limit=1 if c.smoke else c.eval_rollouts
    if any(row['split']!='test' for row in chosen):raise ValueError('Collection requires official held-out test prompts')
    plan=dict(format='e2_e6_shared_verification_v1',generation=generation_kwargs(c),eval_rollouts=c.eval_rollouts,
        seed=c.eval_seed,batch_size=c.batch_size,responses_per_prompt=c.responses_per_prompt,
        prompt_ids=[r['id'] for r in chosen],question_hashes=[r['question_sha256'] for r in chosen],
        training_manifest_sha256=digest(manifest),
        implementation={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in
            [Path(__file__).resolve(),Path(__file__).with_name('events.py'),Path(__file__).resolve().parents[1]/'helper/fastgrpo_generate.py']},
        causal_feedback='Only preceding verification rounds are observable; same-round events are simultaneous',
        timing='Event logging DISABLED during replay. Host phase wall times + separate CUDA stream elapsed times; no new per-round synchronization')
    plan_path=root/'collection/manifest.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Collection identity differs; use a separate experiment output')
    if not plan_path.exists():atomic_json(plan_path,plan)
    completed=[];started=time.perf_counter()
    with stop_signals() as stop:
        for checkpoint in paths:
            step=int(checkpoint.stem.split('_')[1]);info=verify(checkpoint,load=False)
            step_identity=digest(dict(plan=plan,checkpoint=info))
            step_root=root/'collection'/f'step_{step:06d}'
            step_manifest=step_root/'manifest.json'
            step_info=dict(identity_sha256=step_identity,checkpoint=info,shared_plan_sha256=digest(plan))
            if step_manifest.exists() and json.loads(step_manifest.read_text())!=step_info:
                raise ValueError('Checkpoint collection manifest differs')
            if not step_manifest.exists():atomic_json(step_manifest,step_info)
            tasks=[]
            for rollout in range(limit):
                destination=step_root/f'rollout_{rollout:04d}'
                if not validate_rollout(destination,step_identity):tasks.append((rollout,destination))
            if not tasks:completed.append(step);continue
            state=verify(checkpoint)
            if state['manifest_sha256']!=digest(manifest):raise ValueError('Collection checkpoint/training-manifest identity differs')
            if model is None:model=model_factory(c,draft_state=state['draft_weights'],target_lora=state['target_lora'])
            else:
                load_target(model.target_model,state['target_lora']);model.draft_model.load_state_dict(state['draft_weights'])
            del state
            before=digest(dict(target=target_state(model.target_model),draft=model.draft_model.state_dict()))
            with evaluation_state([model]),torch.inference_mode():
                for rollout,destination in tqdm(tasks,desc=f'Collect step {step}',unit='rollout',dynamic_ncols=True):
                    if stop['requested']:break
                    rows=chosen[rollout*c.batch_size:(rollout+1)*c.batch_size]
                    batch=tokenize(rows,tokenizer)
                    if int(batch['attention_mask'].sum(-1).max())>=c.max_length:raise ValueError('Evaluation prompt exceeds max_length')
                    seed=generation_seed(c.eval_seed,step,f'rollout:{rollout}')
                    observer=VerificationLog(step,rollout,[row['id'] for row in rows],c.responses_per_prompt,tokenizer.eos_token_id)
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
                    with isolated_rng(seed):
                        logged=speculative_generate(model=model,tokenizer=tokenizer,input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],verification_hook=observer,**generation_kwargs(c))
                    torch.cuda.synchronize();logging_s=time.perf_counter()-begin
                    observer.validate(logged)
                    # Pure replay: verification event collection is OFF here.
                    timer=ReplayTiming();torch.cuda.synchronize();begin=time.perf_counter()
                    with isolated_rng(seed):
                        timed=speculative_generate(model=model,tokenizer=tokenizer,input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],timing_hook=timer,**generation_kwargs(c))
                    timer.finish();torch.cuda.synchronize();wall_s=time.perf_counter()-begin
                    if logged['generated_token_ids']!=timed['generated_token_ids'] or counters(logged)!=counters(timed):
                        raise ValueError('Event-disabled timing replay changed generated tokens or counters')
                    segments=timer.resolve()
                    destination.mkdir(parents=True,exist_ok=True)
                    atomic_parquet(destination/'events.parquet',observer.events,EVENT_SCHEMA)
                    atomic_parquet(destination/'rounds.parquet',observer.rounds,pa.schema([
                        ('checkpoint_step',pa.int32()),('rollout_id',pa.int32()),('round_index',pa.int32()),
                        ('active_batch_size',pa.int16()),('draft_k',pa.int16()),('tree_budget',pa.int32()),
                        ('tree_nodes',pa.int32()),('accepted_length',pa.int32()),('active_response_rounds',pa.int32())]))
                    atomic_json(destination/'responses.json',dict(tokens=logged['generated_token_ids'],
                        eos_token_id=tokenizer.eos_token_id,counters=counters(logged),prompt_ids=[r['id'] for r in rows],seed=seed))
                    timing=dict(wall_s=wall_s,logging_wall_s=logging_s,segments=segments,
                        host_attribution_remainder_s=wall_s-sum(row['host_wall_s'] for row in segments),
                        measured_wall_clock=True,cuda_stream_intervals_are_not_kernel_only=True,
                        asynchronous_wall_attribution=True,
                        note='Host intervals are measured at CPU phase boundaries; asynchronous GPU carry-over can cross bins. CUDA stream elapsed is reported separately, not substituted for wall time.',
                        token_equivalence=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    atomic_json(destination/'timing.json',timing)
                    names=('events.parquet','rounds.parquet','responses.json','timing.json')
                    atomic_json(destination/'complete.json',dict(identity_sha256=step_identity,
                        files={name:file_hash(destination/name) for name in names},events=len(observer.events),
                        checkpoint_step=step,rollout_id=rollout,seed=seed,reconstruction_passed=True))
                    append_log(root/'logs/collect.jsonl',dict(checkpoint_step=step,rollout_id=rollout,seed=seed,
                        events=len(observer.events),generated_tokens=sum(map(len,logged['generated_token_ids'])),
                        logging_wall_s=logging_s,replay_wall_s=wall_s,checkpoint=str(checkpoint),
                        aal=logged['total_acc_length']/max(logged['total_decoded_token_num'],1),
                        draft_acceptance_rate=logged['draft_acceptance_rate'],failure_reason=None))
            if before!=digest(dict(target=target_state(model.target_model),draft=model.draft_model.state_dict())):
                raise ValueError('Frozen collection changed model weights')
            if stop['requested']:break
            completed.append(step)
    return dict(completed=completed,missing_checkpoints=missing,status='interrupted' if stop['requested'] else 'complete',
                runtime_s=time.perf_counter()-started,requested_rollouts_per_checkpoint=limit,
                planned_rollouts_per_checkpoint=c.eval_rollouts)
