"""Frozen paired SPARK evaluation. No optimizer, training, or timing replay."""
from pathlib import Path
import importlib.metadata
import json
import random
import time
import torch
from tqdm.auto import tqdm
from e2e6.config import Config, generation_kwargs
from e2e6.collect import atomic_parquet, EVENT_SCHEMA, counters, validate_rollout
from e2e6.runtime import build_model
from e2e6.train import stop_signals
from motivation.data import read_split, tokenize
from motivation.runtime import target_state, evaluation_state
from motivation.state import atomic_json, digest, file_hash, generation_seed, isolated_rng
from reviewer_controls.offline import output_directory
from rq2.checkpoints import load
from rq2.events import PackedVerificationLog


def configuration(root, overrides):
    root=Path(root).resolve()
    path=root/'train_manifest.json'
    if not path.is_file():path=root/'external_checkpoint_manifest.json'
    source=json.loads(path.read_text()) if path.is_file() else None
    config=Config(**source['config']) if source else Config()
    plan_path=root/'collection/manifest.json'
    if plan_path.is_file():
        plan=json.loads(plan_path.read_text())
        if plan.get('format')!='e2_e6_shared_verification_v1':raise ValueError('Unsupported original collection plan')
        # Collection CLI overrides can differ from training-manifest defaults.
        # The committed generation plan is authoritative for this comparison.
        for name,value in plan['generation'].items():
            field='responses_per_prompt' if name=='repeated_generate_nums' else name
            if hasattr(config,field):setattr(config,field,value)
        config.batch_size=plan['batch_size'];config.responses_per_prompt=plan['responses_per_prompt']
        config.eval_seed=plan['seed'];config.eval_rollouts=plan['eval_rollouts']
    # No writes to the source manifest or its output paths.
    for name,value in overrides.items():
        if value is not None:setattr(config,name,value)
    config.smoke=False
    config.validate()
    if config.eval_rollouts!=20:raise ValueError('RQ2 production plan requires 20 rollouts; smoke executes only the first')
    return config,source,path if source else None


def generation(c, arm, rank, topk, fast_lr, update_stream):
    return dict(**generation_kwargs(c),method='opd_reflex',opd_rank=rank,opd_topk=topk,
                opd_fast_lr=0. if arm=='off' else fast_lr,opd_train_projector=False,
                opd_visited_weight=1.,opd_frontier_weight=1.,opd_update_stream=update_stream,
                opd_diagnostics=True,opd_profile=False)


def prepare(root,destination,c,source,source_path,steps,template,projector_mode,rank,topk,fast_lr,update_stream,require_b200):
    from motivation.compatibility import model_identity,tokenizer_identity,validate_tokenizer,validate_cuda_runtime
    from transformers import AutoTokenizer
    root=Path(root).resolve();destination=output_directory(root,destination,'rq2_repeated_misses_v1')
    identities={};missing=[]
    # Inspect all checkpoint/projector sources BEFORE allocating any GPU model.
    for step in steps:
        path=Path(template.format(step=step)) if template else root/'checkpoints'/f'step_{step:06d}.pt'
        if not path.is_file():missing.append(step);continue
        state=load(path,step,projector_mode)
        if source is not None and state['checkpoint_format']=='pure_fastgrpo_e2_e6_v1':
            from e2e6.checkpoints import verify
            if verify(path)['manifest_sha256']!=digest(source):raise ValueError('Checkpoint and original source manifest differ')
        identities[str(step)]=state['identity']
        if projector_mode=='initialized':
            identities[str(step)]['projector'].update(rank=rank,
                actual_A_record=f'projectors/step_{step:06d}.json')
        del state
    if not identities:raise FileNotFoundError(f'No compatible checkpoints for {steps}; missing {missing}. No model was loaded.')
    if fast_lr<=0 or topk<max(c.max_draft_k,1):raise ValueError('SPARK ON needs positive fast LR and OPD topk >= proposal k')
    runtime=validate_cuda_runtime(require_b200)
    identity=model_identity(c.model)
    tokenizer=AutoTokenizer.from_pretrained(c.model,padding_side='left',local_files_only=True)
    validate_tokenizer(tokenizer,identity['config'])
    tests,test_info=read_split(c.test_path,'test')
    shuffled=list(tests);random.Random(c.eval_seed).shuffle(shuffled)
    rows=shuffled[:c.eval_rollouts*c.batch_size]
    if len(rows)!=c.eval_rollouts*c.batch_size:raise ValueError('Not enough disjoint held-out prompts for 20 rollout batches')
    if source:
        if source['model']!=identity:raise ValueError('Base model differs from the checkpoint source manifest')
        source_test=source.get('data',{}).get('test',{})
        if source_test.get('sha256') and source_test['sha256']!=test_info['sha256']:raise ValueError('Held-out data differs from checkpoint source manifest')
    plan_path=root/'collection/manifest.json'
    if plan_path.is_file():
        original=json.loads(plan_path.read_text())
        if original['prompt_ids']!=[r['id'] for r in rows] or original['question_hashes']!=[r['question_sha256'] for r in rows]:
            raise ValueError('RQ2 prompt plan differs from original baseline collection')
        if original['generation']!=generation_kwargs(c):raise ValueError('RQ2 sampling/proposal settings differ from original baseline')
    execution=dict(format='spark_rq2_repeated_misses_v1',checkpoints=identities,missing_checkpoints=missing,
        projector_mode=projector_mode,initialized_rank=rank,opd_topk=topk,fast_lr=fast_lr,update_stream=update_stream,
        generation=generation_kwargs(c),eval_rollouts=c.eval_rollouts,eval_seed=c.eval_seed,
        batch_size=c.batch_size,responses_per_prompt=c.responses_per_prompt,
        prompts=[{k:r[k] for k in ('id','question_sha256')} for r in rows],test_data_sha256=test_info['sha256'],
        model=identity,tokenizer=tokenizer_identity(c.model),
        source_manifest_sha256=file_hash(source_path) if source_path else None,
        baseline_plan_sha256=file_hash(plan_path) if plan_path.is_file() else None,
        implementation={name:file_hash(Path(__file__).resolve().parents[1]/name) for name in
            ('rq2/runtime.py','rq2/checkpoints.py','rq2/events.py','helper/opd_generate.py','helper/opd_reflex.py',
             'helper/opd_reflex_kernels.py','helper/opd_sampling.py','helper/tree_verification.py','e2e6/events.py')},
        package_versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','triton')})
    execution['identity_sha256']=digest(execution)
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=execution:raise ValueError('RQ2 identity changed; choose a new output directory')
    if not path.exists():atomic_json(path,execution)
    atomic_json(destination/'runtime_preflight.json',runtime)
    return destination,rows,tokenizer,execution


def frozen_digest(model):
    return digest(dict(target=target_state(model.target_model),draft=model.draft_model.state_dict()))


def run_pair(model, tokenizer, rows, c, step, rollout, destination, manifest,
             topk, fast_lr, update_stream, replay=False, deadline=float('inf'), stop=None):
    from helper.opd_generate import speculative_generate
    initial=frozen_digest(model)
    seed=generation_seed(c.eval_seed,step,f'rollout:{rollout}')
    batch=tokenize(rows,tokenizer)
    if int(batch['attention_mask'].sum(-1).max())>=c.max_length:raise ValueError('Prompt exceeds generation max_length')
    for arm in ('off','on'):
        directory=destination/arm/f'rollout_{rollout:04d}'
        if validate_rollout(directory,manifest['identity_sha256']):continue
        if time.perf_counter()>=deadline or (stop and stop['requested']):return False
        if frozen_digest(model)!=initial:raise RuntimeError('Paired arm initial weights changed')
        observer=PackedVerificationLog(step,rollout,[r['id'] for r in rows],c.responses_per_prompt,tokenizer.eos_token_id)
        kwargs=generation(c,arm,model.opd_projector.shape[1],topk,fast_lr,update_stream)
        torch.cuda.synchronize();start=time.perf_counter()
        with isolated_rng(seed),torch.inference_mode():
            output=speculative_generate(model=model,tokenizer=tokenizer,**batch,verification_hook=observer,**kwargs)
        torch.cuda.synchronize();runtime=time.perf_counter()-start
        observer.validate(output)
        if output['opd_nonfinite_kl_states'] or output['opd_invalid_states']:raise RuntimeError('Invalid real SPARK verification feedback')
        if arm=='off' and (output['opd_final_b_norm']!=0 or output['opd_updates']!=0):raise RuntimeError('OFF updated B')
        # Some legitimate rollouts (e.g. all prefill EOS) contain no feedback.
        feedback_available=output['opd_state_weight']>0
        if arm=='on' and feedback_available and not (output['opd_updates']>0 and output['opd_final_b_norm']>0):
            raise RuntimeError('SPARK ON did not update B despite valid feedback')
        cache=getattr(model,'_opd_runtime_cache',None)
        if cache:
            runtime_state=next(iter(cache.values()))
            if torch.count_nonzero(runtime_state.B_fast).item():raise RuntimeError('B did not reset after rollout')
        equivalent=None
        if replay:
            with isolated_rng(seed),torch.inference_mode():
                plain=speculative_generate(model=model,tokenizer=tokenizer,**batch,**kwargs)
            torch.cuda.synchronize()
            equivalent=plain['generated_token_ids']==output['generated_token_ids'] and counters(plain)==counters(output)
            if not equivalent:raise RuntimeError('Passive SPARK logging changed generation/counters')
        if frozen_digest(model)!=initial:raise RuntimeError('Frozen target/draft/A changed during evaluation')
        atomic_parquet(directory/'events.parquet',observer.events,EVENT_SCHEMA)
        atomic_parquet(directory/'rounds.parquet',observer.rounds)
        atomic_json(directory/'responses.json',dict(tokens=output['generated_token_ids'],eos_token_id=tokenizer.eos_token_id,
            counters=counters(output),prompt_ids=[r['id'] for r in rows],seed=seed,arm=arm,
            initial_weights_sha256=initial,projector_sha256=digest(model.opd_projector)))
        diagnostic={key:output[key] for key in ('opd_updates','opd_state_weight','opd_selected_states','opd_final_b_norm',
                                              'opd_final_b_max_abs','opd_invalid_states','opd_nonfinite_kl_states')}
        atomic_json(directory/'diagnostics.json',dict(**diagnostic,wall_s=runtime,
            logging_token_counter_equivalence=equivalent,B_reset_verified=True,frozen_weights_verified=True,
            valid_feedback_available=feedback_available,projector_sha256=digest(model.opd_projector)))
        atomic_json(directory/'complete.json',dict(identity_sha256=manifest['identity_sha256'],checkpoint_step=step,
            checkpoint_identity=manifest['checkpoints'][str(step)],arm=arm,rollout_id=rollout,seed=seed,
            projector_sha256=digest(model.opd_projector),
            reconstruction_passed=True,files={name:file_hash(directory/name) for name in
                ('events.parquet','rounds.parquet','responses.json','diagnostics.json')}))
    return True


def collect(destination, rows, tokenizer, c, manifest, projector_mode, rank, topk, fast_lr, update_stream,
            wall_minutes=150, smoke_only=False, model_factory=build_model):
    started=time.perf_counter();deadline=started+wall_minutes*60;completed=[]
    with stop_signals() as stop:
        for step,identity in manifest['checkpoints'].items():
            if time.perf_counter()>=deadline or stop['requested']:break
            state=load(identity['path'],int(step),projector_mode)
            expected=dict(identity)
            if projector_mode=='initialized':
                expected['projector']={k:v for k,v in identity['projector'].items() if k not in ('rank','actual_A_record')}
            if state['identity']!=expected:raise ValueError('Checkpoint changed after preflight')
            model=model_factory(c,draft_state=state['draft'],target_lora=state['target'])
            model.enable_opd(state['projector'].shape[1] if state['projector'] is not None else rank)
            if state['projector'] is not None:model.load_opd_projector(state['projector'])
            model.requires_grad_(False);model.eval();del state
            projector_path=destination/'projectors'/f'step_{int(step):06d}.json'
            projector_info=dict(checkpoint_identity=identity,actual_A_sha256=digest(model.opd_projector),
                rank=model.opd_projector.shape[1],initialization=identity['projector'],frozen=True)
            if projector_path.exists() and json.loads(projector_path.read_text())!=projector_info:
                raise ValueError('Frozen projector differs from the committed RQ2 run')
            if not projector_path.exists():atomic_json(projector_path,projector_info)
            # Includes frozen base target weights; checked once per checkpoint.
            all_before=digest(model.state_dict())
            with evaluation_state([model]),torch.inference_mode():
                smoke_root=destination/'smoke'/f'step_{int(step):06d}'
                success=run_pair(model,tokenizer,rows[:c.batch_size],c,int(step),0,smoke_root,manifest,
                    topk,fast_lr,update_stream,replay=True,deadline=deadline,stop=stop)
                if success and not smoke_only:
                    for rollout in tqdm(range(c.eval_rollouts),desc=f'RQ2 OFF/ON step {step}',unit='pair',dynamic_ncols=True):
                        batch_rows=rows[rollout*c.batch_size:(rollout+1)*c.batch_size]
                        success=run_pair(model,tokenizer,batch_rows,c,int(step),rollout,
                            destination/'collection'/f'step_{int(step):06d}',manifest,topk,fast_lr,update_stream,
                            deadline=deadline,stop=stop)
                        if not success:break
            if digest(model.state_dict())!=all_before:raise RuntimeError('Evaluation changed frozen model parameters/buffers')
            if success:completed.append(int(step))
            del model
            if not success:break
    status='complete' if len(completed)==len(manifest['checkpoints']) else ('interrupted' if stop['requested'] else 'wall_budget_exceeded')
    result=dict(status=status,completed_checkpoints=completed,smoke_only=smoke_only,
                wall_minutes=wall_minutes,runtime_s=time.perf_counter()-started,
                missing_checkpoints=manifest['missing_checkpoints'])
    atomic_json(destination/('smoke_execution.json' if smoke_only else 'collection_execution.json'),result)
    return result
