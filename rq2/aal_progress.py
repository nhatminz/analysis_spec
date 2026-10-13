"""Standalone frozen-checkpoint, per-prompt SPARK AAL evaluation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.metadata
import json
import math
import random
import time
import torch
from tqdm.auto import tqdm
from rq2.checkpoints import load
from rq2.events import PackedVerificationLog
from rq2.runtime import generation, feedback_diagnostics
from e2e6.collect import atomic_parquet, EVENT_SCHEMA, counters, validate_rollout
from e2e6.runtime import build_model
from e2e6.train import stop_signals
from motivation.data import read_split, tokenize
from motivation.runtime import evaluation_state
from motivation.state import atomic_json, digest, file_hash, generation_seed, isolated_rng

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT='/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/qwen25_3b__simplelr__method-opd_reflex__seed42__20261008T104039__3885bd48/checkpoints/resume/step234_epoch2_batch0.pt'


@dataclass
class AALConfig:
    model:str='/workspace/storage-shared/models/Qwen2.5-3B-Instruct'
    test_path:str='/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet'
    checkpoint_path:str=CHECKPOINT
    checkpoint_step:int=234
    num_rollouts:int=20
    batch_size:int=8
    responses_per_prompt:int=8
    seed:int=2026
    temperature:float=1.
    top_p:float=.95
    top_k:int|None=None
    max_length:int=2048
    verification_capacity:int=512
    max_verification_num:int=160
    max_draft_token_length:int=5
    max_draft_k:int=8
    min_draft_token_length:int=3
    draft_token_length_c:float=.75
    dtype:str='bf16'
    attention_implementation:str='sdpa'
    opd_topk:int=16
    opd_fast_lr:float=.01
    opd_update_stream:bool=True

    def validate(self):
        for key in ('checkpoint_step','num_rollouts','batch_size','responses_per_prompt','max_length',
                    'verification_capacity','max_verification_num','max_draft_token_length',
                    'max_draft_k','min_draft_token_length','opd_topk'):
            if getattr(self,key)<1:raise ValueError(f'{key} must be positive')
        if not math.isfinite(self.opd_fast_lr) or self.opd_fast_lr<=0:raise ValueError('ON fast LR must be finite and positive')
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<self.top_p<=1:
            raise ValueError('Invalid target sampling settings')
        if self.top_k is not None and self.top_k<1:raise ValueError('top_k must be positive or unset')
        if self.opd_topk<self.max_draft_k:raise ValueError('OPD topk must cover the draft proposal k')
        if self.min_draft_token_length>self.max_draft_token_length:raise ValueError('Invalid tree depths')
        if not math.isfinite(self.draft_token_length_c) or self.draft_token_length_c<=0:raise ValueError('Invalid adaptive tree coefficient')
        if self.seed<0:raise ValueError('Seed must be nonnegative')
        if self.max_length<2 or self.max_verification_num<2 or self.verification_capacity<2*self.batch_size*self.responses_per_prompt:
            raise ValueError('Generation length/capacity cannot support this response batch')


def training_metadata(path):
    """Read metadata only; no optimizer is constructed or resumed."""
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    names=('format','method','step','epoch','next_batch','world_size','used_items','draft_step',
           'draft_accumulated_step','grpo_alignment_version','initial_target_tensor_sha256',
           'opd_ablation_config','opd_initial_a_sha256')
    metadata={k:state.get(k) for k in names}
    config_path=Path(path).parent.parent.parent/'config_resolved.yaml'
    resolved={}
    if config_path.is_file():
        for line in config_path.read_text().splitlines():
            key,sep,value=line.partition(':')
            if sep:
                try:resolved[key.strip()]=json.loads(value.strip())
                except json.JSONDecodeError:resolved[key.strip()]=value.strip()
    declared=(metadata.get('opd_ablation_config') or {}).get('opd_b_sharing_scope')
    sibling=resolved.get('opd_b_sharing_scope')
    # A resume checkpoint is authoritative; absent scope in a v5 checkpoint
    # follows SpecNaacl's legacy batch-wide default, regardless of later edits
    # to the sibling configuration file.
    scope=declared or 'rollout_batch'
    limitation=('This checkpoint was trained with batch-wide B (legacy default when scope is absent). '
        'This is per-prompt adaptation evaluation on that frozen checkpoint, not fully per-prompt-trained SPARK.') if scope!='per_prompt_group' else None
    metadata.update(training_B_scope=scope,training_scope_explicit=declared is not None,
        training_provenance_limitation=limitation,resolved_config=resolved,
        resolved_config_sha256=file_hash(config_path) if config_path.is_file() else None,
        sibling_scope_disagreement=bool(sibling and sibling!=scope))
    return metadata


def validate_cpu_weights(state,model_path):
    """Strictly restore draft and target adapters on CPU/meta before any GPU."""
    from copy import deepcopy
    from transformers import AutoConfig, AutoModelForCausalLM
    from helper.modeling_draft import DraftModel
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    cfg=AutoConfig.from_pretrained(model_path,local_files_only=True)
    dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None
    with torch.device('meta'):
        draft=DraftModel(dc)
        target=AutoModelForCausalLM.from_config(cfg)
        target=get_peft_model(target,LoraConfig(task_type='CAUSAL_LM',r=64,lora_alpha=32,lora_dropout=0.,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    expected=get_peft_model_state_dict(target)
    if set(expected)!=set(state['target']):raise ValueError('Target LoRA keys do not match Qwen/SpecNaacl architecture')
    for key,tensor in expected.items():
        if tensor.shape!=state['target'][key].shape:raise ValueError(f'Target LoRA shape mismatch: {key}')
    draft.load_state_dict(state['draft'],strict=True,assign=True)
    # PEFT's restore helper moves adapters back to the base layer's device;
    # here that base is intentionally meta. Assign only already-validated LoRA
    # tensors to CPU, leaving the 3B base unallocated during preflight.
    adapter_state={key.replace('.lora_A.','.lora_A.default.').replace('.lora_B.','.lora_B.default.'):value
                   for key,value in state['target'].items()}
    target.load_state_dict(adapter_state,strict=False,assign=True)
    if digest(get_peft_model_state_dict(target))!=state['target_lora_sha256']:
        raise ValueError('CPU target LoRA restoration differs from the checkpoint')
    if digest(draft.state_dict())!=state['draft_backbone_sha256']:raise ValueError('CPU draft restoration differs')
    if state['projector'].shape[0]!=cfg.hidden_size or not 1<=state['projector'].shape[1]<=64:
        raise ValueError('Checkpoint A has incompatible hidden dimension/rank')
    if state['projector'].dtype!=torch.float32:raise ValueError('Expected the native FP32 learned projector')
    return dict(target_lora_loaded=True,draft_loaded=True,learned_A_validated=True,optimizer_resumed=False)


def checkpoint_preflight(c):
    c.validate()
    if not Path(c.checkpoint_path).is_file():
        raise FileNotFoundError(f'Required read-only SPARK checkpoint is unavailable: {c.checkpoint_path}. No GPU model was loaded.')
    state=load(c.checkpoint_path,c.checkpoint_step,'learned')
    if state['checkpoint_format']!='opd_fastgrpo_checkpoint_v5':raise ValueError('A SpecNaacl v5 resume checkpoint is required')
    metadata=training_metadata(c.checkpoint_path)
    if metadata['method']!='opd_reflex' or int(metadata['step'])!=c.checkpoint_step:
        raise ValueError('Checkpoint method/step differs from requested frozen evaluation')
    restored=validate_cpu_weights(state,c.model)
    if file_hash(c.checkpoint_path)!=state['identity']['sha256']:raise ValueError('Checkpoint changed during preflight')
    return state,dict(checkpoint=state['identity'],training_metadata=metadata,**restored)


def validate_output_directory(output_dir):
    output=Path(output_dir).resolve()
    if output.parent!=ROOT/'outputs' or not output.name.startswith('rq2_aal_'):
        raise ValueError('Use a separate analysis_spec/outputs/rq2_aal_* directory')
    return output


def prepare(c,output_dir):
    from transformers import AutoTokenizer
    from motivation.compatibility import model_identity,tokenizer_identity,validate_tokenizer
    output=validate_output_directory(output_dir)
    if output.is_dir() and any(output.iterdir()) and not (output/'manifest.json').is_file():
        raise ValueError('Nonempty AAL output directory has no manifest; refusing to overwrite unrelated data')
    state,preflight=checkpoint_preflight(c)
    identity=model_identity(c.model)
    tok=AutoTokenizer.from_pretrained(c.model,padding_side='left',local_files_only=True)
    validate_tokenizer(tok,identity['config'])
    tests,test_info=read_split(c.test_path,'test')
    random.Random(c.seed).shuffle(tests)
    rows=tests[:c.num_rollouts*c.batch_size]
    if len(rows)!=c.num_rollouts*c.batch_size:raise ValueError('Insufficient disjoint held-out prompts')
    files=('rq2/aal_progress.py','rq2/aal_analysis.py','rq2/per_prompt.py','rq2/per_prompt_kernels.py',
        'rq2/checkpoints.py','rq2/runtime.py','rq2/events.py','e2e6/events.py','e2e6/runtime.py',
        'helper/fastgrpo_model.py','helper/opd_generate.py','helper/opd_scheduling.py',
        'helper/opd_reflex.py','helper/opd_reflex_kernels.py','helper/opd_sampling.py','helper/tree_kernels.py')
    manifest=dict(format='spark_rq2_aal_progress_v1',config=asdict(c),**preflight,
        model=identity,tokenizer=tokenizer_identity(c.model),test_data=test_info,
        prompts=[{k:r[k] for k in ('id','question_sha256')} for r in rows],
        seeds=[generation_seed(c.seed,c.checkpoint_step,f'aal-rollout:{i}') for i in range(c.num_rollouts)],
        adaptation_scope='per_prompt_group',B_reset='every rollout',A='loaded learned and frozen',
        planned_sample_counts=dict(rollout_pairs=c.num_rollouts,prompts=len(rows),
            responses_per_arm=len(rows)*c.responses_per_prompt,total_responses=2*len(rows)*c.responses_per_prompt),
        proposal_profile='Uncalibrated per-prompt dispatch; batch-wide timing profiles are not used',
        progress='Round midpoints: 100*(round_index+0.5)/total_rounds; five half-open 20% bins, final bin includes 100%',
        AAL='sum accepted verification path lengths / sum active response-rounds; includes terminal verification token, excludes prefill',
        implementation={name:file_hash(ROOT/name) for name in files},
        package_versions={name:importlib.metadata.version(name) for name in
            ('torch','transformers','peft','triton','numpy','pandas','pyarrow','matplotlib')})
    manifest['identity_sha256']=digest(manifest)
    output.mkdir(parents=True,exist_ok=True)
    existing=output/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())!=manifest:
        raise ValueError('AAL run identity changed; use a new rq2_aal_* output directory')
    if not existing.exists():atomic_json(existing,manifest)
    atomic_json(output/'preflight.json',preflight)
    return output,rows,tok,state,manifest


def validate_counters(observer,output):
    observer.validate(output)
    rounds=observer.rounds
    accepted=sum(r['accepted_length'] for r in rounds)
    active=sum(r['active_response_rounds'] for r in rounds)
    expected=(output['total_acc_length'],output['total_decoded_token_num'],output['verification_batches'])
    if (accepted,active,len(rounds))!=expected:raise ValueError('AAL numerator/denominator/round log disagrees with acceptance counters')
    if active!=sum(output['response_verification_rounds']) or active!=output['active_response_rounds']:
        raise ValueError('Active response-round counters disagree')
    if accepted!=sum(output['response_accepted_length_sum']):raise ValueError('Accepted verification lengths disagree')
    return dict(accepted_length=accepted,active_response_rounds=active,verification_rounds=len(rounds),
                overall_AAL=accepted/active if active else None)


def collect(c,output,rows,tok,state,manifest,*,smoke=False,wall_minutes=150.,require_b200=False,model_factory=build_model):
    from helper.opd_generate import speculative_generate
    from helper.checkpointing import capture_rng_state
    from motivation.compatibility import validate_cuda_runtime
    started=time.perf_counter();deadline=started+wall_minutes*60
    hardware=validate_cuda_runtime(require_b200)
    atomic_json(output/'runtime.json',hardware)
    model=model_factory(c,draft_state=state['draft'],target_lora=state['target'])
    model.enable_opd(state['projector'].shape[1],projector=state['projector'])
    model.requires_grad_(False);model.eval()
    if digest(model.opd_projector)!=state['provenance']['A_sha256']:raise ValueError('Loaded GPU A differs from learned checkpoint A')
    before=digest(model.state_dict());rank=model.opd_projector.shape[1]
    completed=[];runs=1 if smoke else c.num_rollouts
    root=output/('smoke' if smoke else 'collection')
    with stop_signals() as stop, evaluation_state([model]),torch.inference_mode():
        for rollout in tqdm(range(runs),desc='AAL smoke OFF/ON' if smoke else 'AAL rollouts OFF/ON',unit='pair',dynamic_ncols=True):
            batch_rows=rows[rollout*c.batch_size:(rollout+1)*c.batch_size]
            batch=tokenize(batch_rows,tok)
            if int(batch['attention_mask'].sum(-1).max())>=c.max_length:raise ValueError('A held-out prompt exceeds max_length')
            seed=manifest['seeds'][rollout]
            complete=True
            for arm in ('off','on'):
                directory=root/arm/f'rollout_{rollout:04d}'
                if validate_rollout(directory,manifest['identity_sha256']):continue
                if time.perf_counter()>=deadline or stop['requested']:complete=False;break
                observer=PackedVerificationLog(c.checkpoint_step,rollout,[r['id'] for r in batch_rows],c.responses_per_prompt,tok.eos_token_id)
                kwargs=generation(c,arm,rank,c.opd_topk,c.opd_fast_lr,c.opd_update_stream)
                kwargs['opd_b_sharing_scope']='per_prompt_group'
                torch.cuda.synchronize();begin=time.perf_counter()
                target_calls=[]
                handle=model.target_model.get_base_model().model.layers[0].register_forward_pre_hook(lambda *args:target_calls.append(1))
                try:
                    with isolated_rng(seed):
                        out=speculative_generate(model=model,tokenizer=tok,**batch,verification_hook=observer,**kwargs)
                        rng=digest(capture_rng_state())
                finally:handle.remove()
                torch.cuda.synchronize();runtime=time.perf_counter()-begin
                acceptance=validate_counters(observer,out)
                diagnostic=feedback_diagnostics(out)
                diagnostic.update(acceptance=acceptance,wall_s=runtime,
                    group_b_norms=out['opd_group_b_norms'],group_updates=out['opd_group_updates'],
                    group_feedback_weight=out['opd_group_feedback_weight'],
                    initial_b_norm=out['opd_initial_b_norm'],target_forward_count=len(target_calls),
                    sharing_scope=out['opd_b_sharing_scope'])
                errors=list(diagnostic['validation_errors'])
                if diagnostic['sharing_scope']!='per_prompt_group':errors.append('Batch-wide B fallback is forbidden')
                if diagnostic['initial_b_norm']!=0:errors.append('B was not zero at rollout start')
                if len(target_calls)!=1+out['verification_batches']:errors.append('Extra/missing target forward')
                if any(not math.isfinite(v) for v in diagnostic['group_b_norms']):errors.append('Nonfinite per-prompt B')
                if sum(diagnostic['group_updates'])!=out['opd_updates']:errors.append('Per-prompt update counters disagree')
                if sum(diagnostic['group_feedback_weight'])!=out['opd_state_weight']:errors.append('Per-prompt feedback weights disagree')
                for weight,updates,norm in zip(diagnostic['group_feedback_weight'],diagnostic['group_updates'],diagnostic['group_b_norms']):
                    if weight==0 and (updates or norm):errors.append('A prompt without feedback adapted B')
                    if arm=='on' and weight>0 and not updates:errors.append('A prompt with feedback failed to adapt')
                if arm=='off' and (out['opd_updates'] or any(diagnostic['group_b_norms'])):errors.append('OFF adapted B')
                if arm=='on' and out['opd_state_weight']>0 and not out['opd_updates']:errors.append('ON failed to adapt')
                cache=getattr(model,'_opd_runtime_cache',{})
                if any(bool(s.B_fast.any()) for s in cache.values()):errors.append('B was not cleared after rollout')
                if digest(model.state_dict())!=before:errors.append('Frozen target/draft/A weights changed')
                if errors:
                    diagnostic['validation_errors']=errors;diagnostic['feedback_valid']=False
                    # Nonfinite scalar values are JSON strings, not fake zeros.
                    diagnostic['group_b_norms']=[v if math.isfinite(v) else str(v) for v in diagnostic['group_b_norms']]
                    atomic_json(directory/'failure.json',diagnostic)
                    raise RuntimeError(f'AAL validation failed for {arm} rollout {rollout}: {errors}; see {directory}/failure.json')
                equivalence=None
                if smoke:
                    with isolated_rng(seed):
                        plain=speculative_generate(model=model,tokenizer=tok,**batch,**kwargs)
                        plain_rng=digest(capture_rng_state())
                    equivalence=(plain['generated_token_ids']==out['generated_token_ids'] and counters(plain)==counters(out) and rng==plain_rng)
                    if not equivalence:raise RuntimeError('Passive AAL logging changed tokens, counters or RNG')
                    if digest(model.state_dict())!=before:raise RuntimeError('Smoke replay changed frozen weights')
                diagnostic.update(logging_equivalence=equivalence,B_reset_verified=True,frozen_weights_verified=True)
                atomic_parquet(directory/'events.parquet',observer.events,EVENT_SCHEMA)
                import pyarrow as pa
                schema=pa.schema([(key,pa.int64()) for key in ('checkpoint_step','rollout_id','round_index','active_batch_size',
                    'draft_k','tree_budget','tree_nodes','accepted_length','active_response_rounds')])
                atomic_parquet(directory/'rounds.parquet',observer.rounds,schema)
                atomic_json(directory/'responses.json',dict(tokens=out['generated_token_ids'],eos_token_id=tok.eos_token_id,
                    counters=counters(out),seed=seed,prompt_ids=[r['id'] for r in batch_rows],frozen_weights_sha256=before,
                    learned_A_sha256=digest(model.opd_projector)))
                atomic_json(directory/'diagnostics.json',diagnostic)
                atomic_json(directory/'complete.json',dict(identity_sha256=manifest['identity_sha256'],arm=arm,
                    checkpoint_step=c.checkpoint_step,rollout_id=rollout,reconstruction_passed=True,
                    files={name:file_hash(directory/name) for name in ('events.parquet','rounds.parquet','responses.json','diagnostics.json')}))
            if not complete:break
            completed.append(rollout)
    if file_hash(c.checkpoint_path)!=state['identity']['sha256']:raise RuntimeError('Read-only checkpoint changed during evaluation')
    result=dict(status='complete' if len(completed)==runs else 'partial',completed_rollouts=completed,
                requested_rollouts=runs,smoke=smoke,runtime_s=time.perf_counter()-started,wall_minutes=wall_minutes)
    atomic_json(output/('smoke_execution.json' if smoke else 'collection_execution.json'),result)
    # Break the model -> runtime -> model ownership cycle between smoke and
    # collection so the frozen 3B target and retained KV pools can be released.
    getattr(model,'_opd_runtime_cache',{}).clear()
    del model
    import gc
    gc.collect()
    return result
