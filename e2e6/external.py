"""Read-only SpecNaacl target/draft imports for frozen E2–E6 evaluation."""
from pathlib import Path
import json
import torch
import yaml
from motivation.state import atomic_json,atomic_torch,digest,file_hash
from e2e6.checkpoints import FORMAT,verify

MODULES={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def source_info(run,step):
    run=Path(run).expanduser().resolve(strict=True)
    target=run/'checkpoints/target'/f'step{step}'
    draft=run/'checkpoints/draft'/f'step{step}.pth'
    config=run/'config_resolved.yaml'
    metadata=yaml.safe_load(config.read_text())
    if not isinstance(metadata,dict):raise ValueError('Expected a mapping in SpecNaacl resolved config')
    summary_path=run/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    methods=[v for v in (metadata.get('method'),summary.get('method')) if v is not None]
    if not methods or any(v!='fastgrpo' for v in methods):
        raise ValueError('External evaluation requires an explicitly pure fastgrpo source run')
    weights=[target/name for name in ('adapter_model.safetensors','adapter_model.bin') if (target/name).is_file()]
    if not weights:raise FileNotFoundError(f'Target LoRA weights missing: {target}')
    if not draft.is_file():raise FileNotFoundError(draft)
    files=[draft,target/'adapter_config.json',*weights,config]
    if summary_path.exists():files.append(summary_path)
    actual_steps=summary.get('target_optimizer_steps') if summary.get('final_step')==step else None
    return dict(run=str(run),target=str(target),draft=str(draft),checkpoint_label=step,
        checkpoint_label_semantics='SpecNaacl exported step label; NOT assumed to count target optimizer updates',
        actual_target_optimizer_steps=actual_steps,metadata=metadata,
        files_sha256={str(p.relative_to(run)):file_hash(p) for p in files})


def validate_adapter(config,model):
    expected=dict(peft_type='LORA',task_type='CAUSAL_LM',r=64,lora_alpha=32,
                  lora_dropout=0.,bias='none')
    for name,value in expected.items():
        if config.get(name)!=value:raise ValueError(f'Unsupported target LoRA {name}: expected {value!r}')
    if set(config.get('target_modules') or [])!=MODULES:
        raise ValueError('Target LoRA modules differ from reference FastGRPO')
    for name in ('use_dora','use_rslora','fan_in_fan_out','rank_pattern','alpha_pattern','modules_to_save',
                 'trainable_token_indices','layers_to_transform','layers_pattern','target_parameters',
                 'layer_replication','exclude_modules','lora_bias'):
        if config.get(name):raise ValueError(f'Unsupported target LoRA option: {name}')
    base=config.get('base_model_name_or_path')
    if not base or (Path(base).expanduser().resolve()!=Path(model).expanduser().resolve()
                    and base!='Qwen/Qwen2.5-3B-Instruct'):
        raise ValueError('Target adapter base model differs from the supplied Qwen2.5-3B model')


def read_pair(info,c):
    from peft.utils.save_and_load import load_peft_weights
    from motivation.compatibility import validate_draft
    target=Path(info['target'])
    validate_adapter(json.loads((target/'adapter_config.json').read_text()),c.model)
    # Trained draft exports carry no target_config.json. Validate their strict
    # architecture against the validated base model and adapter provenance.
    architecture=validate_draft(info['draft'],c.model,str(Path(c.model)/'config.json'))
    architecture['projector_initialization']='disabled; no Reflex/OPD projector is constructed'
    architecture['target_config_provenance']='supplied base config, cross-checked against source run and target adapter; trained exports have no separate target_config.json'
    state=torch.load(info['draft'],map_location='cpu',weights_only=True)['draft_model']
    if any('opd_' in key for key in state):raise ValueError('External draft contains a Reflex/OPD parameter')
    adapter=load_peft_weights(str(target),device='cpu',local_files_only=True)
    if not adapter or any('lora_' not in key for key in adapter):raise ValueError('Expected pure target LoRA weights')
    if any(not torch.isfinite(value).all() for value in adapter.values()):raise ValueError('Nonfinite target LoRA weights')
    return state,adapter,architecture


def materialize(root,step,manifest,draft,adapter):
    """Write an immutable evaluation-only snapshot, never a training resume pointer."""
    root=Path(root)
    if (root/'train_manifest.json').exists() or (root/'checkpoints/latest.json').exists():
        raise ValueError('Use a separate output directory from the active training run')
    marker=root/'external_checkpoint_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=manifest:
        raise ValueError('External checkpoint/config/source identity differs; use a separate output directory')
    path=root/'checkpoints'/f'step_{step:06d}.pt'
    if path.exists():
        state=verify(path) if path.with_suffix('.json').exists() else torch.load(path,map_location='cpu',weights_only=False)
        if (state.get('format')!=FORMAT or state.get('policy_step')!=step or not state.get('analysis_only')
                or state['manifest_sha256']!=digest(manifest)):
            raise ValueError('Existing snapshot belongs to a different checkpoint or experiment')
        if digest(state['draft_weights'])!=digest(draft) or digest(state['target_lora'])!=digest(adapter):
            raise ValueError('External source weights differ from the immutable evaluation snapshot')
    else:
        state=dict(format=FORMAT,policy_step=step,attempt=0,analysis_only=True,
            checkpoint_label_semantics=manifest['source']['checkpoint_label_semantics'],
            actual_target_optimizer_steps=manifest['source']['actual_target_optimizer_steps'],
            manifest_sha256=digest(manifest),draft_weights=draft,target_lora=adapter)
        atomic_torch(path,state)
    if not path.with_suffix('.json').exists():
        atomic_json(path.with_suffix('.json'),dict(format=FORMAT,policy_step=step,attempt=0,
            analysis_only=True,size=path.stat().st_size,sha256=file_hash(path),
            target_lora_sha256=digest(adapter),draft_weights_sha256=digest(draft)))
    verify(path,load=False)
    if not marker.exists():atomic_json(marker,manifest)
    return path


def prepare_external(c,run,step):
    import importlib.metadata
    from transformers import AutoTokenizer
    from motivation.compatibility import model_identity,validate_tokenizer,tokenizer_identity,validate_cuda_runtime
    from motivation.data import prepare,read_split
    root=Path(c.output_dir)
    if (root/'train_manifest.json').exists() or (root/'checkpoints/latest.json').exists():
        raise ValueError('Use a separate output directory from the active training run')
    info=source_info(run,step)
    c.draft_checkpoint=info['draft'];c.draft_target_config=''
    c.validate()
    for field in ('model','train_path','test_path'):
        setattr(c,field,str(Path(getattr(c,field)).expanduser().resolve(strict=True)))
    runtime=validate_cuda_runtime(c.require_b200)
    for field,source_field in (('model','model'),('train_path','dataset')):
        declared=info['metadata'].get(source_field)
        if declared and Path(declared).expanduser().resolve()!=Path(getattr(c,field)):
            raise ValueError(f'Source run {source_field} differs from the supplied {field}')
    identity=model_identity(c.model)
    draft,adapter,architecture=read_pair(info,c)
    tokenizer=AutoTokenizer.from_pretrained(c.model,padding_side='left',local_files_only=True)
    validate_tokenizer(tokenizer,identity['config'])
    _,_,data=prepare(c.train_path,c.test_path,root/'data',tokenizer,subset_seed=2026)
    tests,_=read_split(c.test_path,'test')
    config=c.to_dict()
    for key in ('smoke','require_b200'):config.pop(key)
    manifest=dict(format='pure_fastgrpo_external_evaluation_v1',analysis_only=True,config=config,
        source=info,model=identity,draft=architecture,data=data,tokenizer=tokenizer_identity(c.model),
        pure_fastgrpo=True,projector_loaded=False,
        implementation={name:file_hash(Path(__file__).parents[1]/name) for name in
            ('e2e6/external.py','run_e2_e6_external.py','e2e6/config.py','e2e6/runtime.py',
             'e2e6/collect.py','e2e6/events.py','helper/fastgrpo_model.py','helper/modeling_draft.py',
             'helper/fastgrpo_generate.py','helper/transformers_compat.py','motivation/runtime.py',
             'motivation/data.py','motivation/state.py')},
        package_versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft')})
    if source_info(run,step)!=info:raise ValueError('Source checkpoint changed while being read')
    materialize(root,step,manifest,draft,adapter)
    atomic_json(root/'runtime_preflight.json',runtime)
    return tests,tokenizer,manifest
