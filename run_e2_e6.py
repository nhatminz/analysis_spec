#!/usr/bin/env python3
"""Pure FastGRPO E2–E6: independently resumable training, logging and analysis."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
if str(ROOT) in sys.path:sys.path.remove(str(ROOT))
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','resume','collect','analyze','all','check'])
    for name in ('model','train-path','test-path','draft-checkpoint','draft-target-config','output-dir','dtype','attention-implementation'):
        p.add_argument('--'+name,default=None)
    for name in ('train-steps','batch-size','responses-per-prompt','accumulation-steps','draft-accumulation-steps','seed',
                 'eval-seed','eval-rollouts','bootstrap-samples','save-every','max-length','max-prompt-length',
                 'max-training-token','max-training-padding-gap','draft-ce-chunk-size','verification-capacity',
                 'max-verification-num','max-draft-token-length','max-draft-k','min-draft-token-length','max-rollout-attempts'):
        p.add_argument('--'+name,type=int,default=None)
    for name in ('target-lr','draft-lr','beta','epsilon','temperature','top-p','draft-token-length-c'):
        p.add_argument('--'+name,type=float,default=None)
    p.add_argument('--checkpoints',default='30,100,200',help='Comma-separated milestones; missing milestones are reported and skipped')
    p.add_argument('--require-b200',action='store_true',default=None)
    p.add_argument('--smoke',action='store_true',default=None,help='One evaluation rollout; still needs a genuine pretrained draft/checkpoint')
    p.add_argument('--target-gradient-checkpointing',action=argparse.BooleanOptionalAction,default=None)
    return p


def config_for(args):
    from e2e6.config import Config
    root=Path(args.output_dir or os.environ.get('OUTPUT_DIR') or ROOT/'outputs/e2_e6_fastgrpo').expanduser().resolve()
    if not root.is_relative_to(ROOT):raise ValueError('All E2–E6 outputs/checkpoints/logs must reside inside analysis_spec')
    manifest=root/'train_manifest.json'
    c=Config(**json.loads(manifest.read_text())['config']) if manifest.exists() else Config()
    env_names=dict(model='MODEL',train_path='TRAIN_DATASET_PATH',test_path='TEST_DATASET_PATH',
        draft_checkpoint='DRAFT_CHECKPOINT',draft_target_config='DRAFT_TARGET_CONFIG',seed='SEED')
    for field,variable in env_names.items():
        if variable in os.environ:
            value=os.environ[variable];setattr(c,field,int(value) if field=='seed' else value)
    for key,value in vars(args).items():
        if hasattr(c,key) and value is not None:setattr(c,key,value)
    c.output_dir=str(root)
    if args.smoke:c.smoke=True
    return c


def preflight(c,training):
    import importlib.metadata
    from transformers import AutoTokenizer
    from motivation.compatibility import model_identity,validate_draft,validate_tokenizer,tokenizer_identity,validate_cuda_runtime
    from motivation.data import prepare,read_split
    from motivation.state import atomic_json,digest,file_hash
    c.validate()
    for name in ('model','draft_checkpoint','train_path','test_path','draft_target_config'):
        value=getattr(c,name)
        if value:setattr(c,name,str(Path(value).expanduser().resolve(strict=True)))
    identity=model_identity(c.model)
    draft=validate_draft(c.draft_checkpoint,c.model,c.draft_target_config)
    draft['projector_initialization']='disabled; no Reflex projector is constructed or loaded'
    c.draft_checkpoint=draft['path']
    tokenizer=AutoTokenizer.from_pretrained(c.model,padding_side='left',local_files_only=True)
    validate_tokenizer(tokenizer,identity['config'])
    rows,heldout,data=prepare(c.train_path,c.test_path,Path(c.output_dir)/'data',tokenizer,subset_seed=2026)
    tests,_=read_split(c.test_path,'test')
    runtime=validate_cuda_runtime(c.require_b200)
    # Collection/analysis settings may change without changing the training identity.
    config=c.to_dict()
    for name in ('eval_rollouts','bootstrap_samples','smoke','require_b200'):config.pop(name)
    source_config=c.to_dict()
    # Immutable training manifest also stores phase defaults for independent commands.
    existing_path=Path(c.output_dir)/'train_manifest.json'
    if existing_path.exists():
        previous=json.loads(existing_path.read_text());source_config=previous['config']
    sources=[ROOT/name for name in ('run_e2_e6.py','e2e6/config.py','e2e6/runtime.py','e2e6/train.py',
        'e2e6/checkpoints.py','helper/fastgrpo_generate.py','helper/fastgrpo_training.py',
        'helper/fastgrpo_model.py','helper/modeling_draft.py','helper/rewards.py','helper/get_QAs.py',
        'helper/opd_optimizer.py','helper/checkpointing.py','helper/transformers_compat.py',
        'motivation/runtime.py','motivation/data.py','motivation/runner.py','motivation/state.py',
        'motivation/compatibility.py')]
    manifest=dict(format='pure_fastgrpo_e2_e6_training_manifest_v1',config=source_config,training_config=config,
        model=identity,draft=draft,data=data,tokenizer=tokenizer_identity(c.model),
        implementation={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        package_versions={name:importlib.metadata.version(name) for name in
            ('torch','transformers','peft','numpy','pandas','pyarrow')},
        reference='SpecNaacl is read-only; no imports; original FastGRPO feature/CE + target GRPO objectives',
        target_step_semantics='Count completed nonzero-gradient target optimizer updates. Reference accumulation_steps is reporting metadata; source steps after every nonempty eligible rollout.',
        pure_fastgrpo=True,projector_loaded=False,pretrained_projector_discarded=draft['projector_pretrained'])
    if existing_path.exists():
        if json.loads(existing_path.read_text())!=manifest:raise ValueError('Immutable training manifest differs; restore original training config/source/data')
    elif not training:raise ValueError('No training manifest; train first, then collect available milestones')
    else:atomic_json(existing_path,manifest)
    atomic_json(Path(c.output_dir)/'runtime_preflight.json',runtime)
    return rows,tests,tokenizer,manifest


def main(argv=None):
    args=parser().parse_args(argv);c=config_for(args)
    steps=[int(s) for s in args.checkpoints.split(',') if s.strip()]
    if any(s not in (30,100,200,500) for s in steps):raise ValueError('Select milestone steps 30,100,200,500')
    root=Path(c.output_dir);root.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',METHOD='fastgrpo')
    if args.mode=='check':
        from e2e6.checkpoints import select,latest,verify
        paths,missing=select(root/'checkpoints',steps)
        result=dict(milestones=[dict(path=str(p),**verify(p,load=False)) for p in paths],missing=missing)
        if (root/'checkpoints/latest.json').exists():result['latest']=dict(path=str(latest(root/'checkpoints')),**verify(latest(root/'checkpoints'),load=False))
        from e2e6.collect import validate_rollout
        committed=list((root/'collection').glob('step_*/rollout_*/complete.json'))
        for marker in committed:validate_rollout(marker.parent)
        result['verified_rollouts']=len(committed)
    elif args.mode=='analyze':
        from e2e6.analysis import analyze
        result=analyze(root,steps,c.bootstrap_samples,c.eval_seed)
    else:
        rows,tests,tokenizer,manifest=preflight(c,args.mode in ('train','resume','all'))
        result={};holder={}
        if args.mode in ('train','resume','all'):
            from e2e6.train import train
            result['training']=train(c,rows,tokenizer,manifest,resume=args.mode=='resume' or (args.mode=='all' and (root/'checkpoints/latest.json').exists()),
                                     model_holder=holder if args.mode=='all' else None)
            if result['training']['stopped']:
                print(json.dumps(result,indent=2));return 0
        if args.mode in ('collect','all'):
            from e2e6.collect import collect
            result['collection']=collect(c,tests,tokenizer,manifest,steps,model=holder.get('model'))
            if result['collection']['status']=='interrupted':
                print(json.dumps(result,indent=2));return 0
        if args.mode=='all':
            from e2e6.analysis import analyze
            result['analysis']=analyze(root,steps,c.bootstrap_samples,c.eval_seed)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':
    started=time.perf_counter()
    try:raise SystemExit(main())
    except Exception as error:
        try:
            args=parser().parse_args();root=Path(config_for(args).output_dir)
            from e2e6.train import append_log
            append_log(root/'logs/failures.jsonl',dict(mode=args.mode,error=repr(error),traceback=traceback.format_exc(),runtime_s=time.perf_counter()-started))
        except Exception:pass
        raise
