#!/usr/bin/env python3
"""Standalone SimpleLR/Qwen2.5-3B-Instruct A1 + A2 experiment."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
if str(ROOT) in sys.path:sys.path.remove(str(ROOT))
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/qwen25_3b/simplelr_a1_a2.json'))
    p.add_argument('--mode',choices=['prepare','validate','run','replot','report'],default='run')
    for name in ('model','draft-checkpoint','draft-target-config','target-adapter','train-path','test-path','output-dir','dtype','attention-implementation'):
        p.add_argument('--'+name,default=None)
    for name in ('train-steps','seed','eval-subset-seed','eval-max-new-tokens','bootstrap-samples','accumulation-steps',
                 'draft-accumulation-steps','batch-size','responses-per-prompt','max-length','max-prompt-length',
                 'max-training-token','max-training-padding-gap','smoke-eval-prompts','save-every'):
        p.add_argument('--'+name,type=int,default=None)
    for name in ('target-lr','draft-lr','opd-projector-lr','opd-fast-lr','replay-hidden-atol','replay-hidden-rtol','replay-distribution-tv'):
        p.add_argument('--'+name,type=float,default=None)
    for name in ('eval-steps','confirmation-steps','zero-update-steps'):
        p.add_argument('--'+name,default=None,help='Comma separated completed target optimizer steps; empty string disables')
    p.add_argument('--resume',default='',help='auto or an explicit experiment checkpoint')
    p.add_argument('--smoke',action='store_true',help='2 steps, 2 held-out prompts; explicitly nonpublishable')
    return p


def resolve_paths(config):
    workspace=ROOT.parent
    def existing_or_error(candidates,label):
        for path in candidates:
            if Path(path).exists():return str(Path(path).resolve())
        raise FileNotFoundError(f'{label} missing. Set its CLI path explicitly. Checked: '+', '.join(map(str,candidates)))
    if not config.model:
        config.model=existing_or_error(['/workspace/storage-shared/models/Qwen2.5-3B-Instruct',workspace/'models/Qwen2.5-3B-Instruct'],'Qwen2.5-3B-Instruct model')
    for split in ('train','test'):
        if not getattr(config,split+'_path'):
            roots=[os.environ.get('DATA_ROOT','/workspace/storage-shared/nlp/minhpn19/data'),workspace/'data']
            setattr(config,split+'_path',existing_or_error([Path(r)/'simplelr_abel_level3to5'/f'{split}.parquet' for r in roots],f'SimpleLR {split}'))
    if not config.output_dir:config.output_dir=str(ROOT/'outputs/qwen25_3b_simplelr_a1_a2')
    # A checkpoint is input data, never executable code or an import path.
    # Do not guess some other model's checkpoint or a validation-only pretrain.
    if not config.draft_checkpoint:
        candidates=[ROOT/'checkpoints/qwen25_3b/draft.pth',workspace/'SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint']
        config.draft_checkpoint=existing_or_error(candidates,'Pretrained Qwen2.5-3B FastGRPO draft')
    for name in ('model','draft_checkpoint','train_path','test_path','output_dir','target_adapter','draft_target_config'):
        value=getattr(config,name)
        if value:setattr(config,name,str(Path(value).expanduser().resolve()))
    return config


def main(argv=None):
    args=parser().parse_args(argv)
    from motivation.config import load_config
    config=load_config(args.config)
    for key,value in vars(args).items():
        if key in ('mode','config','resume','smoke') or value is None:continue
        if key in ('eval_steps','confirmation_steps','zero_update_steps'):
            value=tuple(int(x) for x in value.split(',') if x.strip())
        setattr(config,key,value)
    if args.smoke:
        config.smoke=True
        if args.train_steps is None:config.train_steps=2
        if args.eval_steps is None:config.eval_steps=tuple(sorted({1,config.train_steps}))
        if args.confirmation_steps is None:config.confirmation_steps=()
        if args.zero_update_steps is None:config.zero_update_steps=(1,)
        if args.eval_max_new_tokens is None:config.eval_max_new_tokens=16
    if args.mode in ('replot','report'):
        root=Path(config.output_dir or ROOT/'outputs/qwen25_3b_simplelr_a1_a2')/'analysis'
        if args.mode=='replot':
            from motivation.metrics import replot
            replot(root);print(root/'plots')
        else:
            print((root/'step_metrics.csv').read_text())
        return 0
    # Dataset-only preparation does not require the pretrained draft to exist.
    if args.mode=='prepare':
        original=config.draft_checkpoint;config.draft_checkpoint='__not_required_for_prepare__'
        config=resolve_paths(config);config.draft_checkpoint=original
    else:config=resolve_paths(config)
    config.validate()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    os.environ['OPD_SAMPLER_MODE']=config.opd_sampler_mode
    os.environ['OPD_PROPOSAL_MODE']=config.opd_proposal_mode
    os.environ['OPD_DENSE_IMPLEMENTATION']=config.opd_dense_implementation
    os.environ['OPD_PROPOSAL_PROFILE']=config.opd_proposal_profile
    from transformers import AutoTokenizer
    from motivation.data import prepare
    from motivation.compatibility import model_identity,validate_draft,validate_tokenizer,tokenizer_identity
    from motivation.state import atomic_json,file_hash
    tokenizer=AutoTokenizer.from_pretrained(config.model,padding_side='left',local_files_only=True)
    model=model_identity(config.model);validate_tokenizer(tokenizer,model['config'])
    analysis=Path(config.output_dir)/'analysis'
    train,heldout,data_manifest=prepare(config.train_path,config.test_path,analysis,tokenizer,config.eval_subset_seed)
    if args.mode=='prepare':
        print(json.dumps(dict(train_unique=len(train),test_unique=data_manifest['test']['unique_rows'],subset=64,
                              manifest=str(analysis/'eval_subset_manifest.json')),indent=2));return 0
    draft=validate_draft(config.draft_checkpoint,config.model,config.draft_target_config)
    config.draft_checkpoint=draft['path']
    implementation={str(p.relative_to(ROOT)):file_hash(p) for directory in ('helper','motivation') for p in sorted((ROOT/directory).glob('*.py'))}
    for name in ('teacher_relabel.py','run_policy_lag_motivation.py','PORT_SOURCES.json'):
        implementation[name]=file_hash(ROOT/name)
    import platform,torch,transformers,peft
    runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                 cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    target_adapter_hashes={str(p.relative_to(config.target_adapter)):file_hash(p) for p in sorted(Path(config.target_adapter).rglob('*')) if p.is_file()} if config.target_adapter else {}
    profile_hash=file_hash(config.opd_proposal_profile) if config.opd_proposal_profile else None
    manifest=dict(runtime=runtime,target_adapter_sha256=target_adapter_hashes,opd_profile_sha256=profile_hash,format='simplelr_opd_policy_lag_a1a2_v1',config=config.to_dict(),model=model,draft=draft,
                  tokenizer_files=tokenizer_identity(config.model),data=data_manifest,implementation_sha256=implementation,
                  target_trajectory='OPD-driven',reference_variant='Qwen2.5-3B-Instruct',
                  reference_target_accumulation=4,experiment_target_accumulation=1,research_result=not config.smoke,
                  eval_conditions=6,eval_response_count_per_prompt=1,
                  teacher_distribution='full softmax frozen shared head (distinct from OPD truncated sampler teacher)')
    path=analysis/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Experiment manifest differs; use a new output directory')
    atomic_json(path,manifest)
    if args.mode=='validate':print(json.dumps(dict(model=model['path'],draft=draft['path'],train=config.train_path,test=config.test_path,valid=True),indent=2));return 0
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Production FastGRPO/OPD requires CUDA; run CPU unit tests or --mode prepare/validate here')
    from motivation.runner import run
    result=run(config,train,heldout,tokenizer,manifest,args.resume)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,RuntimeError) as error:
        print(f'ERROR: {error}',file=sys.stderr);raise SystemExit(2)
