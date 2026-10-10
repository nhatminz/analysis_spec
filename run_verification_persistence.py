#!/usr/bin/env python3
"""Frozen FastGRPO verification-root prediction-error persistence; no training."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:sys.path.remove(str(ROOT))
sys.path.insert(0,str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=os.environ.get('ANALYSIS_CHECKPOINT',os.environ.get('CHECKPOINT','')),
                   help='Existing online analysis checkpoint containing target_lora and shadow.weights')
    p.add_argument('--source-manifest',default='',help='Defaults to CHECKPOINT/../../analysis/manifest.json')
    p.add_argument('--model',default=os.environ.get('MODEL',''))
    p.add_argument('--train-path',default=os.environ.get('TRAIN_DATASET_PATH',''))
    p.add_argument('--test-path',default=os.environ.get('TEST_DATASET_PATH',''))
    p.add_argument('--output-dir',default=os.environ.get('OUTPUT_DIR',str(ROOT/'outputs/verification_persistence')))
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--bootstrap-seed',type=int,default=2026)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--num-prompts',type=int,default=None)
    p.add_argument('--donor-prompts',type=int,default=None)
    p.add_argument('--max-new-tokens',type=int,default=None)
    p.add_argument('--wall-clock-minutes',type=float,default=150)
    p.add_argument('--smoke',action='store_true',help='4 prompts (2 donors, 2 eval), 32 tokens; nonpublishable')
    return p


def main(argv=None):
    started = time.perf_counter();args = parser().parse_args(argv)
    args.num_prompts = args.num_prompts if args.num_prompts is not None else (4 if args.smoke else 64)
    args.donor_prompts = args.donor_prompts if args.donor_prompts is not None else (2 if args.smoke else 16)
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (32 if args.smoke else 256)
    if not 1 <= args.donor_prompts < args.num_prompts <= 64 or args.max_new_tokens < 2:
        raise ValueError('Require 1 <= donor-prompts < num-prompts <= 64 and max-new-tokens >= 2')
    if not args.smoke and (args.num_prompts,args.donor_prompts,args.max_new_tokens) != (64,16,256):
        raise ValueError('Production protocol is 64 prompts, 16 donors, 256 tokens; use --smoke for smaller checks')
    if args.wall_clock_minutes <= 0 or args.bootstrap_samples < 1:
        raise ValueError('Positive wall-clock and bootstrap budgets required')
    if not args.checkpoint:
        args.checkpoint = str(ROOT/'outputs/qwen25_3b_simplelr_a1_a2/checkpoints/latest.pt')
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    source_path = Path(args.source_manifest).expanduser().resolve(strict=True) if args.source_manifest else checkpoint.parent.parent/'analysis/manifest.json'
    source = json.loads(source_path.read_text());source_config = source['config']
    args.model = args.model or source_config['model']
    for split in ('train','test'):
        value = getattr(args,split+'_path')
        if not value and os.environ.get('DATA_ROOT'):
            value = str(Path(os.environ['DATA_ROOT'])/'simplelr_abel_level3to5'/f'{split}.parquet')
        setattr(args,split+'_path',value or source_config[split+'_path'])
    for name in ('model','train_path','test_path'):
        setattr(args,name,str(Path(getattr(args,name)).expanduser().resolve(strict=True)))
    root = Path(args.output_dir).expanduser().resolve()
    args.checkpoint = str(checkpoint)
    args.source_manifest = str(source_path.resolve(strict=True))
    args.output_dir = str(root)
    if any((root/name).exists() for name in ('manifest.json','responses','summary.json')):
        raise ValueError('Output already contains observations; use a fresh --output-dir')
    root.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    import torch
    import transformers
    import peft
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    from motivation.compatibility import model_identity,validate_tokenizer,tokenizer_identity
    from motivation.data import prepare,tokenize
    from motivation.runtime import generator_kwargs,evaluation_state,target_state
    from motivation.config import Config
    from motivation.state import atomic_json,digest,file_hash,generation_seed,isolated_rng
    from motivation.verification_persistence import RootObserver,load_frozen_shadow,save_results
    from helper.fastgrpo_generate import speculative_generate
    if not torch.cuda.is_available():raise RuntimeError('FastGRPO generation requires CUDA')
    checkpoint_stat = checkpoint.stat()
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if saved.get('manifest_sha256') != digest(source) or saved.get('config_sha256') != digest(source_config):
        raise ValueError('Checkpoint does not match its source experiment manifest/config')
    if not saved.get('target_lora') or not saved.get('shadow',{}).get('weights'):
        raise ValueError('Checkpoint must contain target_lora and shadow.weights')
    identity = model_identity(args.model)
    if identity['weight_sha256'] != source['model']['weight_sha256']:
        raise ValueError('Target base model weights differ from the checkpoint source model')
    tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side='left',local_files_only=True)
    validate_tokenizer(tokenizer,identity['config'])
    _,heldout,data_manifest = prepare(args.train_path,args.test_path,root/'data',tokenizer,subset_seed=2026)
    if source['data']['test']['sha256'] != data_manifest['test']['sha256']:
        raise ValueError('Test dataset differs from the checkpoint experiment')
    if source['data']['subset'] != data_manifest['subset']:
        raise ValueError('Subset prompt/token alignment differs from the existing deterministic 64-prompt subset')
    config = Config(**{k:(tuple(v) if k in ('eval_steps','confirmation_steps','zero_update_steps') else v)
                       for k,v in source_config.items()})
    kwargs = generator_kwargs(config,method='fastgrpo',feedback=False,train=False)
    kwargs = {k:v for k,v in kwargs.items() if not k.startswith('opd_') and k not in ('method','kv_gather_strategy')}
    if (config.temperature,config.top_p,config.top_k) != (1.,.95,None):
        raise ValueError('Expected original Qwen FastGRPO sampling: temperature=1, top_p=.95, top_k=None')
    model = load_frozen_shadow(args.model,saved,source_config)
    weights_before = digest(dict(target=target_state(model.target_model),shadow=model.draft_model.state_dict()))
    manifest = dict(format='verification_persistence_v1',status='running',config=vars(args),
        started_at_utc=datetime.now(timezone.utc).isoformat(),runtime_s=time.perf_counter()-started,
        seeds=dict(subset=2026,generation=args.seed,controls=args.seed,bootstrap=args.bootstrap_seed),
        checkpoint=dict(path=str(checkpoint),sha256=file_hash(checkpoint),policy_step=saved['policy_step'],
            target_lora_sha256=digest(saved['target_lora']),shadow_weights_sha256=digest(saved['shadow']['weights']),
            source_manifest_path=str(source_path),source_manifest_sha256=file_hash(source_path),
            source_research_result=source.get('research_result',False)),
        model=identity,tokenizer=tokenizer_identity(args.model),data_manifest_sha256=file_hash(root/'data/eval_subset_manifest.json'),
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
            python=sys.version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        generation_kwargs=kwargs,subset=[x['id'] for x in heldout[:args.num_prompts]],
        q_distribution='untempered original FastGRPO root proposal softmax',
        p_distribution='exact target multinomial probabilities after temperature, top-p then top-k; no recomputation',
        alignment='equal root input token, logical target position, prefix cache length; draft shifted-input convention',
        control='one independent donor at identical origin-round index; sparse gather on recipient future context',
        observations='8 strictly positive-error token IDs or missing; sparse future means only; no vocabulary CPU transfers',
        stopping='original FastGRPO EOS and exact max_new_tokens cap, no acceptance-counter changes',
        research_result=not args.smoke and source.get('research_result',False),
        implementation_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in
            [Path(__file__).resolve(),ROOT/'motivation/verification_persistence.py',ROOT/'helper/fastgrpo_generate.py']})
    current_stat = checkpoint.stat()
    if any(getattr(checkpoint_stat, key) != getattr(current_stat, key)
           for key in ('st_dev','st_ino','st_size','st_mtime_ns')):
        raise ValueError('Checkpoint changed during loading/hashing; supply a stable saved checkpoint')
    del saved
    responses = [];donors = {}
    save_results(root,responses,manifest,plot=False)
    interrupted = None
    try:
        with evaluation_state([model]),torch.inference_mode():
            for i,row in enumerate(tqdm(heldout[:args.num_prompts],desc='Verification persistence',unit='response')):
                # Budget includes startup/model loading. Never interrupt a response
                # mid-verification; completed responses are atomically journaled.
                if time.perf_counter()-started >= args.wall_clock_minutes*60:
                    manifest['status']='wall_clock_budget_exceeded';break
                role = 'donor' if i < args.donor_prompts else 'evaluation'
                observer = RootObserver(row['id'],None if role=='donor' else donors,args.seed)
                batch = tokenize([row],tokenizer)
                prompt_length = int(batch['attention_mask'].sum())
                if prompt_length+args.max_new_tokens > identity['config']['max_position_embeddings']:
                    raise ValueError('Prompt plus generation budget exceeds target context capacity')
                seed = generation_seed(args.seed,manifest['checkpoint']['policy_step'],row['id'])
                response_started = time.perf_counter()
                with isolated_rng(seed):
                    output = speculative_generate(model=model,tokenizer=tokenizer,
                        input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                        max_length=prompt_length+args.max_new_tokens,max_new_tokens=args.max_new_tokens,
                        observation_hook=observer,**kwargs)
                torch.cuda.synchronize()
                response = observer.finish(output,tokenizer.eos_token_id)
                response.update(role=role,generation_seed=seed,runtime_s=time.perf_counter()-response_started,
                                question_sha256=row['question_sha256'])
                atomic_json(root/'responses'/f'{i:03d}.json',response)
                responses.append(response)
                if role=='donor':donors[row['id']] = [x['token_ids'] for x in observer.rounds]
                manifest['runtime_s']=time.perf_counter()-started
                save_results(root,responses,manifest,plot=False)
            else:manifest['status']='complete'
    except KeyboardInterrupt as error:
        manifest['status']='interrupted';interrupted=error
    except Exception as error:
        manifest['status']='failed';manifest['error']=f'{type(error).__name__}: {error}';interrupted=error
    finally:
        manifest['runtime_s']=time.perf_counter()-started
        manifest['frozen_weights_verified'] = weights_before == digest(dict(target=target_state(model.target_model),shadow=model.draft_model.state_dict()))
        if not manifest['frozen_weights_verified']:
            manifest['status']='failed';manifest['error']='Frozen checkpoint weights changed'
        summary = save_results(root,responses,manifest)
        manifest['runtime_s']=time.perf_counter()-started
        atomic_json(root/'manifest.json',manifest)
        summary['runtime_s']=manifest['runtime_s'];atomic_json(root/'summary.json',summary)
    print(json.dumps(dict(status=manifest['status'],runtime_s=manifest['runtime_s'],
        sample_counts=manifest['sample_counts'],output_dir=str(root)),indent=2))
    if interrupted is not None:raise interrupted
    if not manifest['frozen_weights_verified']:raise RuntimeError('Frozen checkpoint weights changed')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
