#!/usr/bin/env python3
"""Tune execution-keyed proposal costs once for all compatible pretrained models."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import warnings
import pickle
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from helper.opd_profiles import (active_trials,context_shapes,execution_key,fingerprint,
    inspect_draft,profile_filename,discover_profile,validate_profile)

DEFAULT_MODELS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b')


def measure(fn,iterations,torch):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
    for _ in range(iterations):
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def benchmark_configuration(key,shapes,slots,iterations,progress=True):
    import torch
    import numpy as np
    from types import SimpleNamespace
    from tqdm import tqdm
    from helper.opd_reflex import OPDReflex
    from helper import opd_reflex_kernels as kernels
    dtype=getattr(torch,key['dtype'].split('.')[-1]);v,r,k=key['vocab'],key['rank'],key['topk']
    records=[];thresholds={};dense_implementations={}
    # Already-projected U is the proposal input. H-dependent feature extraction
    # is common to all backends and deliberately excluded from proposal tuning.
    with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        for b,c in tqdm(shapes,desc='Contexts',position=2,leave=False,disable=not progress):
            mapping=torch.arange(v,device='cuda')
            head=SimpleNamespace(weight=SimpleNamespace(shape=(v,r),dtype=dtype),bias=None)
            model=SimpleNamespace(lm_head=head,opd_projector=torch.eye(r,device='cuda'))
            s=OPDReflex(r,k,backend='triton');s.tuning=None;s.profile_selector=None;s._validated_tuning=True
            s.start(model,b,mapping,r,max_contexts=c,max_nodes=b*c,max_path=2,max_proposal_contexts=c)
            s.logits_dtype=dtype
            raw=torch.randn(b,c,v,device='cuda',dtype=dtype);u=torch.randn(b,c,r,device='cuda')
            trials=[]
            for count in tqdm(slots,desc='Active rows',position=3,leave=False,disable=not progress):
                s.B_fast.zero_();s.bitmap.zero_();s.active_count.fill_(count)
                ids=torch.randperm(v,device='cuda')[:count].sort().values
                s.active_ids[:count]=ids;s.B_fast[ids]=torch.randn(count,r,device='cuda')*.05
                cpu_ids=ids.cpu().numpy();bits=np.zeros((v+31)//32,dtype=np.uint32)
                np.bitwise_or.at(bits,cpu_ids//32,np.left_shift(np.uint32(1),(cpu_ids%32).astype(np.uint32)))
                s.bitmap.copy_(torch.from_numpy(bits.view(np.int32)));s._ever_updated=bool(count);s.host_active_count=count
                results={};times={}
                outputs=(s.proposal_q[:b*c*k].view(b,c,k),s.proposal_ids[:b*c*k].view(b,c,k),s.proposal_norm[:b*c*2].view(b,c,2))
                for backend in ('sparse','fused','gemm'):
                    s.proposal_mode='sparse' if backend=='sparse' else 'dense'
                    s.dense_implementation='gemm' if backend=='gemm' else 'fused'
                    def run():kernels.propose(raw,u,s,k,s.proposal_tiles,outputs,bool(count),root=False)
                    run();results[backend]=tuple(x.clone() for x in outputs)
                    times[backend]=measure(run,iterations,torch)
                for backend in ('fused','gemm'):
                    if not all(torch.equal(x,y) for x,y in zip(results['sparse'],results[backend])):
                        raise AssertionError(f'{b*c} contexts S={count}: sparse/{backend} parity failed; no profile')
                trials.append(dict(slots=count,**times,bitwise_parity=True))
            dense=min(('fused','gemm'),key=lambda mode:sum(t[mode] for t in trials))
            threshold=min([t['slots'] for t in trials]+[v+1],key=lambda limit:sum(t['sparse'] if t['slots']<limit else t[dense] for t in trials))
            old_key=f'{b},{c},{v},{r},{dtype}';thresholds[old_key]=threshold;dense_implementations[old_key]=dense
            records.append(dict(shape=f'{b}x{c}',contexts=b*c,threshold=threshold,dense_implementation=dense,trials=trials))
            del s,model,raw,u,outputs,results
    return dict(schema_version=2,execution_key=key,**{name:key[name] for name in ('gpu','compute_capability','torch','triton','cuda','kernel_sha256','vocab','rank','dtype')},
        thresholds=thresholds,dense_implementations=dense_implementations,records=records,
        benchmark_metadata=dict(iterations=iterations,warmup=3,unit='ms',statistic='median',proposal_only=True,hidden_independent=True),
        note='Proposal-only measurements. Dispatch interpolates measured costs in contexts AND active rows. Validate end-to-end; no AAL/speedup claim.')


def inspect_models(args,progress=True):
    from tqdm import tqdm
    models=[];skipped=[]
    for name in tqdm(args.models.split(','),desc='Models inspected',position=0,disable=not progress):
        root=Path(args.pretrain_root)/name
        try:
            if len(args.models.split(','))==1:
                config=args.target_config or root/'latest_target_config.json'
                checkpoint=args.draft_checkpoint or root/'latest_checkpoint'
            else:config,checkpoint=root/'latest_target_config.json',root/'latest_checkpoint'
            detected=inspect_draft(config,checkpoint,rank=args.rank,dtype=args.dtype,topk=args.topk)
            models.append(dict(model=name,**detected))
        except (FileNotFoundError,ValueError,KeyError,RuntimeError,OSError,EOFError,pickle.UnpicklingError) as exc:
            warnings.warn(f'{name}: skip: {exc}');skipped.append(dict(model=name,reason=str(exc)))
    return models,skipped


def tune_models(args,*,hardware=None,benchmark_fn=benchmark_configuration,progress=True):
    from tqdm import tqdm
    models,skipped=inspect_models(args,progress)
    if not models:
        print('No valid draft resources. Models skipped:',len(skipped))
        print('model -> detected V/r/dtype -> profile used')
        for m in skipped:print(f'{m["model"]} -> SKIPPED: {m["reason"]}')
        return dict(models=[],skipped=skipped,unique_configs=0)
    hardware=hardware or fingerprint();groups={}
    for model in models:
        key=execution_key(hardware,model['vocab'],model['rank'],model['dtype'],model['topk'])
        groups.setdefault(profile_filename(key),dict(key=key,models=[]))['models'].append(model)
    if args.output and len(groups)!=1:raise ValueError('--output is one profile; multiple configs require --profile-dir')
    for name,group in tqdm(groups.items(),desc='Unique configs',position=1,disable=not progress):
        key=group['key'];path=Path(args.output) if args.output else Path(args.profile_dir)/name
        existing,payload=discover_profile(args.profile_dir,key)
        if existing and not args.force:
            if args.output and path.resolve()!=existing.resolve():
                if path.exists():raise FileExistsError(f'{path}: choose a new output path')
                path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(existing.resolve())
            else:path=existing
            print(f'Reuse {path}')
        else:
            if path.exists() and not args.force:raise FileExistsError(f'{path}: use --force or a new profile path')
            shapes=([tuple(map(int,shape.split('x'))) for shape in args.shapes.split(',')] if args.shapes else
                    context_shapes(args.batch_size,args.responses,args.max_draft_k,args.context_points))
            slots=(sorted({min(key['vocab'],key['vocab'] if x=='V' else int(x)) for x in args.slots.split(',')}) if args.slots else
                   active_trials(key['vocab'],key['topk'],args.active_points))
            if any(min(shape)<1 for shape in shapes) or min(slots)<0:raise ValueError('invalid workload')
            # Flat proposal workload, not duplicate b*c factorizations.
            shapes=[(n,1) for n in sorted({b*c for b,c in shapes})]
            payload=benchmark_fn(key,shapes,slots,args.iterations,progress)
            validate_profile(payload,key)
            payload['models_inspected']=group['models']
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(path)
        for model in group['models']:model['profile']=str(path.resolve())
    summary=dict(models=models,skipped=skipped,unique_configs=len(groups))
    print('\nmodel -> detected V/r/dtype -> profile used')
    for m in models:print(f'{m["model"]} -> V{m["vocab"]}/r{m["rank"]}/{m["dtype"]} -> {m["profile"]}')
    for m in skipped:print(f'{m["model"]} -> SKIPPED: {m["reason"]}')
    return summary


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default=','.join(DEFAULT_MODELS));p.add_argument('--pretrain-root',default=str(ROOT/'outputs/pretrain'))
    p.add_argument('--profile-dir',default=str(ROOT/'outputs/benchmarks/opd_proposals'));p.add_argument('--output')
    for name in ('target-config','draft-checkpoint'):p.add_argument('--'+name)
    p.add_argument('--rank',type=int);p.add_argument('--dtype',choices=['bf16','fp16','fp32'])
    p.add_argument('--topk',type=int,default=16);p.add_argument('--shapes');p.add_argument('--slots')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--responses',type=int,default=8)
    p.add_argument('--max-draft-k',type=int,default=8);p.add_argument('--context-points',type=int,default=7)
    p.add_argument('--active-points',type=int,default=8);p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--force',action='store_true');p.add_argument('--inspect-only',action='store_true')
    a=p.parse_args(argv)
    if not a.models or min(a.topk,a.batch_size,a.responses,a.max_draft_k,a.context_points,a.active_points,a.iterations)<1:p.error('positive workload sizes required')
    if len(a.models.split(','))!=1 and any((a.target_config,a.draft_checkpoint)):
        p.error('per-model paths require --models <one model>; use --pretrain-root for multiple models')
    return a


def main(argv=None):
    args=parse_args(argv)
    os.environ['OPD_PROPOSAL_PROFILE']='';os.environ['OPD_PROPOSAL_PROFILE_DIR']=''
    if args.inspect_only:
        models,skipped=inspect_models(args);print(json.dumps(dict(models=models,skipped=skipped),indent=2));return
    summary=tune_models(args)
    if summary['models']:
        destination=Path(args.profile_dir)/'models_summary.json'
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
