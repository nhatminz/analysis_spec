#!/usr/bin/env python3
"""Frozen step-234 SPARK OFF/ON: AAL throughout rollout, prompt-group B."""
import argparse
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def parser():
    from rq2.aal_progress import AALConfig
    c=AALConfig();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','smoke','collect','analyze','all'])
    p.add_argument('--output-dir',default=str(ROOT/'outputs/rq2_aal_step234'))
    p.add_argument('--checkpoint-path',default=c.checkpoint_path)
    p.add_argument('--checkpoint-step',type=int,default=c.checkpoint_step)
    p.add_argument('--model',default=os.environ.get('MODEL',c.model))
    p.add_argument('--test-path',default=os.environ.get('TEST_DATASET_PATH',c.test_path))
    p.add_argument('--num-rollouts',type=int,default=c.num_rollouts)
    p.add_argument('--prompts-per-rollout',dest='batch_size',type=int,default=c.batch_size)
    p.add_argument('--responses-per-prompt',type=int,default=c.responses_per_prompt)
    for flag in ('temperature','top_p','draft_token_length_c','opd_fast_lr'):
        p.add_argument('--'+flag.replace('_','-'),type=float,default=getattr(c,flag))
    for flag in ('seed','max_length','verification_capacity','max_verification_num','max_draft_token_length',
                 'min_draft_token_length','max_draft_k','opd_topk'):
        p.add_argument('--'+flag.replace('_','-'),type=int,default=getattr(c,flag))
    p.add_argument('--top-k',type=int,default=None)
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--attention-implementation',choices=['sdpa','eager'],default='sdpa')
    p.add_argument('--opd-update-stream',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=2026)
    p.add_argument('--wall-minutes',type=float,default=150.)
    p.add_argument('--require-b200',action='store_true')
    return p


def main():
    import time
    from dataclasses import fields
    from rq2.aal_progress import AALConfig,prepare,collect,validate_output_directory
    from rq2.aal_analysis import analyze
    started=time.perf_counter();p=parser();a=p.parse_args()
    if a.bootstrap_samples<20 or not math.isfinite(a.wall_minutes) or a.wall_minutes<=0 or a.bootstrap_seed<0:
        p.error('Need >=20 bootstrap samples, a finite positive wall budget and a nonnegative bootstrap seed')
    validate_output_directory(a.output_dir)
    if a.mode=='analyze':result=analyze(a.output_dir,a.bootstrap_samples,a.bootstrap_seed)
    else:
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        c=AALConfig(**{f.name:getattr(a,f.name) for f in fields(AALConfig)})
        output,rows,tok,state,manifest=prepare(c,a.output_dir)
        result=dict(output_dir=str(output),checkpoint=manifest['checkpoint'],preflight='passed')
        if a.mode!='check':
            # Collection always requires a real per-prompt smoke with passive
            # logging replay. Smoke is stored separately from scientific data.
            smoke=collect(c,output,rows,tok,state,manifest,smoke=True,wall_minutes=a.wall_minutes,require_b200=a.require_b200)
            result['smoke']=smoke
            if smoke['status']=='complete':
                result['smoke_analysis']=analyze(output,a.bootstrap_samples,a.bootstrap_seed,smoke=True)
                if a.mode in ('collect','all'):
                    remaining=a.wall_minutes-(time.perf_counter()-started)/60
                    if remaining>0:
                        result['collection']=collect(c,output,rows,tok,state,manifest,wall_minutes=remaining,require_b200=a.require_b200)
                        if a.mode=='all' and list((output/'collection/on').glob('rollout_*/complete.json')):
                            result['analysis']=analyze(output,a.bootstrap_samples,a.bootstrap_seed)
    result['command_runtime_s']=time.perf_counter()-started
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
