#!/usr/bin/env python3
"""Task 2 only: real frozen SPARK OFF/ON repeated-miss generation and analysis."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','smoke','collect','analyze','all'])
    p.add_argument('--root',default=str(ROOT/'outputs/e2_e6_fastgrpo'))
    p.add_argument('--output-dir')
    p.add_argument('--steps',nargs='+',type=int,default=[30,100,200])
    p.add_argument('--checkpoint-template',help='Whole checkpoint path, e.g. /run/step_{step}.pt; never a separate A')
    p.add_argument('--projector-mode',choices=['learned','initialized'],default='learned',
                   help='initialized is an explicit exploratory opt-in for pure FastGRPO checkpoints without A')
    p.add_argument('--opd-rank',type=int,default=8)
    p.add_argument('--opd-topk',type=int,default=16)
    p.add_argument('--opd-fast-lr',type=float,default=.01)
    p.add_argument('--opd-update-stream',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--model',default=os.environ.get('MODEL'))
    p.add_argument('--test-path',default=os.environ.get('TEST_DATASET_PATH'))
    p.add_argument('--dtype',choices=['bf16','fp16'])
    p.add_argument('--attention-implementation',choices=['sdpa','eager'])
    p.add_argument('--wall-minutes',type=float,default=150.)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=2026,help='Analysis bootstrap seed; generation uses the original config eval_seed')
    p.add_argument('--require-b200',action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if any(step not in (30,100,200) for step in a.steps):p.error('This experiment uses steps 30, 100, 200 only')
    if a.wall_minutes<=0 or a.bootstrap_samples<20:p.error('Positive wall budget and >=20 bootstrap samples required')
    from reviewer_controls.offline import output_directory
    destination=output_directory(a.root,a.output_dir,'rq2_repeated_misses_v1')
    if a.mode=='analyze':
        from rq2.analysis import analyze
        result=analyze(destination,a.bootstrap_samples,a.seed)
    else:
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        from rq2.runtime import configuration,prepare,collect
        c,source,source_path=configuration(a.root,{k:getattr(a,k) for k in ('model','test_path','dtype','attention_implementation')})
        destination,rows,tokenizer,manifest=prepare(a.root,a.output_dir,c,source,source_path,a.steps,
            a.checkpoint_template,a.projector_mode,a.opd_rank,a.opd_topk,a.opd_fast_lr,a.opd_update_stream,a.require_b200)
        result=dict(output_dir=str(destination),available_checkpoints=list(manifest['checkpoints']),
                    missing_checkpoints=manifest['missing_checkpoints'],projector_mode=a.projector_mode)
        if a.mode!='check':
            result['collection']=collect(destination,rows,tokenizer,c,manifest,a.projector_mode,
                a.opd_rank,a.opd_topk,a.opd_fast_lr,a.opd_update_stream,a.wall_minutes,a.mode=='smoke')
        if a.mode=='all':
            from rq2.analysis import analyze
            if any((destination/'collection').glob('step_*/on/rollout_*/complete.json')):
                result['analysis']=analyze(destination,a.bootstrap_samples,a.seed)
            else:result['analysis']=dict(status='no_committed_pairs',findings=None)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
