#!/usr/bin/env python3
"""Startup-only exact profile lookup / optional tuning. Stdout is ONLY its path."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from helper.opd_profiles import inspect_draft,execution_key,fingerprint,discover_profile,profile_filename,context_shapes,active_trials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target-config','draft-checkpoint','profile-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--profile',default='');p.add_argument('--rank',type=int,required=True)
    p.add_argument('--dtype',required=True);p.add_argument('--topk',type=int,default=16)
    p.add_argument('--auto-tune',choices=['0','1'],default='0')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--responses',type=int,default=8)
    p.add_argument('--max-draft-k',type=int,default=8);p.add_argument('--iterations',type=int,default=30)
    a=p.parse_args()
    detected=inspect_draft(a.target_config,a.draft_checkpoint,rank=a.rank,dtype=a.dtype,topk=a.topk)
    key=execution_key(fingerprint(),detected['vocab'],detected['rank'],detected['dtype'],detected['topk'])
    path,payload=discover_profile(a.profile_dir,key,a.profile)
    if path is None and a.auto_tune=='1':
        from scripts.tune_opd_proposals import benchmark_configuration
        with contextlib.redirect_stdout(sys.stderr):
            payload=benchmark_configuration(key,context_shapes(a.batch_size,a.responses,a.max_draft_k),active_trials(key['vocab'],key['topk']),a.iterations)
        path=Path(a.profile_dir)/profile_filename(key);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(path)
    print('OPD proposal mode: auto\nprofile: '+str(path or 'NONE: UNCALIBRATED safe fallback; run bash scripts/tune_opd_proposals.sh')+'\n'+json.dumps(key,indent=2),file=sys.stderr,flush=True)
    if path is not None:print(path.resolve())


if __name__=='__main__':main()
