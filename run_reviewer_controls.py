#!/usr/bin/env python3
"""Task 1: validate and analyze existing baseline logs. Never generate data."""
import argparse
import json
from pathlib import Path
from reviewer_controls.offline import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent/'outputs/e2_e6_fastgrpo'))
    p.add_argument('--output-dir')
    p.add_argument('--steps',nargs='+',type=int,default=[30,100,200])
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--permutations',type=int,default=500)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--allow-partial',action='store_true',help='Explicitly analyze fewer committed rollouts; report support')
    a=p.parse_args()
    if a.bootstrap_samples<20:p.error('Use at least 20 bootstrap resamples')
    print(json.dumps(analyze(a.root,a.steps,a.output_dir,a.bootstrap_samples,a.permutations,a.seed,a.allow_partial),indent=2))


if __name__=='__main__':main()
