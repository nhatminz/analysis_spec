#!/usr/bin/env python3
"""Frozen E2–E6 collection from a paired SpecNaacl target/draft export."""
import json
import os
from pathlib import Path
import time
import traceback
from run_e2_e6 import ROOT,parser as baseline_parser,config_for


def parser():
    p=baseline_parser()
    p.description=__doc__
    p.add_argument('--source-run',help='SpecNaacl run containing checkpoints/target and checkpoints/draft')
    p.add_argument('--source-step',type=int,default=235,help='Exported checkpoint label, not necessarily target optimizer steps')
    return p


def configuration(args):
    if args.source_step<1:raise ValueError('External checkpoint label must be positive')
    if args.mode not in ('collect','analyze','check'):raise ValueError('External launcher supports collect/analyze/check only; no training')
    # Deliberately ignore a training OUTPUT_DIR inherited from the user's shell.
    if args.output_dir is None:args.output_dir=str(ROOT/'outputs'/f'e2_e6_external_step{args.source_step}')
    c=config_for(args)
    saved=Path(c.output_dir)/'external_checkpoint_manifest.json'
    if saved.exists():
        from e2e6.config import Config
        previous=Config(**json.loads(saved.read_text())['config'])
        for key,value in vars(args).items():
            if key!='output_dir' and hasattr(previous,key) and value is not None:setattr(previous,key,value)
        previous.output_dir=c.output_dir
        c=previous
    return c


def main(argv=None):
    args=parser().parse_args(argv);c=configuration(args);root=Path(c.output_dir)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',METHOD='fastgrpo')
    if args.mode=='collect':
        if not args.source_run:raise ValueError('--source-run is required for collection')
        from e2e6.external import source_info,prepare_external
        info=source_info(args.source_run,args.source_step)
        for key,convert in (('temperature',float),('top_p',float),('max_length',int),
                            ('max_prompt_length',int),('batch_size',int),('responses_per_prompt',int)):
            if getattr(args,key) is None and info['metadata'].get(key) is not None:
                setattr(c,key,convert(info['metadata'][key]))
        tests,tokenizer,manifest=prepare_external(c,args.source_run,args.source_step)
        from e2e6.collect import collect
        result=collect(c,tests,tokenizer,manifest,[args.source_step])
    else:
        manifest_path=root/'external_checkpoint_manifest.json'
        if not manifest_path.exists():raise ValueError('No external evaluation manifest; collect first')
        manifest=json.loads(manifest_path.read_text())
        if manifest['source']['checkpoint_label']!=args.source_step:raise ValueError('Output belongs to a different source checkpoint label')
        if args.mode=='analyze':
            from e2e6.analysis import analyze
            result=analyze(root,[args.source_step],c.bootstrap_samples,c.eval_seed)
            from motivation.state import atomic_bytes,atomic_json
            destination=Path(result['output_dir']);report=destination/'REPORT.md'
            note=('\nExternal checkpoint labels are source export labels, not counts of target optimizer updates. '
                  f'Source label: {args.source_step}; source-reported target optimizer steps: '
                  f'{manifest["source"]["actual_target_optimizer_steps"]}.\n')
            text=report.read_text()
            if note not in text:atomic_bytes(report,(text+note).encode())
            # The report is part of the offline output commit; update its hash.
            from motivation.state import file_hash
            complete=destination/'complete.json';commit=json.loads(complete.read_text())
            commit['files']['REPORT.md']=file_hash(report);atomic_json(complete,commit)
        else:
            from e2e6.checkpoints import select,verify
            from e2e6.collect import validate_rollout
            paths,missing=select(root/'checkpoints',[args.source_step])
            commits=list((root/'collection').glob('step_*/rollout_*/complete.json'))
            for marker in commits:validate_rollout(marker.parent)
            result=dict(checkpoints=[dict(path=str(p),**verify(p,load=False)) for p in paths],
                missing=missing,verified_rollouts=len(commits),analysis_only=True)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':
    started=time.perf_counter()
    try:raise SystemExit(main())
    except Exception as error:
        try:
            args=parser().parse_args();c=configuration(args)
            from e2e6.train import append_log
            append_log(Path(c.output_dir)/'logs/failures.jsonl',dict(mode='external_'+args.mode,
                error=repr(error),traceback=traceback.format_exc(),runtime_s=time.perf_counter()-started))
        except Exception:pass
        raise
