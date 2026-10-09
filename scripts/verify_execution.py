#!/usr/bin/env python3
"""Verify durable evidence from an actual one-boundary 8x8 A1+A2 validation."""
from pathlib import Path
import argparse
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motivation.metrics import CONDITIONS,aal
from motivation.state import atomic_json


def verify(root,require_b200=False):
    root=Path(root);a=root/'analysis'
    manifest=json.loads((a/'manifest.json').read_text())
    config=manifest['config']
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    check((config['batch_size'],config['responses_per_prompt'])==(8,8),'Training must be 8x8 for both learners')
    check(not config['smoke'],'A reduced smoke is not production integration validation')
    if require_b200:
        check('B200' in manifest['runtime']['gpu'],'No B200 execution evidence')
    metrics=[json.loads(s) for s in (a/'step_metrics.jsonl').read_text().splitlines()]
    responses=[json.loads(s) for s in (a/'per_response.jsonl').read_text().splitlines()]
    training=json.loads((root/'training_metrics.json').read_text())
    check(bool(metrics),'No completed measurement boundary')
    resume=root/'resume_audit.json'
    check(resume.is_file() and json.loads(resume.read_text()).get('state_identity_passed',False),
          'No successful checkpoint resume state audit')
    keys=[(r['policy_step'],r['condition'],r['prompt_id'],r['sample_index']) for r in responses]
    check(len(keys)==len(set(keys)),'Duplicate response records after resume')
    real=[r for r in training if r['target_optimizer_steps']==1 and not r['zero_drift']]
    check(bool(real),'No genuine target parameter update')
    for r in real:
        check(math.isfinite(r['target_gradient_norm']) and r['target_gradient_norm']>0,'Missing finite nonzero GRPO gradients')
        check(r['target_old_sha256']!=r['target_new_sha256'],'Target parameters did not change')
        for learner in ('reflex','shadow'):
            check(r['rollout_counters'][learner]['responses']==64,f'{learner} did not generate 64 training responses')
            check(r[learner+'_loss']['optimizer_steps']==1,f'{learner} draft optimizer did not update')
        check(r['reflex_loss']['projector_gradient_norm'] is not None and r['reflex_loss']['projector_gradient_norm']>0,'No learned A gradient')
    for m in metrics:
        step=m['policy_step'];rows=[r for r in responses if r['policy_step']==step]
        n=64 if step in config['confirmation_steps'] else 16
        check(m['eval_prompts']==n and len(rows)==6*n,'Incorrect six-condition 16/64 evaluation budget')
        check(set(r['condition'] for r in rows)==set(CONDITIONS),'Missing condition')
        check(m['target_parameters_changed'],'Measurement has no target parameter update')
        check(any(r['policy_step']==step for r in real),'Measurement is not tied to a genuine target update')
        check(m['old_replay_distribution_tv_max']<=config['replay_distribution_tv'],'Teacher distribution alignment failed')
        for condition in CONDITIONS:
            group=[r for r in rows if r['condition']==condition]
            check(len({r['prompt_id'] for r in group})==n,'Duplicate/missing held-out prompts')
            try:aal(group)
            except ValueError as error:errors.append(str(error))
            check(all(r['response_count']==1 for r in group),'Multiple evaluation responses per prompt')
        four=[r for r in rows if r['condition'].startswith('reflex_')]
        check(len({r['draft_sha256'] for r in four})==1,'A2 backbone changed')
        check(len({r['learned_A_sha256'] for r in four})==1,'A2 learned A changed')
        check(all(r['opd_updates']==0 and r['opd_final_b_norm']==0 for r in four if r['condition'].startswith('reflex_off')),'OFF adapted B')
        for target in ('old','new'):
            check(any(r['opd_updates']>0 and r['opd_final_b_norm']>0 for r in four
                      if r['condition']=='reflex_on/'+target),f'No real ON feedback under {target} target')
        if step in config['zero_update_steps']:
            control=json.loads((a/f'boundaries/step_{step}/zero_drift_control.json').read_text())
            check(control['passed'] and control['target_optimizer_steps']==0,'Isolated zero-drift control failed')
        check((a/f'boundaries/step_{step}/complete.json').exists(),'Boundary not durably committed')
    report=dict(status='PASS' if not errors else 'FAIL',errors=errors,device=manifest['runtime']['gpu'],
                draft_checkpoint=manifest['draft']['path'],draft_sha256=manifest['draft']['sha256'],
                projector_pretrained=manifest['draft']['projector_pretrained'],
                train_prompts=config['batch_size'],responses_per_prompt=config['responses_per_prompt'],
                train_max_length=config['max_length'],eval_max_new_tokens=config['eval_max_new_tokens'],
                real_target_updates=len(real),measurement_boundaries=len(metrics),evaluation_responses=len(responses),
                training_peak_cuda_allocated_bytes=max((r['peak_training_cuda_allocated_bytes'] for r in training),default=0),
                evaluation_peak_cuda_allocated_bytes=max((r['peak_cuda_allocated_bytes'] for r in responses),default=0),
                b200_execution_verified=bool(require_b200 and not errors))
    atomic_json(root/'execution_validation.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--require-b200',action='store_true')
    args=p.parse_args();report=verify(args.output_dir,args.require_b200);print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
