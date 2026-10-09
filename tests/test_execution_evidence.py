"""The evidence checker must reject incomplete or inconsistent journals."""
import json
from copy import deepcopy
from scripts.verify_execution import verify
from motivation.metrics import CONDITIONS


def evidence(tmp_path):
    root=tmp_path/'fixture';a=root/'analysis';a.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    write(a/'manifest.json',dict(config=dict(batch_size=8,responses_per_prompt=8,smoke=False,
        confirmation_steps=[],zero_update_steps=[],replay_distribution_tv=.02,max_length=2048,eval_max_new_tokens=256),
        runtime={'gpu':'NVIDIA GeForce RTX 3090'},
        draft=dict(path='/fixture/existing/draft.pth',sha256='fixture',projector_pretrained=False)))
    rows=[]
    for condition in CONDITIONS:
        for i in range(16):
            on=condition.startswith('reflex_on')
            rows.append(dict(policy_step=1,condition=condition,prompt_id=str(i),sample_index=0,response_count=1,
                accepted_draft_length_sum=2,verification_rounds=1,draft_sha256='R' if condition.startswith('reflex_') else condition,
                learned_A_sha256='A',opd_updates=int(on),opd_final_b_norm=float(on),peak_cuda_allocated_bytes=100))
    (a/'per_response.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (a/'step_metrics.jsonl').write_text(json.dumps(dict(policy_step=1,eval_prompts=16,
        target_parameters_changed=True,old_replay_distribution_tv_max=0.))+'\n')
    write(root/'resume_audit.json',{'state_identity_passed':True})
    write(root/'training_metrics.json',[dict(policy_step=1,target_optimizer_steps=1,zero_drift=False,
        target_gradient_norm=.1,target_old_sha256='old',target_new_sha256='new',
        rollout_counters={name:{'responses':64} for name in ('reflex','shadow')},
        reflex_loss=dict(optimizer_steps=1,projector_gradient_norm=.01),shadow_loss={'optimizer_steps':1},
        peak_training_cuda_allocated_bytes=200)])
    boundary=a/'boundaries/step_1';boundary.mkdir(parents=True);write(boundary/'complete.json',{'policy_step':1})
    return root,rows


def test_evidence_requires_real_updates_and_both_feedback_targets(tmp_path):
    root,rows=evidence(tmp_path)
    report=verify(root)
    assert report['status']=='PASS' and not report['b200_execution_verified']
    assert verify(root,require_b200=True)['status']=='FAIL'
    for r in rows:
        if r['condition']=='reflex_on/new':r.update(opd_updates=0,opd_final_b_norm=0.)
    (root/'analysis/per_response.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=verify(root)
    assert report['status']=='FAIL' and 'No real ON feedback under new target' in report['errors']


def test_evidence_rejects_duplicate_records_and_missing_resume(tmp_path):
    root,rows=evidence(tmp_path);rows.append(deepcopy(rows[0]))
    (root/'analysis/per_response.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'resume_audit.json').unlink()
    report=verify(root)
    assert report['status']=='FAIL'
    assert 'Duplicate response records after resume' in report['errors']
    assert 'No successful checkpoint resume state audit' in report['errors']
