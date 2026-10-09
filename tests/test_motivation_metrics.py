import json
import random
import numpy as np
import pytest
from motivation.metrics import CONDITIONS,aal,summarize,rebuild_exports
from motivation.state import atomic_json,digest,isolated_rng


def records(n=16,step=20):
    return [dict(policy_step=step,condition=c,prompt_id=str(i),sample_index=0,
                 accepted_draft_length_sum=(i+1)*(j+1),verification_rounds=i+1)
            for j,c in enumerate(CONDITIONS) for i in range(n)]


def test_six_conditions_ratio_of_sums_and_paired_interaction():
    before=digest(np.random.get_state());rows=records();out=summarize(rows,20,16,samples=200)
    assert out['eval_trajectories']==96;assert out['A1_delta_lag']==1
    assert out['A2_gain_old']==out['A2_gain_new']==1;assert out['A2_drift_interaction']==0
    assert out['A2_drift_interaction_ci_low']==out['A2_drift_interaction_ci_high']==0
    assert before==digest(np.random.get_state())
    assert summarize(records(64),20,64,samples=1)['eval_trajectories']==384
    assert aal([dict(accepted_draft_length_sum=10,verification_rounds=1),
                dict(accepted_draft_length_sum=10,verification_rounds=10)])==20/11
    with pytest.raises(ValueError):aal([dict(accepted_draft_length_sum=1,verification_rounds=0)])
    with pytest.raises(ValueError):summarize(rows[:-1],20,16)
    rows[-1]['prompt_id']='0'
    with pytest.raises(ValueError):summarize(rows,20,16)


def test_atomic_journal_resume_dedup(tmp_path):
    rows=records();metric=summarize(rows,20,16,samples=1)
    atomic_json(tmp_path/'boundaries/step_20/results.json',dict(per_response=rows,metrics=metric))
    rebuild_exports(tmp_path,[]);assert not (tmp_path/'per_response.jsonl').read_text()
    rebuild_exports(tmp_path,[20]);rebuild_exports(tmp_path,[20])
    assert len((tmp_path/'per_response.jsonl').read_text().splitlines())==96
    assert len((tmp_path/'step_metrics.jsonl').read_text().splitlines())==1


def test_isolated_rng_restores_all_sources_on_exception():
    import torch
    from helper.checkpointing import capture_rng_state
    before=digest(capture_rng_state())
    with pytest.raises(RuntimeError):
        with isolated_rng(42):
            random.random();np.random.rand();torch.rand(3);raise RuntimeError('intentional')
    assert digest(capture_rng_state())==before
