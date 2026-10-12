import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from e2e6.analysis import annotate, ClusterBootstrap
from e2e6.events import VerificationLog
from e2e6.collect import atomic_parquet, EVENT_SCHEMA
from motivation.state import atomic_json, digest, file_hash
from reviewer_controls.statistics import joint_match, SAME_STRATA
from reviewer_controls.offline import (paired_controls,e2_controls,e3_controls,PermutationHistory,
                                       e6_frequency,e6_donor,analyze,output_directory)
from test_e2e6 import event


def test_independent_history_is_exclusive_and_strictly_prior():
    frames={0:annotate(pd.DataFrame([event(1,0,'donor0',rollout=0)])),
            1:annotate(pd.DataFrame([event(0,0,'donor1',rollout=1)])),
            2:annotate(pd.DataFrame([event(0,0,'recipient2',rollout=2),event(1,0,'recipient2',rollout=2)])),
            3:annotate(pd.DataFrame([event(0,0,'recipient3',miss=False,rollout=3),
                                     event(1,0,'recipient3',rollout=3)]))}
    control,pairs,odd=paired_controls(frames)
    assert pairs==[dict(donor=0,recipient=2),dict(donor=1,recipient=3)] and not odd
    first=control[control.rollout_id==2]
    assert not first.cross_prior.any()  # round-1 donor miss cannot explain recipient round 1
    assert first.iloc[-1].token_same
    second=control[control.rollout_id==3]
    assert not second.iloc[0].cross_prior and second.iloc[-1].cross_prior
    assert not second.iloc[-1].token_same
    misses=e6_donor(control,ClusterBootstrap([2,3],100,7))
    assert len({r['recipient_misses'] for r in misses})==1
    assert misses[2]['availability']['numerator']==1
    frames[0]['prompt_id']='recipient2'
    with pytest.raises(ValueError,match='share prompts'):paired_controls(frames)


def matched_frame(cells=('current_response','independent_only','no_prior')):
    base=dict(y=5,round_bin='2–4',position_bin='0–15',tree_depth=0,draft_k=2,active_bin='64',
              exposure_bin=3,same_events_exposure=2,same_y_exposure=1,
              cross_exposure_bin=3,cross_y_exposure_bin=1)
    return pd.DataFrame([dict(base,rollout_id=r,cell=cell,is_miss=bool(i))
                         for r in (0,1,2) for cell in cells for i in (0,1)])


def test_joint_support_weights_and_rollout_bootstrap():
    frame=matched_frame();cells=list(frame.cell.unique());boot=ClusterBootstrap([0,1,2],200,42)
    result=joint_match(frame,cells,boot,SAME_STRATA)
    assert result['support']['matched_events']==18
    assert result['support']['common_strata']==1
    assert all(r['estimate']==.5 and r['ci95']==[.5,.5] for r in result['risks'].values())
    assert result['contrasts']['current_response_minus_independent_only']['risk_difference']['ci95']==[0.,0.]
    # Whole-rollout reweighting: duplicating all events changes neither risk nor CI.
    doubled=joint_match(pd.concat([frame,frame],ignore_index=True),cells,boot,SAME_STRATA)
    assert doubled['risks']==result['risks']
    assert (boot.weights.sum(1)==3).all()
    broken=frame.copy();broken.loc[broken.cell==cells[0],'y']=99
    empty=joint_match(broken,cells,boot,SAME_STRATA)
    assert empty['support']['common_strata']==0
    assert all(r['estimate'] is None for r in empty['risks'].values())
    one=joint_match(frame[frame.rollout_id==0],cells,ClusterBootstrap([0],100,42),SAME_STRATA)
    assert one['risks'][cells[0]]['ci95']==[None,None]


def test_e2_all_conditions_use_same_population_and_distance_bins():
    frame=matched_frame();frame['token_same']=frame.cell=='current_response'
    frame['cross_prior']=frame.cell!='no_prior';frame['distance']='1'
    result=e2_controls(frame,ClusterBootstrap([0,1,2],100,42))
    assert len(result)==6
    assert result[0]['support']['matched_events']==18
    assert result[1]['support']['matched_events']==18
    assert result[2]['support']['common_strata']==0
    assert result[0]['risks']['no_prior']['estimate']==.5


def test_e3_separates_token_context_and_excludes_same_response():
    raw=pd.DataFrame([event(0,0,'a'),event(1,0,'a'),event(1,1,'a',ctx=[3]*8),event(1,2,'b')])
    frame=annotate(raw)
    assert frame.iloc[2].token_source=='sibling_response'
    assert frame.iloc[2].context4_source=='none'
    result=e3_controls(frame,ClusterBootstrap([0],100,42))
    assert {(r['definition'],r['source']) for r in result}=={
        (d,s) for d in ('token','context4') for s in ('sibling_response','different_prompt')}
    # Same-response row never enters the primary token-only populations.
    assert all(r['support']['eligible_events']<len(frame) for r in result if r['definition']=='token')


def test_frequency_null_preserves_token_totals_and_censor_mask():
    raw=pd.DataFrame([event(0,0,'a'),event(0,1,'a'),event(1,0,'a',miss=False),
                      event(2,1,'a',y=7,miss=False),event(3,0,'a',y=7),
                      dict(event(4,0,'a'),is_miss=None,reason='EOS')])
    history=PermutationHistory(raw);before=raw.copy(deep=True)
    for seed in range(20):
        labels=history.permute(np.random.default_rng(seed))
        for token in set(history.tokens):
            idx=history.tokens==token
            assert labels[idx].sum()==history.miss[idx].sum()
    assert history.counts(history.miss)==[0,0,0]  # simultaneous misses cannot supply prior feedback
    pd.testing.assert_frame_equal(raw,before)
    boot=ClusterBootstrap([0,1],100,7)
    a=e6_frequency({0:raw,1:raw.copy()},boot,40,17)
    b=e6_frequency({0:raw,1:raw.copy()},boot,40,17)
    assert a==b
    assert all(r['censored_events']==2 and r['miss_events']==6 for r in a)
    assert all(r['observed']['estimate']==0 for r in a)
    assert any(r['null_mean']>0 for r in a)


def make_collection(root):
    plan=dict(format='e2_e6_shared_verification_v1',batch_size=1,responses_per_prompt=1,
              eval_rollouts=4,prompt_ids=[f'p{i}' for i in range(4)],question_hashes=[f'h{i}' for i in range(4)])
    atomic_json(root/'collection/manifest.json',plan)
    folder=root/'collection/step_000030'
    checkpoint=dict(policy_step=30,sha256='fixture',test_fixture=True)
    identity=digest(dict(plan=plan,checkpoint=checkpoint))
    atomic_json(folder/'manifest.json',dict(identity_sha256=identity,shared_plan_sha256=digest(plan),
        checkpoint=checkpoint))
    for r in range(4):
        log=VerificationLog(30,r,[f'p{r}'],1,99)
        log.prefill(torch.tensor([[1,2,3,4]]),torch.ones(1,4,dtype=torch.long),torch.tensor([[5]]),1)
        for round_index in range(3):
            log.verification(trees=[[]],chosen=[[]],proposals=[{-1:[1]}],sampled=[[5]],paths=[[10]],
                emitted=[[5]],canonical=[0],round_index=round_index,past_kv_len=10,active=1,draft_k=1,budget=0,depth_limit=3)
        directory=folder/f'rollout_{r:04d}'
        atomic_parquet(directory/'events.parquet',log.events,EVENT_SCHEMA)
        atomic_parquet(directory/'rounds.parquet',log.rounds)
        atomic_json(directory/'responses.json',dict(tokens=log.tokens,eos_token_id=99,prompt_ids=[f'p{r}'],
            counters=dict(response_verification_rounds=[3],response_accepted_length_sum=[3])))
        atomic_json(directory/'timing.json',dict(test_fixture=True))
        atomic_json(directory/'complete.json',dict(identity_sha256=identity,checkpoint_step=30,rollout_id=r,
            files={p.name:file_hash(p) for p in directory.iterdir()}))


def test_offline_preserves_originals_and_never_loads_models(tmp_path,monkeypatch):
    root=tmp_path/'experiment';make_collection(root)
    original={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
    import e2e6.runtime
    def forbidden(*args,**kwargs):raise AssertionError('Offline analysis loaded a model')
    monkeypatch.setattr(e2e6.runtime,'build_model',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    result=analyze(root,[30],bootstrap_samples=40,permutations=20,allow_partial=True)
    assert result['checkpoints']==['30']
    assert all(file_hash(p)==sha for p,sha in original.items())
    output=Path(result['output_dir'])
    assert set(p.name for p in output.glob('*.pdf'))=={'E2_step_30.pdf','E3_step_30.pdf','E6_step_30.pdf'}
    assert json.loads((output/'manifest.json').read_text())['zero_model_forwards']
    assert not (root/'analysis_e2_e6').exists()
    with pytest.raises(ValueError,match='separate'):output_directory(root,root/'collection','unused')
    with pytest.raises(FileNotFoundError,match='unavailable'):analyze(root/'missing',[30])
