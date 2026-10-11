from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from e2e6.events import classify,VerificationLog,validate_reconstruction,canonical_ids
from e2e6.analysis import annotate,exclusive_source,ClusterBootstrap,matched_comparison,distance_bin,add_controls,analyze
from e2e6.checkpoints import save,verify,select,latest,FORMAT
from e2e6.config import Config


def test_coverage_pruning_hits_and_invalid_retained_candidates():
    args=dict(eos=99,depth=0,depth_limit=3)
    assert classify(4,[1,2],[1],**args)==('coverage',True)
    assert classify(2,[1,2],[1],**args)==('pruned',True)
    assert classify(1,[1,2],[1],**args)==('hit',False)
    with pytest.raises(ValueError):classify(1,[1],[2],**args)
    with pytest.raises(ValueError):classify(1,[1],[1,1],**args)


def test_eos_depth_beam_cap_are_censored():
    assert classify(99,[1],[1],eos=99,depth=0,depth_limit=3)==('EOS',None)
    assert classify(4,None,[],eos=99,depth=3,depth_limit=3)==('depth_cap',None)
    assert classify(4,None,[],eos=99,depth=1,depth_limit=3)==('beam_cap',None)
    assert classify(4,[1],[],eos=99,depth=1,depth_limit=3,at_cap=True)==('length_cap',None)
    with pytest.raises(ValueError):classify(4,None,[1],eos=99,depth=1,depth_limit=3)


def test_visited_tree_path_labels_and_response_reconstruction():
    log=VerificationLog(30,0,['p'],1,99)
    log.prefill(torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.long),torch.tensor([[5]]),1)
    node=SimpleNamespace(input_id=7,depth=0,child=[],target_index=11)
    log.verification(trees=[[node]],chosen=[[0]],proposals=[{-1:[7,6],0:[9,6]}],
        sampled=[[7,8]],paths=[[10,11]],emitted=[[7,8]],canonical=[0],round_index=0,
        past_kv_len=10,active=1,draft_k=2,budget=1,depth_limit=3)
    out=dict(generated_token_ids=[[5,7,8]],response_verification_rounds=[1],response_accepted_length_sum=[2])
    assert log.validate(out)
    assert [e['reason'] for e in log.events]==['prefill','hit','coverage']
    log.events[-1]['target_token']=6
    with pytest.raises(ValueError):validate_reconstruction(log.events,out['generated_token_ids'],99)


def test_swap_removal_keeps_canonical_prompt_response_mapping():
    assert canonical_ids(['a','b'],2)==[('a',0),('a',1),('b',0),('b',1)]
    log=VerificationLog(30,0,['a','b'],2,99)
    log.prefill(torch.tensor([[1],[2]]),torch.ones(2,1,dtype=torch.long),torch.tensor([[99],[7]]),2)
    # Rows 0/1 terminate; active row zero now refers to canonical response 2.
    log.verification(trees=[[],[]],chosen=[[],[]],proposals=[{-1:[1]},{-1:[1]}],sampled=[[8],[9]],
        paths=[[3],[3]],emitted=[[8],[9]],canonical=[2,3],round_index=0,past_kv_len=3,
        active=2,draft_k=1,budget=0,depth_limit=3)
    assert log.tokens==[[99],[99],[7,8],[7,9]]
    assert [e['prompt_id'] for e in log.events[-2:]]==['b','b']
    assert log.validate(dict(generated_token_ids=log.tokens,response_verification_rounds=[0,0,1,1],
                             response_accepted_length_sum=[0,0,1,1]))


def event(r,response,prompt,y=5,miss=True,rollout=0,ctx=None):
    ctx=ctx or list(range(8))
    return dict(rollout_id=rollout,round_index=r,response_id=response,prompt_id=prompt,
        tree_depth=0,draft_k=2,token_position=10+r,reason='coverage' if miss else 'hit',
        is_miss=miss,target_token=y,draft_top1=1,active_batch_size=64,
        context2=ctx[-2:],context4=ctx[-4:],context8=ctx,full_context_sha256='full:'+str(ctx))


def test_feedback_strictly_earlier_and_exclusive_sources():
    raw=pd.DataFrame([event(0,0,'a'),event(0,1,'a'),event(1,1,'a'),event(1,2,'b'),event(2,2,'b')])
    rows=annotate(raw)
    assert list(rows['token_source'])==['none','none','same_response','different_prompt','same_response']
    assert rows.iloc[0]['context8_source']=='none'
    assert rows.iloc[2]['context8_source']=='same_response'
    assert exclusive_source({0:'a'},1,'a')=='sibling_response'
    assert exclusive_source({0:'a',1:'b'},1,'b')=='same_response'
    assert exclusive_source({0:'a'},3,'c')=='different_prompt'
    assert exclusive_source({},3,'c')=='none'
    sibling=annotate(pd.DataFrame([event(0,0,'a'),event(1,1,'a')]))
    assert sibling.iloc[-1]['token_source']=='sibling_response'


def test_round_distance_buckets():
    assert [distance_bin(d) for d in (None,1,2,4,5,16,17,64,65)]==['none','1','2–4','2–4','5–16','5–16','17–64','17–64','>64']


def test_exposure_opportunities_count_hits_and_exclude_same_round():
    raw=pd.DataFrame([event(0,0,'a',miss=False),event(0,1,'a'),event(0,2,'b',y=9),
                      event(1,0,'a'),event(1,2,'b')])
    rows=annotate(raw)
    assert not rows.iloc[:3]['batch_events_exposure'].any()
    a,b=rows.iloc[3],rows.iloc[4]
    # Prior hits are opportunities, even though they cannot supply miss feedback.
    assert a['same_y_exposure']==1 and not a['token_same']
    assert a['sibling_y_exposure']==1 and a['token_source']=='sibling_response'
    assert a['different_y_exposure']==0
    assert b['same_y_exposure']==0 and b['different_y_exposure']==2
    assert b['token_source']=='different_prompt'
    assert a['batch_events_exposure']==2 and b['batch_events_exposure']==2


def test_one_rollout_reports_point_estimate_without_fabricated_ci():
    frame=pd.DataFrame(dict(rollout_id=[0,0],value=[1.,0.]))
    stat=ClusterBootstrap([0],2000,42).ratio(frame,frame['value'],np.ones(2))
    assert stat['estimate']==.5 and stat['ci95']==[None,None]
    assert stat['ci_status']=='insufficient_rollout_clusters'


def test_bootstrap_whole_rollout_clusters_not_individual_tokens():
    frame=pd.DataFrame(dict(rollout_id=[0,0,1],value=[1.,0.,1.]))
    boot=ClusterBootstrap([0,1],2000,42)
    stat=boot.ratio(frame,frame['value'],np.ones(3))
    assert stat['estimate']==pytest.approx(2/3)
    nums=boot.weights@np.array([1.,1.]);dens=boot.weights@np.array([2.,1.])
    assert stat['ci95']==pytest.approx(np.quantile(nums/dens,[.025,.975]))
    empty=boot.ratio(frame,np.zeros(3),np.zeros(3))
    assert empty['estimate'] is None and empty['ci95']==[None,None]


def test_matched_lifts_condition_on_actual_target_token_and_exposure():
    base=dict(y=5,round_bin='2–4',position_bin='0–15',tree_depth=0,draft_k=2,active_bin='64',exposure_bin=3)
    frame=pd.DataFrame([dict(base,rollout_id=r,is_miss=miss,flag=exposed) for r in (0,1)
        for exposed,miss in [(True,True),(True,False),(False,True),(False,False)]])
    stat=matched_comparison(frame,'flag',ClusterBootstrap([0,1],2000,42))
    assert stat['lift']==pytest.approx(1.) and stat['lift_ci95']==pytest.approx([1,1])
    frame.loc[frame['flag'],'y']=999
    assert matched_comparison(frame,'flag',ClusterBootstrap([0,1]))['lift'] is None


def test_cross_controls_disjoint_rollout_halves_and_no_future_feedback():
    frames={i:annotate(pd.DataFrame([event(0,0,f'p{i}',rollout=i),event(1,0,f'p{i}',rollout=i)])) for i in range(4)}
    control,pairs=add_controls(frames)
    assert pairs==2 and set(control['rollout_id'])=={2,3}
    assert not control[control['round_index']==0]['cross_prior'].any()
    assert control[control['round_index']==1]['cross_prior'].all()


def state(step,attempt):
    return dict(format=FORMAT,policy_step=step,attempt=attempt,target_lora={'x':torch.tensor([step])},
                draft_weights={'x':torch.tensor([attempt])})


def test_checkpoint_integrity_selection_latest_and_milestone_survival(tmp_path):
    save(tmp_path,state(30,30),milestone=True)
    milestone_hash=verify(tmp_path/'step_000030.pt',load=False)['sha256']
    for n in (31,40,50):save(tmp_path,state(n,n))
    selected,missing=select(tmp_path,[30,100,200])
    assert selected==[tmp_path/'step_000030.pt'] and missing==[100,200]
    assert verify(latest(tmp_path))['policy_step']==50
    assert verify(tmp_path/'step_000030.pt',load=False)['sha256']==milestone_hash
    assert len(list(tmp_path.glob('resume_step*.pt')))==2
    # Re-saving at the same milestone preserves the first checkpoint.
    save(tmp_path,state(30,999),milestone=True)
    assert verify(tmp_path/'step_000030.pt',load=False)['sha256']==milestone_hash
    with (tmp_path/'step_000030.pt').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='integrity'):verify(tmp_path/'step_000030.pt')


def test_no_validation_draft_fallback_and_outputs_confined_to_analysis_spec():
    c=Config(draft_checkpoint='/production/pretrained/latest_checkpoint');c.validate()
    with pytest.raises(ValueError,match='validation-only'):
        Config(draft_checkpoint='/x/validation/draft.pth').validate()
    from run_e2_e6 import parser,config_for
    with pytest.raises(ValueError,match='inside analysis_spec'):
        config_for(parser().parse_args(['train','--output-dir','/tmp/not-authorized']))


def test_offline_empty_partial_analysis_uses_no_gpu(tmp_path):
    result=analyze(tmp_path,[30],100,42)
    assert result['checkpoints']==[]
    assert (tmp_path/'analysis_e2_e6/recurrence.pdf').read_bytes().startswith(b'%PDF')
    assert 'No valid committed rollouts' in (tmp_path/'analysis_e2_e6/REPORT.md').read_text()


def test_crash_after_latest_commit_recovers_milestone_without_replaying_step(tmp_path,monkeypatch):
    import e2e6.checkpoints as cp
    native=cp._link
    def crash(source,target):
        if target.name=='step_000030.pt':raise OSError('simulated crash after durable commit')
        return native(source,target)
    monkeypatch.setattr(cp,'_link',crash)
    with pytest.raises(OSError):save(tmp_path,state(30,30),milestone=True)
    committed=latest(tmp_path)
    assert verify(committed)['policy_step']==30
    assert not (tmp_path/'step_000030.pt').exists()
    monkeypatch.setattr(cp,'_link',native)
    cp.repair_milestone(tmp_path,committed)
    assert select(tmp_path,[30])[0]==[tmp_path/'step_000030.pt']
