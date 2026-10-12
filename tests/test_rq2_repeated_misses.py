from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from e2e6.analysis import ClusterBootstrap
from e2e6.checkpoints import FORMAT
from helper.tree_verification import PackedTree, VerifiedPath
from motivation.state import digest
from rq2.checkpoints import decode
from rq2.events import PackedVerificationLog
from rq2.analysis import comparison
from test_reviewer_controls import matched_frame


def test_checkpoint_requires_same_trajectory_learned_A_or_explicit_exploratory():
    pure=dict(format=FORMAT,policy_step=30,draft_weights={'weight':torch.ones(4,4)},target_lora={'lora':torch.ones(4)})
    with pytest.raises(ValueError,match='No learned A'):decode(pure,30,'learned')
    decoded=decode(pure,30,'initialized')
    assert decoded['projector'] is None and 'EXPLORATORY' in decoded['provenance']['interpretation']
    with pytest.raises(ValueError,match='counter'):decode(pure,100,'initialized')
    a=torch.ones(4,2)
    trained=dict(format='opd_fastgrpo_checkpoint_v5',step=30,method='opd_reflex',target_lora=pure['target_lora'],
        draft_model={**pure['draft_weights'],'opd_projector':a},opd_projector=a,
        opd_projector_initial=torch.zeros_like(a))
    learned=decode(trained,30,'learned')
    assert learned['provenance']['A_sha256']==digest(a)
    assert 'opd_projector' not in learned['draft']
    with pytest.raises(ValueError,match='discarding'):decode(trained,30,'initialized')
    corrupted=dict(trained,opd_projector=torch.zeros_like(a))
    with pytest.raises(ValueError,match='Conflicting'):decode(corrupted,30,'learned')
    del trained['opd_projector_initial']
    with pytest.raises(ValueError,match='provenance'):decode(trained,30,'learned')
    trained['optimizer_draft']={'state':{0:dict(step=torch.tensor(3),exp_avg=torch.zeros_like(a))}}
    assert decode(trained,30,'learned')['provenance']['mode']=='learned'
    lag=dict(format='simplelr_opd_policy_lag_a1a2_v2',policy_step=30,training_trajectory='OPD-driven',
        target_lora=pure['target_lora'],reflex=dict(weights=trained['draft_model'],optimizer=trained['optimizer_draft']))
    assert decode(lag,30,'learned')['provenance']['A_sha256']==digest(a)


def test_configuration_reuses_actual_collection_settings_not_training_defaults(tmp_path):
    from e2e6.config import Config,generation_kwargs
    from motivation.state import atomic_json
    from rq2.runtime import configuration
    original=Config();actual=Config(max_length=1024,temperature=.8,eval_seed=77)
    atomic_json(tmp_path/'train_manifest.json',dict(config=original.to_dict()))
    atomic_json(tmp_path/'collection/manifest.json',dict(format='e2_e6_shared_verification_v1',
        generation=generation_kwargs(actual),batch_size=8,responses_per_prompt=8,seed=77,eval_rollouts=20))
    loaded,_,_=configuration(tmp_path,{})
    assert loaded.max_length==1024 and loaded.temperature==.8 and loaded.eval_seed==77


@pytest.mark.parametrize('ending',['coverage','pruned','EOS','depth_cap','beam_cap','length_cap'])
def test_packed_log_exact_alignment_pruning_and_censoring(ending):
    log=PackedVerificationLog(30,0,['p'],1,99)
    log.prefill(torch.tensor([[1,2,3,4]]),torch.ones(1,4,dtype=torch.long),torch.tensor([[5]]),1)
    # Only ONE root child survived pruning despite k=2. Its child's packed
    # index 2 must NOT be mistaken for another root child by VerificationLog.
    last=99 if ending=='EOS' else (4 if ending=='pruned' else 5)
    feedback= -1 if ending in ('depth_cap','beam_cap') else 2
    tree=PackedTree(torch.tensor([[-1,0,1,2]]),torch.tensor([[-1,7,8,9]]),torch.tensor([[0,1,feedback,-1]]),3)
    if ending in ('depth_cap','beam_cap'):
        tree=PackedTree(tree.parents[:,:3],tree.tokens[:,:3],tree.feedback_contexts[:,:3],tree.max_depth)
    path=VerifiedPath(torch.tensor([[7,8,last,-1]]),torch.tensor([[0,1,2,-1]]),
                      torch.tensor([[0,1,feedback,-1]]),torch.tensor([3]))
    proposals=[(0,torch.tensor([[[7,6]]])),(1,torch.tensor([[[8,5],[9,4]]]))]
    log.packed_verification(tree=tree,proposals=proposals,sampled=torch.tensor([[7,8,last,0]]),path=path,
        canonical=[0],round_index=0,past_kv_len=10,active=1,draft_k=2,budget=3,
        depth_limit=2 if ending=='depth_cap' else 3,capped=ending=='length_cap')
    assert [e['reason'] for e in log.events]==['prefill','hit','hit',ending]
    assert log.events[1]['retained_children']==[7]
    assert log.validate(dict(generated_token_ids=[[5,7,8,last]],response_verification_rounds=[1],response_accepted_length_sum=[3]))


def test_packed_observer_rejects_token_or_context_misalignment():
    log=PackedVerificationLog(30,0,['p'],1,99)
    log.prefill(torch.tensor([[1]]),torch.ones(1,1,dtype=torch.long),torch.tensor([[5]]),1)
    tree=PackedTree(torch.tensor([[-1]]),torch.tensor([[-1]]),torch.tensor([[0]]),1)
    path=VerifiedPath(torch.tensor([[8]]),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor([1]))
    args=dict(tree=tree,proposals=[(0,torch.tensor([[[1]]]))],sampled=torch.tensor([[9]]),path=path,
              canonical=[0],round_index=0,past_kv_len=10,active=1,draft_k=1,budget=0,depth_limit=1)
    with pytest.raises(ValueError,match='alignment'):log.packed_verification(**args)
    args.update(proposals=[],sampled=torch.tensor([[8]]))
    with pytest.raises(ValueError,match='missing'):log.packed_verification(**args)


def test_rq2_four_cell_common_support_and_paired_bootstrap():
    frame=matched_frame(('prior','no_prior'))
    frame['token_same']=frame.cell=='prior';frame['distance']='1'
    off=frame.copy();on=frame.copy()
    on.loc[on.token_same,'is_miss']=False
    boot=ClusterBootstrap([0,1,2],200,42)
    rows=comparison(off,on,boot)
    row=rows[1]
    assert row['matched']['support']['common_strata']==1
    assert row['recurrence_absolute_change']['estimate']==-.5
    assert row['recurrence_absolute_change']['ci95']==[-.5,-.5]
    assert row['recurrence_relative_change']['estimate']==-1
    assert len(rows)==6 and all(r['distance'] in ('all','1','2–4','5–16','17–64','>64') for r in rows)
    on['y']=99
    assert comparison(off,on,boot)[0]['recurrence_absolute_change']['estimate'] is None


def test_paired_clusters_cancel_shared_variation_despite_different_event_counts():
    frame=matched_frame(('prior','no_prior'))
    frame['token_same']=frame.cell=='prior';frame['distance']='1'
    frame.loc[(frame.rollout_id==0)&frame.token_same,'is_miss']=False
    frame.loc[(frame.rollout_id==1)&frame.token_same,'is_miss']=True
    # Different event composition/size, identical within-rollout risks. Pairing
    # whole rollouts must cancel their shared variability in the ON–OFF contrast.
    on=pd.concat([frame,frame],ignore_index=True)
    row=comparison(frame,on,ClusterBootstrap([0,1,2],500,17))[0]
    assert row['recurrence_absolute_change']['ci95']==[0.,0.]
    off_risk=row['matched']['risks']['off_prior']
    assert off_risk['ci95'][0]<off_risk['estimate']<off_risk['ci95'][1]


def test_cpu_real_feedback_off_zero_on_nonzero_frozen_A_and_reset():
    from test_opd_reflex import state
    for lr in (0.,.05):
        runtime,model,mapping=state('cpu',v=13,h=32,lr=lr,topk=4)
        before=digest(dict(head=model.lm_head.state_dict(),A=model.opd_projector))
        hidden=torch.ones(3,1,32);logits=model.lm_head(hidden)
        runtime.propose(logits,hidden,2,mapping,root=True)
        tree=PackedTree(torch.full((3,1),-1,dtype=torch.long),torch.full((3,1),-1,dtype=torch.long),torch.zeros(3,1,dtype=torch.long),1)
        path=SimpleNamespace(packed_indices=torch.zeros(3,1,dtype=torch.long))
        teacher=torch.zeros(3,1,13);teacher[...,12]=1.
        runtime.feedback(tree,path,teacher)
        assert bool(runtime.B_fast.any())==(lr>0)
        assert digest(dict(head=model.lm_head.state_dict(),A=model.opd_projector))==before
        runtime.clear();assert not runtime.B_fast.any()
        runtime.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
        assert not runtime.B_fast.any() and not runtime._ever_updated


def test_saved_paired_analysis_integrity_and_figures(tmp_path):
    import json
    import shutil
    from test_reviewer_controls import make_collection
    from motivation.state import atomic_json,file_hash
    from rq2.analysis import analyze
    baseline=tmp_path/'baseline';make_collection(baseline)
    original={str(p):file_hash(p) for p in baseline.rglob('*') if p.is_file()}
    output=tmp_path/'rq2';identity='paired-fixture'
    checkpoint=dict(projector=dict(mode='initialized',interpretation='EXPLORATORY tiny fixture'))
    atomic_json(output/'manifest.json',dict(identity_sha256=identity,checkpoints={'30':checkpoint}))
    atomic_json(output/'projectors/step_000030.json',dict(actual_A_sha256='fixture-A'))
    for arm in ('off','on'):
        for rollout in range(4):
            folder=output/'collection/step_000030'/arm/f'rollout_{rollout:04d}'
            shutil.copytree(baseline/'collection/step_000030'/f'rollout_{rollout:04d}',folder)
            response=json.loads((folder/'responses.json').read_text())
            response.update(seed=rollout,initial_weights_sha256='same-initial-weights',projector_sha256='fixture-A')
            atomic_json(folder/'responses.json',response)
            commit=json.loads((folder/'complete.json').read_text())
            commit.update(identity_sha256=identity,arm=arm,checkpoint_identity=checkpoint,projector_sha256='fixture-A')
            commit['files']['responses.json']=file_hash(folder/'responses.json')
            atomic_json(folder/'complete.json',commit)
    result=analyze(output,100,42)
    assert result['checkpoints']==['30']
    summary=json.loads((output/'rq2_repeated_misses.json').read_text())
    assert summary['checkpoints']['30']['counts']['off']['paired_rollouts']==4
    assert summary['checkpoints']['30']['overall']['recurrence_absolute_change']['estimate'] is None
    assert (output/'rq2_repeated_misses.pdf').read_bytes().startswith(b'%PDF')
    assert (output/'rq2_repeated_misses.png').is_file()
    assert all(file_hash(p)==sha for p,sha in original.items())
    broken=output/'collection/step_000030/on/rollout_0000/complete.json'
    commit=json.loads(broken.read_text());commit['projector_sha256']='other-A';atomic_json(broken,commit)
    with pytest.raises(ValueError,match='Projector integrity'):analyze(output,100,42)
