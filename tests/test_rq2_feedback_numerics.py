"""Peaked real draft feedback: diagnostic KL may diverge while q-p is valid."""
import json
from types import SimpleNamespace
import pytest
import torch
from helper.tree_verification import PackedTree
from rq2.runtime import feedback_diagnostics
from test_opd_reflex import state

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def peaked_feedback(device,peak,lr,compact=False,corrupt=None):
    runtime,model,mapping=state(device,v=37,h=32,lr=lr,topk=16)
    model.lm_head.bfloat16()
    runtime.diagnostics=True
    runtime.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
    model.lm_head.weight.data.zero_();model.lm_head.weight.data[0,0]=peak
    hidden=torch.zeros(3,1,32,device=device,dtype=torch.bfloat16);hidden[...,0]=1.
    runtime.propose(model.lm_head(hidden),hidden,8,mapping,root=True)
    if corrupt=='q_nan':runtime.q_cache[:3,:1,0]=float('nan')
    elif corrupt=='q_negative':runtime.q_cache[:3,:1,0]=-1.
    elif corrupt=='u_nan':runtime.u_cache[:3,:1,0]=float('nan')
    root=torch.full((3,1),-1,device=device,dtype=torch.long)
    tree=PackedTree(root,root.clone(),torch.zeros_like(root),1)
    path=SimpleNamespace(packed_indices=torch.zeros_like(root))
    target=torch.full((3,1,37),1./37,device=device)
    if corrupt=='teacher_mass':target[0,0].zero_()
    metadata=runtime.prepare_compact_teacher(tree,path,target) if compact else None
    runtime.feedback(tree,path,target,sampling_metadata=metadata)
    return runtime,runtime.finish()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('peak',[20.,120.],ids=['rounded_zero_tail','underflowed_token_q'])
@pytest.mark.parametrize('lr',[0.,.01],ids=['off','on'])
@pytest.mark.parametrize('compact',[False,True])
def test_nonfinite_kl_with_finite_feedback_is_valid(device,peak,lr,compact):
    runtime,output=peaked_feedback(device,peak,lr,compact)
    diagnostic=feedback_diagnostics(output)
    assert output['opd_nonfinite_kl_states']==3
    assert output['opd_invalid_states']==0
    assert torch.isfinite(runtime.B_fast).all()
    assert bool(runtime.B_fast.any())==(lr>0)
    assert output['opd_updates']==int(lr>0)
    assert diagnostic['feedback_valid']
    assert diagnostic['kl_status']=='nonfinite_probability_space'
    assert diagnostic['kl_mean'] is None  # Finite-term sum is not an overall KL mean.
    if peak==120:assert (runtime.q_cache[0,0,1:]==0).all()
    json.dumps(diagnostic,allow_nan=False)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('corrupt',['q_nan','q_negative','u_nan','teacher_mass'])
def test_actual_invalid_feedback_still_fails_including_off(device,corrupt):
    _,output=peaked_feedback(device,20.,0.,compact=True,corrupt=corrupt)
    diagnostic=feedback_diagnostics(output)
    assert output['opd_invalid_states']>0
    assert not diagnostic['feedback_valid']
    assert diagnostic['validation_errors']
    json.dumps(diagnostic,allow_nan=False)


@pytest.mark.parametrize('name',['opd_final_b_norm','opd_final_b_max_abs'])
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_adapter_is_rejected_even_with_no_invalid_teacher_states(name,value):
    _,output=peaked_feedback('cpu',20.,0.)
    output[name]=value
    diagnostic=feedback_diagnostics(output)
    assert not diagnostic['feedback_valid']
    assert any(name in error for error in diagnostic['validation_errors'])
    json.dumps(diagnostic,allow_nan=False)


def test_finite_and_zero_feedback_kl_reporting():
    _,output=peaked_feedback('cpu',1.,.01)
    diagnostic=feedback_diagnostics(output)
    assert diagnostic['feedback_valid'] and diagnostic['kl_status']=='finite'
    assert diagnostic['kl_mean']==pytest.approx(output['opd_kl_sum']/output['opd_state_weight'])
    for key in ('opd_state_weight','opd_selected_states','opd_updates','opd_kl_sum',
                'opd_final_b_norm','opd_final_b_max_abs'):output[key]=0.
    diagnostic=feedback_diagnostics(output)
    assert diagnostic['feedback_valid'] and diagnostic['kl_status']=='no_feedback'
    assert diagnostic['kl_mean'] is None


@pytest.mark.parametrize('invalid',[False,True])
def test_run_pair_commits_valid_kl_inf_but_preserves_invalid_failure_details(tmp_path,monkeypatch,invalid):
    import rq2.runtime as pipeline
    import helper.opd_generate as generator
    from e2e6.config import Config
    _,packet=peaked_feedback('cpu',20.,0.)
    packet.update(generated_token_ids=[[4]],response_accepted_length_sum=[0],response_verification_rounds=[0],
        total_acc_length=0,total_decoded_token_num=1,total_proposed_draft_tokens=0,total_accepted_draft_tokens=0)
    if invalid:packet['opd_invalid_states']=1.
    outputs=[]
    def generate(**kwargs):
        output=dict(packet)
        if kwargs['opd_fast_lr']>0:output.update(opd_updates=1.,opd_final_b_norm=.01,opd_final_b_max_abs=.005)
        outputs.append(output)
        return output
    monkeypatch.setattr(generator,'speculative_generate',generate)
    monkeypatch.setattr(pipeline,'frozen_digest',lambda model:'frozen')
    monkeypatch.setattr(pipeline,'tokenize',lambda *args:dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3)))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(pipeline,'PackedVerificationLog',lambda *args:SimpleNamespace(events=[],rounds=[],validate=lambda out:True))
    model=SimpleNamespace(opd_projector=torch.ones(32,8))
    c=Config(batch_size=1,responses_per_prompt=1)
    manifest=dict(identity_sha256='fixture',checkpoints={'30':{'fixture':True}})
    args=(model,SimpleNamespace(eos_token_id=99),[dict(id='test:0')],c,30,0,tmp_path,manifest,16,.01,False)
    if invalid:
        with pytest.raises(RuntimeError,match='step=30, rollout=0, arm=off, invalid_states=1.0'):
            pipeline.run_pair(*args)
        failure=tmp_path/'off/rollout_0000/failure.json'
        assert json.loads(failure.read_text())['opd_nonfinite_kl_states']==3
        assert not list(tmp_path.glob('*/rollout_*/complete.json'))
    else:
        assert pipeline.run_pair(*args)
        for arm in ('off','on'):
            directory=tmp_path/arm/'rollout_0000'
            diagnostic=json.loads((directory/'diagnostics.json').read_text())
            assert diagnostic['feedback_valid'] and diagnostic['kl_mean'] is None
            assert (directory/'complete.json').is_file()
            assert json.loads((directory/'responses.json').read_text())['tokens']==[[4]]
        assert len(outputs)==2
