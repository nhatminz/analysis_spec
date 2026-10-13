from types import SimpleNamespace
import pytest
import torch
from helper.opd_reflex import union_reference
from helper.tree_verification import PackedTree
from rq2.per_prompt import PerPromptReflex
from test_opd_reflex import state

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def grouped(device='cpu',lr=.03):
    _,model,mapping=state(device,v=37,h=32,topk=4)
    engine=PerPromptReflex(8,4,lr,diagnostics=True)
    engine.start(model,4,mapping,32,prompt_groups=2,responses_per_prompt=2,canonical=[0,1,2,3],
        max_contexts=4,max_nodes=32,max_path=4,max_proposal_contexts=4)
    return engine,model,mapping


@pytest.mark.parametrize('device',DEVICES)
def test_group_weighted_update_matches_independent_prompt_oracle(device):
    engine,model,mapping=grouped(device)
    h=torch.randn(4,2,32,generator=torch.Generator().manual_seed(71)).to(device)
    raw=model.lm_head(h);engine.propose(raw,h,2,mapping,root=True)
    parents=torch.tensor([[-1,0]]*4,device=device)
    contexts=torch.tensor([[0,1],[0,-1],[0,1],[0,1]],device=device)
    tree=PackedTree(parents,parents.clone(),contexts,1)
    path=SimpleNamespace(packed_indices=torch.tensor([[0,1]]*4,device=device))
    target=torch.randn(4,2,37,generator=torch.Generator().manual_seed(13)).to(device).softmax(-1)
    q=raw.float().softmax(-1).reshape(8,37)
    di=engine.ids_cache[:4,:2].reshape(8,4)
    ti=torch.argsort(target.reshape(8,37),descending=True,stable=True)[:,:4]
    ids,valid,p,qq,*_=union_reference(target.reshape(8,37),q,ti,di)
    weights=(contexts>=0).flatten().float()
    groups=engine.row_groups.repeat_interleave(2)
    u=engine.u_cache[:4,:2].reshape(8,8).clone()
    expected=torch.zeros_like(engine.B_fast)
    for group in range(2):
        w=weights*(groups==group)
        gradient=torch.where(valid,(qq-p)*w[:,None],0.)
        expected[group].index_add_(0,ids.clamp_min(0).flatten(),(gradient[...,None]*u[:,None,:]).reshape(-1,8))
        expected[group]*=-engine.fast_lr/w.sum()
    before=model.opd_projector.clone()
    engine.feedback(tree,path,target)
    torch.testing.assert_close(engine.B_fast,expected,rtol=5e-5,atol=2e-7)
    assert engine.group_weight.tolist()==[3.,4.]
    assert engine.group_updates.tolist()==[1,1]
    assert torch.equal(before,model.opd_projector)


@pytest.mark.parametrize('device',DEVICES)
def test_ownership_after_prefill_removal_swap_and_canonical_teacher_order(device):
    engine,model,mapping=grouped(device)
    engine.set_response_order([3,0,2])  # Physical KV rows, after arbitrary removal.
    assert engine.row_groups.tolist()==[1,0,1]
    assert engine.group_first.tolist()==[1,0]
    h=torch.randn(3,1,32,device=device);engine.propose(model.lm_head(h),h,2,mapping,root=True)
    engine.feedback_row_map=torch.tensor([1,2,0],device=device)  # Canonical target sampler order.
    root=torch.full((3,1),-1,device=device,dtype=torch.long)
    contexts=torch.tensor([[0],[-1],[-1]],device=device)  # Feedback only for original response 0.
    tree=PackedTree(root,root.clone(),contexts,1)
    target=torch.ones(3,1,37,device=device)/37
    engine.feedback(tree,SimpleNamespace(packed_indices=torch.zeros_like(root)),target)
    assert engine.B_fast[0].any() and not engine.B_fast[1].any()
    assert engine.group_updates.tolist()==[1,0]
    assert engine.group_weight.tolist()==[1.,0.]
    assert engine.finish()['opd_invalid_states']==0
    engine.clear();assert not engine.B_fast.any()
    engine.start(model,2,mapping,32,prompt_groups=2,responses_per_prompt=2,canonical=[1,3],
        max_contexts=4,max_nodes=32,max_path=4,max_proposal_contexts=4)
    assert engine.row_groups.tolist()==[0,1] and not engine.B_fast.any()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['sparse','fused','gemm'])
def test_proposal_group_isolation_and_backend_parity(device,mode):
    engine,model,mapping=grouped(device)
    engine.set_response_order([3,0,2])
    engine.B_fast.zero_();engine.B_fast[0,4]=.4
    engine.active_ids.zero_();engine.active_ids[0,0]=4
    engine.active_count.copy_(torch.tensor([1,0],device=device))
    engine.bitmap.zero_();engine.bitmap[0,0]=1<<4
    engine.dispatch_count.fill_(1);engine.host_active_count=1;engine._ever_updated=True
    engine.proposal_mode='sparse' if mode=='sparse' else 'dense'
    engine.dense_implementation='fused' if mode=='sparse' else mode
    h=torch.randn(3,2,32,generator=torch.Generator().manual_seed(39)).to(device)
    logits=model.lm_head(h)
    q,ids,_=engine.propose(logits,h,4,mapping,root=True)
    expected=logits.float()+torch.bmm(h.float().matmul(model.opd_projector),engine.B_fast[engine.row_groups].transpose(1,2))
    order=torch.argsort(expected,descending=True,stable=True)[...,:4]
    assert torch.equal(ids,order)
    torch.testing.assert_close(q,expected.softmax(-1).gather(-1,order),rtol=3e-5,atol=2e-7)
    torch.testing.assert_close(q[[0,2]],logits.float().softmax(-1)[[0,2]].gather(-1,ids[[0,2]]),rtol=3e-5,atol=2e-7)


@pytest.mark.parametrize('device',DEVICES)
def test_other_prompt_teacher_cannot_change_this_prompt_B(device):
    results=[]
    for change in (False,True):
        engine,model,mapping=grouped(device)
        h=torch.ones(4,1,32,device=device)
        engine.propose(model.lm_head(h),h,2,mapping,root=True)
        root=torch.full((4,1),-1,device=device,dtype=torch.long)
        target=torch.ones(4,1,37,device=device)/37
        if change:target[:2].zero_();target[:2,:,17]=1.
        engine.feedback(PackedTree(root,root.clone(),torch.zeros_like(root),1),
            SimpleNamespace(packed_indices=torch.zeros_like(root)),target)
        results.append(engine.B_fast.clone())
    assert not torch.equal(results[0][0],results[1][0])
    torch.testing.assert_close(results[0][1],results[1][1],rtol=1e-6,atol=1e-9)


def test_per_prompt_refuses_new_A_and_training():
    with pytest.raises(ValueError,match='frozen'):PerPromptReflex(train_projector=True)
    engine,model,mapping=grouped()
    model.opd_projector=None
    with pytest.raises(ValueError,match='loaded'):
        engine.start(model,4,mapping,32,prompt_groups=2,responses_per_prompt=2,canonical=[0,1,2,3])
    with pytest.raises(ValueError,match='ownership'):engine.set_response_order([0,0])
