from copy import deepcopy
import json
import pytest
import torch
from helper.rewards import accuracy_reward_func,parse_gold_answer
from helper.fastgrpo_training import training_draft_model,compute_target_loss
from motivation.config import Config
from motivation.runtime import update_draft,update_target,target_state
from motivation.state import digest
from motivation.metrics import CONDITIONS,summarize,aal
from test_teacher_relabel import tiny_cpu,sequence


def test_invalid_gold_never_gets_accuracy_one():
    for gold in ('',r'\boxed{}','unparseable gold ???'):
        with pytest.raises(ValueError,match='ground truth|Ground truth'):
            accuracy_reward_func(['anything'],[gold])
    assert accuracy_reward_func([r'\boxed{4}',r'\boxed{5}'],[r'\boxed{4}']*2)==[1.,0.]
    assert str(parse_gold_answer(r'\boxed{\dfrac{1}{29,\!322,\!216}}')[0])=='1/29322216'
    with pytest.raises(ValueError,match='counts differ'):accuracy_reward_func([],['4'])


def test_invalid_data_answers_are_recorded_or_rejected(tmp_path):
    from test_motivation_data import files,Tokenizer
    from motivation.data import prepare,extract_row
    import pandas as pd
    train,test=files(tmp_path)
    rows=pd.read_parquet(train).to_dict('records')
    rows.append(dict(prompt=[dict(role='user',content='invalid distinct question')],reward_model={'ground_truth':''}))
    pd.DataFrame(rows).to_parquet(train)
    a,_,m=prepare(train,test,tmp_path/'exclude',Tokenizer())
    assert len(a)==8 and m['train']['invalid_answers'][0]['id']=='train:8'
    with pytest.raises(ValueError,match='train:8'):prepare(train,test,tmp_path/'error',Tokenizer(),invalid_answer_policy='error')
    with pytest.raises(ValueError,match='Cannot parse'):extract_row(rows[-1],'train',8)


def histories(model):
    generator=torch.Generator().manual_seed(9)
    return dict(all_draft_input_ids=[torch.randint(1,96,(n,),generator=generator) for n in (3,7,11)],
                all_draft_input_states=[torch.randn(n,32,generator=generator) for n in (3,7,11)])


def test_draft_padding_and_final_microbatch_normalization_are_invariant():
    model=tiny_cpu();out=histories(model);mask=torch.ones(3,3,dtype=torch.long)
    results=[]
    for budget,accumulation in ((1,1),(100,1),(1,2),(100,2)):
        other=deepcopy(model)
        loss=training_draft_model(other,out,mask,repeated_generate_nums=1,max_training_token=budget,
                                  max_training_padding_gap=100,draft_accumulation_steps=accumulation,ce_chunk_size=2)
        results.append((loss,[p.grad.clone() for p in other.draft_model.parameters()]))
    for loss,grads in results[1:]:
        torch.testing.assert_close(torch.tensor(loss),torch.tensor(results[0][0]),rtol=1e-6,atol=1e-6)
    for i,(_,grads) in enumerate(results[1:],1):
        scale=2 if i>=2 else 1
        for a,b in zip(results[0][1],grads):torch.testing.assert_close(a,b*scale,rtol=2e-5,atol=1e-6)


def test_empty_draft_masks_skip_without_momentum_decay_or_nan():
    model=tiny_cpu();opt=torch.optim.AdamW(model.draft_model.parameters(),lr=.01)
    out=histories(model);c=Config(batch_size=3,responses_per_prompt=1,smoke=True)
    update_draft(model,opt,out,torch.ones(3,3,dtype=torch.long),c)
    before=digest(dict(model=model.draft_model.state_dict(),optimizer=opt.state_dict()))
    empty={k:[x[:3] for x in v] for k,v in out.items()}
    loss=update_draft(model,opt,empty,torch.ones(3,3,dtype=torch.long),c)
    assert loss['valid_examples']==0 and loss['empty_examples']==3 and loss['optimizer_steps']==0
    assert loss['feature_loss']==loss['distribution_loss']==0.
    assert before==digest(dict(model=model.draft_model.state_dict(),optimizer=opt.state_dict()))
    # Captured early EOS still has a valid prompt/shift, with zero loss positions.
    s=sequence(model,[7,4,9,96],3)
    assert not s.loss_mask[:-1].any()


def test_soft_ce_survives_extreme_logits_and_nonfinite_labels_fail():
    model=tiny_cpu();model.lm_head.weight.data.mul_(10000)
    out=histories(model)
    loss=training_draft_model(model,out,torch.ones(3,3,dtype=torch.long),repeated_generate_nums=1,
        max_training_token=100,max_training_padding_gap=100,draft_accumulation_steps=1)
    assert all(torch.isfinite(torch.tensor(loss)))
    assert all(torch.isfinite(p.grad).all() for p in model.draft_model.parameters())
    out['all_draft_input_states'][1][4]=float('nan')
    with pytest.raises(RuntimeError,match='Nonfinite'):
        training_draft_model(model,out,torch.ones(3,3,dtype=torch.long),repeated_generate_nums=1,
            max_training_token=100,max_training_padding_gap=100,draft_accumulation_steps=1)


def test_grpo_empty_mask_and_masked_nan_do_not_corrupt_gradients():
    logits=torch.randn(2,5,13,requires_grad=True);ref=logits.detach().clone()
    logits.data[0]=float('nan');ref[0]=float('nan')
    labels=torch.ones(2,5,dtype=torch.long);mask=torch.tensor([[0]*5,[0,1,1,1,1]])
    loss,*_=compute_target_loss(logits,ref,None,labels,mask,torch.tensor([[1.],[-1.]]),.1,.04,0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0].count_nonzero()==0


def test_zero_round_eos_is_not_an_invented_verification():
    assert aal([dict(accepted_draft_length_sum=0,verification_rounds=0),
                dict(accepted_draft_length_sum=3,verification_rounds=2)])==1.5
    rows=[dict(policy_step=1,condition=c,prompt_id=str(i),sample_index=0,
               accepted_draft_length_sum=i,verification_rounds=i) for c in CONDITIONS for i in (0,1)]
    out=summarize(rows,1,2,samples=100)
    assert out['bootstrap_undefined_replicates']>0 and out['A1_delta_lag_ci_low'] is None
    with pytest.raises(ValueError):aal([dict(accepted_draft_length_sum=0,verification_rounds=0)])


def test_placebo_checks_allow_precision_roundoff_but_reject_real_state_changes():
    from motivation.state import compare_training_values
    expected={'grad':torch.tensor([1.,.125],dtype=torch.bfloat16),'step':1}
    actual=deepcopy(expected)
    actual['grad'][0]=torch.nextafter(actual['grad'][0],torch.tensor(float('inf'),dtype=torch.bfloat16))
    report=compare_training_values(actual,expected)
    assert report['passed'] and not report['bitwise_equal'] and report['different_tensors']==1
    actual['grad'][0]+=4*torch.finfo(torch.bfloat16).eps
    assert not compare_training_values(actual,expected)['passed']
    for altered in ({'grad':expected['grad'],'step':2},
                    {'grad':expected['grad'].float(),'step':1},
                    {'grad':torch.full((2,),float('nan'),dtype=torch.bfloat16),'step':1}):
        assert not compare_training_values(altered,expected)['passed']
