from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from helper.fastgrpo_model import FastGRPOModel
from helper.opd_optimizer import draft_optimizer
from motivation.config import Config
from motivation.runtime import update_draft
from motivation.state import digest,cpu_copy
from teacher_relabel import CapturedSequence,teacher_features,replay_old_gate,relabel,training_outputs,capture


def tiny_cpu():
    torch.manual_seed(123)
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
                    attention_dropout=0.,torch_dtype=torch.float32)
    cfg._attn_implementation='sdpa';target=Qwen2ForCausalLM(cfg).eval()
    dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None
    return FastGRPOModel(dc,target).eval()


def sequence(model,ids,prompt_len,index=0):
    l=len(ids)-1
    row=CapturedSequence(index,f'train:{index}',prompt_len,torch.tensor(ids[:prompt_len]),torch.tensor(ids),
                         torch.tensor(ids[1:]),torch.ones(l,dtype=torch.long),torch.arange(l),
                         (torch.arange(l)>=prompt_len).long(),torch.zeros(l,32))
    row.teacher_features=teacher_features(model.target_model,row);return row.validate()


def test_shift_original_token_masks_both_channels_and_synthetic_drift():
    model=tiny_cpu();s=sequence(model,[7,4,9,3,6,2,8,1],3)
    conf=Config(replay_hidden_atol=1e-6,replay_hidden_rtol=1e-5,replay_distribution_tv=1e-5)
    gate=replay_old_gate(model.target_model,[s],conf)
    assert gate['passed'];assert gate['examples'][0]['supervised_positions']==3
    same=relabel(model.target_model,[s],zero_drift=True)[0]
    assert digest(s)==digest(same)
    with torch.no_grad():model.target_model.model.layers[0].self_attn.q_proj.weight.add_(.1)
    new=relabel(model.target_model,[s])[0]
    assert new.invariant()==s.invariant();assert not torch.equal(new.teacher_features,s.teacher_features)
    broken=deepcopy(s);broken.teacher_features[3]+=1
    with pytest.raises(RuntimeError,match='hidden replay mismatch'):replay_old_gate(model.target_model,[broken],conf)
    # Hidden gate can be permissive, but the independent CE distribution gate
    # must still reject a teacher label mismatch.
    conf.replay_hidden_atol=100.;conf.replay_distribution_tv=1e-8
    with pytest.raises(RuntimeError,match='distribution replay mismatch'):replay_old_gate(model.target_model,[broken],conf)
    lost=deepcopy(s);lost.context_ids=lost.context_ids[1:]
    with pytest.raises(ValueError):lost.validate()


def test_capture_left_padding_variable_lengths_preserves_exact_shift():
    model=tiny_cpu()
    a=sequence(model,[7,4,9,3,6,2,8,1],3);b=sequence(model,[5,9,3,7,8,4,2],2,1)
    out=dict(all_draft_input_ids=[a.draft_input_ids,b.draft_input_ids],
             all_draft_input_states=[a.teacher_features,b.teacher_features],generated_token_ids=[[3,6,2,8,1],[3,7,8,4,2]])
    saved=capture(out,torch.tensor([[7,4,9],[0,5,9]]),torch.tensor([[1,1,1],[0,1,1]]),['a','b'],1)
    assert torch.equal(saved[1].context_ids,b.context_ids)
    assert saved[1].original_prompt_ids.tolist()==[5,9]


def test_zero_drift_placebo_loss_gradients_weights_optimizer_identical():
    model=tiny_cpu();other=deepcopy(model)
    seq=[sequence(model,[7,4,9,3,6,2,8,1],3),sequence(model,[5,9,3,7,8,4,2],2,1)]
    old=training_outputs(seq,model.device);fresh=training_outputs(relabel(model.target_model,seq,zero_drift=True),other.device)
    conf=Config(batch_size=2,responses_per_prompt=1,smoke=True,max_training_token=8)
    oa=draft_optimizer(model.draft_model,1e-4);ob=draft_optimizer(other.draft_model,1e-4)
    mask=torch.tensor([[1,1,1],[0,1,1]])
    a=update_draft(model,oa,old,mask,conf);b=update_draft(other,ob,fresh,mask,conf)
    assert a==b;assert digest(model.draft_model.state_dict())==digest(other.draft_model.state_dict())
    assert digest(oa.state_dict())==digest(ob.state_dict())
