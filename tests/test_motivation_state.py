from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch
from motivation.config import Config
from motivation.runtime import update_target,target_state
from motivation.state import digest,cpu_copy
from motivation.compatibility import QWEN_3B,validate_draft,validate_tokenizer


def test_one_actual_target_step_and_control_cannot_change_weights():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    cfg=Qwen2Config(vocab_size=17,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=1,torch_dtype=torch.float32)
    target=get_peft_model(Qwen2ForCausalLM(cfg).eval(),LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,
                                                                  lora_dropout=0.,target_modules=['q_proj','v_proj']))
    optimizer=torch.optim.AdamW(target.parameters(),lr=.01)
    calls=[];step=optimizer.step
    optimizer.step=lambda *a,**kw:(calls.append(1),step(*a,**kw))[1]
    records=[dict(ids=[2,3,4,5,6],mask=[0,0,1,1,1],advantage=-1.),
             dict(ids=[2,3,7,8,9],mask=[0,0,1,1,1],advantage=1.)]
    c=Config(max_training_token=5)
    before=digest(target_state(target));data=digest(records)
    result=update_target(target,optimizer,records,c)
    assert len(calls)==1 and result['target_optimizer_steps']==1
    assert digest(target_state(target))!=before;assert digest(records)==data
    before=digest(target_state(target))
    update_target(target,optimizer,records,c,zero_update=True)
    assert len(calls)==2;assert digest(target_state(target))==before
    update_target(target,optimizer,[],c)
    assert len(calls)==3;assert digest(target_state(target))==before


def test_checkpoint_format_architecture_dimensions_fail_closed(tmp_path):
    from transformers import AutoConfig
    from helper.modeling_draft import DraftModel
    config=dict(QWEN_3B,rope_theta=1000000.,rms_norm_eps=1e-6,attention_dropout=0.,max_position_embeddings=32768,
                hidden_act='silu')
    model=tmp_path/'target';model.mkdir();(model/'config.json').write_text(json.dumps(config))
    draft=tmp_path/'draft';draft.mkdir();(draft/'target_config.json').write_text(json.dumps(config))
    torch.save({'model':{}},draft/'draft.pth')
    with pytest.raises(ValueError,match='EAGLE3/SpecForge'):validate_draft(draft,model)
    torch.save({'draft_model':{'compact_head.weight':torch.zeros(1)}},draft/'draft.pth')
    with pytest.raises(ValueError,match='architecture mismatch'):validate_draft(draft,model)
    dc=AutoConfig.from_pretrained(model,local_files_only=True);dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=torch.bfloat16
    with torch.device('meta'):keys=DraftModel(dc).state_dict().keys()
    torch.save({'draft_model':{k:torch.zeros(1) for k in keys}},draft/'draft.pth')
    with pytest.raises(ValueError,match='shape'):validate_draft(draft,model)
    config['hidden_size']=1536;(draft/'target_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='3B'):validate_draft(draft,model)


def test_tokenizer_ids_and_chat_template_validation():
    from types import SimpleNamespace
    t=SimpleNamespace(eos_token_id=151645,pad_token_id=151643,padding_side='left',chat_template='<|im_start|>',
                      get_vocab=lambda:{'x':1},convert_tokens_to_ids=lambda s:{'<|endoftext|>':151643,'<|im_start|>':151644,'<|im_end|>':151645}[s])
    validate_tokenizer(t,QWEN_3B)
    t.eos_token_id=2
    with pytest.raises(ValueError,match='EOS/pad'):validate_tokenizer(t,QWEN_3B)
