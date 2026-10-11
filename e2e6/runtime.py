"""Pure model initialization and the reference decode/retemplate reward buffer."""
from copy import deepcopy
import numpy as np
import torch
from motivation.data import messages
from motivation.runtime import load_target,target_state
from motivation.state import digest


def build_model(c, draft_state=None, target_lora=None, training=False):
    from transformers import AutoConfig,AutoModelForCausalLM
    from peft import get_peft_model,LoraConfig,TaskType
    from helper.fastgrpo_model import FastGRPOModel
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16}[c.dtype]
    target=AutoModelForCausalLM.from_pretrained(c.model,local_files_only=True,torch_dtype=dtype,
                                               attn_implementation=c.attention_implementation).cuda().eval()
    dc=AutoConfig.from_pretrained(c.model,local_files_only=True)
    dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=dtype
    model=FastGRPOModel(dc,target).cuda()
    if draft_state is None:
        draft_state=torch.load(c.draft_checkpoint,map_location='cpu',weights_only=True)['draft_model']
        # A pretrained A, if present, is explicitly discarded; this is a pure draft.
        draft_state={k:v for k,v in draft_state.items() if k!='opd_projector'}
    if any('opd_' in k for k in draft_state):raise ValueError('Pure draft state cannot contain an OPD projector')
    model.draft_model.load_state_dict(draft_state,strict=True)
    for p in target.parameters():p.requires_grad_(False)
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=32,lora_dropout=0.,
                   target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    target=get_peft_model(target,lora)
    if target_lora is not None:load_target(target,target_lora)
    model.target_model=target
    model.eval()
    if training:
        if c.target_gradient_checkpointing:
            target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        for p in model.draft_model.parameters():p.requires_grad_(True)
    else:model.requires_grad_(False)
    if getattr(model,'opd_projector',None) is not None:raise ValueError('Reflex must be disabled')
    return model


def reference_target_batch(rows,outputs,tokenizer,c):
    """Match SpecNaacl's decoded assistant/retemplated target loss examples."""
    from helper.rewards import accuracy_reward_func,format_reward_func
    records=[];all_rewards=[]
    for index,row in enumerate(rows):
        ids=outputs['generated_token_ids'][index*c.responses_per_prompt:(index+1)*c.responses_per_prompt]
        texts=[tokenizer.decode(tokens,skip_special_tokens=True) for tokens in ids]
        rewards=np.array(accuracy_reward_func(texts,[row['answer']]*len(texts)))+.2*np.array(format_reward_func(texts))
        all_rewards.append(rewards.tolist())
        if rewards.std()==0:continue
        advantages=(rewards-rewards.mean())/rewards.std()
        for text,advantage in zip(texts,advantages):
            prompt=messages(row)
            full=prompt+[dict(role='assistant',content=text)]
            rendered=tokenizer.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
            tokens=tokenizer(rendered,padding=False)['input_ids']
            prefix=tokenizer.encode(tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True))
            mask=[0]*(len(prefix)-1)+[1]*(len(tokens)-len(prefix)+1)
            if len(mask)!=len(tokens) or not any(mask[:-1]):raise ValueError('Retemplated target supervision misaligned')
            records.append(dict(ids=tokens,mask=mask,advantage=float(advantage),prompt_id=row['id']))
    return records,all_rewards
