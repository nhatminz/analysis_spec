"""Small execution checks for the installed training APIs; no production weights."""
from copy import deepcopy
import io


def probe_training_runtime(device='cpu'):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, get_scheduler
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from helper.fastgrpo_model import FastGRPOModel
    from helper.transformers_compat import DynamicCache
    from train_draft import pretrain_loss

    device=torch.device(device)
    devices=[torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17)
        dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
        config=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=64,attention_dropout=0.,torch_dtype=dtype)
        config._attn_implementation='sdpa'
        target=Qwen2ForCausalLM(config).to(device=device,dtype=dtype).eval()
        draft_config=deepcopy(config);draft_config.num_hidden_layers=1;draft_config.rope_scaling=None
        model=FastGRPOModel(draft_config,target).to(device)
        batch=dict(input_ids=torch.tensor([[3,4,5,6,7]],device=device),
                   attention_mask=torch.ones(1,5,device=device,dtype=torch.long),
                   loss_mask=torch.tensor([[0,0,1,1,1]],device=device))
        optimizer=torch.optim.AdamW(model.draft_model.parameters(),lr=1e-4)
        scheduler=get_scheduler('cosine_with_min_lr',optimizer=optimizer,
            num_warmup_steps=0,num_training_steps=2,scheduler_specific_kwargs={'min_lr_rate':0.})
        loss=sum(pretrain_loss(model,batch))
        if not torch.isfinite(loss):raise RuntimeError('pretrain probe returned nonfinite loss')
        loss.backward()
        if not any(p.grad is not None for p in model.draft_model.parameters()):
            raise RuntimeError('pretrain probe produced no draft gradient')
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)

        # The upstream generator calls each decoder directly with the singular
        # cache argument. Normal target forwards above exercise the plural API.
        cache=DynamicCache()
        hidden=target.model.embed_tokens(batch['input_ids'])
        positions=torch.arange(5,device=device)[None,:]
        rotary=target.model.rotary_emb(hidden,positions)
        mask=torch.zeros(1,1,5,5,device=device,dtype=dtype)
        with torch.no_grad():
            for layer in target.model.layers:
                result=layer(hidden,attention_mask=mask,position_ids=positions,
                    past_key_value=cache,use_cache=True,output_attentions=False,
                    cache_position=positions[0],position_embeddings=rotary)
                if not isinstance(result,tuple) or result[0].shape!=hidden.shape:
                    raise RuntimeError('decoder does not support the FastGRPO output/cache API')
                hidden=result[0]
        if cache.get_seq_length()!=5 or len(cache.key_cache)!=2:
            raise RuntimeError('target cache was not updated by the decoder')
        cache.batch_repeat_interleave(2);cache.crop(3)
        cache.key_cache[0]=cache.key_cache[0][:1]
        cache.value_cache[0]=cache.value_cache[0][:1]
        if cache[0][0].shape[:3]!=(1,2,3):raise RuntimeError('legacy cache writes did not update HF cache')

        # Exercise the actual LoRA state APIs rather than accept PEFT by version.
        adapted=get_peft_model(target,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,
                                               target_modules=['q_proj'],lora_dropout=0.))
        adapter_optimizer=torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad],lr=1e-6)
        output=adapted(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
        adapter_loss=output.logits.float().square().mean()
        adapter_loss.backward();adapter_optimizer.step()
        state={k:v.detach().clone() for k,v in get_peft_model_state_dict(adapted).items()}
        adapted.disable_adapter_layers();adapted.enable_adapter_layers()
        set_peft_model_state_dict(adapted,state)
        restored=get_peft_model_state_dict(adapted)
        if not all(torch.equal(value,restored[key]) for key,value in state.items()):
            raise RuntimeError('PEFT adapter state roundtrip failed')
        buffer=io.BytesIO()
        torch.save(dict(draft_model=model.draft_model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),target_lora=state),buffer)
        buffer.seek(0)
        checkpoint=torch.load(buffer,map_location=device,weights_only=False)
        model.draft_model.load_state_dict(checkpoint['draft_model'])
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        if device.type=='cuda':torch.cuda.synchronize(device)
    return 'pretrain backward/optimizer/scheduler, target decoder/cache, LoRA and checkpoint roundtrip'
