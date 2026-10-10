"""Real CUDA FastGRPO invariance: same tokens/counters/RNG/forward counts."""
import pytest
import torch
from motivation.verification_persistence import RootObserver
from motivation.state import isolated_rng,digest
from helper.checkpointing import capture_rng_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA FastGRPO smoke')


@pytest.mark.parametrize('eos_mode',['ordinary','prefill','verified'])
def test_observation_preserves_tokens_counters_rng_and_forward_count(eos_mode,monkeypatch):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    import helper.fastgrpo_generate as fast
    from motivation.runtime import generator_kwargs
    from helper.specualtive_generate import speculative_generate
    _,model,*_ = two_models(conf('unused'))
    model.eval();model.requires_grad_(False)
    original=fast.sampling;calls=[]
    def sample(logits,*args,**kwargs):
        calls.append(1)
        force = (eos_mode=='prefill' and len(calls)==1) or (eos_mode=='verified' and len(calls)==2)
        if force:
            logits=logits.clone();logits[0,0].fill_(-10000.);logits[0,0,96]=10000.
        return original(logits,*args,**kwargs)
    monkeypatch.setattr(fast,'sampling',sample)
    forwards=[]
    target_hook=model.target_model.base_model.model.model.layers[0].register_forward_hook(lambda *args:forwards.append('target'))
    draft_hook=model.draft_model.register_forward_hook(lambda *args:forwards.append('draft'))
    kwargs=generator_kwargs(conf('unused'),method='fastgrpo',train=False)
    outputs=[];states=[];counts=[]
    observer=RootObserver('smoke',{'donor':[list(range(8))]*40})
    try:
        for observation in (None,observer):
            calls.clear();forwards.clear()
            with isolated_rng(71),torch.inference_mode():
                outputs.append(speculative_generate(model=model,input_ids=torch.tensor([[2,7,9]]),
                    attention_mask=torch.ones(1,3,dtype=torch.long),tokenizer=TinyTokenizer(),max_length=35,max_new_tokens=32,
                    **({'observation_hook':observation} if observation is not None else {}),**kwargs))
                states.append(digest(capture_rng_state()));counts.append(list(forwards))
    finally:
        target_hook.remove();draft_hook.remove()
    counters=['generated_token_ids','response_accepted_length_sum','response_verification_rounds','verification_batches',
              'active_response_rounds','verified_tree_nodes','total_accepted_draft_tokens','total_proposed_draft_tokens',
              'draft_acceptance_rate','total_acc_length','total_decoded_token_num']
    assert all(outputs[0][key]==outputs[1][key] for key in counters)
    assert states[0]==states[1] and counts[0]==counts[1]
    response=observer.finish(outputs[1],96)
    if eos_mode=='prefill':assert response['zero_round']
    else:assert response['rounds']>0
    if eos_mode=='verified':assert response['termination']=='eos'
