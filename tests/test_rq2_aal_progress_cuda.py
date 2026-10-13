import json
from pathlib import Path
import pytest
import torch
from motivation.state import digest,isolated_rng,atomic_json,cpu_copy
from helper.checkpointing import capture_rng_state

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA/Triton evaluation required')


@pytest.mark.parametrize('async_update',[False,True])
@pytest.mark.parametrize('edge',['ordinary','prefill_removal','verified_removal','all_eos'])
def test_actual_per_prompt_generation_passive_logs_EOS_reset_frozen_and_target_forwards(edge,async_update,monkeypatch):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from motivation.runtime import generator_kwargs
    from rq2.events import PackedVerificationLog
    import helper.opd_generate as spark
    from rq2.per_prompt import PerPromptReflex
    model,*_=two_models(conf('unused'));model.requires_grad_(False)
    before=digest(model.state_dict());global_rng=digest(capture_rng_state())
    native=spark.sample_target_with_metadata;sample_calls=[];target_calls=[];orders=[]
    set_order=PerPromptReflex.set_response_order
    def owner_spy(self,canonical):
        result=set_order(self,canonical)
        assert self.row_groups.tolist()==[i//2 for i in canonical]
        orders.append(list(canonical))
        return result
    monkeypatch.setattr(PerPromptReflex,'set_response_order',owner_spy)
    def sample(logits,*args,**kwargs):
        sample_calls.append(1)
        if (len(sample_calls)==1 and edge in ('prefill_removal','all_eos')) or (len(sample_calls)==2 and edge=='verified_removal'):
            logits=logits.clone()
            for row in (range(len(logits)) if edge=='all_eos' else [0]):
                logits[row,0].fill_(-10000);logits[row,0,96]=10000
        return native(logits,*args,**kwargs)
    monkeypatch.setattr(spark,'sample_target_with_metadata',sample)
    handle=model.target_model.get_base_model().model.layers[0].register_forward_pre_hook(lambda *args:target_calls.append(1))
    try:
        for lr in (0.,.01):
            outputs=[];rng=[]
            observer=PackedVerificationLog(234,0,['p0','p1'],2,96)
            for hook in (None,observer):
                sample_calls.clear();target_calls.clear()
                kwargs=generator_kwargs(conf('unused'),method='opd_reflex',train=False)
                kwargs.update(opd_fast_lr=lr,opd_train_projector=False,opd_diagnostics=True,
                    opd_update_stream=async_update,repeated_generate_nums=2,opd_b_sharing_scope='per_prompt_group')
                with isolated_rng(51),torch.inference_mode():
                    out=spark.speculative_generate(model=model,input_ids=torch.tensor([[0,7,9],[3,5,8]]),
                        attention_mask=torch.tensor([[0,1,1],[1,1,1]]),tokenizer=TinyTokenizer(),max_length=19,
                        **({'verification_hook':hook} if hook else {}),**kwargs)
                    rng.append(digest(capture_rng_state()))
                assert len(target_calls)==1+out['verification_batches']
                assert out['opd_b_sharing_scope']=='per_prompt_group' and out['opd_initial_b_norm']==0
                assert out['opd_invalid_states']==0
                assert digest(model.state_dict())==before
                assert all(not e.B_fast.any() for e in getattr(model,'_opd_runtime_cache',{}).values())
                outputs.append(out)
            for key in ('generated_token_ids','total_acc_length','total_decoded_token_num',
                        'response_accepted_length_sum','response_verification_rounds','opd_group_updates'):
                assert outputs[0][key]==outputs[1][key],key
            assert rng[0]==rng[1] and observer.validate(outputs[1])
            out=outputs[1]
            if lr==0:assert out['opd_final_b_norm']==0 and not any(out['opd_group_updates'])
            elif edge!='all_eos':assert out['opd_final_b_norm']>0 and any(out['opd_group_updates'])
            if edge=='prefill_removal':assert out['opd_group_updates'][0]==0
    finally:handle.remove()
    assert digest(capture_rng_state())==global_rng


def test_checkpoint_A_install_does_not_initialize_projector(monkeypatch):
    from test_motivation_cuda import two_models,conf
    import helper.opd_reflex as helper
    reference,pure,*_=two_models(conf('unused'))
    a=reference.opd_projector.detach().cpu().clone()
    monkeypatch.setattr(helper,'initialize_projector',lambda *args,**kwargs:pytest.fail('New A initialized'))
    pure.enable_opd(a.shape[1],projector=a)
    assert digest(pure.opd_projector)==digest(a)


def test_actual_standalone_AAL_smoke_collection_analysis_and_resume(tmp_path):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from motivation.runtime import target_state,load_target
    from rq2.checkpoints import load
    from rq2.aal_progress import AALConfig,collect
    from rq2.aal_analysis import analyze
    model,*_=two_models(conf('unused'))
    cp=tmp_path/'step234.pt'
    torch.save(dict(format='opd_fastgrpo_checkpoint_v5',method='opd_reflex',step=234,
        target_lora=target_state(model.target_model),draft_model=cpu_copy(model.draft_model.state_dict()),
        opd_projector=cpu_copy(model.opd_projector),opd_projector_initial=torch.zeros_like(model.opd_projector).cpu()),cp)
    state=load(cp,234,'learned')
    c=AALConfig(checkpoint_path=str(cp),num_rollouts=2,batch_size=2,responses_per_prompt=2,max_length=19,
        verification_capacity=28,max_verification_num=7,max_draft_k=2,max_draft_token_length=3,min_draft_token_length=3)
    output=tmp_path/'rq2_aal_fixture';output.mkdir()
    rows=[dict(id=f'test:{i}',question='q') for i in range(4)]
    manifest=dict(identity_sha256='actual-tiny-AAL',config=dict(checkpoint_step=234,num_rollouts=2),
        checkpoint=state['identity'],training_metadata=dict(training_B_scope='rollout_batch'),seeds=[51,52])
    atomic_json(output/'manifest.json',manifest)
    def factory(config,**kwargs):
        _,pure,*_=two_models(conf('unused'))
        pure.draft_model.load_state_dict(kwargs['draft_state']);load_target(pure.target_model,kwargs['target_lora'])
        return pure
    result=collect(c,output,rows,TinyTokenizer(),state,manifest,smoke=True,model_factory=factory)
    assert result['status']=='complete' and result['completed_rollouts']==[0]
    summary=analyze(output,100,2026,smoke=True)
    assert summary['paired_rollouts']==1 and not (output/'collection').exists()
    commit=list((output/'smoke').glob('*/rollout_*/complete.json'))
    stamps={p:p.stat().st_mtime_ns for p in commit}
    assert len(stamps)==2
    collect(c,output,rows,TinyTokenizer(),state,manifest,smoke=True,model_factory=factory)
    assert all(p.stat().st_mtime_ns==stamp for p,stamp in stamps.items())
    validation=json.loads((output/'smoke_analysis/validation.json').read_text())
    assert validation['smoke_only'] and validation['events_reconstructed'] and validation['counters_match']


def test_actual_eight_prompts_eight_responses_OFF_matches_legacy_zero_B_and_ON_updates_every_prompt():
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from rq2.events import PackedVerificationLog
    from rq2.aal_progress import validate_counters
    from helper.opd_generate import speculative_generate
    model,*_=two_models(conf('unused'));model.requires_grad_(False)
    before=digest(model.state_dict())
    ids=torch.tensor([[3,5,8]]*8);mask=torch.ones_like(ids)
    kwargs=dict(model=model,input_ids=ids,attention_mask=mask,tokenizer=TinyTokenizer(),do_sample=True,
        repeated_generate_nums=8,temperature=1.,top_p=.95,max_length=13,verification_capacity=512,
        max_verification_num=160,max_draft_k=8,max_draft_token_length=5,min_draft_token_length=3,
        opd_train_projector=False,opd_diagnostics=True,opd_update_stream=True)
    with isolated_rng(2026):legacy=speculative_generate(**kwargs,opd_fast_lr=0.)
    log=PackedVerificationLog(234,0,[f'p{i}' for i in range(8)],8,96)
    with isolated_rng(2026):off=speculative_generate(**kwargs,opd_fast_lr=0.,opd_b_sharing_scope='per_prompt_group',verification_hook=log)
    for key in ('generated_token_ids','total_acc_length','total_decoded_token_num','response_verification_rounds',
                'total_accepted_draft_tokens','total_proposed_draft_tokens'):
        assert off[key]==legacy[key],key
    assert validate_counters(log,off)['active_response_rounds']>0
    log=PackedVerificationLog(234,0,[f'p{i}' for i in range(8)],8,96)
    with isolated_rng(2026):on=speculative_generate(**kwargs,opd_fast_lr=.01,opd_b_sharing_scope='per_prompt_group',verification_hook=log)
    assert validate_counters(log,on)['active_response_rounds']>0
    assert len(on['generated_token_ids'])==64
    assert all(u>0 for u in on['opd_group_updates'])
    assert all(n>0 for n in on['opd_group_b_norms'])
    assert on['opd_invalid_states']==0
    assert digest(model.state_dict())==before
