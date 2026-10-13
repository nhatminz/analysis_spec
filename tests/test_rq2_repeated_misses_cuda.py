"""Actual tiny Qwen CUDA/Triton generation. No 3B scientific claims."""
import json
import os
from pathlib import Path
import pytest
import torch
from motivation.state import digest, isolated_rng
from helper.checkpointing import capture_rng_state
from rq2.events import PackedVerificationLog

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA/Triton SPARK generation required')


@pytest.mark.parametrize('async_update',[False,True])
@pytest.mark.parametrize('edge',['ordinary','swap_removal','all_eos','verified_eos'])
def test_actual_spark_logging_passive_off_on_reset_and_frozen(edge,async_update,monkeypatch):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from motivation.runtime import generator_kwargs
    import helper.opd_generate as spark
    from helper.opd_reflex import OPDReflex
    c=conf('unused');model,*_=two_models(c);model.requires_grad_(False)
    before=digest(model.state_dict());global_rng=digest(capture_rng_state())
    native=spark.sample_target_with_metadata;sample_calls=[];target_calls=[];starts=[]
    original_start=OPDReflex.start
    def start(self,*args,**kwargs):
        result=original_start(self,*args,**kwargs)
        starts.append(int(torch.count_nonzero(self.B_fast)))
        return result
    monkeypatch.setattr(OPDReflex,'start',start)
    def sample(logits,*args,**kwargs):
        sample_calls.append(1)
        if (len(sample_calls)==1 and edge in ('swap_removal','all_eos')) or (len(sample_calls)==2 and edge=='verified_eos'):
            logits=logits.clone()
            for row in (range(len(logits)) if edge=='all_eos' else [0]):
                logits[row,0].fill_(-10000);logits[row,0,96]=10000
        return native(logits,*args,**kwargs)
    monkeypatch.setattr(spark,'sample_target_with_metadata',sample)
    target_layer=model.target_model.get_base_model().model.layers[0]
    handle=target_layer.register_forward_pre_hook(lambda *args:target_calls.append(1))
    diagnostics=[]
    try:
        for lr in (0.,.01):
            results=[];rngs=[];forwards=[]
            observer=PackedVerificationLog(30,0,['p0','p1'],2,96)
            for hook in (None,observer,None):
                sample_calls.clear();target_calls.clear()
                kwargs=generator_kwargs(c,method='opd_reflex',train=False)
                kwargs.update(opd_fast_lr=lr,opd_train_projector=False,opd_diagnostics=True,
                              opd_update_stream=async_update,repeated_generate_nums=2)
                with isolated_rng(51),torch.inference_mode():
                    output=spark.speculative_generate(model=model,input_ids=torch.tensor([[0,7,9],[3,5,8]]),
                        attention_mask=torch.tensor([[0,1,1],[1,1,1]]),tokenizer=TinyTokenizer(),max_length=17,
                        **({'verification_hook':hook} if hook is not None else {}),**kwargs)
                    rngs.append(digest(capture_rng_state()))
                results.append(output);forwards.append(len(target_calls))
                assert digest(model.state_dict())==before
                cache=getattr(model,'_opd_runtime_cache',{})
                assert all(not state.B_fast.any() for state in cache.values())
            for key in ('generated_token_ids','response_accepted_length_sum','response_verification_rounds',
                        'total_proposed_draft_tokens','total_accepted_draft_tokens','verified_tree_nodes','opd_updates'):
                assert results[0][key]==results[1][key]==results[2][key],key
            assert rngs[0]==rngs[1]==rngs[2]
            assert forwards[0]==forwards[1]==forwards[2]==1+results[0]['verification_batches']
            assert observer.validate(results[1])
            if edge=='swap_removal':assert observer.tokens[:2]==[[96],[96]]
            if edge=='all_eos':assert not observer.rounds
            if lr==0:assert results[1]['opd_final_b_norm']==0 and results[1]['opd_updates']==0
            elif edge!='all_eos':assert results[1]['opd_final_b_norm']>0 and results[1]['opd_updates']>0
            diagnostics.append(dict(fast_lr=lr,ordinary_events=sum(e['is_miss'] is not None for e in observer.events),
                misses=sum(e['is_miss'] is True for e in observer.events),rounds=results[1]['verification_batches'],
                B_final_norm=results[1]['opd_final_b_norm'],updates=results[1]['opd_updates'],target_forwards=forwards[1]))
    finally:handle.remove()
    assert all(value==0 for value in starts)
    assert digest(capture_rng_state())==global_rng
    print(json.dumps(dict(edge=edge,async_update=async_update,diagnostics=diagnostics)))
    output_path=os.environ.get('RQ2_CUDA_EVIDENCE')
    if output_path:
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:stream.write(json.dumps(dict(edge=edge,async_update=async_update,
            diagnostics=diagnostics,tiny_validation_fixture=True,gpu=torch.cuda.get_device_name()))+'\n')


def test_real_tiny_pipeline_smoke_commit_and_resume(tmp_path):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from e2e6.config import Config
    from e2e6.checkpoints import save
    from motivation.runtime import target_state,load_target
    from motivation.state import cpu_copy,atomic_json
    from rq2.checkpoints import load
    from rq2.runtime import collect
    c=Config(output_dir=str(tmp_path),smoke=True,batch_size=2,responses_per_prompt=2,eval_rollouts=20,
        max_length=17,verification_capacity=28,max_verification_num=7,max_draft_k=2,
        max_draft_token_length=3,min_draft_token_length=3)
    _,pure,*_=two_models(conf('unused'));pure.requires_grad_(False)
    save(tmp_path/'checkpoints',dict(policy_step=30,attempt=30,target_lora=target_state(pure.target_model),
        draft_weights=cpu_copy(pure.draft_model.state_dict()),test_fixture=True),milestone=True)
    del pure
    checkpoint=load(tmp_path/'checkpoints/step_000030.pt',30,'initialized')['identity']
    checkpoint['projector'].update(rank=8,actual_A_record='projectors/step_000030.json')
    manifest=dict(identity_sha256='tiny-pipeline-fixture',checkpoints={'30':checkpoint},missing_checkpoints=[])
    output=tmp_path/'rq2';atomic_json(output/'manifest.json',manifest)
    rows=[dict(id=f'test:{i}',split='test',question='q',question_sha256=f'h{i}') for i in range(40)]
    calls=[]
    def factory(config,**kwargs):
        calls.append(1);_,model,*_=two_models(conf('unused'));model.requires_grad_(False)
        model.draft_model.load_state_dict(kwargs['draft_state']);load_target(model.target_model,kwargs['target_lora'])
        return model
    first=collect(output,rows,TinyTokenizer(),c,manifest,'initialized',8,16,.01,True,
                  smoke_only=True,model_factory=factory)
    assert first['status']=='complete' and first['smoke_only']
    # Both arms are committed independently, including their projector identity.
    commits=list((output/'smoke').glob('step_*/*/rollout_*/complete.json'))
    assert len(commits)==2
    stamps={p:p.stat().st_mtime_ns for p in commits}
    second=collect(output,rows,TinyTokenizer(),c,manifest,'initialized',8,16,.01,True,
                   smoke_only=True,model_factory=factory)
    assert second['status']=='complete'
    assert all(p.stat().st_mtime_ns==stamp for p,stamp in stamps.items())
    assert not (output/'collection').exists()  # Smoke never masquerades as 20 production rollouts.


@pytest.mark.parametrize('async_update',[False,True])
def test_real_peaked_draft_kl_inf_collects_both_arms_and_replays_identically(tmp_path,async_update):
    """Reproduce the old fatal KL guard in actual Qwen/Triton generation."""
    from copy import deepcopy
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from e2e6.config import Config
    from rq2.runtime import run_pair
    c=Config(smoke=True,batch_size=2,responses_per_prompt=2,eval_rollouts=20,max_length=17,
        verification_capacity=28,max_verification_num=7,max_draft_k=2,
        max_draft_token_length=3,min_draft_token_length=3)
    model,*_=two_models(conf('unused'))
    # A fixture-only independent head gives the draft a finite, concentrated
    # distribution while leaving the target sampler's native logits intact.
    model.lm_head=deepcopy(model.lm_head)
    with torch.no_grad():
        model.lm_head.weight.zero_();model.lm_head.bias.zero_();model.lm_head.bias[7]=120.
    model.requires_grad_(False);before=digest(model.state_dict())
    rng_before=digest(capture_rng_state())
    rows=[dict(id=f'test:{i}',question='q') for i in range(2)]
    manifest=dict(identity_sha256='tiny-peaked-draft',checkpoints={'30':{'fixture':True}})
    assert run_pair(model,TinyTokenizer(),rows,c,30,0,tmp_path,manifest,16,.01,async_update,replay=True)
    for arm in ('off','on'):
        directory=tmp_path/arm/'rollout_0000'
        assert (directory/'complete.json').is_file()
        diagnostic=json.loads((directory/'diagnostics.json').read_text())
        assert diagnostic['opd_nonfinite_kl_states']>0
        assert diagnostic['opd_invalid_states']==0
        assert diagnostic['feedback_valid'] and diagnostic['kl_mean'] is None
        assert diagnostic['logging_token_counter_equivalence']
        assert (diagnostic['opd_updates']>0)==(arm=='on')
        assert (diagnostic['opd_final_b_norm']>0)==(arm=='on')
    assert digest(model.state_dict())==before
    assert digest(capture_rng_state())==rng_before
