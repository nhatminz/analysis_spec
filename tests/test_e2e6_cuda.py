import pytest
import torch
from motivation.state import isolated_rng,digest
from helper.checkpointing import capture_rng_state
from e2e6.events import VerificationLog,ReplayTiming

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA FastGRPO generation')

@pytest.mark.parametrize('edge',['ordinary','swap_removal','all_eos','verified_eos'])
def test_shared_path_log_preserves_generation_and_reconstructs(edge,monkeypatch):
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    import helper.fastgrpo_generate as fast
    from motivation.runtime import generator_kwargs
    from helper.specualtive_generate import speculative_generate
    _,model,*_=two_models(conf('unused'));model.requires_grad_(False)
    native=fast.sampling;calls=[]
    def sample(logits,*args,**kwargs):
        calls.append(1)
        if (len(calls)==1 and edge in ('swap_removal','all_eos')) or (len(calls)==2 and edge=='verified_eos'):
            logits=logits.clone()
            for i in (range(len(logits)) if edge=='all_eos' else [0]):
                logits[i,0].fill_(-10000);logits[i,0,96]=10000
        return native(logits,*args,**kwargs)
    monkeypatch.setattr(fast,'sampling',sample)
    logger=VerificationLog(30,0,['test:a','test:b'],2,96)
    results=[];rng=[]
    for hook in (None,logger):
        calls.clear()
        with isolated_rng(51),torch.inference_mode():
            results.append(speculative_generate(model=model,input_ids=torch.tensor([[0,7,9],[3,5,8]]),
                attention_mask=torch.tensor([[0,1,1],[1,1,1]]),tokenizer=TinyTokenizer(),max_length=17,
                **({'verification_hook':hook} if hook else {}),**generator_kwargs(conf('unused'),method='fastgrpo',train=True)))
            rng.append(digest(capture_rng_state()))
    for key in ('generated_token_ids','response_accepted_length_sum','response_verification_rounds',
                'total_proposed_draft_tokens','total_accepted_draft_tokens','verified_tree_nodes'):
        assert results[0][key]==results[1][key]
    assert rng[0]==rng[1]
    assert logger.validate(results[1])
    if edge=='swap_removal':assert logger.tokens[:2]==[[96],[96]]
    if edge=='all_eos':assert not logger.rounds
    assert all(row['is_miss'] is None for row in logger.events if row['reason'] in ('EOS','depth_cap','beam_cap','prefill'))


def test_timing_disabled_event_log_replay_is_identical():
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from motivation.runtime import generator_kwargs
    from helper.specualtive_generate import speculative_generate
    _,model,*_=two_models(conf('unused'));model.requires_grad_(False)
    timer=ReplayTiming();out=[]
    for timing in (None,timer):
        with isolated_rng(123),torch.inference_mode():
            out.append(speculative_generate(model=model,input_ids=torch.tensor([[2,7,9],[3,5,8]]),
                attention_mask=torch.ones(2,3,dtype=torch.long),tokenizer=TinyTokenizer(),max_length=17,
                **({'timing_hook':timing} if timing else {}),**generator_kwargs(conf('unused'),method='fastgrpo',train=True)))
    timer.finish();torch.cuda.synchronize()
    assert out[0]['generated_token_ids']==out[1]['generated_token_ids']
    assert all(s['host_wall_s']>=0 and s['cuda_stream_elapsed_s']>=0 for s in timer.resolve())


def test_collect_three_fixture_checkpoints_analyze_and_skip_completed(tmp_path):
    """Checkpoint IDs are test fixtures, NOT evidence of 30/100/200 training steps."""
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from e2e6.config import Config
    from e2e6.checkpoints import save
    from e2e6.collect import collect
    from e2e6.analysis import analyze
    from motivation.runtime import target_state
    from motivation.state import cpu_copy,atomic_json
    c=Config(output_dir=str(tmp_path),smoke=True,batch_size=2,responses_per_prompt=2,
        eval_rollouts=1,max_length=17,verification_capacity=28,max_verification_num=7,
        max_draft_k=2,max_draft_token_length=3,min_draft_token_length=3,bootstrap_samples=100)
    _,model,*_=two_models(conf('unused'));model.requires_grad_(False)
    manifest=dict(test_fixture=True,config=c.to_dict())
    for step in (30,100,200):
        save(tmp_path/'checkpoints',dict(policy_step=step,attempt=step,
            target_lora=target_state(model.target_model),draft_weights=cpu_copy(model.draft_model.state_dict()),
            manifest_sha256=digest(manifest)),milestone=True)
    del model
    loads=[]
    def factory(config,**kwargs):
        loads.append(1);_,model,*_=two_models(conf('unused'));model.requires_grad_(False)
        model.draft_model.load_state_dict(kwargs['draft_state'])
        from motivation.runtime import load_target
        load_target(model.target_model,kwargs['target_lora']);return model
    rows=[dict(id=f'test:{i}',split='test',question='q',question_sha256=f'h{i}') for i in range(4)]
    result=collect(c,rows,TinyTokenizer(),manifest,[30,100,200],model_factory=factory)
    assert result['completed']==[30,100,200] and len(loads)==1
    second=collect(c,rows,TinyTokenizer(),manifest,[30,100,200,500],model_factory=factory)
    assert second['missing_checkpoints']==[500] and len(loads)==1
    # Add a new fixture checkpoint later without rewriting finalized earlier logs.
    from e2e6.checkpoints import verify
    prior=tmp_path/'collection/step_000030/rollout_0000/events.parquet'
    prior_stamp=prior.stat().st_mtime_ns
    new=verify(tmp_path/'checkpoints/step_000200.pt')
    new.update(policy_step=500,attempt=500)
    save(tmp_path/'checkpoints',new,milestone=True)
    added=collect(c,rows,TinyTokenizer(),manifest,[30,100,200,500],model_factory=factory)
    assert added['completed']==[30,100,200,500] and len(loads)==2
    assert prior.stat().st_mtime_ns==prior_stamp
    result=analyze(tmp_path,[30,100,200],100,42)
    assert result['checkpoints']==['30','100','200']
    cached=tmp_path/'analysis_e2_e6/steps/step_000030.json'
    stamp=cached.stat().st_mtime_ns
    expanded=analyze(tmp_path,[30,100,200,500],100,42)
    assert expanded['checkpoints']==['30','100','200','500']
    assert cached.stat().st_mtime_ns==stamp
    for name in ('recurrence.pdf','response_sharing.pdf','long_tail.pdf','miss_types.pdf'):
        assert (tmp_path/'analysis_e2_e6'/name).read_bytes().startswith(b'%PDF')


def test_pure_training_exact_resume_and_zero_variance_skip(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from test_motivation_cuda import two_models,conf,TinyTokenizer
    from e2e6.config import Config
    import e2e6.train as training
    from e2e6.checkpoints import latest,verify
    full=tmp_path/'full';split=tmp_path/'split'
    calls={'n':0};flag={'requested':False,'signal':None}
    def factory(config,**kwargs):
        _,model,*_=two_models(conf('unused'));return model
    def real_test_rewards(rows,out,tokenizer,c):
        calls['n']+=1
        if calls['n']==1:return [],[[0.,0.],[0.,0.]]
        records=[]
        for i,row in enumerate(rows):
            for j,ids in enumerate(out['generated_token_ids'][i*2:i*2+2]):
                records.append(dict(ids=[2,7,9]+ids,mask=[0,0]+[1]*(len(ids)+1),
                                    advantage=-1. if j==0 else 1.,prompt_id=row['id']))
        return records,[[0.,1.],[0.,1.]]
    monkeypatch.setattr(training,'reference_target_batch',real_test_rewards)
    rows=[dict(id=f'train:{i}',split='train',question='q',question_sha256=f'h{i}') for i in range(8)]
    def config(path):
        return Config(output_dir=str(path),smoke=True,train_steps=2,save_every=1,batch_size=2,
            responses_per_prompt=2,max_length=13,max_training_token=24,max_training_padding_gap=32,
            verification_capacity=28,max_verification_num=7,max_draft_k=2,max_draft_token_length=3,
            min_draft_token_length=3,target_lr=1e-3)
    # Identical manifest identity intentionally excludes each test output path.
    manifest={'test_fixture':True}
    training.train(config(full),rows,TinyTokenizer(),manifest,model_factory=factory)
    expected=verify(latest(full/'checkpoints'))
    calls['n']=0
    @contextmanager
    def signals():yield flag
    monkeypatch.setattr(training,'stop_signals',signals)
    native_save=training.save
    def stop_at_one(directory,state,**kwargs):
        result=native_save(directory,state,**kwargs)
        if state['policy_step']==1:flag['requested']=True
        return result
    monkeypatch.setattr(training,'save',stop_at_one)
    interrupted=training.train(config(split),rows,TinyTokenizer(),manifest,model_factory=factory)
    assert interrupted['policy_step']==1 and interrupted['attempt']>=2
    flag['requested']=False;monkeypatch.setattr(training,'save',native_save)
    resumed=training.train(config(split),rows,TinyTokenizer(),manifest,resume=True,model_factory=factory)
    actual=verify(latest(split/'checkpoints'))
    assert resumed['policy_step']==2
    for key in ('target_lora','draft_weights','optimizer_target','optimizer_draft','sampler','rng','policy_step','attempt','draft_steps'):
        assert digest(actual[key])==digest(expected[key]),key
