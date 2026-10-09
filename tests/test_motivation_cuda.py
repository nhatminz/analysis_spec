"""Real tiny CUDA/Triton protocol pilot; never substitutes for a 3B result."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace
import pytest
import torch
from motivation.config import Config
from motivation.state import digest,cpu_copy,isolated_rng
from motivation.runtime import evaluate,learner_snapshot,load_learner,update_target,target_state
from teacher_relabel import capture,replay_old_gate

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Production generator requires CUDA')


class TinyTokenizer:
    eos_token_id=96;pad_token_id=0
    def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
    def apply_chat_template(self,*args,**kwargs):return 'tiny prompt'
    def __call__(self,texts,**kwargs):
        return dict(input_ids=torch.tensor([[2,7,9]]*len(texts)),attention_mask=torch.ones(len(texts),3,dtype=torch.long))


def two_models(config):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from helper.fastgrpo_model import FastGRPOModel
    from helper.opd_optimizer import draft_optimizer
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(431)
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
                   num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.,torch_dtype=torch.bfloat16)
    cfg._attn_implementation='sdpa';target=Qwen2ForCausalLM(cfg).cuda().bfloat16().eval()
    target.lm_head.bias=torch.nn.Parameter(torch.zeros(97,device='cuda',dtype=torch.bfloat16))
    target.lm_head.bias.data[96]=-20.  # valid EOS used for padding, suppress incidental early fixture EOS
    dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None
    r=FastGRPOModel(deepcopy(dc),target).cuda();s=FastGRPOModel(deepcopy(dc),target).cuda()
    s.draft_model.load_state_dict(r.draft_model.state_dict());r.enable_opd(8)
    target=get_peft_model(target,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj']))
    r.target_model=target;s.target_model=target;r.eval();s.eval()
    return r,s,torch.optim.AdamW(target.parameters(),lr=1e-3),draft_optimizer(r.draft_model,1e-4),draft_optimizer(s.draft_model,1e-4)


def conf(output,steps=2):
    return Config(output_dir=str(output),train_steps=steps,eval_steps=(1,2),confirmation_steps=(),smoke=True,
                  batch_size=2,responses_per_prompt=2,max_length=13,max_prompt_length=10,max_training_token=24,
                  max_training_padding_gap=32,eval_max_new_tokens=6,bootstrap_samples=50,opd_update_stream=False,
                  verification_capacity=28,max_verification_num=7,max_draft_k=2,max_draft_token_length=3,min_draft_token_length=3,
                  zero_update_steps=(1,))


def test_actual_rollout_alignment_and_exact_cap_both_methods():
    from helper.specualtive_generate import speculative_generate
    from motivation.runtime import generator_kwargs
    from teacher_relabel import TeacherTrace
    c=conf('unused');r,s,*_=two_models(c);tok=TinyTokenizer()
    trace=TeacherTrace(2)
    ids=torch.tensor([[0,7,9],[3,5,8]]);mask=torch.tensor([[0,1,1],[1,1,1]])
    with isolated_rng(42),torch.inference_mode():
        out=speculative_generate(model=s,input_ids=ids,attention_mask=mask,tokenizer=tok,max_length=13,teacher_trace=trace,
                                 **generator_kwargs(c,method='fastgrpo',train=True))
    captured=capture(out,ids,mask,['a','b'],2)
    assert replay_old_gate(s.target_model,captured,c)['passed']
    gate=replay_old_gate(s.target_model,captured,c,teacher_trace=trace)
    assert gate['passed']
    assert max(x['hidden_max_abs'] for x in gate['examples'])==0.
    assert max(x['distribution_tv_max'] for x in gate['examples'])==0.
    for method,model in (('fastgrpo',s),('opd_reflex',r)):
        with isolated_rng(12),torch.inference_mode():
            out=speculative_generate(model=model,input_ids=ids[:1],attention_mask=mask[:1],tokenizer=tok,max_length=10,max_new_tokens=5,
                                     **generator_kwargs(c,method=method,train=False))
        assert len(out['generated_token_ids'][0])==5
        assert out['response_accepted_length_sum'][0]==4
        assert out['response_verification_rounds'][0]>0


def test_a2_order_invariance_frozen_A_and_rng_restore():
    from helper.checkpointing import capture_rng_state
    c=conf('unused');r,*_=two_models(c);tok=TinyTokenizer()
    rows=[dict(id='test:0',question='q',answer='\\boxed{1}',question_sha256='abc')]
    before=digest(dict(model=r.draft_model.state_dict(),rng=capture_rng_state()))
    old=evaluate(r,rows,tok,c,1,'reflex_off/old','old','new')
    on=evaluate(r,rows,tok,c,1,'reflex_on/old','old','new')
    on_again=evaluate(r,rows,tok,c,1,'reflex_on/old','old','new')
    again=evaluate(r,rows,tok,c,1,'reflex_off/old','old','new')
    assert old[0]['generated_token_ids']==again[0]['generated_token_ids']
    assert old[0]['learned_A_sha256']==on[0]['learned_A_sha256']
    assert old[0]['opd_final_b_norm']==0;assert on[0]['opd_final_b_norm']>0
    assert on[0]['generated_token_ids']==on_again[0]['generated_token_ids']
    assert on[0]['opd_final_b_norm']==on_again[0]['opd_final_b_norm']
    assert before==digest(dict(model=r.draft_model.state_dict(),rng=capture_rng_state()))


def test_completed_rollout_releases_idle_KV_without_changing_tokens_or_feedback():
    from helper.specualtive_generate import speculative_generate
    from helper.checkpointing import capture_rng_state
    from motivation.runtime import generator_kwargs,rollout
    c=conf('unused');r,*_=two_models(c);tok=TinyTokenizer()
    batch=dict(input_ids=torch.tensor([[2,7,9],[3,5,8]]),attention_mask=torch.ones(2,3,dtype=torch.long))
    before=digest(capture_rng_state())
    with isolated_rng(42),torch.inference_mode():
        raw=speculative_generate(model=r,**batch,tokenizer=tok,max_length=c.max_length,
                                 **generator_kwargs(c,method='opd_reflex',train=True))
    assert r._opd_target_kv_pool.layers  # Reference end_rollout(0) retains pools.
    pending=r.opd_projector_grad_sum.clone();weight=r.opd_projector_grad_weight.clone()
    released=rollout(r,batch,tok,c,'opd_reflex',42)
    for name in ('_opd_target_kv_pool','_opd_draft_kv_pool'):
        assert not getattr(r,name).layers
    for key in ('generated_token_ids','response_accepted_length_sum','response_verification_rounds'):
        assert raw[key]==released[key]
    for key in ('all_draft_input_states','all_draft_input_ids'):
        for a,b in zip(raw[key],released[key]):assert torch.equal(a,b)
    torch.testing.assert_close(r.opd_projector_grad_sum,2*pending,rtol=3e-6,atol=1e-12)
    torch.testing.assert_close(r.opd_projector_grad_weight,2*weight,rtol=0,atol=0)
    assert digest(capture_rng_state())==before


def test_single_target_step_resume_vs_uninterrupted_and_unscheduled_no_eval(tmp_path,monkeypatch):
    import motivation.runner as runner
    import motivation.runtime as runtime
    monkeypatch.setattr(runner,'build_models',two_models)
    # Rewards deliberately varied for a tiny model, so the second boundary has
    # an actual nonzero target update. This fixture does not claim math accuracy.
    def training_only(rows,batch,out,tok,c):
        assert all(row['split']=='train' for row in rows)
        records=[]
        for i,ids in enumerate(out['generated_token_ids']):
            prompt=batch['input_ids'][i//c.responses_per_prompt].tolist()
            records.append(dict(ids=prompt+ids,mask=[0]*(len(prompt)-1)+[1]*(len(ids)+1),
                                advantage=float(2*(i%2)-1),prompt_id=rows[i//c.responses_per_prompt]['id']))
        return records,[[0.,1.] for _ in rows]
    monkeypatch.setattr(runner,'target_batch',training_only)
    train=[dict(id=f'train:{i}',question=str(i),answer='\\boxed{1}',split='train',question_sha256=str(i)) for i in range(8)]
    test=[dict(id=f'test:{i}',question=str(i),answer='\\boxed{1}',split='test',question_sha256=str(i)) for i in range(64)]
    manifest={'fixture':'tiny'};tok=TinyTokenizer()
    full=tmp_path/'full';interrupted=tmp_path/'interrupted'
    runner.run(conf(full),train,test,tok,manifest)
    atomic=runner.atomic_torch
    def crash(path,value):
        atomic(path,value)
        if str(path).endswith('checkpoints/latest.pt') and value['policy_step']==1:raise RuntimeError('simulated crash after durable checkpoint')
    monkeypatch.setattr(runner,'atomic_torch',crash)
    with pytest.raises(RuntimeError,match='simulated crash'):runner.run(conf(interrupted),train,test,tok,manifest)
    monkeypatch.setattr(runner,'atomic_torch',atomic)
    runner.run(conf(interrupted),train,test,tok,manifest,resume='auto')
    a=torch.load(full/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(interrupted/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('target_lora','optimizer_target','shadow','rng','sampler'):
        assert digest(a[key])==digest(b[key]),key
    # Reference OPD uses FP32 atomic addition for A feedback. Its rounding can
    # differ across launches; verify all learned tensors/moments numerically,
    # while target, shadow, RNG and accepted response paths remain bitwise equal.
    def close(x,y):
        if torch.is_tensor(x):torch.testing.assert_close(x,y,rtol=3e-6,atol=1e-12)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:close(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for u,v in zip(x,y):close(u,v)
        else:assert x==y
    close(a['reflex'],b['reflex'])
    def numerical_logs(x):
        # CUDA allocation telemetry depends on process allocator history, and
        # analytical A feedback uses the reference FP32 atomic reduction.
        if isinstance(x,dict):return {k:numerical_logs(v) for k,v in x.items()
            if k not in ('gradient_sha256','peak_training_cuda_allocated_bytes')}
        if isinstance(x,list):return [numerical_logs(v) for v in x]
        if isinstance(x,float):return pytest.approx(x,rel=3e-6,abs=1e-12)
        return x
    assert numerical_logs(a['training_rows'])==numerical_logs(b['training_rows'])
    assert json.loads((interrupted/'resume_audit.json').read_text())['state_identity_passed']
    ar=[json.loads(x) for x in (full/'analysis/per_response.jsonl').read_text().splitlines()]
    br=[json.loads(x) for x in (interrupted/'analysis/per_response.jsonl').read_text().splitlines()]
    for x,y in zip(ar,br):
        for k in ('generated_token_ids','accepted_draft_length_sum','verification_rounds','generation_seed'):
            assert x[k]==y[k]
    assert a['policy_step']==2;assert a['completed_analysis_steps']==[1,2]
    assert not a['training_rows'][0]['zero_drift'];assert not a['training_rows'][1]['zero_drift']
    control=json.loads((full/'analysis/boundaries/step_1/zero_drift_control.json').read_text())
    assert control['passed'] and control['target_optimizer_steps']==0
    assert len((interrupted/'analysis/per_response.jsonl').read_text().splitlines())==24
    c=conf(tmp_path/'unscheduled');c.eval_steps=();c.zero_update_steps=()
    def forbidden(*args,**kwargs):raise AssertionError('unscheduled analysis was invoked')
    monkeypatch.setattr(runner,'evaluate',forbidden);monkeypatch.setattr(runner,'relabel',forbidden)
    runner.run(c,train,test,tok,manifest)


@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
@pytest.mark.parametrize('edge',['all_prefill','one_prefill','verified_eos'])
def test_early_eos_history_counters_and_teacher_trace(method,edge,monkeypatch):
    from helper.specualtive_generate import speculative_generate
    from motivation.runtime import generator_kwargs
    from teacher_relabel import TeacherTrace
    import helper.fastgrpo_generate as fast
    import helper.opd_generate as opd
    c=conf('unused');r,s,*_=two_models(c);model=s if method=='fastgrpo' else r
    module=fast if method=='fastgrpo' else opd
    name='sampling' if method=='fastgrpo' else 'sample_target_with_metadata'
    native=getattr(module,name);calls=[]
    def sample(logits,*args,**kwargs):
        calls.append(1)
        if (len(calls)==1 and edge!='verified_eos') or (len(calls)==2 and edge=='verified_eos'):
            logits=logits.clone()
            rows=range(logits.shape[0]) if edge=='all_prefill' else [0]
            for row in rows:
                logits[row,0].fill_(-10000.);logits[row,0,96]=10000.
        return native(logits,*args,**kwargs)
    monkeypatch.setattr(module,name,sample)
    ids=torch.tensor([[0,7,9],[3,5,8]]);mask=torch.tensor([[0,1,1],[1,1,1]])
    trace=TeacherTrace(2)
    with isolated_rng(71),torch.inference_mode():
        out=speculative_generate(model=model,input_ids=ids,attention_mask=mask,tokenizer=TinyTokenizer(),max_length=13,
            **({'teacher_trace':trace} if method=='fastgrpo' else {}),**generator_kwargs(c,method=method,train=True))
    assert len(out['generated_token_ids'])==4
    for tokens,num,den in zip(out['generated_token_ids'],out['response_accepted_length_sum'],out['response_verification_rounds']):
        assert len(tokens)==num+1 and num>=den>=0
    if edge=='all_prefill':
        assert len(calls)==1 and out['verification_batches']==0
        assert out['generated_token_ids']==[[96]]*4
    elif edge=='one_prefill':
        assert out['generated_token_ids'][:2]==[[96],[96]]
        assert out['response_verification_rounds'][:2]==[0,0]
        assert all(x>0 for x in out['response_verification_rounds'][2:])
    if method=='fastgrpo':
        sequences=capture(out,ids,mask,['a','b'],2)
        gate=replay_old_gate(s.target_model,sequences,c,teacher_trace=trace)
        assert gate['passed'] and max(x['hidden_max_abs'] for x in gate['examples'])==0.


def test_resume_after_skipped_reward_group_does_not_invent_policy_step(tmp_path,monkeypatch):
    import motivation.runner as runner
    monkeypatch.setattr(runner,'build_models',two_models)
    calls=[]
    def batch(rows,batch,out,tok,c):
        calls.append(1)
        if len(calls)==1:return [],[[0.,0.]]*len(rows)
        records=[]
        for i,ids in enumerate(out['generated_token_ids']):
            prompt=batch['input_ids'][i//c.responses_per_prompt].tolist()
            records.append(dict(ids=prompt+ids,mask=[0]*(len(prompt)-1)+[1]*(len(ids)+1),
                                advantage=float(2*(i%2)-1),prompt_id=rows[i//c.responses_per_prompt]['id']))
        return records,[[0.,1.]]*len(rows)
    monkeypatch.setattr(runner,'target_batch',batch)
    train=[dict(id=f'train:{i}',question=str(i),answer=r'\boxed{1}',split='train',question_sha256=str(i)) for i in range(8)]
    test=[dict(id=f'test:{i}',question=str(i),answer=r'\boxed{1}',split='test',question_sha256=str(i)) for i in range(64)]
    config=conf(tmp_path,steps=1);config.eval_steps=(1,)
    atomic=runner.atomic_torch
    def crash(path,value):
        atomic(path,value)
        if str(path).endswith('checkpoints/latest.pt') and value['rollout_attempt']==1:raise RuntimeError('skip checkpoint crash')
    monkeypatch.setattr(runner,'atomic_torch',crash)
    with pytest.raises(RuntimeError,match='skip checkpoint crash'):runner.run(config,train,test,TinyTokenizer(),{})
    saved=torch.load(tmp_path/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert saved['policy_step']==0 and saved['rollout_attempt']==1 and not saved['optimizer_target']['state']
    assert saved['completed_analysis_steps']==[]
    monkeypatch.setattr(runner,'atomic_torch',atomic)
    runner.run(config,train,test,TinyTokenizer(),{},resume='auto')
    saved=torch.load(tmp_path/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert saved['policy_step']==1 and saved['rollout_attempt']==2
    assert [x['target_optimizer_steps'] for x in saved['training_rows']]==[0,1]
    assert len((tmp_path/'analysis/per_response.jsonl').read_text().splitlines())==12


def test_load_existing_pretrained_A_preserves_backbone_and_pure_shadow(tmp_path,monkeypatch):
    from transformers import AutoConfig,AutoModelForCausalLM,Qwen2ForCausalLM
    from motivation.runtime import build_models
    config=conf(tmp_path);r,_,*_=two_models(config)
    cfg=r.target_model.get_base_model().config
    checkpoint=tmp_path/'draft.pth';torch.save({'draft_model':cpu_copy(r.draft_model.state_dict())},checkpoint)
    config.draft_checkpoint=str(checkpoint)
    monkeypatch.setattr(AutoConfig,'from_pretrained',lambda *a,**kw:deepcopy(cfg))
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:Qwen2ForCausalLM(deepcopy(cfg)).bfloat16())
    loaded,shadow,*_=build_models(config)
    assert torch.equal(loaded.opd_projector,r.opd_projector)
    assert shadow.opd_projector is None
    for key,value in shadow.draft_model.state_dict().items():assert torch.equal(value,r.draft_model.state_dict()[key])
