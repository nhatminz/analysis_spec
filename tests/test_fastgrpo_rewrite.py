"""Behavioral parity against the frozen ORIGINAL source, using real tiny transformers."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from helper.fastgrpo_model import FastGRPOModel
from helper.specualtive_generate import speculative_generate
from helper.fastgrpo_training import training_draft_model
from helper.opd_sampling import sample_target_with_metadata

ROOT=Path(__file__).resolve().parents[1]

def original(name):
    path=ROOT/'sources/FastGRPO'/name
    spec=importlib.util.spec_from_file_location('original_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if hasattr(module, 'DynamicCache'):
        from helper.transformers_compat import DynamicCache
        module.DynamicCache = DynamicCache
    return module


def tiny(family='qwen2',dtype=torch.bfloat16):
    torch.manual_seed(321)
    cls,model_cls=(Qwen2Config,Qwen2ForCausalLM) if family=='qwen2' else (Qwen3Config,Qwen3ForCausalLM)
    config=cls(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
               num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
               attention_dropout=0.,torch_dtype=dtype)
    config._attn_implementation='sdpa'
    target=model_cls(config).to(device='cuda',dtype=dtype).eval()
    dc=deepcopy(config);dc.num_hidden_layers=1;dc.rope_scaling=None
    model=FastGRPOModel(dc,target).cuda().eval()
    return model


def run(generate, model, **kwargs):
    counts={'target':0,'draft':0};masks=[]
    def th(module,args,kw):counts['target']+=1;masks.append(kw['attention_mask'].clone())
    def dh(*_):counts['draft']+=1
    h=model.target_model.model.layers[0].register_forward_pre_hook(th,with_kwargs=True)
    d=model.draft_model.register_forward_pre_hook(dh)
    torch.manual_seed(715)
    with torch.inference_mode():
        result=generate(model,torch.tensor([[0,7,9],[3,5,8]]),torch.tensor([[0,1,1],[1,1,1]]),
            SimpleNamespace(eos_token_id=96),do_sample=True,repeated_generate_nums=2,
            temperature=.8,top_p=.95,max_length=14,verification_capacity=28,
            max_verification_num=7,max_draft_k=2,max_draft_token_length=3,min_draft_token_length=3,
            return_all_draft_input=True,statistical_time=False,**kwargs)
    rng=torch.cuda.get_rng_state();h.remove();d.remove()
    return result,rng,counts,masks


@pytest.mark.skipif(not torch.cuda.is_available(),reason='original source is CUDA-only')
@pytest.mark.parametrize('family',['qwen2','qwen3'])
def test_baseline_matches_original_tokens_rng_tree_forward_counts_and_history(family):
    model=tiny(family)
    a=run(original('helper/specualtive_generate.py').speculative_generate,model)
    b=run(speculative_generate,model,method='fastgrpo')
    for key in ('generated_token_ids','total_acc_length','total_decoded_token_num','max_sequence_length'):
        assert a[0][key]==b[0][key]
    assert torch.equal(a[1],b[1]);assert a[2]==b[2]
    assert b[0]['verification_batches']==b[2]['target']-1
    for x,y in zip(a[3],b[3]):assert torch.equal(x,y)
    for key in ('all_draft_input_states','all_draft_input_ids'):
        for x,y in zip(a[0][key],b[0][key]):assert torch.equal(x,y)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='original CUDA sampler parity')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('top_p,top_k',[(.95,None),(1.,5),(None,None),(.9,5)])
def test_opd_sampler_matches_source_rng_and_tokens(dtype,top_p,top_k):
    source=original('helper/specualtive_generate.py')
    logits=torch.randn(3,4,97,dtype=dtype,device='cuda')
    logits[0,1]=float('nan')
    torch.manual_seed(34);a=source.sampling(logits,top_k,top_p,.8,96);after=torch.cuda.get_rng_state()
    torch.manual_seed(34);b,_,_=sample_target_with_metadata(logits,do_sample=True,temperature=.8,top_p=top_p,top_k=top_k,eos_token_id=96)
    assert torch.equal(a,b);assert torch.equal(after,torch.cuda.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
@pytest.mark.parametrize('accumulation',[1,2])
def test_online_draft_loss_gradients_and_update_match_source(accumulation):
    s=(ROOT/'sources/FastGRPO/grpo_speculative.py').read_text();tree=ast.parse(s)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_draft_model')
    scope=dict(torch=torch,repeated_generate_nums=2,max_training_token=12,max_training_padding_gap=4,draft_accumulation_steps=accumulation)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'source','exec'),scope)
    model=tiny();other=deepcopy(model)
    gen=torch.Generator(device='cuda').manual_seed(56)
    outputs=dict(all_draft_input_states=[torch.randn(n,32,device='cuda',dtype=torch.bfloat16,generator=gen) for n in [7,9,11,13]],
                 all_draft_input_ids=[torch.randint(0,97,(n,),device='cuda',generator=gen) for n in [7,9,11,13]])
    mask=torch.ones(2,3,dtype=torch.long)
    a=scope['training_draft_model'](model,outputs,mask)
    b=training_draft_model(other,outputs,mask,repeated_generate_nums=2,max_training_token=12,max_training_padding_gap=4,draft_accumulation_steps=accumulation)
    assert a==b
    for x,y in zip(model.draft_model.parameters(),other.draft_model.parameters()):assert torch.equal(x.grad,y.grad)
    for m in (model,other):torch.optim.AdamW(m.draft_model.parameters(),lr=1e-4).step()
    for x,y in zip(model.draft_model.parameters(),other.draft_model.parameters()):assert torch.equal(x,y)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
@pytest.mark.parametrize('stream',[False,True])
def test_opd_real_transformer_rollout_and_training_without_extra_forward(stream):
    model=tiny();model.enable_opd(8)
    out,rng,counts,masks=run(speculative_generate,model,method='opd_reflex',opd_update_stream=stream,opd_train_projector=True)
    assert counts['target']==out['verification_batches']+1
    assert counts['draft']==out['verification_batches']*3
    assert out['opd_selected_states']>0
    assert model._opd_target_kv_pool.get_seq_length()==0
    assert torch.isfinite(model.opd_projector_grad_sum).all()
    # Inference storage is cloned by the trainer before the upstream objective.
    for key in ('all_draft_input_states','all_draft_input_ids'):out[key]=[x.clone() for x in out[key]]
    loss=training_draft_model(model,out,torch.tensor([[0,1,1],[1,1,1]]),repeated_generate_nums=2,
        max_training_token=64,max_training_padding_gap=64,draft_accumulation_steps=1)
    assert all(torch.isfinite(torch.tensor(loss)))
    model.apply_opd_projector_gradient()
    assert torch.isfinite(model.opd_projector.grad).all()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_pretrain_loss_and_gradient_match_original_slices_masks_normalization():
    from train_draft import pretrain_loss
    source=(ROOT/'sources/FastGRPO/train_draft.py').read_text()
    start=source.index('        with torch.no_grad():\n            target_outputs=')
    end=source.index('        if torch.isnan(loss)',start)
    import textwrap
    body=textwrap.dedent(source[start:end])
    wrapper='def reference(model,batch):\n    input_ids=batch["input_ids"].cuda()\n    attention_mask=batch["attention_mask"].cuda()\n    loss_mask=batch["loss_mask"].cuda()\n    l1_loss=torch.nn.SmoothL1Loss(reduction="none")\n'+textwrap.indent(body,'    ')+'    return loss1,loss2\n'
    scope={'torch':torch};exec(wrapper,scope)
    a=tiny();b=deepcopy(a)
    batch=dict(input_ids=torch.tensor([[2,3,4,7,8,0,0],[6,8,3,9,11,12,13]]),
        attention_mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]]),
        loss_mask=torch.tensor([[0,0,1,1,1,0,0],[0,0,0,1,1,1,1]]))
    x=scope['reference'](a,batch);y=pretrain_loss(b,batch)
    for u,v in zip(x,y):assert torch.equal(u,v)
    sum(x).backward();sum(y).backward()
    for u,v in zip(a.draft_model.parameters(),b.draft_model.parameters()):assert torch.equal(u.grad,v.grad)
    for model in (a,b):torch.optim.AdamW(model.parameters(),lr=5e-5).step()
    for u,v in zip(a.draft_model.parameters(),b.draft_model.parameters()):assert torch.equal(u,v)


def test_pretrain_collation_matches_original_sharegpt_mask():
    from helper.pretrain_data import DataCollator
    source=(ROOT/'sources/FastGRPO/train_draft.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='DataCollator')
    scope={'torch':torch,'model_type':'qwen2'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'source-collator','exec'),scope)
    tokenizer=SimpleNamespace(encode=lambda text,**kw:[ord(c)%97 for c in text],eos_token_id=96)
    batch=[{'conversations':[{'from':'human','value':'What is 2+2?'},{'from':'gpt','value':'4'}]},
           {'conversations':[{'from':'human','value':'Short'},{'from':'gpt','value':'Longer answer here'}]}]
    a=scope['DataCollator'](tokenizer,max_length=200)(batch)
    b=DataCollator(tokenizer,max_length=200)(batch)
    for key in a:assert torch.equal(a[key],b[key])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_async_sync_feedback_persistent_pool_reuse_matches_rng_tokens():
    outputs=[]
    for stream in (False,True):
        m=tiny();m.enable_opd(8)
        first=run(speculative_generate,m,method='opd_reflex',opd_update_stream=stream,opd_train_projector=True)
        second=run(speculative_generate,m,method='opd_reflex',opd_update_stream=stream,opd_train_projector=True)
        assert first[0]['generated_token_ids']==second[0]['generated_token_ids']
        assert torch.equal(first[1],second[1])
        assert second[0]['opd_target_pool_allocations']==0
        assert second[0]['opd_draft_pool_allocations']==0
        outputs.append(second)
    assert outputs[0][0]['generated_token_ids']==outputs[1][0]['generated_token_ids']
    assert torch.equal(outputs[0][1],outputs[1][1]);assert outputs[0][2]==outputs[1][2]

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Triton full-vocabulary tests')
@pytest.mark.parametrize('slots',[0,16,1024])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_full_target_vocab_sparse_fused_gemm_and_zero_B(slots,dtype):
    from helper.opd_reflex import OPDReflex
    from test_opd_reflex import seed
    v,h,r=151936,32,8
    head=torch.nn.Linear(h,v,bias=False,device='cuda',dtype=dtype)
    model=SimpleNamespace(lm_head=head,opd_projector=torch.randn(h,r,device='cuda')*.01)
    state=OPDReflex(rank=r);ids=torch.arange(v,device='cuda')
    state.start(model,1,ids,h,max_contexts=2,max_nodes=2,max_path=2,max_proposal_contexts=2)
    active=torch.arange(slots,device='cuda')*31
    seed(state,active,torch.randn(slots,r,device='cuda')*.01)
    hidden=torch.randn(1,2,h,device='cuda',dtype=dtype)
    raw=head(hidden).detach()
    results=[]
    for backend in ('sparse','fused','gemm'):
        state.proposal_mode='sparse' if backend=='sparse' else 'dense'
        state.dense_implementation='gemm' if backend=='gemm' else 'fused'
        values,tokens,_=state.propose(raw,hidden,16,ids)
        results.append((values.clone(),tokens.clone()))
    for a,b in zip(results,results[1:]):
        assert torch.equal(a[0],b[0]);assert torch.equal(a[1],b[1])
    corrected=raw.float()+state.u_cache[:1,:2]@state.B_fast.t()
    if slots==0:assert torch.equal(corrected,raw.float())
    expected=torch.argsort(corrected,descending=True,stable=True)[...,:16]
    assert torch.equal(results[0][1],expected)
    torch.testing.assert_close(results[0][0],corrected.softmax(-1).gather(-1,expected),rtol=3e-5,atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA sampler metadata')
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('top_p,top_k',[(.95,None),(.95,8),(None,None)])
def test_teacher_reads_actual_source_sampler_without_second_sort_softmax(monkeypatch,dtype,top_p,top_k):
    from helper.opd_reflex import OPDReflex
    from helper.tree_verification import PackedTree,VerifiedPath
    from helper import opd_reflex_kernels as kernels
    v,h=97,32
    head=torch.nn.Linear(h,v,bias=False,device='cuda',dtype=dtype)
    model=SimpleNamespace(lm_head=head,opd_projector=torch.randn(h,8,device='cuda')*.01)
    state=OPDReflex();identity=torch.arange(v,device='cuda')
    state.start(model,2,identity,h,max_contexts=1,max_nodes=2,max_path=1,max_proposal_contexts=1)
    hidden=torch.randn(2,1,h,device='cuda',dtype=dtype)
    state.propose(head(hidden),hidden,8,identity)
    root=torch.full((2,1),-1,device='cuda',dtype=torch.long)
    tree=PackedTree(root,root,torch.zeros_like(root),0)
    path=SimpleNamespace(packed_indices=torch.zeros_like(root))
    def forbidden(*args,**kwargs):raise AssertionError('second teacher vocabulary scan')
    if top_p:monkeypatch.setattr(kernels,'teacher',forbidden)
    logits=torch.randn(2,1,v,device='cuda',dtype=dtype)
    def capture(tokens,p,sort):return state.prepare_compact_teacher(tree,path,p,sort)
    _,probs,meta=sample_target_with_metadata(logits,do_sample=True,temperature=.8,top_p=top_p,top_k=top_k,
        eos_token_id=96,metadata_builder=capture)
    p,ids,mass,draft_p=meta
    expected=probs.reshape(2,v).float()
    order=torch.argsort(expected,descending=True,stable=True)[:,:16]
    target_p=expected.gather(-1,order)
    assert torch.equal(ids,torch.where(target_p>0,order,-1))
    torch.testing.assert_close(p,target_p/mass[:,None],rtol=1e-6,atol=1e-7)
    assert torch.equal(draft_p,expected.gather(-1,state.ids_cache[:2,:1].reshape(2,16)))
    # Async feedback receives only O(N*K) owned metadata, never [N,V].
    assert all(x.numel()<=2*16 for x in meta)
