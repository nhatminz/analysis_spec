"""Execution identity, model inspection/dedup, exact discovery and host dispatch."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_profiles import (ProposalProfile,execution_key,profile_filename,validate_profile,
    discover_profile,inspect_draft,context_shapes,active_trials,fingerprint)
from scripts.tune_opd_proposals import parse_args,tune_models,DEFAULT_MODELS

HW=dict(gpu='Test GPU',compute_capability=[10,0],torch='test',triton='test',cuda='test',kernel_sha256='abc')


def payload(key):
    return dict(execution_key=key,records=[dict(contexts=n,trials=[
        dict(slots=0,sparse=1.,fused=5.,gemm=9.),
        dict(slots=key['vocab']//2,sparse=12.,fused=3.,gemm=6.),
        dict(slots=key['vocab'],sparse=22.,fused=8.,gemm=2.)]) for n in (1,37)],
        thresholds={f'1,1,{key["vocab"]},{key["rank"]},{key["dtype"]}':1})


def draft_fixture(root,name,v=41,hidden=16,rank=None,dtype=torch.bfloat16):
    path=root/name;path.mkdir(parents=True)
    (path/'latest_target_config.json').write_text(json.dumps(dict(vocab_size=v,hidden_size=hidden)))
    state={'fs.up_proj.weight':torch.zeros(hidden*2,hidden,dtype=dtype),
           'states_last_norm.weight':torch.ones(hidden,dtype=dtype)}
    if rank is not None:state['opd_projector']=torch.zeros(hidden,rank)
    torch.save(dict(draft_model=state),path/'latest_checkpoint')
    return path


@pytest.mark.parametrize('v',[41,127,521])
@pytest.mark.parametrize('rank',[4,8,16])
def test_host_only_interpolation_varying_contexts_and_active_counts(v,rank,monkeypatch):
    key=execution_key(HW,v,rank,'bf16');p=validate_profile(payload(key),key)
    monkeypatch.setattr(torch.Tensor,'item',lambda *a:pytest.fail('GPU scalar read'))
    for contexts in (1,7,37,139):
        assert p.choose(contexts,0)=='sparse'
        assert p.choose(contexts,v//2)=='fused'
        assert p.choose(contexts,v)=='gemm'
        assert p.choose(contexts,v)==p.choose(contexts,v)
    assert profile_filename(key)!=profile_filename(execution_key(HW,v+1,rank,'bf16'))


def test_interpolates_context_costs_not_global_threshold():
    key=execution_key(HW,101,8,'bf16');p=payload(key)
    p['records'][1]['trials'][1].update(sparse=2.,fused=8.,gemm=9.)
    selector=ProposalProfile(p)
    assert selector.choose(1,50)=='fused'
    assert selector.choose(37,50)=='sparse'
    assert selector.costs(7,50)!=selector.costs(1,50)


@pytest.mark.parametrize('field,value',[('gpu','other'),('compute_capability',[9,0]),('vocab',17),('rank',4),
    ('dtype','torch.float32'),('kernel_sha256','different'),('triton','other'),('topk',8)])
def test_mismatch_never_silently_reused(tmp_path,field,value):
    key=execution_key(HW,41,8,'bf16');p=payload(key)
    path=tmp_path/'arbitrary_model_name.json';path.write_text(json.dumps(p))
    wrong=dict(key,**{field:value})
    assert discover_profile(tmp_path,wrong)==(None,None)
    with pytest.raises(ValueError,match='incompatible'):discover_profile(tmp_path,wrong,str(path))
    selected,_=discover_profile(tmp_path,key)
    assert selected==path


def test_missing_or_corrupt_profile(tmp_path):
    key=execution_key(HW,41,8,'bf16')
    assert discover_profile(tmp_path,key)==(None,None)
    (tmp_path/profile_filename(key)).write_text('{broken')
    assert discover_profile(tmp_path,key)==(None,None)
    with pytest.raises(ValueError):discover_profile(tmp_path,key,str(tmp_path/profile_filename(key)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='startup resolver with actual hardware key')
def test_startup_resolver_finds_shared_profile_without_auto_tuning(tmp_path,monkeypatch,capsys):
    from scripts.resolve_opd_profile import main
    root=draft_fixture(tmp_path,'model');profiles=tmp_path/'profiles';profiles.mkdir()
    key=execution_key(fingerprint(),41,8,'bf16')
    profile=profiles/profile_filename(key);profile.write_text(json.dumps(payload(key)))
    monkeypatch.setattr('sys.argv',['resolve','--target-config',str(root/'latest_target_config.json'),
        '--draft-checkpoint',str(root/'latest_checkpoint'),
        '--profile-dir',str(profiles),'--rank','8','--dtype','bf16'])
    main();captured=capsys.readouterr()
    assert captured.out.strip()==str(profile.resolve())
    assert 'OPD proposal mode: auto' in captured.err and key['kernel_sha256'] in captured.err


@pytest.mark.skipif(not torch.cuda.is_available(),reason='runtime dtype/vocab re-discovery')
def test_runtime_does_not_reuse_old_profile_after_execution_shape_changes(tmp_path,monkeypatch):
    from test_opd_reflex import state
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE','')
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',str(tmp_path))
    for v,dtype in ((41,torch.float32),(41,torch.bfloat16),(47,torch.bfloat16)):
        key=execution_key(fingerprint(),v,8,dtype)
        (tmp_path/profile_filename(key)).write_text(json.dumps(payload(key)))
    s,model,mapping=state('cuda',v=41)
    for v,dtype in ((41,torch.bfloat16),(47,torch.bfloat16)):
        model.lm_head=torch.nn.Linear(32,v,bias=False,dtype=dtype,device='cuda')
        mapping=torch.arange(v,device='cuda')
        s.start(model,3,mapping,32,max_contexts=8,max_nodes=24,max_path=5,max_proposal_contexts=4)
        assert s.tuning['execution_key']['vocab']==v and s.tuning['execution_key']['dtype']==str(dtype)
        assert s.head_cache.dtype==dtype


def test_valid_profile_does_not_use_legacy_global_fallback(monkeypatch):
    from helper.opd_reflex import OPDReflex
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE','')
    s=OPDReflex();s.profile_selector=ProposalProfile(payload(execution_key(HW,101,8,'bf16')))
    s.host_active_count=50
    monkeypatch.setattr(s,'proposal_threshold',lambda *a:pytest.fail('used global threshold despite valid profile'))
    assert s.selected_proposal_backend(7,1)=='dense'
    assert s.selected_dense_implementation(7,1)=='fused'


def test_detects_real_checkpoint_full_vocab_dtype_rank_and_rejects_disagreement(tmp_path):
    root=draft_fixture(tmp_path,'saved',rank=4,dtype=torch.float16)
    args=[root/'latest_target_config.json',root/'latest_checkpoint']
    detected=inspect_draft(*args)
    assert (detected['vocab'],detected['rank'],detected['dtype'])==(41,4,'torch.float16')
    assert inspect_draft(*args,dtype='bf16')['dtype']=='torch.bfloat16' # runtime cast override
    with pytest.raises(ValueError,match='projector rank'):inspect_draft(*args,rank=8)
    (root/'latest_target_config.json').write_text(json.dumps(dict(vocab_size=41,hidden_size=17)))
    with pytest.raises(ValueError,match='hidden size mismatch'):inspect_draft(*args)


def test_multi_model_dedup_hidden_independent_and_reuse_without_retune(tmp_path):
    root=tmp_path/'pretrain'
    for name,hidden in zip(DEFAULT_MODELS[:-1],[16,24,32,40,64]):draft_fixture(root,name,hidden=hidden)
    args=parse_args(['--pretrain-root',str(root),'--profile-dir',str(tmp_path/'profiles'),'--dtype','bf16'])
    calls=[]
    def bench(key,*rest):calls.append(key);return payload(key)
    with pytest.warns(UserWarning,match='skip'):
        result=tune_models(args,hardware=HW,benchmark_fn=bench,progress=False)
    assert result['unique_configs']==1 and len(calls)==1 and len(result['models'])==5
    assert len({m['profile'] for m in result['models']})==1
    with pytest.warns(UserWarning,match='skip'):
        again=tune_models(args,hardware=HW,benchmark_fn=lambda *a:pytest.fail('retuned reused config'),progress=False)
    assert again['unique_configs']==1


def test_different_vocab_and_rank_produce_different_config_groups(tmp_path):
    draft_fixture(tmp_path/'pretrain','a',v=41,rank=4)
    draft_fixture(tmp_path/'pretrain','b',v=41,rank=8)
    draft_fixture(tmp_path/'pretrain','c',v=47,rank=8)
    args=parse_args(['--models','a,b,c','--pretrain-root',str(tmp_path/'pretrain'),'--profile-dir',str(tmp_path/'profiles')])
    calls=[]
    def bench(key,*rest):calls.append(key);return payload(key)
    result=tune_models(args,hardware=HW,benchmark_fn=bench,progress=False)
    assert result['unique_configs']==3 and len(calls)==3


def test_directory_checkpoint_reads_fastgrpo_weights(tmp_path):
    root=draft_fixture(tmp_path,'exported',rank=4)
    weights=root/'weights';weights.mkdir()
    (root/'latest_checkpoint').rename(weights/'draft.pth')
    result=inspect_draft(root/'latest_target_config.json',weights)
    assert result['vocab']==41 and result['rank']==4 and result['dtype']=='torch.bfloat16'


def test_mixed_missing_and_corrupt_models_do_not_abort_valid_config(tmp_path):
    root=tmp_path/'pretrain';draft_fixture(root,'good')
    broken=draft_fixture(root,'broken');(broken/'latest_checkpoint').write_bytes(b'corrupt')
    args=parse_args(['--models','missing,broken,good','--pretrain-root',str(root),'--profile-dir',str(tmp_path/'profiles')])
    with pytest.warns(UserWarning):
        summary=tune_models(args,hardware=HW,benchmark_fn=lambda key,*a:payload(key),progress=False)
    assert len(summary['models'])==1 and len(summary['skipped'])==2


def test_workload_trials_scale_with_actual_config():
    assert context_shapes(3,5,7)[-1]==(105,1)
    assert context_shapes(2,3,5)[-1]==(30,1)
    assert active_trials(41)[-1]==41 and active_trials(139)[-1]==139
    assert parse_args([]).models==','.join(DEFAULT_MODELS)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real proposal tuning smoke')
def test_real_gpu_tuner_deduplicates_two_hidden_sizes_and_validates_results(tmp_path,monkeypatch):
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE','')
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',str(tmp_path/'profiles'))
    for name,h in (('a',16),('b',24)):draft_fixture(tmp_path/'pretrain',name,v=41,hidden=h)
    args=parse_args(['--models','a,b','--pretrain-root',str(tmp_path/'pretrain'),
        '--profile-dir',str(tmp_path/'profiles'),'--shapes','1x1,3x2','--slots','0,16,V','--iterations','2'])
    result=tune_models(args,progress=False)
    assert result['unique_configs']==1 and result['models'][0]['profile']==result['models'][1]['profile']
    p=json.loads(Path(result['models'][0]['profile']).read_text())
    assert p['benchmark_metadata']['hidden_independent']
    assert all(t['bitwise_parity'] for r in p['records'] for t in r['trials'])
    assert p['execution_key']==execution_key(fingerprint(),41,8,'bf16')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU dispatch counters and only-selected launches')
def test_profile_selects_all_three_backends_with_identical_proposals(tmp_path,monkeypatch):
    from helper.opd_reflex import OPD_COUNTER_NAMES
    from test_opd_reflex import state,seed
    from helper import opd_reflex_kernels as kernels
    key=execution_key(fingerprint(),41,8,'fp32')
    p=payload(key);p['thresholds']={'3,1,41,8,torch.float32':1}
    path=tmp_path/'measured_fixture.json';path.write_text(json.dumps(p))
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE',str(path));monkeypatch.setenv('OPD_PROPOSAL_MODE','auto')
    monkeypatch.setenv('OPD_DENSE_IMPLEMENTATION','auto')
    s,_,mapping=state('cuda',v=41)
    raw=torch.randn(3,1,41,device='cuda');hidden=torch.randn(3,1,32,device='cuda')
    original_sparse,original_gemm=kernels._sparse_scores,kernels._dense_gemm
    class Forbid:
        def __getitem__(self,grid):pytest.fail('unselected proposal preparation launched')
    for count,selected in ((0,'sparse'),(20,'fused'),(41,'gemm')):
        seed(s,torch.arange(count,device='cuda'),torch.randn(count,8,device='cuda')*.1)
        s._ever_updated=True
        monkeypatch.setattr(kernels,'_sparse_scores',original_sparse if selected=='sparse' else Forbid())
        monkeypatch.setattr(kernels,'_dense_gemm',original_gemm if selected=='gemm' else Forbid())
        q,ids,_=s.propose(raw,hidden,16,mapping,root=True);reference=(q.clone(),ids.clone())
        monkeypatch.setattr(kernels,'_sparse_scores',original_sparse)
        monkeypatch.setattr(kernels,'_dense_gemm',original_gemm)
        s.proposal_mode='sparse';actual=s.propose(raw,hidden,16,mapping)
        assert all(torch.equal(a,b) for a,b in zip(reference,actual))
        s.proposal_mode='auto'
    counts=s.finish()
    assert counts['opd_proposal_mode_sparse_rounds']==1
    assert counts['opd_proposal_mode_fused_rounds']==1
    assert counts['opd_proposal_mode_gemm_rounds']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA packet/update stream')
def test_synchronous_packet_uses_current_count_async_packet_uses_stable_snapshot():
    from helper import tree_kernels
    from helper.opd_scheduling import schedule
    from helper.tree_verification import VerifiedPath
    path=VerifiedPath(torch.tensor([[4,5]],device='cuda'),torch.tensor([[0,1]],device='cuda'),
                      torch.tensor([[0,1]],device='cuda'),torch.tensor([2],device='cuda'))
    workspace=[torch.empty(1,2,device='cuda',dtype=torch.bool if i==2 else torch.long) for i in range(3)]+[torch.empty(1,1,device='cuda',dtype=torch.long)]
    packet=torch.empty(1,6,device='cuda',dtype=torch.long)
    s=SimpleNamespace(host_sync_count=0,active_count=torch.tensor([9],device='cuda'),dispatch_snapshot=torch.tensor([3],device='cuda'))
    for async_mode,expected in ((False,9),(True,3)):
        s.async_updates=async_mode;schedule(path,0,99,workspace,packet,tree_kernels,s)
        assert s.host_active_count==expected
    assert s.host_sync_count==2
