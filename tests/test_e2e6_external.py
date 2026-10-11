from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch
from e2e6.config import Config
from e2e6.external import MODULES,source_info,validate_adapter,read_pair,materialize
from e2e6.checkpoints import select,verify
from motivation.state import digest,file_hash


def source_fixture(path,step=235):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model
    from helper.modeling_draft import DraftModel
    model_path=path/'base';model_path.mkdir(parents=True)
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    cfg.save_pretrained(model_path)
    target=Qwen2ForCausalLM(cfg);target.name_or_path=str(model_path)
    target.save_pretrained(model_path)
    target=get_peft_model(target,LoraConfig(task_type='CAUSAL_LM',r=64,lora_alpha=32,lora_dropout=0.,target_modules=sorted(MODULES)))
    run=path/'source';adapter=run/'checkpoints/target'/f'step{step}';target.save_pretrained(adapter)
    dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=torch.float32
    draft=DraftModel(dc).state_dict()
    (run/'checkpoints/draft').mkdir(parents=True)
    torch.save(dict(draft_model=draft),run/'checkpoints/draft'/f'step{step}.pth')
    metadata=dict(method='fastgrpo',model=str(model_path),temperature='1.0',top_p='0.95',
        max_length='17',batch_size='2',responses_per_prompt='2')
    (run/'config_resolved.yaml').write_text(''.join(f'{k}: {json.dumps(v)}\n' for k,v in metadata.items()))
    (run/'summary.json').write_text(json.dumps(dict(method='fastgrpo',final_step=step,target_optimizer_steps=901)))
    return run,model_path,cfg


def test_source_pair_paths_and_label_semantics_and_pure_method(tmp_path):
    run,model_path,_=source_fixture(tmp_path)
    info=source_info(run,235)
    assert info['actual_target_optimizer_steps']==901
    assert info['checkpoint_label']==235 and 'NOT assumed' in info['checkpoint_label_semantics']
    assert Path(info['draft']).name=='step235.pth'
    assert Path(info['target']).name=='step235'
    # Accept ordinary unquoted YAML as well as SpecNaacl's JSON-valued YAML.
    (run/'config_resolved.yaml').write_text(f'method: fastgrpo\nmodel: {model_path}\n')
    assert source_info(run,235)['metadata']['method']=='fastgrpo'
    with pytest.raises(FileNotFoundError):source_info(run,234)
    (run/'summary.json').write_text(json.dumps(dict(method='opd_reflex')))
    with pytest.raises(ValueError,match='pure fastgrpo'):source_info(run,235)


def test_adapter_validation_rejects_wrong_base_and_incompatible_lora(tmp_path):
    run,model_path,_=source_fixture(tmp_path)
    config=json.loads((run/'checkpoints/target/step235/adapter_config.json').read_text())
    validate_adapter(config,model_path)
    for key,value in [('r',8),('target_modules',['q_proj']),('base_model_name_or_path','other-model'),('use_dora',True)]:
        bad=deepcopy(config);bad[key]=value
        with pytest.raises(ValueError):validate_adapter(bad,model_path)


def test_real_peft_export_import_snapshot_and_immutability(tmp_path,monkeypatch):
    # Strict production Qwen3B identity is tested elsewhere. This test substitutes
    # only that identity gate to exercise real tiny checkpoint I/O and shapes.
    import motivation.compatibility as compatibility
    monkeypatch.setattr(compatibility,'validate_model_config',lambda config:None)
    run,model_path,_=source_fixture(tmp_path)
    info=source_info(run,235);source_hashes=info['files_sha256']
    draft,adapter,architecture=read_pair(info,Config(model=str(model_path)))
    assert not architecture['projector_pretrained']
    output=tmp_path/'evaluation';manifest=dict(source=info,analysis_only=True)
    snapshot=materialize(output,235,manifest,draft,adapter)
    stamp=snapshot.stat().st_mtime_ns;before=file_hash(snapshot)
    assert select(output/'checkpoints',[235])[0]==[snapshot]
    assert verify(snapshot)['analysis_only']
    assert not (output/'checkpoints/latest.json').exists()
    materialize(output,235,manifest,draft,adapter)
    assert snapshot.stat().st_mtime_ns==stamp and file_hash(snapshot)==before
    # Recover interruption between atomic tensor write and sidecar publication.
    snapshot.with_suffix('.json').unlink()
    materialize(output,235,manifest,draft,adapter)
    assert file_hash(snapshot)==before
    assert source_info(run,235)['files_sha256']==source_hashes
    changed=deepcopy(adapter);next(iter(changed.values())).add_(1)
    with pytest.raises(ValueError,match='weights differ'):materialize(output,235,manifest,draft,changed)
    with pytest.raises(ValueError,match='identity differs'):materialize(output,235,dict(source=info,changed=True),draft,adapter)


def test_external_import_refuses_active_training_directory(tmp_path):
    (tmp_path/'train_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='active training'):
        materialize(tmp_path,235,{},dict(x=torch.tensor([1.])),dict(x=torch.tensor([2.])))


def test_external_cli_separate_output_and_no_training(monkeypatch):
    from run_e2_e6_external import parser,configuration
    from run_e2_e6 import ROOT
    monkeypatch.setenv('OUTPUT_DIR',str(ROOT/'outputs/active_training'))
    c=configuration(parser().parse_args(['collect','--source-run','/existing/run']))
    assert c.output_dir==str(ROOT/'outputs/e2_e6_external_step235')
    with pytest.raises(ValueError,match='no training'):configuration(parser().parse_args(['train']))
    with pytest.raises(ValueError,match='inside analysis_spec'):
        configuration(parser().parse_args(['collect','--output-dir','/tmp/external-output']))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA FastGRPO generation')
def test_external_step235_frozen_collect_and_offline_cli(tmp_path,monkeypatch):
    from helper.fastgrpo_model import FastGRPOModel
    from transformers import Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model
    from motivation.runtime import load_target
    from e2e6.collect import collect
    from run_e2_e6_external import main
    from test_motivation_cuda import TinyTokenizer
    import motivation.compatibility as compatibility
    monkeypatch.setattr(compatibility,'validate_model_config',lambda config:None)
    run,model_path,cfg=source_fixture(tmp_path)
    info=source_info(run,235)
    draft,adapter,_=read_pair(info,Config(model=str(model_path)))
    c=Config(output_dir=str(tmp_path/'evaluation'),smoke=True,model=str(model_path),batch_size=2,
        responses_per_prompt=2,eval_rollouts=1,max_length=17,verification_capacity=28,
        max_verification_num=7,max_draft_k=2,max_draft_token_length=3,min_draft_token_length=3,bootstrap_samples=100)
    manifest=dict(format='pure_fastgrpo_external_evaluation_v1',analysis_only=True,source=info,config=c.to_dict())
    materialize(c.output_dir,235,manifest,draft,adapter)
    loads=[]
    def factory(config,**kwargs):
        loads.append(1)
        target=Qwen2ForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval()
        dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=torch.bfloat16
        model=FastGRPOModel(dc,target).cuda()
        model.target_model=get_peft_model(target,LoraConfig(task_type='CAUSAL_LM',r=64,
            lora_alpha=32,lora_dropout=0.,target_modules=sorted(MODULES)))
        load_target(model.target_model,kwargs['target_lora'])
        model.draft_model.load_state_dict(kwargs['draft_state'])
        model.eval().requires_grad_(False)
        return model
    rows=[dict(id=f'test:{i}',split='test',question='q',question_sha256=f'h{i}') for i in range(2)]
    result=collect(c,rows,TinyTokenizer(),manifest,[235],model_factory=factory)
    assert result['completed']==[235] and loads==[1]
    collect(c,rows,TinyTokenizer(),manifest,[235],model_factory=factory)
    assert loads==[1]
    assert main(['analyze','--source-step','235','--output-dir',c.output_dir,'--bootstrap-samples','100'])==0
    report=(Path(c.output_dir)/'analysis_e2_e6/REPORT.md').read_text()
    assert 'not counts of target optimizer updates' in report and '901' in report
    assert main(['check','--source-step','235','--output-dir',c.output_dir])==0
    assert source_info(run,235)==info
