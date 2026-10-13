import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from rq2.aal_analysis import bin_rounds,paired_summary
from rq2.aal_progress import AALConfig,checkpoint_preflight,training_metadata,validate_counters
from motivation.state import digest,atomic_json,file_hash


def rounds(active=(8,8,4,2,1),accepted=(8,24,8,8,2)):
    return [dict(round_index=i,active_batch_size=a,active_response_rounds=a,accepted_length=n)
            for i,(a,n) in enumerate(zip(active,accepted))]


def test_round_bin_AAL_ratio_of_sums_and_exact_midpoint_indexing():
    bins=bin_rounds(rounds(),0,'off')
    assert [b['verification_rounds'] for b in bins]==[1]*5
    assert [b['AAL'] for b in bins]==[1,3,2,4,2]
    data=rounds((8,4,4,2,2,1),(8,12,8,8,4,2))
    bins=bin_rounds(data,0,'off')
    assert [b['verification_rounds'] for b in bins]==[1,1,2,1,1]
    assert bins[2]['AAL']==pytest.approx(16/6)
    assert bins[2]['AAL']!=(2+4)/2
    assert sum(b['accepted_length'] for b in bins)==42
    assert sum(b['active_response_rounds'] for b in bins)==21


def test_zero_and_short_rollouts_never_invent_missing_bins():
    bins=bin_rounds([],0,'off')
    assert all(b['AAL'] is None and b['verification_rounds']==0 for b in bins)
    bins=bin_rounds(rounds((2,),(4,)),0,'off')
    assert [b['AAL'] for b in bins]==[None,None,2,None,None]
    bad=rounds();bad[1]['round_index']=3
    with pytest.raises(ValueError,match='Noncontiguous'):bin_rounds(bad,0,'off')
    bad=rounds();bad[0]['accepted_length']=float('nan')
    with pytest.raises(ValueError,match='counter'):bin_rounds(bad,0,'off')


def test_bootstrap_is_paired_by_rollout_and_reproducible():
    rows=[]
    for rollout,aal in enumerate((1,2,4,8)):
        for arm,delta in (('off',0),('on',1)):
            rows.extend(bin_rounds(rounds((2,)*5,(2*(aal+delta),)*5),rollout,arm))
    frame=pd.DataFrame(rows)
    first=paired_summary(frame,[0,1,2,3],2000,2026)
    assert first==paired_summary(frame,[0,1,2,3],2000,2026)
    for row in first:
        assert row['off_AAL_mean']==3.75 and row['on_AAL_mean']==4.75
        assert row['paired_ON_minus_OFF']==1
        assert row['paired_difference_ci95_low']==row['paired_difference_ci95_high']==1
        assert row['off_ci95_low']<row['off_ci95_high']


def test_missing_measurements_have_common_paired_support_and_no_one_cluster_CI():
    rows=[]
    for rollout in (0,1):
        for arm in ('off','on'):
            rows.extend(bin_rounds([] if rollout==1 and arm=='on' else rounds(),rollout,arm))
    summary=paired_summary(pd.DataFrame(rows),[0,1],100,11)
    assert all(r['paired_rollouts_valid']==1 and r['paired_rollouts_missing']==[1] for r in summary)
    assert all(r['off_ci95_low'] is None and r['paired_difference_ci95_high'] is None for r in summary)


def test_acceptance_log_counter_consistency():
    observer=SimpleNamespace(rounds=rounds(),validate=lambda output:True)
    out=dict(total_acc_length=50,total_decoded_token_num=23,verification_batches=5,
        response_verification_rounds=[23],active_response_rounds=23,response_accepted_length_sum=[50])
    assert validate_counters(observer,out)['overall_AAL']==pytest.approx(50/23)
    out['total_decoded_token_num']=5
    with pytest.raises(ValueError,match='denominator'):validate_counters(observer,out)


def test_training_metadata_is_checkpoint_authoritative_and_read_only(tmp_path):
    cp=tmp_path/'run/checkpoints/resume/step.pt';cp.parent.mkdir(parents=True)
    torch.save(dict(format='opd_fastgrpo_checkpoint_v5',step=234,method='opd_reflex',epoch=1),cp)
    sibling=cp.parent.parent.parent/'config_resolved.yaml'
    sibling.write_text('opd_b_sharing_scope: "per_prompt_group"\n')
    before=file_hash(cp);meta=training_metadata(cp)
    assert meta['training_B_scope']=='rollout_batch' and meta['sibling_scope_disagreement']
    assert 'not fully per-prompt-trained' in meta['training_provenance_limitation']
    assert file_hash(cp)==before


def test_output_scope_refuses_SpecNaacl_and_existing_analysis_roots():
    from rq2.aal_progress import ROOT,validate_output_directory
    assert validate_output_directory(ROOT/'outputs/rq2_aal_step234')==ROOT/'outputs/rq2_aal_step234'
    for path in (ROOT.parent/'SpecNaacl/outputs/rq2_aal_step234',ROOT/'outputs/e2_e6_fastgrpo',ROOT/'outputs/rq2_aal_x/child'):
        with pytest.raises(ValueError,match='separate'):validate_output_directory(path)


def test_native_resume_preflight_restores_CPU_weights_A_without_optimizer_or_GPU(tmp_path,monkeypatch):
    from copy import deepcopy
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from helper.modeling_draft import DraftModel
    cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,max_position_embeddings=128)
    model_path=tmp_path/'model';cfg.save_pretrained(model_path)
    target=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(task_type='CAUSAL_LM',r=64,lora_alpha=32,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    dc=deepcopy(cfg);dc.num_hidden_layers=1;dc.rope_scaling=None
    draft=DraftModel(dc);a=torch.ones(32,8)
    cp=tmp_path/'step234.pt'
    payload=dict(format='opd_fastgrpo_checkpoint_v5',step=234,method='opd_reflex',
        target_lora=get_peft_model_state_dict(target),draft_model={**draft.state_dict(),'opd_projector':a},
        opd_projector=a,opd_projector_initial=torch.zeros_like(a),optimizer_draft={'state':{}})
    torch.save(payload,cp);before=file_hash(cp)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:pytest.fail('GPU called during CPU preflight'))
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *args,**kwargs:pytest.fail('Optimizer was constructed'))
    loaded,preflight=checkpoint_preflight(AALConfig(checkpoint_path=str(cp),model=str(model_path)))
    assert preflight['target_lora_loaded'] and preflight['draft_loaded'] and preflight['learned_A_validated']
    assert not preflight['optimizer_resumed']
    assert loaded['provenance']['A_sha256']==digest(a) and file_hash(cp)==before
    with pytest.raises(ValueError,match='counter'):checkpoint_preflight(AALConfig(checkpoint_path=str(cp),checkpoint_step=235,model=str(model_path)))


def test_saved_analysis_integrity_outputs_and_no_overwrite_of_other_experiments(tmp_path):
    import pyarrow as pa
    from e2e6.collect import atomic_parquet,EVENT_SCHEMA
    from rq2.aal_analysis import analyze
    from e2e6.events import VerificationLog
    output=tmp_path/'rq2_aal_fixture';output.mkdir()
    manifest=dict(identity_sha256='fixture',config=dict(checkpoint_step=234,num_rollouts=2),
        seeds=[1,2],checkpoint=dict(projector=dict(A_sha256='own-A')),training_metadata=dict(training_B_scope='rollout_batch'))
    atomic_json(output/'manifest.json',manifest)
    for rollout in (0,1):
        for arm in ('off','on'):
            directory=output/'collection'/arm/f'rollout_{rollout:04d}'
            # One zero-verification response (prefill EOS), explicitly empty.
            event=dict(checkpoint_step=234,rollout_id=rollout,response_id=0,event_order=0,token_position=0,
                       target_token=99,reason='prefill',is_miss=None)
            atomic_parquet(directory/'events.parquet',[event],EVENT_SCHEMA)
            atomic_parquet(directory/'rounds.parquet',[],pa.schema([('round_index',pa.int64())]))
            counter=dict(total_acc_length=0,total_decoded_token_num=0,response_accepted_length_sum=[0],response_verification_rounds=[0])
            atomic_json(directory/'responses.json',dict(tokens=[[99]],eos_token_id=99,counters=counter,
                seed=rollout+1,prompt_ids=[f'p{rollout}'],frozen_weights_sha256='fixed',learned_A_sha256='own-A'))
            atomic_json(directory/'diagnostics.json',dict(feedback_valid=True,sharing_scope='per_prompt_group',
                B_reset_verified=True,frozen_weights_verified=True,opd_nonfinite_kl_states=0,
                group_b_norms=[0.],group_updates=[0],acceptance=dict(accepted_length=0,active_response_rounds=0,verification_rounds=0,overall_AAL=None)))
            atomic_json(directory/'complete.json',dict(identity_sha256='fixture',arm=arm,rollout_id=rollout,
                checkpoint_step=234,files={name:file_hash(directory/name) for name in ('events.parquet','rounds.parquet','responses.json','diagnostics.json')}))
    result=analyze(output,100,2026)
    assert result['paired_rollouts']==2
    for name in ('aal_by_rollout.csv','aal_summary.csv','aal_during_rollout.pdf','aal_during_rollout.png','active_responses.pdf','validation.json'):
        assert (output/name).is_file()
    summary=json.loads((output/'validation.json').read_text())['summary']
    assert all(row['off_AAL_mean'] is None for row in summary)
    assert all(row['paired_rollouts_valid']==0 for row in summary)
