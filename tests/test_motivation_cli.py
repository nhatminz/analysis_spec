from pathlib import Path
import os
import subprocess
import sys
import pytest
from run_policy_lag_motivation import main,resolve_paths
from motivation.config import Config

ROOT=Path(__file__).resolve().parents[1]


def test_help_is_standalone_and_legacy_settings_rejected():
    result=subprocess.run([sys.executable,str(ROOT/'run_policy_lag_motivation.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0;assert '--draft-checkpoint' in result.stdout
    for variable,value in [('DATASET','dapo'),('REFLEX_MODE','active')]:
        env=dict(os.environ,**{variable:value})
        result=subprocess.run(['bash',str(ROOT/'run_policy_lag_motivation.sh')],env=env,capture_output=True,text=True)
        assert result.returncode==2;assert 'ERROR' in result.stderr


def test_local_only_paths_never_fall_back_to_other_model(tmp_path):
    c=Config(model=str(tmp_path/'missing_3b'),draft_checkpoint=str(tmp_path/'missing_draft'),
             train_path=str(tmp_path/'train.parquet'),test_path=str(tmp_path/'test.parquet'))
    resolved=resolve_paths(c)
    assert resolved.model.endswith('missing_3b')
    assert resolved.draft_checkpoint.endswith('missing_draft')


def test_launcher_shell_syntax():
    for path in ROOT.glob('*.sh'):
        assert subprocess.run(['bash','-n',str(path)]).returncode==0


def test_zero_control_is_preregistered_measured_boundary():
    assert Config().zero_update_steps==(20,)
    with pytest.raises(ValueError,match='measured boundaries'):
        Config(zero_update_steps=(21,)).validate()


def test_validate_manifest_can_be_reopened_and_detects_config_change(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    import motivation.compatibility as compatibility
    import motivation.data as data
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k: object())
    monkeypatch.setattr(compatibility,'model_identity',lambda path: dict(path=path,config={}))
    monkeypatch.setattr(compatibility,'validate_tokenizer',lambda *a: None)
    monkeypatch.setattr(compatibility,'validate_draft',lambda path,*a: dict(path=path))
    monkeypatch.setattr(compatibility,'tokenizer_identity',lambda *a: {})
    monkeypatch.setattr(data,'prepare',lambda *a: (list(range(8)),list(range(64)),{}))
    args=['--mode','validate','--model',str(tmp_path/'model'),
          '--draft-checkpoint',str(tmp_path/'draft.pth'),
          '--train-path',str(tmp_path/'train.parquet'),'--test-path',str(tmp_path/'test.parquet'),
          '--output-dir',str(tmp_path/'output')]
    assert main(args)==0
    assert main(args)==0
    with pytest.raises(ValueError,match='manifest differs'):
        main(args+['--seed','43'])
