import re
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherContractTests(unittest.TestCase):
    def test_generic_launcher_only_uses_parser_flags(self):
        launcher = (ROOT / 'run_policy_lag_analysis.sh').read_text()
        source = (ROOT / 'grpo_speculative.py').read_text()
        command = launcher.split('cmd=(\n', 1)[1].split('\n)', 1)[0]
        launcher_flags = set(re.findall(r'(?m)^\s+(--[a-z_]+)\b', command))
        parser_flags = set(re.findall(r"parser\.add_argument\('(--[a-z_]+)'", source))
        self.assertFalse(launcher_flags - parser_flags, launcher_flags - parser_flags)
        self.assertIn('--analysis_eval_prompts', launcher_flags)
        self.assertIn('--analysis_eval_batch_size', launcher_flags)
        self.assertNotIn('--analysis_seeds', launcher_flags)

    def test_draft_budget_does_not_raise_target_packing_limit(self):
        launcher = (ROOT / 'run_policy_lag_analysis.sh').read_text()
        self.assertIn('DRAFT_TOKEN_BUDGET="${DRAFT_TOKEN_BUDGET:-8192}"', launcher)
        self.assertIn('TARGET_MAX_TRAINING_TOKEN="${TARGET_MAX_TRAINING_TOKEN:-1024}"', launcher)
        self.assertIn('--max_training_token "$TARGET_MAX_TRAINING_TOKEN"', launcher)
        self.assertIn('--analysis_training_token_budget "$DRAFT_TOKEN_BUDGET"', launcher)
        self.assertIn('DRAFT_LR="${DRAFT_LR:-1e-5}"', launcher)
        self.assertIn('ANALYSIS_EVAL_PROMPTS="${ANALYSIS_EVAL_PROMPTS:-8}"', launcher)

    def test_b200_default_assets_schedule_and_reflex_flags(self):
        launcher = (ROOT / 'run_policy_lag_analysis_b200.sh').read_text()
        for name in ('latest_checkpoint', 'latest_draft_config.json', 'latest_vocab_mapping.pt'):
            self.assertIn(f'$SPECNAACL_DIR/outputs/pretrain/qwen25_3b/{name}', launcher)
        self.assertIn('ANALYSIS_BOUNDARIES="${ANALYSIS_BOUNDARIES:-1}"', launcher)
        self.assertIn('ANALYSIS_INTERVAL="${ANALYSIS_INTERVAL:-5}"', launcher)
        self.assertIn('TOTAL_POLICY_STEPS="${TOTAL_POLICY_STEPS:-600}"', launcher)
        for setting in ('REFLEX_BACKEND:-triton', 'REFLEX_FEEDBACK_SCOPE:-root',
                        'REFLEX_PROPOSAL_STRATEGY:-fused', 'REFLEX_CORRECTION_STRATEGY:-serial',
                        'REFLEX_FEEDBACK_STRATEGY:-serial', 'REFLEX_FEATURE_STRATEGY:-auto',
                        'REFLEX_UPDATE_STREAM:-1', 'REFLEX_FEATURE_DIM:-8', 'REFLEX_LR:-0.05'):
            self.assertIn(setting, launcher)

    def test_b200_dry_run_respects_overrides_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, 'DRY_RUN': 'true', 'OUTPUT_DIR': directory,
                   'TRAIN_DATASET_PATH': '/test/train.jsonl', 'EVAL_DATASET_PATH': '/test/eval.jsonl',
                   'TARGET_MODEL_PATH': '/test/target', 'DRAFT_CHECKPOINT': '/test/draft',
                   'DRAFT_CONFIG': '/test/draft.json', 'VOCAB_MAPPING': '/test/mapping.pt',
                   'ANALYSIS_BOUNDARIES': '1', 'ANALYSIS_INTERVAL': '5', 'TOTAL_POLICY_STEPS': '600',
                   'SMOKE_TEST': 'false', 'DRAFT_TOKEN_BUDGET': '8192', 'TARGET_MAX_TRAINING_TOKEN': '1024'}
            result = subprocess.run(['bash', str(ROOT / 'run_policy_lag_analysis_b200.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for expected in ('--model_dir /test/target', '--adapter_path /test/draft',
                             '--max_grpo_steps 600', '--analysis_interval 5', '--reflex_backend triton',
                             '--reflex_update_stream 1', '--analysis_training_token_budget 8192',
                             '--max_training_token 1024'):
                self.assertIn(expected, result.stdout)

    def test_no_runtime_sibling_python_imports(self):
        for path in [ROOT / 'grpo_speculative.py', *ROOT.glob('helper/*.py')]:
            self.assertNotRegex(path.read_text(), r'(?m)^\s*(?:from|import)\s+SpecNaacl\b')


if __name__ == '__main__':
    unittest.main()
