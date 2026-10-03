import re
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


if __name__ == '__main__':
    unittest.main()
