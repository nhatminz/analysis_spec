import builtins
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from policy_lag_analysis import BranchSummary, load_completed_results, write_results


def summary(policy_step: int) -> BranchSummary:
    return BranchSummary(
        policy_step=policy_step,
        seed=11,
        branch="stale",
        aal=2.0,
        delta_aal=None,
        verification_rounds=2,
        generated_tokens=4,
        actual_training_token_count=8,
        optimizer_steps=1,
        policy_checkpoint_id="policy",
        draft_checkpoint_id="draft",
        feature_policy_version="policy",
        teacher_shift_tv=0.0,
    )


class PolicyLagExportTests(unittest.TestCase):
    def test_missing_matplotlib_does_not_abort_data_export(self):
        real_import = builtins.__import__

        def without_matplotlib(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch("builtins.__import__", side_effect=without_matplotlib):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    status = write_results(
                        output,
                        [{"policy_step": 1, "prompt_id": "p0"}],
                        [summary(1)],
                    )

            self.assertEqual(status["status"], "skipped")
            self.assertTrue((output / "per_response.jsonl").is_file())
            self.assertTrue((output / "summary.jsonl").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            self.assertEqual(
                json.loads((output / "plot_status.json").read_text())["status"],
                "skipped",
            )
            self.assertTrue(any("JSONL/CSV results were exported" in str(item.message) for item in caught))

    def test_resume_ignores_rows_without_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "boundaries" / "step_1" / "complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"policy_step": 1}), encoding="utf-8")
            response_rows = [
                {"policy_step": 1, "prompt_id": "complete"},
                {"policy_step": 5, "prompt_id": "interrupted"},
            ]
            (output / "per_response.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in response_rows),
                encoding="utf-8",
            )
            summary_rows = [summary(1).__dict__, summary(5).__dict__]
            (output / "summary.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in summary_rows),
                encoding="utf-8",
            )

            completed, responses, summaries = load_completed_results(output)

            self.assertEqual(completed, {1})
            self.assertEqual([row["policy_step"] for row in responses], [1])
            self.assertEqual([row.policy_step for row in summaries], [1])


if __name__ == "__main__":
    unittest.main()
