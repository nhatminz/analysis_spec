import builtins
import json
import tempfile
import unittest
import warnings
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

from policy_lag_analysis import (
    BranchSummary,
    bootstrap_delta_by_prompt,
    dataset_disjointness_report,
    load_completed_results,
    main,
    weighted_aal,
    write_results,
)


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
    def test_split_disjointness_is_recorded_and_source_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.jsonl"
            evaluation = Path(directory) / "eval.jsonl"
            train.write_text(json.dumps({"source_index": 1, "question": "one"}) + "\n")
            evaluation.write_text(json.dumps({"source_index": 2, "question": "two"}) + "\n")
            report = dataset_disjointness_report(train, evaluation)
            self.assertTrue(report["source_disjoint"])
            self.assertEqual(report["source_index_overlap"], 0)
            self.assertIn("next real GRPO rollout", report["evaluation_usage"])
            evaluation.write_text(json.dumps({"source_index": 1, "question": "two"}) + "\n")
            with self.assertRaisesRegex(ValueError, "source indices overlap"):
                dataset_disjointness_report(train, evaluation)

    def test_aal_uses_total_accepted_over_total_verification_rounds(self):
        records = [
            {"prompt_id": "p0", "accepted_length_sum": 5, "verification_rounds": 2},
            {"prompt_id": "p1", "accepted_length_sum": 3, "verification_rounds": 2},
        ]
        self.assertEqual(weighted_aal(records), (2.0, 8, 4, 0))

    def test_bootstrap_rejects_unpaired_rollouts(self):
        stale = [
            {"prompt_id": "p0", "response_index": 0, "accepted_length_sum": 2, "verification_rounds": 1},
            {"prompt_id": "p1", "response_index": 0, "accepted_length_sum": 2, "verification_rounds": 1},
        ]
        fresh = [
            {"prompt_id": "p0", "response_index": 0, "accepted_length_sum": 3, "verification_rounds": 1},
            {"prompt_id": "p2", "response_index": 0, "accepted_length_sum": 3, "verification_rounds": 1},
        ]
        with self.assertRaisesRegex(ValueError, "prompt sets differ"):
            bootstrap_delta_by_prompt(stale, fresh, seed=1, samples=10)
        fresh[1]["prompt_id"] = "p1"
        fresh[1]["response_index"] = 1
        with self.assertRaisesRegex(ValueError, "response indices differ"):
            bootstrap_delta_by_prompt(stale, fresh, seed=1, samples=10)

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

    def test_v3_resume_rejects_incomplete_completed_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "protocol.json").write_text(
                json.dumps({"format": "fastgrpo_policy_lag_protocol_v3"}), encoding="utf-8"
            )
            marker = output / "boundaries" / "step_1" / "complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"policy_step": 1}), encoding="utf-8")
            (output / "per_response.jsonl").write_text(
                json.dumps({"policy_step": 1, "branch": "stale"}) + "\n", encoding="utf-8"
            )
            (output / "summary.jsonl").write_text(
                json.dumps(summary(1).__dict__) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "missing paired"):
                load_completed_results(output)

    def test_v3_resume_preserves_paired_completed_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "protocol.json").write_text(
                json.dumps({"format": "fastgrpo_policy_lag_protocol_v3"}), encoding="utf-8"
            )
            marker = output / "boundaries" / "step_1" / "complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"policy_step": 1}), encoding="utf-8")
            branches = ("base", "stale", "fresh")
            (output / "per_response.jsonl").write_text(
                "".join(json.dumps({"policy_step": 1, "branch": branch, "prompt_id": 0}) + "\n"
                        for branch in branches),
                encoding="utf-8",
            )
            (output / "summary.jsonl").write_text(
                "".join(json.dumps(replace(summary(1), branch=branch).__dict__) + "\n"
                        for branch in branches),
                encoding="utf-8",
            )
            completed, responses, summaries = load_completed_results(output)
            self.assertEqual(completed, {1})
            self.assertEqual({row["branch"] for row in responses}, set(branches))
            self.assertEqual({row.branch for row in summaries}, set(branches))

    def test_plot_mode_uses_only_completed_results_without_dependency_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "boundaries" / "step_1" / "complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"policy_step": 1}), encoding="utf-8")
            (output / "summary.jsonl").write_text(
                json.dumps(summary(1).__dict__) + "\n" +
                json.dumps(summary(5).__dict__) + "\n",
                encoding="utf-8",
            )
            (output / "per_response.jsonl").write_text(
                json.dumps({"policy_step": 1, "branch": "stale", "prompt_id": "p0"}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(sys, "argv", ["policy_lag_analysis.py", "--mode", "plot", "--output-dir", str(output)]), \
                 mock.patch("policy_lag_analysis.dependency_report", side_effect=AssertionError("should not validate dependencies")), \
                 mock.patch("policy_lag_analysis.plot_results", return_value={"status": "generated", "path": "plot.png"}) as plot:
                main()
            self.assertEqual([row["policy_step"] for row in plot.call_args.args[0]], [1])
            self.assertEqual(json.loads((output / "plot_status.json").read_text())["status"], "generated")


if __name__ == "__main__":
    unittest.main()
