import csv
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from policy_lag_analysis import (
    AnalysisIO, BranchSummary, PROTOCOL_VERSION, STEP_METRIC_COLUMNS,
    append_boundary_results, atomic_json, bootstrap_delta_by_prompt,
    cleanup_branch_checkpoints, load_completed_results, mark_durable_boundaries,
    plot_results, weighted_aal, write_results,
)


def boundary_fixture(step=1):
    raw, summaries = [], []
    branches = {'stale': [(9, 3), (1, 1)], 'fresh': [(8, 2), (4, 2)], 'reflex': [(7, 2), (5, 1)]}
    for branch, values in branches.items():
        rows = [dict(policy_step=step, prompt_id=index, response_index=0, branch=branch,
                     accepted_length_sum=accepted, verification_rounds=rounds,
                     generated_tokens=accepted, initial_rng_id='paired', used_for_grpo=branch == 'stale')
                for index, (accepted, rounds) in enumerate(values)]
        raw.extend(rows)
        aal, accepted, rounds, generated = weighted_aal(rows)
        summaries.append(BranchSummary(step, 42, branch, aal, None, rounds, generated, 8192, 1,
            'theta_t_plus_1', 'fresh' if branch == 'fresh' else 'stale', 'theta_t', 0.01,
            accepted_length_sum=accepted, requested_training_token_budget=8192,
            effective_draft_lr=1e-5, online_draft_update_gpu_ms=12.0,
            reflex_update_gpu_ms=4.0 if branch == 'reflex' else None,
            reflex_update_count=3 if branch == 'reflex' else None,
            reflex_update_gpu_ms_per_update=4.0 / 3 if branch == 'reflex' else None,
            reflex_update_exposed_ms=0.5 if branch == 'reflex' else None))
    for index in (1, 2):
        branch = summaries[index].branch
        boot = bootstrap_delta_by_prompt([row for row in raw if row['branch'] == 'stale'],
            [row for row in raw if row['branch'] == branch], seed=42, samples=20)
        summaries[index] = replace(summaries[index], delta_aal=boot['delta_aal'],
                                 ci_low=boot['delta_aal_ci_low'], ci_high=boot['delta_aal_ci_high'])
    return raw, summaries


class V5ExportTests(unittest.TestCase):
    def test_append_exact_metrics_both_cis_and_no_plot_per_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, summaries = boundary_fixture()
            with mock.patch('policy_lag_analysis.plot_results', side_effect=AssertionError('plot in loop')):
                metrics = append_boundary_results(root, raw, summaries,
                    dict(policy_step=1, next_target_optimizer_step=2, analysis_io_wall_ms=7.0,
                         durable_target_checkpoint=False), io_meter=AnalysisIO())
            self.assertEqual(list(metrics), list(STEP_METRIC_COLUMNS))
            self.assertEqual(metrics['stale_aal'], 10 / 4)  # NOT mean(9/3, 1/1)=2
            self.assertEqual(metrics['fresh_aal'], 12 / 4)
            self.assertEqual(metrics['reflex_aal'], 12 / 3)
            self.assertEqual(metrics['fresh_minus_stale_aal'], 0.5)
            self.assertEqual(metrics['reflex_minus_stale_aal'], 1.5)
            self.assertIsNotNone(metrics['fresh_minus_stale_ci_low'])
            self.assertIsNotNone(metrics['reflex_minus_stale_ci_low'])
            self.assertEqual(metrics['stale_online_draft_update_gpu_ms'], 12.0)
            self.assertEqual(metrics['reflex_update_gpu_ms_per_update'], 4 / 3)
            self.assertGreaterEqual(metrics['analysis_io_wall_ms'], 7.0)
            with (root / 'step_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0]), list(STEP_METRIC_COLUMNS))

    def test_resume_recovers_interrupted_append_from_only_durable_journals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'protocol.json', {'format': PROTOCOL_VERSION})
            raw, summaries = boundary_fixture()
            append_boundary_results(root, raw, summaries,
                dict(policy_step=1, next_target_optimizer_step=2, durable_target_checkpoint=False), io_meter=AnalysisIO())
            self.assertEqual(load_completed_results(root), (set(), [], []))
            # Crash before the training checkpoint: branch files MUST survive.
            boundary = root / 'boundaries/step_1'
            for name in ('phi_base.pt', 'draft_stale.pt', 'draft_fresh.pt'):
                (boundary / name).write_bytes(b'toy checkpoint')
            self.assertEqual(cleanup_branch_checkpoints(root, 1), [])
            # Training save happened but append was truncated/duplicated.
            mark_durable_boundaries(root, 2)
            (root / 'per_response.jsonl').write_text('truncated invalid JSON')
            completed, recovered, recovered_summaries = load_completed_results(root)
            self.assertEqual(completed, {1})
            self.assertEqual(len(recovered), 6)
            write_results(root, recovered, recovered_summaries)
            write_results(root, recovered, recovered_summaries)
            self.assertEqual(len((root / 'step_metrics.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len((root / 'per_response.jsonl').read_text().splitlines()), 6)
            self.assertEqual(cleanup_branch_checkpoints(root, 2, keep=True), [])
            self.assertEqual(len(cleanup_branch_checkpoints(root, 2)), 3)
            self.assertTrue((boundary / 'results.json').is_file())
            self.assertTrue((root / 'per_response.jsonl').is_file())
            self.assertEqual(load_completed_results(root)[0], {1})

    def test_partial_three_way_journal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'protocol.json', {'format': PROTOCOL_VERSION})
            raw, summaries = boundary_fixture()
            boundary = root / 'boundaries/step_1'
            atomic_json(boundary / 'complete.json', dict(policy_step=1, durable_target_checkpoint=True))
            atomic_json(boundary / 'results.json', dict(policy_step=1, per_response=raw[:-2],
                                                       summaries=[asdict(row) for row in summaries[:-1]]))
            with self.assertRaisesRegex(RuntimeError, 'missing paired'):
                load_completed_results(root)

    def test_io_meter_is_separate_from_compute(self):
        meter = AnalysisIO()
        with mock.patch('policy_lag_analysis.time.perf_counter', side_effect=[5.0, 5.025]):
            with meter.measure():
                pass
        self.assertAlmostEqual(meter.wall_ms, 25.0)

    def test_plot_has_exactly_three_branch_curves(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.axes import Axes
        except ImportError:
            self.skipTest('matplotlib unavailable')
        _, summaries = boundary_fixture()
        labels = []
        original_plot = Axes.plot
        def spy(axis, *args, **kwargs):
            labels.append(kwargs['label'])
            return original_plot(axis, *args, **kwargs)
        with mock.patch.object(Axes, 'plot', new=spy):
            with tempfile.TemporaryDirectory() as directory:
                status = plot_results([asdict(item) for item in summaries], Path(directory) / 'plot.png')
            self.assertEqual(status['status'], 'generated')
            self.assertEqual(labels, ['stale', 'fresh', 'reflex'])


if __name__ == '__main__':
    unittest.main()
