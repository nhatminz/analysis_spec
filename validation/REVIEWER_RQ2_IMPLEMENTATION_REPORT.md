# Reviewer controls and RQ2 implementation / execution report

The implementation is complete for exactly two experiments: offline E2/E3/E6
reviewer controls and real frozen SPARK OFF/ON repeated draft misses. Production
scientific experiments have **not** completed locally: neither the requested
B200 collection nor the compatible production checkpoints are mounted here.
There are no production numerical findings, confidence intervals or supported
claims that SPARK reduces recurrence.

## Files changed by this task

Only files under `analysis_spec` were written. Existing runtime modification:

- `helper/opd_generate.py`: optional passive packed-verification observation.

New implementation, launchers, plotting, documentation and tests:

- `README_REVIEWER_RQ2.md`
- `plot_motivation_controls.py`
- `reviewer_controls/__init__.py`
- `reviewer_controls/offline.py`
- `reviewer_controls/plotting.py`
- `reviewer_controls/statistics.py`
- `rq2/__init__.py`
- `rq2/analysis.py`
- `rq2/checkpoints.py`
- `rq2/events.py`
- `rq2/runtime.py`
- `run_reviewer_controls.py`
- `run_reviewer_controls_b200.sh`
- `run_rq2_repeated_misses.py`
- `run_rq2_repeated_misses_b200.sh`
- `tests/test_reviewer_controls.py`
- `tests/test_rq2_repeated_misses.py`
- `tests/test_rq2_repeated_misses_cuda.py`

Validation outputs created under `validation/`:

- This report and `reviewer_rq2_validation.json` (hashes, counts, runtimes).
- `reviewer_controls_production_attempt.log`
- `reviewer_rq2_production_preflight.log`
- `reviewer_rq2_cpu.log`, `reviewer_rq2_cpu_final.log`
- `reviewer_rq2_cuda.log`, `reviewer_rq2_cuda_evidence.jsonl`
- `reviewer_rq2_gpu_regression.log`

Existing pure FastGRPO generation, model, training, collector, event schema,
E2–E6 analysis, checkpoint code, sampling and correction kernels match their
original `analysis_spec` Git HEAD hashes. Original training manifest source
identity therefore remains usable. The task issued no writes to `SpecNaacl`,
paper LaTeX or any other project. Concurrent external `SpecNaacl` working-tree
edits were observed and left untouched; the old reference-hash file is not a
valid before/after audit for those concurrent edits.

## Checks actually run

| Check | Result | Actual runtime |
|---|---:|---:|
| New CPU statistical, integrity, projector and event tests, final run | 19 passed | 10.59 s |
| Broader CPU regression (earlier run; overlaps focused tests) | 60 passed, 161 CUDA cases skipped | 42.99 s |
| New real CUDA SPARK observer edge cases | 8 passed | 11.36 s |
| Integrated CUDA save/resume smoke + baseline reconstruction, cap and sampler regression | 157 passed | 20.63 s |
| Shell syntax and Git whitespace checks | passed | — |

CPU tests verify strict prior-round history; independent/exclusive controls;
joint strata/weights; all five distance bins; token-frequency invariants and
determinism; missing support; whole-rollout bootstrap and paired cancellation
under unequal event counts; preservation of original collection bytes; saved
analysis and PDF/PNG outputs; checkpoint/projector provenance; packed alignment,
coverage/pruning and EOS/depth/beam/length censoring.

CUDA validation used an RTX 3090 and a **tiny randomly initialized Qwen2 fixture**,
not the Qwen2.5-3B online checkpoints. The 8 observer cases cover ordinary
generation, prefill swap-removal, all-prefill EOS and verification EOS, each
with synchronous and asynchronous B updates. Tests assert:

- Identical tokens, acceptance counters and final RNG with observation on/off.
- Identical target forward counts: prefill plus existing verification rounds.
- Frozen target/draft/A weights and zero initial B for each rollout.
- B remains zero OFF, updates from actual verification feedback ON, and resets.
- Exact canonical response reconstruction, including swap-removal and EOS.

The integrated checkpoint-loader/collector smoke committed one real rollout per
arm, validated its replay, and resumed without rewriting the commits. It used
initialized A. OFF had B norm 0 and 0 updates; ON had B norm
0.0008160977740772069 and 12 updates, with identical A hashes in both arms.
These are correctness diagnostics, not evidence of a scientific effect.

As a useful negative check, the ordinary observer fixture had 48 ordinary
misses out of 48 valid events in **both** OFF and ON. It does not show a reduction.
Its single independent fixture rollout cannot provide a valid scientific
rollout-cluster CI; the edge variants are not pooled as independent evidence.

## Requested production execution status

Phase 1 was implemented and CPU-tested before any new GPU model was launched.
The production offline CLI failed safely because
`outputs/e2_e6_fastgrpo/collection` does not exist on this local machine. The
last recorded attempt took 1.415 s. The B200 absolute collection path is also
unavailable here. **Zero of the three production checkpoints were analyzed.**
No baseline collection was repeated or replaced with fixture logs.

Phase 2 production preflight reported missing checkpoints 30, 100 and 200,
before GPU model loading. The last recorded attempt took 3.376 s. Only the tiny
CUDA correctness smoke ran; the 20-rollout, 8×8, Qwen2.5-3B B200 experiment did
not run. No GRPO or other long training was launched.

The default projector policy requires a learned A from the exact same whole
checkpoint. For pure FastGRPO checkpoints, the B200 command below explicitly
opts into deterministic initialized A and labels every result exploratory.
It must not be presented as fully trained SPARK. Learned-checkpoint loading is
implemented and CPU-tested, but no actual learned-A production checkpoint was
available for GPU evaluation here.

## Exact B200 commands

Use the existing activated B200 Python environment.

Task 1 (offline; launcher hides CUDA):

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
PYTHON_BIN=python3 bash run_reviewer_controls_b200.sh
```

Task 2 (existing pure FastGRPO checkpoints, explicit exploratory initialized A):

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
CUDA_VISIBLE_DEVICES=1 PYTHON_BIN=python3 \
  bash run_rq2_repeated_misses_b200.sh --projector-mode initialized
```

The second command performs the real per-checkpoint smoke before 20 paired
rollouts, uses the original sampling/prompt/seed plan, checks a 150-minute
collection budget between arm-rollout batches and safely resumes committed
outputs. Missing checkpoints are reported. A different model/config/checkpoint
identity requires a new output directory.

## Artifact locations after production runs

Task 1:
`/workspace/storage-shared/nlp/minhpn19/analysis_spec/outputs/e2_e6_fastgrpo/reviewer_controls_v1/`

Contains `E2.csv/json`, `E3.csv/json`, `E6_frequency.csv/json`, `E6_donor.csv/json`,
`summary.json`, `REVIEWER_CONTROLS_REPORT.md`, integrity manifests and E2/E3/E6
vector PDFs for each checkpoint. These production files were not fabricated
locally. Synthetic end-to-end artifacts exist only under `.cache/` test folders.

Task 2:
`/workspace/storage-shared/nlp/minhpn19/analysis_spec/outputs/e2_e6_fastgrpo/rq2_repeated_misses_v1/`

Contains separate OFF/ON event commits, actual A identity records,
`rq2_repeated_misses.csv/json`, `RQ2_REPEATED_MISSES_REPORT.md`,
`rq2_repeated_misses.pdf/png` and execution/integrity metadata. The shared
`plot_motivation_controls.py` entrypoint reproduces figures from JSON alone.

No E4/E5, rollout-progress/AAL, scope-ablation or speed figures are added.
Conditional recurrence is analyzed independently of the overall miss-rate
sanity metric. Empty common support or unsupported intervals are explicitly
reported, including outcomes that do not favor SPARK.
