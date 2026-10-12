# Offline reviewer controls and RQ2 repeated draft misses

These are exactly two new experiments. Existing collection, checkpoints,
training manifests and `analysis_e2_e6` are read-only inputs. No training,
E4/E5, rollout-progress/AAL figures, scope ablation or speed experiment is added.
Only `helper/opd_generate.py` changes existing runtime code, adding an optional
passive observer. Pure FastGRPO training files and their manifest hashes stay
unchanged, so an ongoing pure FastGRPO training run can continue/resume.

## Task 1: existing logs only, CPU

In the activated B200 project environment:

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
PYTHON_BIN=python3 bash run_reviewer_controls_b200.sh
```

Reads `outputs/e2_e6_fastgrpo/collection/step_000030`, `step_000100` and
`step_000200`, including original event/round Parquet, response metadata and
checkpoint manifests. It requires all planned rollouts unless `--allow-partial`
is explicitly supplied. The launcher hides CUDA. It never calls model loading,
generation, the baseline collector, training, or target forward passes.

The deterministic donor protocol is the existing `add_controls()` protocol:
sorted rollout IDs, first half donors, second half recipients, one-to-one,
without reuse. An odd last rollout is explicitly excluded. Original prompt and
question hashes must be disjoint. A donor can provide a miss only at a round
index **strictly smaller** than the recipient round index; these are independent
evaluations at a frozen checkpoint, not sequential policy updates.

- E2 has three exclusive conditions on the recipient population: a prior miss
  in this response; a prior donor miss with no prior miss in this response; and
  neither. All three risks, RD and RR use jointly supported strata and the same
  fixed empirical weights. Distances are `1`, `2–4`, `5–16`, `17–64`, `>64`,
  plus an aggregate. The aggregate current-versus-donor RD directly answers the
  reviewer question; older estimates from incompatible populations are never
  subtracted.
- E3 reports token-only and context4 separately, each for sibling responses and
  different prompts versus an unexposed control. It preserves the existing
  exclusive source priority and excludes same-response sources from the main
  figure. Events with fewer than four context tokens are ineligible for context4.
- E6 permutes ordinary miss labels only within each token and rollout. Target
  sequences, positions, response/group IDs, round indices, censored masks and
  per-token miss totals stay fixed. Histories are recomputed for every draw.
  The default is 500 permutations (`PERMUTATIONS=2000` increases it). Null
  intervals are empirical permutation quantiles, **not SPARK measurements**.
  The two-sided Monte Carlo p-value compares absolute departures from the null
  mean, uses the plus-one correction, and is unavailable for a degenerate null,
  fewer than 19 permutations or fewer than two contributing rollout clusters.
- E6 donor availability and donor availability without prior recipient-batch
  availability use exactly the same recipient-miss denominator as their
  within-batch comparison. These estimates use the recipient half; the frequency
  null uses all rollouts. Donor-control bootstrap units are independent pairs,
  indexed by recipient rollout ID.

Outputs under `outputs/e2_e6_fastgrpo/reviewer_controls_v1/`:

```text
E2.csv/json, E3.csv/json, E6_frequency.csv/json, E6_donor.csv/json
summary.json, REVIEWER_CONTROLS_REPORT.md, manifest.json, complete.json
E2_step_{30,100,200}.pdf
E3_step_{30,100,200}.pdf
E6_step_{30,100,200}.pdf
```

## Task 2: real paired SPARK OFF/ON, CUDA device 1

Original pure FastGRPO checkpoints have no learned A. The following is an
**explicitly exploratory initialized-projector experiment**, not the effect of
fully trained SPARK:

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
CUDA_VISIBLE_DEVICES=1 PYTHON_BIN=python3 \
  bash run_rq2_repeated_misses_b200.sh --projector-mode initialized
```

This uses the existing `checkpoints/step_000030.pt`, `step_000100.pt` and
`step_000200.pt`, matching target LoRA and draft weights from each entire
checkpoint. Missing checkpoints are reported and skipped. Checkpoints are
validated before loading a GPU model. It reads the original source config and
uses the B200 model/data environment variables:

```text
MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
TEST_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet
ANALYSIS_ROOT=$PWD/outputs/e2_e6_fastgrpo
```

There is no pretrained-draft fallback. It loads the online checkpoint directly.
The initialized projector is the existing deterministic head-aligned QR basis,
rank 8, from the same frozen target LM head; it consumes no random draws. Both
arms freeze exactly the same target, draft and A. Actual A hashes are saved in
`projectors/step_*.json` and independently in every arm's response and commit
metadata. No A from another trajectory is accepted.

For learned SPARK checkpoints, use `--projector-mode learned` (the safe default)
and `--checkpoint-template '/absolute/own_spark_run/step_{step}.pt'`. Supported
whole-checkpoint formats are `opd_fastgrpo_checkpoint_v5` and
`simplelr_opd_policy_lag_a1a2_v2`. The loader extracts target, its SPARK draft and
its own A together, verifies rank/weights and requires training provenance
(changed A relative to that trajectory's initial A or saved positive optimizer
state for an A-shaped parameter). It refuses to label an unexplained projector
as learned and refuses to replace a checkpoint's existing A with an initialized
one. For the policy-lag format, it uses `reflex.weights`, never `shadow.weights`.

The launcher runs `all`: one real smoke rollout per arm at every available
checkpoint, checking logging against an unobserved replay, then 20 paired
rollouts of 8 prompts × 8 responses. Seeds follow the original
`generation_seed(eval_seed, step, 'rollout:{id}')` protocol. It matches the original
160-prompt plan and generation settings when the baseline plan exists. OFF uses
the **same SPARK generator** with fast LR zero; ON uses actual CUDA/Triton B
updates at LR 0.01. A, target and draft are never optimized. EOS and native
batch-level stopping behavior remain intact. No event correspondence between
the two generated trajectories is assumed.

The default 150-minute collection budget is checked at safe arm-rollout
boundaries. Each arm is committed atomically after reconstruction and integrity
checks. A partial arm pair remains saved but is excluded from paired analysis.
Repeat the same command to resume; existing valid commits are not rewritten.
Changing checkpoint/config/code identity requires a new output directory.
Re-running `all` analyzes all completed pairs, including safe partial results.

To smoke only or analyze saved RQ2 logs without any GPU model:

```bash
CUDA_VISIBLE_DEVICES=1 python3 run_rq2_repeated_misses.py smoke \
  --projector-mode initialized --require-b200
python3 run_rq2_repeated_misses.py analyze
```

Outputs under `outputs/e2_e6_fastgrpo/rq2_repeated_misses_v1/`:

```text
manifest.json, runtime_preflight.json, projectors/step_*.json
smoke/step_*/{off,on}/rollout_0000/...
collection/step_*/{off,on}/rollout_*/events.parquet
collection/step_*/{off,on}/rollout_*/rounds.parquet
collection/step_*/{off,on}/rollout_*/responses.json
collection/step_*/{off,on}/rollout_*/diagnostics.json
collection/step_*/{off,on}/rollout_*/complete.json
collection_execution.json
rq2_repeated_misses.csv, rq2_repeated_misses.json
RQ2_REPEATED_MISSES_REPORT.md
rq2_repeated_misses.pdf, rq2_repeated_misses.png, analysis_complete.json
```

All four arm/exposure cells (OFF prior/control, ON prior/control) must coexist
in a stratum. Matching uses exact token, round/position bins, tree depth, proposal
k, active size, total earlier exposure, response exposure and target-token
exposure. Weights are pooled empirical event counts on joint support, fixed
across cells. ON–OFF absolute and relative recurrence changes use paired
rollout-cluster bootstrap draws, with 2,000 resamples by default. Events lost to
common-support matching and coverage for each arm/exposure cell are reported.
No event is artificially paired across trajectories.

For both tasks, a bootstrap draw missing any required cell in any originally
supported stratum is invalid rather than silently changing the population or
imputing a risk. Fewer than two contributing clusters per cell or fewer than
20 valid resamples gives no CI. This deliberately conservative rule may produce
insufficient-support findings. Overall miss rates are sanity metrics; only the
conditional recurrence estimates can support the RQ2 conclusion.

## Replot and tests

```bash
python3 plot_motivation_controls.py reviewer outputs/e2_e6_fastgrpo/reviewer_controls_v1
python3 plot_motivation_controls.py rq2 outputs/e2_e6_fastgrpo/rq2_repeated_misses_v1
python3 -m pytest tests/test_reviewer_controls.py tests/test_rq2_repeated_misses.py -q
CUDA_VISIBLE_DEVICES=1 python3 -m pytest tests/test_rq2_repeated_misses_cuda.py -q
```

Figures use DejaVu Serif, 9 pt, vector PDF. Reviewer figures are single-column
(3.35 inches), with the two-panel E6 plot spanning two columns. The RQ2 figure
fits all three checkpoint panels into 6.8 inches; PNG is included for review. Plotting code only
reads JSON and does not load a model.

Actual local validation, runtime, unavailable production inputs and the tiny
CUDA smoke measurements are recorded in
`validation/REVIEWER_RQ2_IMPLEMENTATION_REPORT.md`. Tiny synthetic/model fixtures
are correctness evidence only, never substituted for the requested experiments.
