# Policy-supervision lag experiment

`run_policy_lag_analysis.sh` measures the effect of one policy-update of draft
supervision lag on FastGRPO accepted length. The main training trajectory still
uses FastGRPO's GRPO loss/reward code, target-update schedule, tree verification,
and concurrency-aware `verification_capacity / active_batch_size` scheduler.
The legacy FastGRPO draft head and SmoothL1/CE draft trainer are replaced, for
this entrypoint only, by SpecForge EAGLE-3 at pinned commit
`3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83`: its architecture, three-layer
feature projection, KL/LK objective, and training-time TTT unrolling are called
directly. `helper/eagle3_specforge.py` is only the target-vocabulary and flat-KV
runtime adapter needed by the unchanged FastGRPO verifier.

At target update boundary `t`, the run saves `theta_t` and `phi_base` (draft plus
optimizer), uses the just-collected `R_t` for stale supervision, applies GRPO to
obtain `theta_{t+1}`, and separately collects fresh supervision under
`theta_{t+1}` on those same source prompts using `phi_base`. Stale and fresh
branches start from the identical `phi_base`/optimizer, consume the same token
budget, and take the same number of draft optimizer steps. The stale branch
becomes the actual training draft.

On the **next usable GRPO batch**, before another target update, the run
rolls out `phi_base` and fresh as shadow controls under the fixed
`theta_{t+1}`, using the exact prompts and captured RNG state that the real
stale rollout then uses. The real stale rollout is measured for AAL and is the
one passed to GRPO to update `theta_{t+2}`. Shadow rollouts never enter GRPO.
If a batch has no reward variation and cannot update target, its measurements
are discarded and the experiment retries on the next usable batch.
Because stale becomes the live draft, later target updates follow that branch's
trajectory. This estimates a local next-rollout AAL effect, not a multi-step
comparison of two independently trained targets.

The primary metric is

```text
delta_aal = aal_fresh - aal_stale
```

FastGRPO `total_acc_length` includes the verified root/bonus target token. AAL
is therefore `sum(total_acc_length) / sum(total_decoded_token_num)` over all
sequence verification rounds, never an unweighted mean of batch averages.
`teacher_shift_tv` is exact full-vocabulary TV in FP32 at temperature 1 before
top-p/top-k, evaluated on the source batch's prompt prefixes. A positive delta
means fresh supervision helped on the *next actual training batch* under the
same target; the code does not assume its sign. `delta_vs_base` shows whether
each trained branch improves over the pre-update draft. Confidence intervals
for fresh-minus-stale use a prompt-cluster bootstrap over the next batch's
prompts. This is a local next-rollout diagnostic, not held-out generalization.

## Install and run

For the no-Internet B200 deployment, use `README_B200_POLICY_LAG.md`. The exact
SpecForge source is already present in `third_party/SpecForge`; no clone is
performed at runtime.

On a connected preparation machine, the Python dependency manifest is:

```bash
python -m pip install -r fastgrpo/requirements-policy-lag.txt
python -m pip install --no-deps -e fastgrpo/third_party/SpecForge
```

This pin requires Python 3.11 and the exact Torch/Transformers versions listed
in `DEPENDENCIES_POLICY_LAG.md`.

Pretrained initialization is the safe default and requires an actual SpecForge
runtime checkpoint. Missing or incompatible weights are an error. To initialize
once from a compatible config instead, opt in explicitly with
`DRAFT_INITIALIZATION_MODE=random`; that same saved `phi_base` is cloned into
both branches.

```bash
DRAFT_CHECKPOINT=/path/to/specforge-checkpoint \
TARGET_MODEL_PATH=/workspace/storage-shared/models/Qwen2.5-7B-Instruct \
bash fastgrpo/run_policy_lag_analysis.sh
```

Small end-to-end run (it still loads the real target and SpecForge model; it is
not a reported experiment):

```bash
DRAFT_INITIALIZATION_MODE=random \
SMOKE_TEST=true \
TARGET_MODEL_PATH="$PWD/models/Qwen2.5-1.5B-Instruct" \
DRAFT_CONFIG="$PWD/fastgrpo/configs/qwen25_1p5b/eagle3_full_vocab.json" \
DATASET_PATH="$PWD/data/gsm8k/main" \
OUTPUT_DIR="$PWD/outputs/fastgrpo/policy_lag/smoke" \
bash fastgrpo/run_policy_lag_analysis.sh
```

Important overrides include `TARGET_ADAPTER_PATH`,
`TARGET_RESUME_CHECKPOINT`, `DATASET_PATH`, `TRAIN_OPTION`,
`ANALYSIS_BOUNDARIES`, `ANALYSIS_INTERVAL`, `TOTAL_POLICY_STEPS`, `NUM_EPOCHS`,
`TRAINING_TOKEN_BUDGET`, `DRAFT_LR`, `TRAIN_BATCH_SIZE`,
`GRADIENT_ACCUMULATION`, `RESPONSES_PER_PROMPT`,
`MAX_LENGTH`, `TEMPERATURE`, `TOP_P`, and all existing
FastGRPO concurrency controls. `RESUME=true` and `ANALYSIS_RESUME=true` reuse
completed main/analysis checkpoints. Multi-process target inference is not
silently moved to SGLang: this upstream decoder remains single-process and the
launcher rejects `NPROC_PER_NODE != 1`.

Outputs include per-boundary base/stale/fresh checkpoints, per-response JSONL,
summary CSV/JSONL, exact token/optimizer counts, checkpoint/feature policy IDs,
bootstrap intervals, and `aal_policy_lag.png` with the zero line. The
per-response records mark the stale rollout `used_for_grpo=true`; both shadow
branches are false. Protocol v3 results cannot be mixed with older results in
one `OUTPUT_DIR`. For boundaries `1,5,10`, the default target-update budget is
11 so boundary 10 can be evaluated on a real subsequent GRPO update. Plotting
is a derived, best-effort artifact: a missing or broken Matplotlib installation
is recorded in `plot_status.json` and does not abort training or invalidate the
JSONL/CSV results. When Matplotlib is installed, the derived artifact is
`aal_policy_lag.png` with the zero line.

After installing Matplotlib, regenerate a skipped plot without rerunning the
GPU experiment:

```bash
python fastgrpo/policy_lag_analysis.py --mode plot \
  --output-dir /path/to/run/analysis
```
