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

At boundary `t`, the run saves `theta_t`'s ID and `phi_base` (draft plus
optimizer; FastGRPO has no draft scheduler, recorded explicitly as `null`), uses
the already-collected `R_t` for stale supervision, applies the normal GRPO
update, collects `R_{t+1}` with the new policy on the same training prompts, and
replays both branches from `phi_base`. Both consume the same valid-token budget
and make the configured `DRAFT_UPDATE_STEPS` optimizer steps. Evaluation uses held-out test prompts, the same
current target `theta_{t+1}`, sampler, seeds, batch/concurrency settings, and no
draft adaptation. It then restores the stale branch and RNG/module state; fresh
rollouts and evaluation never enter the GRPO buffer.

The primary metric is

```text
delta_aal = aal_fresh - aal_stale
```

FastGRPO `total_acc_length` includes the verified root/bonus target token. AAL
is therefore `sum(total_acc_length) / sum(total_decoded_token_num)` over all
sequence verification rounds, never an unweighted mean of batch averages.
`teacher_shift_tv` is exact full-vocabulary TV in FP32 at temperature 1 before
top-p/top-k, evaluated on a fixed held-out prefix set. A positive delta means
fresh supervision helped under this comparison; the code does not assume its
sign. Confidence intervals use a prompt-cluster bootstrap.

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
`ANALYSIS_BOUNDARIES`, `ANALYSIS_INTERVAL`, `TOTAL_POLICY_STEPS`,
`TRAINING_TOKEN_BUDGET`, `DRAFT_LR`, `TRAIN_BATCH_SIZE`,
`EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION`, `RESPONSES_PER_PROMPT`,
`MAX_LENGTH`, `TEMPERATURE`, `TOP_P`, `SAMPLING_SEEDS`, and all existing
FastGRPO concurrency controls. `RESUME=true` and `ANALYSIS_RESUME=true` reuse
completed main/analysis checkpoints. Multi-process target inference is not
silently moved to SGLang: this upstream decoder remains single-process and the
launcher rejects `NPROC_PER_NODE != 1`.

Outputs include per-boundary base/stale/fresh checkpoints, per-response JSONL,
summary CSV/JSONL, exact token/optimizer counts, checkpoint/feature policy IDs,
bootstrap intervals, and `aal_policy_lag.png` with the zero line.
