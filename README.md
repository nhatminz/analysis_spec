# SimpleLR + Qwen2.5-3B-Instruct: A1/A2 in one run

`run_policy_lag_motivation.py` is the standalone entrypoint. Production FastGRPO and latest Reflex OPD are ported locally; `SpecNaacl` is a read-only source reference, never a runtime Python dependency. There is one OPD-driven target GRPO trajectory, persistent Reflex R/A, persistent pure FastGRPO shadow S, and a disposable matched-sequence Fresh fork only at measurement boundaries.

Read [DESIGN_A1_A2.md](DESIGN_A1_A2.md) for the exact timeline, replay gates, counter semantics and statistical limits, and [PORT_MANIFEST.md](PORT_MANIFEST.md) for sources, hashes and retired code. Old policy-lag launcher names forward to the new entrypoint; obsolete experiment arguments/settings fail. Historical EAGLE3 pretraining tools are segregated in `legacy_pretraining/` and cannot initialize this experiment.

[CORRECTNESS_VALIDATION.md](CORRECTNESS_VALIDATION.md) records the current fixes, executed checks, memory measurements and exact B200/checkpoint blockers. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) is the historical initial-port report; its reduced zero-update pilot does not validate a genuine production transition.

Use an installed CUDA PyTorch/Transformers/PEFT/Triton environment compatible with the reference `requirements.txt`. Both Transformers 4.51 and 5.x API adapters are retained. No dependencies or models are downloaded automatically. CPU unit tests and dataset preparation do not require CUDA; generation does.

B200 preflight requires compute capability 10.x, CUDA >=12.8 and a PyTorch build containing the device's `sm_100` architecture. For the pinned Torch 2.8.0 environment, explicitly choose its CUDA 12.8 wheel when installing dependencies (`python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`); see the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). The locally tested Torch 2.6/CUDA 12.4 environment is for the available 3090 and cannot validate B200 execution.

```bash
cd analysis_spec
export PYTHON_BIN=/path/to/your/cuda-env/bin/python
export MODEL=/path/to/Qwen2.5-3B-Instruct
export DRAFT_CHECKPOINT=/path/to/SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint
export TRAIN_DATASET_PATH=/path/to/data/simplelr_abel_level3to5/train.parquet
export TEST_DATASET_PATH=/path/to/data/simplelr_abel_level3to5/test.parquet
export OUTPUT_DIR="$PWD/outputs/qwen25_3b_simplelr_a1_a2"

# Validate the official train/test split, provenance and fixed 64-test selection.
# No draft is needed for prepare. validate additionally checks model/draft compatibility.
bash run_policy_lag_motivation.sh --mode prepare
bash run_policy_lag_motivation.sh --mode validate

# Nonpublishable smoke; use its own output directory.
OUTPUT_DIR="$PWD/outputs/smoke" bash run_policy_lag_motivation.sh --smoke

# B200 validation: actual 8x8, one genuine boundary, 16 held-out prompts,
# natural rewards, checkpoint resume and an evidence checker. Separate output.
OUTPUT_DIR="$PWD/outputs/b200_execution_validation" bash scripts/validate_b200.sh

# Main: 200 completed target optimizer steps, 8 x 8 responses per learner.
# Measures 20/50/100/150/200; 100/200 use 64 prompts.
# Step 20 additionally checks an isolated zero-drift control.
bash run_policy_lag_motivation.sh --require-b200

# Same config, input paths, environment and output directory as the interrupted run.
bash run_policy_lag_motivation.sh --require-b200 --resume auto

# No model construction or GPU is needed for plotting/reporting.
# Replot reads the durable checkpoint on CPU to recover committed boundaries.
bash run_policy_lag_motivation.sh --mode replot
bash run_policy_lag_motivation.sh --mode report
```

Equivalent direct Python flags are `--model`, `--draft-checkpoint`, `--draft-target-config`, `--train-path`, `--test-path`, `--output-dir`, `--train-steps`, `--eval-steps`, `--confirmation-steps`, `--zero-drift-control-steps`, `--seed`, `--eval-max-new-tokens`, and `--resume`. Use `--config configs/qwen25_3b/simplelr_a1_a2.json` to override reference optimizer/decoder/OPD settings. An empty step list disables that schedule, e.g. `--zero-drift-control-steps ''` (`--zero-update-steps` remains a legacy alias). Changing an existing run's manifest is rejected. Accumulation other than 1 is rejected.

Gold parsing is validated before training. Default `--invalid-answer-policy exclude` records the two empty labels in the supplied train split and excludes them; `--invalid-answer-policy error` stops instead. Nonempty exact-text parser fallbacks are recorded. Uniform reward groups or zero target gradients never call the target optimizer or consume a policy step; the next sampled batch retries, with an explicit `--max-attempts-per-step` limit (default 32). Natural zero parameter/policy drift is reported separately. Target gradient checkpointing, chunked CE/log-probabilities and releasing completed KV pools between learners reduce memory while preserving the configured rollout/evaluation budgets. Zero-drift checks log numerical loss/gradient/weight/moment errors at dtype precision; evaluation freeze and resume guards require exact state hashes.

An optional pretrained `opd_projector` is loaded only into Reflex. Without it, A uses the reference QR initialization and the manifest explicitly records that A was not pretrained. The pretrained draft backbone is always strictly loaded. The runner never trains or downloads a replacement draft.

Defaults are selected only from existing local reference-compatible paths: first `/workspace/storage-shared/models/Qwen2.5-3B-Instruct`, then `../models/Qwen2.5-3B-Instruct`; data from `DATA_ROOT` or `../data`; draft from `checkpoints/qwen25_3b/draft.pth` or the sibling reference pretrain `latest_checkpoint`. A draft checkpoint is input data only. Copy it and its `target_config.json` anywhere and pass that location; the source tree can be absent. Missing resources produce explicit errors; no other model or validation-only checkpoint is chosen.

The supplied environment has the local 3B target and official SimpleLR files, but no production draft at the standard `SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint` path. The discovered `/mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/latest_checkpoint` contains only a 2-step pretrain validation artifact; it is used solely for compatibility/pilot checks and is **not** a production default. Point `DRAFT_CHECKPOINT` at your actual pretrained checkpoint on the training machine.

```bash
# Tests (from this directory; use your installed pytest environment).
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m pytest -q
bash -n run_policy_lag_motivation.sh
```

Outputs live exclusively below the chosen output directory: `analysis/manifest.json`, `analysis/eval_subset_manifest.json`, `analysis/per_response.jsonl`, `analysis/step_metrics.csv`, `analysis/step_metrics.jsonl`, `analysis/boundaries/step_*/`, `analysis/plots/` and `checkpoints/latest.pt`. Source-tree validation logs are in `validation/`; they contain software checks, not published research results.

CPU snapshots avoid duplicating the 3B target on GPU. The local one-boundary validation passed real 8×8 rollouts, length 2048, N=16/cap 256 evaluation and exact resume on a 3090 after releasing idle KV pools. Peak training allocation was 18.643 GiB; whole-device usage reached 21045 MiB. It used the existing validation draft, so production-draft behavior, later training steps and B200 execution remain unverified. CPU replay snapshots and atomic latest checkpoints also need disk space; the measured checkpoint/trace are about 4.0/0.87 GB, and checkpoint replacement temporarily needs another checkpoint-sized allocation. No multi-GPU/distributed training is claimed by this single-process causal protocol.
