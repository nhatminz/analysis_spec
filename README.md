# SimpleLR + Qwen2.5-3B-Instruct: A1/A2 in one run

`run_policy_lag_motivation.py` is the standalone entrypoint. Production FastGRPO and latest Reflex OPD are ported locally; `SpecNaacl` is a read-only source reference, never a runtime Python dependency. There is one OPD-driven target GRPO trajectory, persistent Reflex R/A, persistent pure FastGRPO shadow S, and a disposable matched-sequence Fresh fork only at measurement boundaries.

Read [DESIGN_A1_A2.md](DESIGN_A1_A2.md) for the exact timeline, replay gates, counter semantics and statistical limits, and [PORT_MANIFEST.md](PORT_MANIFEST.md) for sources, hashes and retired code. Old policy-lag launcher names forward to the new entrypoint; obsolete experiment arguments/settings fail. Historical EAGLE3 pretraining tools are segregated in `legacy_pretraining/` and cannot initialize this experiment.

[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) records the executed CPU/CUDA tests, real 3B smoke/resume, input paths/hashes, output columns and concurrent external reference updates detected by the final audit.

Use an installed CUDA PyTorch/Transformers/PEFT/Triton environment compatible with the reference `requirements.txt`. Both Transformers 4.51 and 5.x API adapters are retained. No dependencies or models are downloaded automatically. CPU unit tests and dataset preparation do not require CUDA; generation does.

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

# Main: 200 completed target steps, 8 x 8 responses per learner.
# Measures 20/50/100/150/200; 100/200 use 64 prompts; step 20 is a zero-update control.
bash run_policy_lag_motivation.sh

# Same config, input paths, environment and output directory as the interrupted run.
bash run_policy_lag_motivation.sh --resume auto

# No model construction or GPU is needed for plotting/reporting.
# Replot reads the durable checkpoint on CPU to recover committed boundaries.
bash run_policy_lag_motivation.sh --mode replot
bash run_policy_lag_motivation.sh --mode report
```

Equivalent direct Python flags are `--model`, `--draft-checkpoint`, `--draft-target-config`, `--train-path`, `--test-path`, `--output-dir`, `--train-steps`, `--eval-steps`, `--confirmation-steps`, `--zero-update-steps`, `--seed`, `--eval-max-new-tokens`, and `--resume`. Use `--config configs/qwen25_3b/simplelr_a1_a2.json` to override any reference optimizer/decoder/OPD setting. An empty step list disables that schedule, e.g. `--zero-update-steps ''`. Changing an existing run's manifest is rejected. Accumulation other than 1 is rejected.

Defaults are selected only from existing local reference-compatible paths: first `/workspace/storage-shared/models/Qwen2.5-3B-Instruct`, then `../models/Qwen2.5-3B-Instruct`; data from `DATA_ROOT` or `../data`; draft from `checkpoints/qwen25_3b/draft.pth` or the sibling reference pretrain `latest_checkpoint`. A draft checkpoint is input data only. Copy it and its `target_config.json` anywhere and pass that location; the source tree can be absent. Missing resources produce explicit errors; no other model or validation-only checkpoint is chosen.

The supplied environment has the local 3B target and official SimpleLR files, but no production draft at the standard `SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint` path. The discovered `/mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/latest_checkpoint` contains only a 2-step pretrain validation artifact; it is used solely for compatibility/pilot checks and is **not** a production default. Point `DRAFT_CHECKPOINT` at your actual pretrained checkpoint on the training machine.

```bash
# Tests (from this directory; use your installed pytest environment).
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m pytest -q
bash -n run_policy_lag_motivation.sh
```

Outputs live exclusively below the chosen output directory: `analysis/manifest.json`, `analysis/eval_subset_manifest.json`, `analysis/per_response.jsonl`, `analysis/step_metrics.csv`, `analysis/step_metrics.jsonl`, `analysis/boundaries/step_*/`, `analysis/plots/` and `checkpoints/latest.pt`. Source-tree validation logs are in `validation/`; they contain software checks, not published research results.

CPU snapshots avoid duplicating the 3B target on GPU. Two draft optimizers, long 64-response rollouts and full-vocabulary draft CE still require substantial GPU/host memory. Tiny CUDA tests do not establish that the default 8×8, length-2048 production run fits a 24 GB 3090. Prefer the intended larger-memory training GPU for that run. CPU replay snapshots and atomic latest checkpoints also need disk space. No multi-GPU/distributed training is claimed by this single-process causal protocol.
