# B200 offline: ShareGPT EAGLE-3 pretrain → DAPO policy-lag analysis

This folder is self-contained with respect to project source code.  It vendors
`sgl-project/SpecForge@3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` under
`third_party/SpecForge`; the B200 machine does not need Git or Internet access.
CUDA/PyTorch/SGLang binaries are not vendored: the offline image must already
provide the versions in `DEPENDENCIES_POLICY_LAG.md` (or an administrator must
provide a local wheelhouse).

Matplotlib is optional at runtime. If it is absent, the launcher warns before
loading the model and still writes all JSONL/CSV/checkpoint results; only
`aal_policy_lag.png` is skipped and the reason is saved to
`analysis/plot_status.json`.

Expected placement:

```text
/workspace/storage-shared/nlp/minhpn19/
├── fastgrpo/                         # this entire folder
└── data/
    ├── sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
    └── DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet
```

The target model path is the source of truth. The default is
`/workspace/storage-shared/models/Qwen2.5-3B-Instruct`; set only
`TARGET_MODEL_PATH` to use another local compatible Qwen2/Qwen2.5
checkpoint. The launcher reads that model's `config.json` and generates the
matching one-layer EAGLE-3 config and capture layers automatically.

## Commands

Activate the preinstalled Python 3.11 environment, then run both stages:

```bash
cd /workspace/storage-shared/nlp/minhpn19/fastgrpo
PYTHON_BIN="$(command -v python)" bash run_b200_policy_lag_pipeline.sh
```

For Qwen2.5-3B-Instruct:

```bash
TARGET_MODEL_PATH=/workspace/storage-shared/models/Qwen2.5-3B-Instruct \
PYTHON_BIN="$(command -v python)" \
bash run_b200_policy_lag_pipeline.sh
```

The launcher keeps an installed CUDA Torch `2.11.x` build. It does not ask pip
to replace it with the upstream `2.13.0` lock. Before loading the model it
checks the concrete EAGLE-3, FlexAttention, and SGLang capture APIs and records
the actual versions in `<run-directory>/dependencies.json`.

The vendored backend includes narrow compatibility shims for the installed
SGLang 0.5.14: it does not require `runtime_context.get_flags` or the newer
`ParallelState.attn_dcp_*` fields on the default DP/DCP-disabled capture path.
DP/DCP modes are not silently emulated: requesting an unsupported mode produces
an explicit error. The `flash_attn is not found` message is only a warning;
SpecForge then uses the supported PyTorch FlexAttention backend.

For Qwen2.5-7B-Instruct, only change the path:

```bash
TARGET_MODEL_PATH=/workspace/storage-shared/models/Qwen2.5-7B-Instruct \
PYTHON_BIN="$(command -v python)" \
bash run_b200_policy_lag_pipeline.sh
```

The model directory basename becomes a filesystem-safe model slug. Every
pretrain invocation receives a UTC timestamp with nanoseconds in its run ID,
so runs do not overwrite one another. On success, stable `latest_*` symlinks
are updated. The analysis launcher reads the latest completed run manifest and
therefore does not need the run ID copied by hand.

Or run separately:

```bash
TARGET_MODEL_PATH=/workspace/storage-shared/models/Qwen2.5-3B-Instruct \
PYTHON_BIN="$(command -v python)" \
bash pretrain_eagle3_sharegpt_b200.sh

PYTHON_BIN="$(command -v python)" bash run_policy_lag_analysis_b200.sh
```

The first command converts the local ShareGPT JSON, captures the exact
SpecForge EAGLE-3 teacher tensors, creates one fixed vocabulary mapping, and
trains one epoch. It uses explicit random initialization by default. To warm
start instead, set both `DRAFT_INITIALIZATION_MODE=pretrained` and
`INITIAL_DRAFT_CHECKPOINT=/absolute/local/checkpoint`.

The second command deterministically shuffles the English DAPO parquet with
seed 42 and uses exactly 5,000 rows as the GRPO/analysis training pool. The
split utility also creates 512 other rows in `eval.jsonl` for compatibility,
but the policy-lag AAL analysis does **not** use them. At each selected target
update boundary it trains stale and fresh draft branches; on the **next GRPO
batch**, it compares their AAL under the same target and prompts. The stale
rollout is the real rollout used to update target again; fresh and base are
shadow controls. `ANALYSIS_BOUNDARIES=1,5,10` refers to actual target optimizer
updates; `TOTAL_POLICY_STEPS=11` leaves an update after boundary 10 for its
live measurement.

Useful B200 overrides:

```bash
CAPTURE_CUDA_VISIBLE_DEVICES=0,1 \
CAPTURE_NPROC_PER_NODE=2 \
TRAIN_CUDA_VISIBLE_DEVICES=2 \
CUDA_VISIBLE_DEVICES=3 \
PYTHON_BIN="$(command -v python)" \
bash run_b200_policy_lag_pipeline.sh
```

Feature capture can use multiple data-parallel workers. The SpecForge trainer
can also use multiple GPUs by setting matching `TRAIN_CUDA_VISIBLE_DEVICES` and
`PRETRAIN_NPROC_PER_NODE`. FastGRPO rollout/verification remains the upstream
single-process decoder and therefore intentionally uses one GPU.

## Outputs

SpecForge pretraining:

```text
/workspace/storage-shared/nlp/minhpn19/outputs/specforge/
├── latest_run -> <latest completed run>
└── qwen2_5_3b_instruct/
    ├── latest_run -> runs/<run-id>
    ├── latest_checkpoint -> runs/<run-id>/checkpoints/<run-id>-latest
    ├── latest_draft_config.json -> runs/<run-id>/config/eagle3.json
    ├── latest_vocab_mapping.pt -> runs/<run-id>/features/vocab_mapping/vocab_mapping.pt
    └── runs/<run-id>/
        ├── config/eagle3.json
        ├── data/sharegpt_train.jsonl
        ├── features/{**/*.ckpt,vocab_mapping/vocab_mapping.pt,capture_complete.json}
        ├── checkpoints/{<run-id>-step<N>,<run-id>-latest,pretrain_complete.json}
        └── logs/{capture.log,train.log}
```

DAPO split and policy-lag results:

```text
/workspace/storage-shared/nlp/minhpn19/outputs/fastgrpo/policy_lag/
├── dapo_math_seed42/{train.jsonl,eval.jsonl,split_manifest.json}
└── qwen2_5_3b_instruct/<pretrain-run-id>_dapo5k/
    ├── analysis/{per_response.jsonl,summary.jsonl,summary.csv,aal_policy_lag.png}
    ├── analysis/boundaries/step_*/{phi_base.pt,draft_stale.pt,draft_fresh.pt,pending.json,complete.json}
    ├── checkpoints/latest.pt
    └── logs/{console.log,train.jsonl,summary.json,summary.txt}
```

Successful stages are reused. `RESUME_PRETRAIN=true`, `RESUME=true`, and
`ANALYSIS_RESUME=true` are defaults. Protocol v3 refuses to mix next-rollout
results with older analyses in one output directory: set a new
`OUTPUT_DIR` (and `RESUME=false` for a fresh training trajectory).
Use a new output directory for a genuinely new experiment instead of
overwriting an existing run.

The analysis directory also contains `plot_status.json`. The PNG is present
only when Matplotlib can render it.

After Matplotlib is installed into the same `PYTHON_BIN` environment, a skipped
PNG can be generated without rerunning training:

```bash
"$PYTHON_BIN" policy_lag_analysis.py --mode plot --output-dir /path/to/run/analysis
```
