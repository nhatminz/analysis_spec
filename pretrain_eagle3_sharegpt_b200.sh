#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$SCRIPT_DIR/third_party/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Primary knob: edit/override only this path to select the target model.
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
SHAREGPT_PATH="${SHAREGPT_PATH:-$WORKSPACE/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
MODEL_BASENAME="$(basename "${TARGET_MODEL_PATH%/}")"
MODEL_SLUG="$(printf '%s' "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
MODEL_SLUG="${MODEL_SLUG%_}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-$WORKSPACE/outputs/specforge/$MODEL_SLUG}"
RESUME_PRETRAIN="${RESUME_PRETRAIN:-true}"
REQUESTED_RUN_ID="${RUN_ID:-}"
REQUESTED_PRETRAIN_ROOT="${PRETRAIN_ROOT:-}"
ACTIVE_RUN_LINK="$MODEL_OUTPUT_ROOT/active_run"
if [[ -n "$REQUESTED_PRETRAIN_ROOT" ]]; then
  PRETRAIN_ROOT="$REQUESTED_PRETRAIN_ROOT"
  RUN_ID="${REQUESTED_RUN_ID:-$(basename "$PRETRAIN_ROOT")}"
elif [[ -n "$REQUESTED_RUN_ID" ]]; then
  RUN_ID="$REQUESTED_RUN_ID"
  PRETRAIN_ROOT="$MODEL_OUTPUT_ROOT/runs/$RUN_ID"
elif [[ "$RESUME_PRETRAIN" == "true" && -e "$ACTIVE_RUN_LINK" && ! -f "$(readlink -f "$ACTIVE_RUN_LINK")/checkpoints/pretrain_complete.json" ]]; then
  PRETRAIN_ROOT="$(readlink -f "$ACTIVE_RUN_LINK")"
  RUN_ID="$(basename "$PRETRAIN_ROOT")"
else
  RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%S_%N)}"
  RUN_ID="${MODEL_SLUG}_eagle3_sharegpt_1ep_${RUN_TAG}"
  PRETRAIN_ROOT="$MODEL_OUTPUT_ROOT/runs/$RUN_ID"
fi
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-16000}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_ROOT/config/eagle3.json}"

CONVERTED_DATA_DIR="${CONVERTED_DATA_DIR:-$PRETRAIN_ROOT/data}"
CONVERTED_DATA_PATH="${CONVERTED_DATA_PATH:-$CONVERTED_DATA_DIR/sharegpt_train.jsonl}"
FEATURE_DIR="${FEATURE_DIR:-$PRETRAIN_ROOT/features}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PRETRAIN_ROOT/checkpoints}"

DRAFT_INITIALIZATION_MODE="${DRAFT_INITIALIZATION_MODE:-random}"
INITIAL_DRAFT_CHECKPOINT="${INITIAL_DRAFT_CHECKPOINT:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
# Keep 1 as the safe default: SpecForge intentionally rejects an epoch whose
# number of micro-batches is not divisible by the accumulation window.
PRETRAIN_GRADIENT_ACCUMULATION="${PRETRAIN_GRADIENT_ACCUMULATION:-1}"
PRETRAIN_LR="${PRETRAIN_LR:-5e-5}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-500}"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
PRETRAIN_NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-1}"

CAPTURE_CUDA_VISIBLE_DEVICES="${CAPTURE_CUDA_VISIBLE_DEVICES:-0}"
CAPTURE_NPROC_PER_NODE="${CAPTURE_NPROC_PER_NODE:-1}"
CAPTURE_TP_SIZE="${CAPTURE_TP_SIZE:-1}"
CAPTURE_BATCH_SIZE="${CAPTURE_BATCH_SIZE:-8}"
CAPTURE_MAX_LENGTH="${CAPTURE_MAX_LENGTH:-2048}"
CAPTURE_MEM_FRACTION_STATIC="${CAPTURE_MEM_FRACTION_STATIC:-0.70}"
CAPTURE_NUM_WORKERS="${CAPTURE_NUM_WORKERS:-4}"
CAPTURE_IO_THREADS="${CAPTURE_IO_THREADS:-16}"
CAPTURE_COMPRESS="${CAPTURE_COMPRESS:-false}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
FORCE_CAPTURE="${FORCE_CAPTURE:-false}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"

EXPECTED_SPECFORGE_COMMIT="3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83"
CAPTURE_MARKER="$FEATURE_DIR/capture_complete.json"
PRETRAIN_MARKER="$CHECKPOINT_DIR/pretrain_complete.json"
LATEST_CHECKPOINT="$CHECKPOINT_DIR/$RUN_ID-latest"
VOCAB_MAPPING="$FEATURE_DIR/vocab_mapping/vocab_mapping.pt"

fail() { echo "ERROR: $*" >&2; exit 2; }
run() {
  printf 'Run:'; printf ' %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" != "true" ]]; then "$@"; fi
}
publish_latest() {
  mkdir -p "$MODEL_OUTPUT_ROOT"
  ln -sfn "$PRETRAIN_ROOT" "$MODEL_OUTPUT_ROOT/latest_run"
  ln -sfn "$LATEST_CHECKPOINT" "$MODEL_OUTPUT_ROOT/latest_checkpoint"
  ln -sfn "$VOCAB_MAPPING" "$MODEL_OUTPUT_ROOT/latest_vocab_mapping.pt"
  ln -sfn "$DRAFT_CONFIG" "$MODEL_OUTPUT_ROOT/latest_draft_config.json"
  ln -sfn "$PRETRAIN_ROOT" "$WORKSPACE/outputs/specforge/latest_run"
}

[[ -f "$TARGET_MODEL_PATH/config.json" ]] || fail "target model not found: $TARGET_MODEL_PATH"
[[ -f "$SHAREGPT_PATH" ]] || fail "ShareGPT file not found: $SHAREGPT_PATH"
[[ -f "$SPECFORGE_DIR/VENDORED_COMMIT" ]] || fail "vendored SpecForge is incomplete: $SPECFORGE_DIR"
grep -q "commit=$EXPECTED_SPECFORGE_COMMIT" "$SPECFORGE_DIR/VENDORED_COMMIT" || \
  fail "vendored SpecForge commit does not match $EXPECTED_SPECFORGE_COMMIT"
if [[ "$DRAFT_INITIALIZATION_MODE" != "random" && "$DRAFT_INITIALIZATION_MODE" != "pretrained" ]]; then
  fail "DRAFT_INITIALIZATION_MODE must be random or pretrained"
fi
if [[ "$DRAFT_INITIALIZATION_MODE" == "pretrained" && -z "$INITIAL_DRAFT_CHECKPOINT" ]]; then
  fail "pretrained initialization requires INITIAL_DRAFT_CHECKPOINT"
fi

export PYTHONPATH="$SPECFORGE_DIR:$SCRIPT_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$MODEL_OUTPUT_ROOT" "$PRETRAIN_ROOT"
ln -sfn "$PRETRAIN_ROOT" "$ACTIVE_RUN_LINK"
printf 'Target model : %s\nModel slug   : %s\nRun ID       : %s\nRun directory: %s\n' \
  "$TARGET_MODEL_PATH" "$MODEL_SLUG" "$RUN_ID" "$PRETRAIN_ROOT"

if [[ ! -f "$DRAFT_CONFIG" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/generate_eagle3_config.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --output "$DRAFT_CONFIG" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE"
fi
[[ -f "$DRAFT_CONFIG" ]] || fail "EAGLE-3 config was not created: $DRAFT_CONFIG"

"$PYTHON_BIN" - "$PRETRAIN_ROOT/dependencies.json" <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path
from packaging.version import Version

if sys.version_info < (3, 11):
    raise SystemExit(f"Python >=3.11 is required; found {sys.version.split()[0]}")
modules = {}
for name in ("torch", "transformers", "datasets", "accelerate", "yaml", "sglang", "specforge"):
    try:
        modules[name] = importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"missing/incompatible offline dependency {name}: {type(exc).__name__}: {exc}")
torch_version = modules["torch"].__version__
transformers_version = modules["transformers"].__version__
sglang_version = getattr(modules["sglang"], "__version__", "unknown")
torch_base = Version(torch_version.split("+", 1)[0])
transformers_base = Version(transformers_version.split("+", 1)[0])
# SpecForge contains an explicit Torch-2.11 CuteDSL compatibility shim. 2.13 is
# its upstream lockfile version; both stacks exercise the same EAGLE-3 code.
if (torch_base.major, torch_base.minor) not in {(2, 11), (2, 13)}:
    raise SystemExit(
        "unsupported torch stack: expected a validated 2.11.x or upstream "
        f"2.13.x build, found {torch_version}"
    )
if not (Version("5.8.0") <= transformers_base < Version("6.0.0")):
    raise SystemExit(
        "unsupported transformers stack: expected >=5.8,<6.0, found "
        f"{transformers_version}"
    )

# Import the concrete APIs used by EAGLE-3 training and local SGLang feature
# capture. This is more useful than rejecting a working locally installed build
# solely because its package version differs from the upstream lockfile.
capabilities = (
    "specforge.modeling.auto",
    "specforge.algorithms.eagle3.model",
    "torch.nn.attention.flex_attention",
)
for name in capabilities:
    try:
        importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"dependency capability check failed for {name}: {type(exc).__name__}: {exc}")
try:
    backend = importlib.import_module("specforge.offline_capture.sglang_backend")
    getattr(backend, "OfflineSGLangCaptureBackend")
except Exception as exc:
    raise SystemExit(
        "installed SGLang does not provide the APIs required for offline "
        f"feature capture: {type(exc).__name__}: {exc}"
    )

report = {
    "python": sys.version.split()[0],
    "torch": torch_version,
    "transformers": transformers_version,
    "sglang": sglang_version,
    "specforge": getattr(modules["specforge"], "__version__", "vendored"),
    "compatibility_mode": "torch_2_11" if torch_base.minor == 11 else "upstream_lock",
    "capability_checks": list(capabilities) + ["specforge.offline_capture.sglang_backend"],
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
print("Runtime dependency check passed:", json.dumps(report, sort_keys=True))
PY

"$PYTHON_BIN" - "$TARGET_MODEL_PATH/config.json" "$DRAFT_CONFIG" <<'PY'
import json, sys
target = json.load(open(sys.argv[1], encoding="utf-8"))
draft = json.load(open(sys.argv[2], encoding="utf-8"))
checks = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size")
mismatch = {key: (target.get(key), draft.get(key)) for key in checks if target.get(key) != draft.get(key)}
if mismatch:
    raise SystemExit(f"target/draft architecture mismatch: {mismatch}")
layers = draft.get("eagle_config", {}).get("eagle_aux_hidden_state_layer_ids")
expected = [1, target["num_hidden_layers"] // 2 - 1, target["num_hidden_layers"] - 4]
if layers is not None and layers != expected:
    raise SystemExit(f"EAGLE-3 feature layers mismatch: configured={layers}, expected={expected}")
print(f"Validated {target['hidden_size']}-wide target/draft config; feature layers={layers or expected}")
PY

mkdir -p "$CONVERTED_DATA_DIR" "$PRETRAIN_ROOT/logs" "$CHECKPOINT_DIR"

if [[ ! -f "$CONVERTED_DATA_PATH" ]]; then
  run "$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_data.py" \
    --dataset sharegpt \
    --data-path "$SHAREGPT_PATH" \
    --output-path "$CONVERTED_DATA_DIR"
else
  echo "Reuse converted ShareGPT: $CONVERTED_DATA_PATH"
fi

if [[ ! -f "$CAPTURE_MARKER" ]]; then
  if [[ -d "$FEATURE_DIR" ]] && find "$FEATURE_DIR" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print -quit | grep -q .; then
    [[ "$FORCE_CAPTURE" == "true" ]] || fail \
      "partial feature directory exists without completion marker: $FEATURE_DIR; use a new FEATURE_DIR or set FORCE_CAPTURE=true"
  fi
  capture_cmd=(
    "$PYTHON_BIN" -m torch.distributed.run --standalone
    "--nproc_per_node=$CAPTURE_NPROC_PER_NODE"
    "$SPECFORGE_DIR/scripts/prepare_hidden_states.py"
    --strategy eagle3
    --target-model-path "$TARGET_MODEL_PATH"
    --draft-model-config "$DRAFT_CONFIG"
    --data-path "$CONVERTED_DATA_PATH"
    --output-path "$FEATURE_DIR"
    --chat-template qwen
    --max-length "$CAPTURE_MAX_LENGTH"
    --tp-size "$CAPTURE_TP_SIZE"
    --batch-size "$CAPTURE_BATCH_SIZE"
    --num-workers "$CAPTURE_NUM_WORKERS"
    --num-io-threads "$CAPTURE_IO_THREADS"
    --sglang-mem-fraction-static "$CAPTURE_MEM_FRACTION_STATIC"
  )
  [[ "$CAPTURE_COMPRESS" == "true" ]] && capture_cmd+=(--compress)
  printf 'Capture CUDA_VISIBLE_DEVICES=%s\n' "$CAPTURE_CUDA_VISIBLE_DEVICES"
  if [[ "$DRY_RUN" == "true" ]]; then
    printf 'Run:'; printf ' %q' env CUDA_VISIBLE_DEVICES="$CAPTURE_CUDA_VISIBLE_DEVICES" "${capture_cmd[@]}"; printf '\n'
  else
    env CUDA_VISIBLE_DEVICES="$CAPTURE_CUDA_VISIBLE_DEVICES" "${capture_cmd[@]}" \
      2>&1 | tee -a "$PRETRAIN_ROOT/logs/capture.log"
    [[ -f "$VOCAB_MAPPING" ]] || fail "capture completed without vocabulary mapping: $VOCAB_MAPPING"
    "$PYTHON_BIN" - "$CAPTURE_MARKER" "$CONVERTED_DATA_PATH" "$FEATURE_DIR" <<'PY'
import json, os, sys
from pathlib import Path
marker, data, features = map(Path, sys.argv[1:])
records = sum(1 for path in features.rglob('*') if path.suffix == '.ckpt' or path.name.endswith('.ckpt.gz'))
payload = {"status": "complete", "input_jsonl": str(data), "feature_dir": str(features), "feature_records": records}
tmp = marker.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
os.replace(tmp, marker)
PY
  fi
else
  echo "Reuse completed feature capture: $FEATURE_DIR"
fi

if [[ "$PREPARE_ONLY" == "true" || "$DRY_RUN" == "true" ]]; then
  echo "Prepared data/features. Vocabulary mapping: $VOCAB_MAPPING"
  exit 0
fi

[[ -f "$VOCAB_MAPPING" ]] || fail "vocabulary mapping not found: $VOCAB_MAPPING"
if [[ -f "$PRETRAIN_MARKER" && -f "$LATEST_CHECKPOINT/training_state.pt" ]]; then
  publish_latest
  echo "Reuse completed one-epoch checkpoint: $LATEST_CHECKPOINT"
  exit 0
fi

train_overrides=(
  "model.target_model_path=$TARGET_MODEL_PATH"
  "model.draft_model_config=$DRAFT_CONFIG"
  "model.vocab_mapping_path=$VOCAB_MAPPING"
  "data.hidden_states_path=$FEATURE_DIR"
  "data.max_length=$CAPTURE_MAX_LENGTH"
  "training.num_epochs=$PRETRAIN_EPOCHS"
  "training.batch_size=$PRETRAIN_BATCH_SIZE"
  "training.accumulation_steps=$PRETRAIN_GRADIENT_ACCUMULATION"
  "training.learning_rate=$PRETRAIN_LR"
  "training.save_interval=$PRETRAIN_SAVE_INTERVAL"
  "training.seed=$PRETRAIN_SEED"
  "deployment.trainer.nproc_per_node=$PRETRAIN_NPROC_PER_NODE"
  "run_id=$RUN_ID"
  "output_dir=$CHECKPOINT_DIR"
)
if [[ "$DRAFT_INITIALIZATION_MODE" == "pretrained" ]]; then
  train_overrides+=("model.draft_checkpoint_path=$INITIAL_DRAFT_CHECKPOINT")
elif [[ "$RESUME_PRETRAIN" == "true" && -f "$LATEST_CHECKPOINT/training_state.pt" ]]; then
  train_overrides+=("training.resume_from=$LATEST_CHECKPOINT")
fi

train_cmd=(
  "$PYTHON_BIN" -m specforge.cli train
  --config "$SPECFORGE_DIR/examples/configs/offline/colocated/qwen2.5-7b-eagle3-offline.yaml"
  "${train_overrides[@]}"
)
printf 'Train CUDA_VISIBLE_DEVICES=%s\n' "$TRAIN_CUDA_VISIBLE_DEVICES"
printf 'Run:'; printf ' %q' env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" "${train_cmd[@]}"; printf '\n'
env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" "${train_cmd[@]}" \
  2>&1 | tee -a "$PRETRAIN_ROOT/logs/train.log"

[[ -f "$LATEST_CHECKPOINT/training_state.pt" ]] || fail "SpecForge did not create $LATEST_CHECKPOINT/training_state.pt"
"$PYTHON_BIN" - "$PRETRAIN_MARKER" "$LATEST_CHECKPOINT" "$VOCAB_MAPPING" "$TARGET_MODEL_PATH" "$DRAFT_CONFIG" "$RUN_ID" "$PRETRAIN_ROOT/dependencies.json" <<'PY'
import json, os, sys
from pathlib import Path
marker, checkpoint, mapping, target, config = map(Path, sys.argv[1:6])
run_id = sys.argv[6]
dependencies_path = Path(sys.argv[7])
payload = {
    "status": "complete",
    "run_id": run_id,
    "target_model_path": str(target.resolve()),
    "checkpoint": str(checkpoint.resolve()),
    "vocab_mapping": str(mapping.resolve()),
    "draft_config": str(config.resolve()),
    "dependencies": json.loads(dependencies_path.read_text(encoding="utf-8")),
}
tmp = marker.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
os.replace(tmp, marker)
PY
publish_latest
echo "Run ID: $RUN_ID"
echo "Run directory: $PRETRAIN_ROOT"
echo "Draft checkpoint: $LATEST_CHECKPOINT"
echo "Vocabulary mapping: $VOCAB_MAPPING"
