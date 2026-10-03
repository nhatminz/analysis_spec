#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$SCRIPT_DIR/third_party/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Policy-lag v4 evaluates the next real GRPO batch of eight training prompts.
# The stale rollout of that batch is reused for the target update.
ANALYSIS_EVAL_PROMPTS="${ANALYSIS_EVAL_PROMPTS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
DRAFT_TOKEN_BUDGET="${DRAFT_TOKEN_BUDGET:-8192}"
TARGET_MAX_TRAINING_TOKEN="${TARGET_MAX_TRAINING_TOKEN:-1024}"
DRAFT_LR="${DRAFT_LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$ANALYSIS_EVAL_PROMPTS}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${DATASET_PATH:-}}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-}"
ANALYSIS_BOUNDARIES="${ANALYSIS_BOUNDARIES:-1,5,10}"
ANALYSIS_INTERVAL="${ANALYSIS_INTERVAL:-0}"
TOTAL_POLICY_STEPS="${TOTAL_POLICY_STEPS:-11}"
TRACE_SEED="${TRACE_SEED:-42}"

GLOBAL_LATEST_RUN="$WORKSPACE/outputs/specforge/latest_run"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-}"
if [[ -z "$TARGET_MODEL_PATH" && -z "$PRETRAIN_ROOT" && -e "$GLOBAL_LATEST_RUN" ]]; then
  PRETRAIN_ROOT="$(readlink -f "$GLOBAL_LATEST_RUN")"
fi
if [[ -z "$TARGET_MODEL_PATH" && -n "$PRETRAIN_ROOT" && -f "$PRETRAIN_ROOT/checkpoints/pretrain_complete.json" ]]; then
  TARGET_MODEL_PATH="$($PYTHON_BIN - "$PRETRAIN_ROOT/checkpoints/pretrain_complete.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['target_model_path'])
PY
)"
fi
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
MODEL_BASENAME="$(basename "${TARGET_MODEL_PATH%/}")"
MODEL_SLUG="$(printf '%s' "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
MODEL_SLUG="${MODEL_SLUG%_}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-$WORKSPACE/outputs/specforge/$MODEL_SLUG}"
if [[ -z "$PRETRAIN_ROOT" && -e "$MODEL_OUTPUT_ROOT/latest_run" ]]; then
  PRETRAIN_ROOT="$(readlink -f "$MODEL_OUTPUT_ROOT/latest_run")"
fi
PRETRAIN_RUN_ID="$(basename "${PRETRAIN_ROOT:-unresolved_latest}")"
if [[ -n "$PRETRAIN_ROOT" && -f "$PRETRAIN_ROOT/checkpoints/pretrain_complete.json" ]]; then
  "$PYTHON_BIN" - "$PRETRAIN_ROOT/checkpoints/pretrain_complete.json" "$TARGET_MODEL_PATH" <<'PY'
import json, os, sys
recorded = os.path.realpath(json.load(open(sys.argv[1], encoding='utf-8'))['target_model_path'])
requested = os.path.realpath(sys.argv[2])
if recorded != requested:
    raise SystemExit(f"pretrain target mismatch: checkpoint={recorded}, requested={requested}")
PY
fi
DAPO_PARQUET="${DAPO_PARQUET:-$WORKSPACE/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}"
DAPO_SPLIT_DIR="${DAPO_SPLIT_DIR:-$WORKSPACE/outputs/fastgrpo/policy_lag/dapo_math_seed42}"
DAPO_ANALYSIS_SAMPLES="${DAPO_ANALYSIS_SAMPLES:-5000}"
DAPO_EVAL_SAMPLES="${DAPO_EVAL_SAMPLES:-512}"
DAPO_SPLIT_SEED="${DAPO_SPLIT_SEED:-42}"

DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$MODEL_OUTPUT_ROOT/latest_checkpoint}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$MODEL_OUTPUT_ROOT/latest_draft_config.json}"
VOCAB_MAPPING="${VOCAB_MAPPING:-$MODEL_OUTPUT_ROOT/latest_vocab_mapping.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/outputs/fastgrpo/policy_lag/$MODEL_SLUG/${PRETRAIN_RUN_ID}_dapo5k}"

FORCE_DAPO_SPLIT="${FORCE_DAPO_SPLIT:-false}"
PREPARE_DAPO_ONLY="${PREPARE_DAPO_ONLY:-false}"

# Catch the pre-v4 launcher before expensive data preparation. Evaluation
# prompt count is now a supported flag; only the old seeds flag is invalid.
if grep -Eq -- '--analysis_seeds([[:space:]]|$)|--num_epochs[[:space:]]+1000000([[:space:]]|$)' "$SCRIPT_DIR/run_policy_lag_analysis.sh"; then
  echo "Incompatible run_policy_lag_analysis.sh: old policy-lag CLI/epoch settings detected." >&2
  echo "Sync run_policy_lag_analysis.sh and run_policy_lag_analysis_b200.sh from the same fastgrpo revision." >&2
  exit 2
fi
for required_flag in --analysis_eval_prompts --analysis_eval_batch_size --analysis_training_token_budget; do
  if ! grep -Fq -- "$required_flag" "$SCRIPT_DIR/run_policy_lag_analysis.sh"; then
    echo "Incompatible generic launcher: missing $required_flag. Deploy both v4 launchers together." >&2
    exit 2
  fi
done

export PYTHONPATH="$SPECFORGE_DIR:$SCRIPT_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

custom_train_path="$TRAIN_DATASET_PATH"
if [[ -n "$custom_train_path" || -n "${EVAL_DATASET_PATH:-}" ]]; then
  if [[ -z "$custom_train_path" || -z "${EVAL_DATASET_PATH:-}" ]]; then
    echo "Override both TRAIN_DATASET_PATH and EVAL_DATASET_PATH together." >&2
    exit 2
  fi
else
  [[ -f "$DAPO_PARQUET" ]] || { echo "DAPO parquet not found: $DAPO_PARQUET" >&2; exit 2; }
  split_cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_dapo_policy_lag.py"
    --input-parquet "$DAPO_PARQUET"
    --output-dir "$DAPO_SPLIT_DIR"
    --train-samples "$DAPO_ANALYSIS_SAMPLES"
    --eval-samples "$DAPO_EVAL_SAMPLES"
    --seed "$DAPO_SPLIT_SEED"
  )
  [[ "$FORCE_DAPO_SPLIT" == "true" ]] && split_cmd+=(--force)
  printf 'Run:'; printf ' %q' "${split_cmd[@]}"; printf '\n'
  "${split_cmd[@]}"
fi

if [[ "$PREPARE_DAPO_ONLY" == "true" ]]; then
  exit 0
fi

if [[ "${DRY_RUN:-false}" != "true" ]]; then
  [[ -f "$DRAFT_CHECKPOINT/training_state.pt" ]] || {
    echo "SpecForge draft checkpoint not found: $DRAFT_CHECKPOINT/training_state.pt" >&2
    echo "Run: bash $SCRIPT_DIR/pretrain_eagle3_sharegpt_b200.sh" >&2
    exit 2
  }
  [[ -f "$VOCAB_MAPPING" ]] || { echo "Vocabulary mapping not found: $VOCAB_MAPPING" >&2; exit 2; }
fi

export SPECFORGE_DIR PYTHON_BIN TARGET_MODEL_PATH DRAFT_CHECKPOINT DRAFT_CONFIG VOCAB_MAPPING OUTPUT_DIR
export DRAFT_INITIALIZATION_MODE=pretrained
export DATASET_PATH="${custom_train_path:-$DAPO_SPLIT_DIR/train.jsonl}"
export EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-$DAPO_SPLIT_DIR/eval.jsonl}"
export TRAIN_SPLIT=train TRAIN_OPTION=DAPO-math
export TRAIN_DATA_FRACTION=1.0 MAX_TRAIN_SAMPLES="$DAPO_ANALYSIS_SAMPLES"
export TRACE_SEED ANALYSIS_BOUNDARIES ANALYSIS_INTERVAL TOTAL_POLICY_STEPS
export DRAFT_TOKEN_BUDGET TARGET_MAX_TRAINING_TOKEN DRAFT_LR ANALYSIS_EVAL_PROMPTS
export DRAFT_UPDATE_STEPS="${DRAFT_UPDATE_STEPS:-1}"
export TRAIN_BATCH_SIZE EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
export DRAFT_GRADIENT_ACCUMULATION="${DRAFT_GRADIENT_ACCUMULATION:-1}"
export RESPONSES_PER_PROMPT
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}" TOP_P="${TOP_P:-0.95}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export RESUME="${RESUME:-true}" ANALYSIS_RESUME="${ANALYSIS_RESUME:-true}"

exec bash "$SCRIPT_DIR/run_policy_lag_analysis.sh"
