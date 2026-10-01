#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$SCRIPT_DIR/third_party/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/workspace/storage-shared/models/Qwen2.5-7B-Instruct}"
DAPO_PARQUET="${DAPO_PARQUET:-$WORKSPACE/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}"
DAPO_SPLIT_DIR="${DAPO_SPLIT_DIR:-$WORKSPACE/outputs/fastgrpo/policy_lag/dapo_math_seed42}"
DAPO_ANALYSIS_SAMPLES="${DAPO_ANALYSIS_SAMPLES:-5000}"
DAPO_EVAL_SAMPLES="${DAPO_EVAL_SAMPLES:-512}"
DAPO_SPLIT_SEED="${DAPO_SPLIT_SEED:-42}"

PRETRAIN_ROOT="${PRETRAIN_ROOT:-$WORKSPACE/outputs/specforge/qwen25_7b_sharegpt_1ep}"
RUN_ID="${RUN_ID:-qwen25-7b-eagle3-sharegpt-1ep}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$PRETRAIN_ROOT/checkpoints/$RUN_ID-latest}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$SPECFORGE_DIR/configs/qwen2.5-7b-eagle3.json}"
VOCAB_MAPPING="${VOCAB_MAPPING:-$PRETRAIN_ROOT/features/vocab_mapping/vocab_mapping.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/outputs/fastgrpo/policy_lag/qwen25_7b_dapo5k}"

FORCE_DAPO_SPLIT="${FORCE_DAPO_SPLIT:-false}"
PREPARE_DAPO_ONLY="${PREPARE_DAPO_ONLY:-false}"

[[ -f "$DAPO_PARQUET" ]] || { echo "DAPO parquet not found: $DAPO_PARQUET" >&2; exit 2; }
export PYTHONPATH="$SPECFORGE_DIR:$SCRIPT_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

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
export DATASET_PATH="$DAPO_SPLIT_DIR/train.jsonl"
export EVAL_DATASET_PATH="$DAPO_SPLIT_DIR/eval.jsonl"
export TRAIN_SPLIT=train EVAL_SPLIT=eval TRAIN_OPTION=DAPO-math
export TRAIN_DATA_FRACTION=1.0 MAX_TRAIN_SAMPLES="$DAPO_ANALYSIS_SAMPLES"
export TRACE_SEED="${TRACE_SEED:-42}"
export ANALYSIS_BOUNDARIES="${ANALYSIS_BOUNDARIES:-1,5,10}"
export TOTAL_POLICY_STEPS="${TOTAL_POLICY_STEPS:-10}"
export EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
export TRAINING_TOKEN_BUDGET="${TRAINING_TOKEN_BUDGET:-1024}"
export DRAFT_UPDATE_STEPS="${DRAFT_UPDATE_STEPS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
export DRAFT_GRADIENT_ACCUMULATION="${DRAFT_GRADIENT_ACCUMULATION:-1}"
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}" TOP_P="${TOP_P:-0.95}"
export SAMPLING_SEEDS="${SAMPLING_SEEDS:-11,29,47}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export RESUME="${RESUME:-true}" ANALYSIS_RESUME="${ANALYSIS_RESUME:-true}"

exec bash "$SCRIPT_DIR/run_policy_lag_analysis.sh"
