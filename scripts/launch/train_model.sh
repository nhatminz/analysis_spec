#!/usr/bin/env bash

# Sourced by fastgrpo/train_<model>.sh after MODEL_KEY is set.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"

FASTGRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(cd "$FASTGRPO_DIR/.." && pwd)"

COMMON_ENV="${COMMON_ENV:-$FASTGRPO_DIR/configs/_shared/b200_common.env}"
MODEL_ENV="${MODEL_ENV:-$FASTGRPO_DIR/configs/${MODEL_KEY}/b200.env}"
[[ -f "$COMMON_ENV" ]] && source "$COMMON_ENV"
[[ -f "$MODEL_ENV" ]] && source "$MODEL_ENV"
: "${MODEL:?MODEL is required}"

cd "$WORKSPACE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$FASTGRPO_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

case "${DATASET,,}" in
  simplelr|simplerl|simplelr_abel|simplelr_abel_level3to5)
    TRAIN_OPTION="simplelr_abel_level3to5"
    DATASET_SLUG="simplelr_abel_l3to5"
    ;;
  simplelr_qwen|simplelr_qwen_level3to5)
    TRAIN_OPTION="simplelr_qwen_level3to5"
    DATASET_SLUG="simplelr_qwen_l3to5"
    ;;
  gsm8k)
    TRAIN_OPTION="gsm8k"
    DATASET_SLUG="gsm8k"
    ;;
  dapo|dapo-math|dapo_math)
    TRAIN_OPTION="DAPO-math"
    DATASET_SLUG="dapo_math"
    ;;
  *)
    TRAIN_OPTION="$DATASET"
    DATASET_SLUG="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
    DATASET_SLUG="${DATASET_SLUG%_}"
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
FRACTION_TAG="${TRAIN_DATA_FRACTION//./p}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
FASTGRPO_EXP="${FASTGRPO_EXP:-${EXP:-${MODEL_KEY}_fastgrpo_${DATASET_SLUG}_f${FRACTION_TAG}_${RUN_TAG}}}"

DRAFT_OUTPUT_DIR="${DRAFT_OUTPUT_DIR:-outputs/fastgrpo/pretrain/${MODEL_KEY}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/fastgrpo/train/${MODEL_KEY}/${FASTGRPO_EXP}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs/fastgrpo/train/${MODEL_KEY}/${FASTGRPO_EXP}}"
FASTGRPO_LOG_FILE="${FASTGRPO_LOG_FILE:-$TRAIN_LOG_DIR/fastgrpo.jsonl}"
FASTGRPO_SUMMARY_FILE="${FASTGRPO_SUMMARY_FILE:-$TRAIN_LOG_DIR/summary.json}"
FASTGRPO_SAVED_MODEL_DIR="${FASTGRPO_SAVED_MODEL_DIR:-$TRAIN_OUTPUT_DIR/target_lora}"
FASTGRPO_SAVED_DRAFT_MODEL_DIR="${FASTGRPO_SAVED_DRAFT_MODEL_DIR:-$TRAIN_OUTPUT_DIR/draft_model}"
FASTGRPO_SAVED_STATISTICS_DIR="${FASTGRPO_SAVED_STATISTICS_DIR:-$TRAIN_OUTPUT_DIR/statistics}"
FASTGRPO_CHECKPOINT_DIR="${FASTGRPO_CHECKPOINT_DIR:-$TRAIN_OUTPUT_DIR/checkpoints}"

if [[ "${RESUME_CHECKPOINT:-}" == "auto" ]]; then
  RESUME_CHECKPOINT=""
  if [[ -d "$FASTGRPO_CHECKPOINT_DIR" ]]; then
    RESUME_CHECKPOINT="$(find "$FASTGRPO_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step*.pt' | sort -V | tail -n 1 || true)"
  fi
fi

if [[ -z "${DRAFT_ADAPTER:-}" ]]; then
  DRAFT_ADAPTER=""
  if [[ -d "$DRAFT_OUTPUT_DIR" ]]; then
    DRAFT_ADAPTER="$(find "$DRAFT_OUTPUT_DIR" -maxdepth 1 -type f -name 'step*.pth' | sort -V | tail -n 1 || true)"
  fi
fi

cmd=(
  "$PYTHON_BIN" "$FASTGRPO_DIR/grpo_speculative.py"
  --model_dir "$MODEL"
  --adapter_path "$DRAFT_ADAPTER"
  --dtype "$MODEL_DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --load_lora_path "$LOAD_LORA_PATH"
  --model_type "$MODEL_TYPE"
  --train_option "$TRAIN_OPTION"
  --train_data_fraction "$TRAIN_DATA_FRACTION"
  --train_subset_seed "$TRAIN_SUBSET_SEED"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --version_name "$FASTGRPO_EXP"
  --batch_size "$BATCH_SIZE"
  --num_epochs "$NUM_EPOCHS"
  --sample_num "$SAMPLE_NUM"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --draft_accumulation_steps "$DRAFT_ACCUMULATION_STEPS"
  --target_lr "$TARGET_LR"
  --draft_lr "$FASTGRPO_DRAFT_LR"
  --is_train_draft "$IS_TRAIN_DRAFT"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_length "$GEN_MAX_LENGTH"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_training_padding_gap "$MAX_TRAINING_PADDING_GAP"
  --max_training_token "$MAX_TRAINING_TOKEN"
  --logps_chunk_size "$LOGPS_CHUNK_SIZE"
  --grpo_iteration_num "$GRPO_ITERATION_NUM"
  --repeated_generate_nums "$REPEATED_GENERATE_NUMS"
  --beta "$BETA"
  --epsilon "$EPSILON"
  --verification_capacity "$VERIFICATION_CAPACITY"
  --max_draft_token_length "$MAX_DRAFT_TOKEN_LENGTH"
  --max_draft_k "$MAX_DRAFT_K"
  --max_verification_num "$MAX_VERIFICATION_NUM"
  --min_draft_token_length "$MIN_DRAFT_TOKEN_LENGTH"
  --draft_token_length_c "$DRAFT_TOKEN_LENGTH_C"
  --statistical_time "${STATISTICAL_TIME:-False}"
  --num_workers "$NUM_WORKERS"
  --persistent_workers "$PERSISTENT_WORKERS"
  --log_file "$FASTGRPO_LOG_FILE"
  --summary_file "$FASTGRPO_SUMMARY_FILE"
  --saved_model_dir "$FASTGRPO_SAVED_MODEL_DIR"
  --saved_draft_model_dir "$FASTGRPO_SAVED_DRAFT_MODEL_DIR"
  --saved_statistics_dir "$FASTGRPO_SAVED_STATISTICS_DIR"
  --checkpoint_dir "$FASTGRPO_CHECKPOINT_DIR"
  --save_checkpoint_steps "$FASTGRPO_SAVE_CHECKPOINT_STEPS"
  --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
  --resume_checkpoint "$RESUME_CHECKPOINT"
  --append_log "${APPEND_LOG:-}"
  --seed "${TRACE_SEED:-$TRAIN_SUBSET_SEED}"
  --reset_rng_on_resume "${RESET_RNG_ON_RESUME:-false}"
  --max_grpo_steps "${MAX_GRPO_STEPS:-0}"
  --drift_topk "${DRIFT_TOPK:-16}"
  --drift_temperature "${DRIFT_TEMPERATURE:-1.0}"
  --drift_row_chunk_size "${DRIFT_ROW_CHUNK_SIZE:-32}"
  --draft_lr_multiplier "${DRAFT_LR_MULTIPLIER:-1.0}"
)
if (($#)); then
  cmd+=("$@")
fi

printf 'Run name : %s\nModel    : %s\nModel key: %s\nDataset  : %s (fraction=%s)\nDraft    : %s\nLogs     : %s\nSummary  : %s\nOutputs  : %s\n' \
  "$FASTGRPO_EXP" "$MODEL" "$MODEL_KEY" "$TRAIN_OPTION" "$TRAIN_DATA_FRACTION" "${DRAFT_ADAPTER:-<empty>}" "$TRAIN_LOG_DIR" "$FASTGRPO_SUMMARY_FILE" "$TRAIN_OUTPUT_DIR"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  return 0 2>/dev/null || exit 0
fi

[[ -f "$MODEL/config.json" ]] || { echo "Model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$DRAFT_ADAPTER" ]] || {
  echo "Cannot find FastGRPO draft adapter: ${DRAFT_ADAPTER:-<empty>}" >&2
  echo "Run: bash fastgrpo/pretrain_${MODEL_KEY}.sh" >&2
  echo "or pass: DRAFT_ADAPTER=/path/to/stepXXXX.pth bash fastgrpo/train_${MODEL_KEY}.sh" >&2
  exit 2
}
mkdir -p "$TRAIN_LOG_DIR" "$FASTGRPO_SAVED_MODEL_DIR" "$FASTGRPO_SAVED_DRAFT_MODEL_DIR" "$FASTGRPO_SAVED_STATISTICS_DIR" "$FASTGRPO_CHECKPOINT_DIR"
"${cmd[@]}" 2>&1 | tee -a "$TRAIN_LOG_DIR/console.log"
