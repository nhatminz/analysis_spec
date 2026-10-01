#!/usr/bin/env bash

# Sourced by fastgrpo/pretrain_<model>.sh after MODEL_KEY is set.
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRAFT_DATASET_SLUG="$(printf '%s' "$DRAFT_DATASET_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
DRAFT_DATASET_SLUG="${DRAFT_DATASET_SLUG%_}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRAFT_EXP="${DRAFT_EXP:-${EXP:-${MODEL_KEY}_draft_${DRAFT_DATASET_SLUG}_${RUN_TAG}}}"

DRAFT_OUTPUT_DIR="${DRAFT_OUTPUT_DIR:-outputs/fastgrpo/pretrain/${MODEL_KEY}}"
DRAFT_LOG_DIR="${DRAFT_LOG_DIR:-logs/fastgrpo/pretrain/${MODEL_KEY}/${DRAFT_EXP}}"
DRAFT_CHECKPOINT_DIR="${DRAFT_CHECKPOINT_DIR:-$DRAFT_OUTPUT_DIR/checkpoints}"

if [[ "${RESUME_CHECKPOINT:-}" == "auto" ]]; then
  RESUME_CHECKPOINT=""
  if [[ -d "$DRAFT_CHECKPOINT_DIR" ]]; then
    RESUME_CHECKPOINT="$(find "$DRAFT_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step*.pt' | sort -V | tail -n 1 || true)"
  fi
fi

cmd=(
  "$PYTHON_BIN" "$FASTGRPO_DIR/train_draft.py"
  --model_dir "$MODEL"
  --version_name "$DRAFT_EXP"
  --model_type "$MODEL_TYPE"
  --dtype "$MODEL_DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --batch_size "$DRAFT_PRETRAIN_BATCH_SIZE"
  --num_epochs "$DRAFT_PRETRAIN_EPOCHS"
  --lr "$DRAFT_PRETRAIN_LR"
  --accumulation_steps "$DRAFT_PRETRAIN_ACCUMULATION_STEPS"
  --warmup_ratio "$DRAFT_PRETRAIN_WARMUP_RATIO"
  --sample_num "$DRAFT_PRETRAIN_SAMPLE_NUM"
  --max_seq_len "$DRAFT_PRETRAIN_MAX_SEQ_LEN"
  --num_workers "$NUM_WORKERS"
  --persistent_workers "$PERSISTENT_WORKERS"
  --log_dir "$DRAFT_LOG_DIR"
  --saved_model_dir "$DRAFT_OUTPUT_DIR"
  --dataset_dir "$DRAFT_DATASET"
  --checkpoint_dir "$DRAFT_CHECKPOINT_DIR"
  --save_checkpoint_steps "$DRAFT_SAVE_CHECKPOINT_STEPS"
  --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
  --resume_checkpoint "$RESUME_CHECKPOINT"
)
if (($#)); then
  cmd+=("$@")
fi

printf 'Run name : %s\nModel    : %s\nModel key: %s\nDataset  : %s\nLogs     : %s\nOutputs  : %s\n' \
  "$DRAFT_EXP" "$MODEL" "$MODEL_KEY" "$DRAFT_DATASET" "$DRAFT_LOG_DIR" "$DRAFT_OUTPUT_DIR"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  return 0 2>/dev/null || exit 0
fi

[[ -f "$MODEL/config.json" ]] || { echo "Model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$DRAFT_DATASET" ]] || { echo "ShareGPT draft pretrain data not found: $DRAFT_DATASET" >&2; exit 2; }
mkdir -p "$DRAFT_LOG_DIR" "$DRAFT_OUTPUT_DIR" "$DRAFT_CHECKPOINT_DIR"
"${cmd[@]}" 2>&1 | tee -a "$DRAFT_LOG_DIR/console.log"
