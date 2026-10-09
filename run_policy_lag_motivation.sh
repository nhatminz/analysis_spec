#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_KEY:=qwen25_3b}"
[[ "$MODEL_KEY" == qwen25_3b ]] || { echo 'ERROR: this experiment requires MODEL_KEY=qwen25_3b' >&2; exit 2; }
for variable in REFLEX_MODE REFLEX_LR REFLEX_FEATURE_DIM DAPO_PARQUET DAPO_SPLIT_DIR VOCAB_MAPPING DRAFT_CONFIG; do
  if [[ -v "$variable" ]]; then echo "ERROR: retired experiment setting $variable; use the A1/A2 CLI/config" >&2;exit 2;fi
 done
[[ "${DATASET:-simplelr}" == simplelr ]] || { echo 'ERROR: only DATASET=simplelr is supported' >&2;exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# All runtime imports live in this checkout; external PYTHONPATH is unnecessary.
export PYTHONPATH="$PROJECT_DIR"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
cmd=("$PYTHON_BIN" "$PROJECT_DIR/run_policy_lag_motivation.py")
for pair in MODEL:model DRAFT_CHECKPOINT:draft-checkpoint TRAIN_DATASET_PATH:train-path TEST_DATASET_PATH:test-path OUTPUT_DIR:output-dir TOTAL_POLICY_STEPS:train-steps EVAL_STEPS:eval-steps CONFIRMATION_STEPS:confirmation-steps SEED:seed; do
  variable="${pair%%:*}";flag="${pair#*:}"
  if [[ -v "$variable" ]]; then cmd+=("--$flag" "${!variable}"); fi
 done
cmd+=("$@")
if [[ "${DRY_RUN:-false}" == true ]]; then printf '%q ' "${cmd[@]}";printf '\n';exit 0;fi
exec "${cmd[@]}"
