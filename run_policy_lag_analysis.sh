#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$SCRIPT_DIR/third_party/SpecForge}"
SPECNAACL_DIR="${SPECNAACL_DIR:-$SCRIPT_DIR/../SpecNaacl}"

# All experiment knobs are overrideable here or as environment variables.
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
TARGET_ADAPTER_PATH="${TARGET_ADAPTER_PATH:-}"
TARGET_RESUME_CHECKPOINT="${TARGET_RESUME_CHECKPOINT:-}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$SPECNAACL_DIR/outputs/pretrain/qwen25_3b/latest_checkpoint}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$SPECNAACL_DIR/outputs/pretrain/qwen25_3b/latest_draft_config.json}"
DRAFT_INITIALIZATION_MODE="${DRAFT_INITIALIZATION_MODE:-pretrained}" # pretrained|random (explicit)
VOCAB_MAPPING="${VOCAB_MAPPING-$SPECNAACL_DIR/outputs/pretrain/qwen25_3b/latest_vocab_mapping.pt}" # explicit empty: full vocab

DATASET_PATH="${TRAIN_DATASET_PATH:-${DATASET_PATH:-$WORKSPACE/data/gsm8k/main}}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TRAIN_OPTION="${TRAIN_OPTION:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/outputs/fastgrpo/policy_lag/qwen25_3b_reflex_v5_$(date -u +%Y%m%dT%H%M%S)}"
ANALYSIS_BOUNDARIES="${ANALYSIS_BOUNDARIES:-1}"
ANALYSIS_INTERVAL="${ANALYSIS_INTERVAL:-5}"
TOTAL_POLICY_STEPS="${TOTAL_POLICY_STEPS:-600}" # absolute target step; boundary 600 excluded
NUM_EPOCHS="${NUM_EPOCHS:-2}" # minimum; analysis auto-extends until the step limit
ANALYSIS_KEEP_BRANCH_CHECKPOINTS="${ANALYSIS_KEEP_BRANCH_CHECKPOINTS:-0}"
ANALYSIS_EVAL_PROMPTS="${ANALYSIS_EVAL_PROMPTS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

DRAFT_TOKEN_BUDGET="${DRAFT_TOKEN_BUDGET:-8192}"
TARGET_MAX_TRAINING_TOKEN="${TARGET_MAX_TRAINING_TOKEN:-1024}"
DRAFT_UPDATE_STEPS="${DRAFT_UPDATE_STEPS:-1}"
DRAFT_LR="${DRAFT_LR:-1e-5}"
TARGET_LR="${TARGET_LR:-1e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$ANALYSIS_EVAL_PROMPTS}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
DRAFT_GRADIENT_ACCUMULATION="${DRAFT_GRADIENT_ACCUMULATION:-1}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TRACE_SEED="${TRACE_SEED:-42}"
REFLEX_BACKEND="${REFLEX_BACKEND:-triton}"
REFLEX_FEEDBACK_SCOPE="${REFLEX_FEEDBACK_SCOPE:-root}"
REFLEX_PROPOSAL_STRATEGY="${REFLEX_PROPOSAL_STRATEGY:-fused}"
REFLEX_CORRECTION_STRATEGY="${REFLEX_CORRECTION_STRATEGY:-serial}"
REFLEX_FEEDBACK_STRATEGY="${REFLEX_FEEDBACK_STRATEGY:-serial}"
REFLEX_FEATURE_STRATEGY="${REFLEX_FEATURE_STRATEGY:-auto}"
REFLEX_UPDATE_STREAM="${REFLEX_UPDATE_STREAM:-1}"
REFLEX_FEATURE_DIM="${REFLEX_FEATURE_DIM:-8}"
REFLEX_LR="${REFLEX_LR:-0.05}"
REFLEX_WEIGHT_DECAY="${REFLEX_WEIGHT_DECAY:-0.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Existing FastGRPO concurrency-aware scheduler defaults.
VERIFICATION_CAPACITY="${VERIFICATION_CAPACITY:-512}"
MAX_VERIFICATION_NUM="${MAX_VERIFICATION_NUM:-512}"
MAX_DRAFT_TOKEN_LENGTH="${MAX_DRAFT_TOKEN_LENGTH:-5}"
MAX_DRAFT_K="${MAX_DRAFT_K:-8}"
MIN_DRAFT_TOKEN_LENGTH="${MIN_DRAFT_TOKEN_LENGTH:-3}"
DRAFT_TOKEN_LENGTH_C="${DRAFT_TOKEN_LENGTH_C:-0.75}"

RESUME="${RESUME:-true}"
SMOKE_TEST="${SMOKE_TEST:-false}"
ANALYSIS_RESUME="${ANALYSIS_RESUME:-true}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STATISTICAL_TIME="${STATISTICAL_TIME:-false}"
DRY_RUN="${DRY_RUN:-false}"

if [[ "$NPROC_PER_NODE" != "1" ]]; then
  echo "FastGRPO upstream decoder is single-process; set NPROC_PER_NODE=1 and select one GPU with CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
if [[ -n "${TRAINING_TOKEN_BUDGET:-}" ]]; then
  echo "TRAINING_TOKEN_BUDGET is obsolete; use DRAFT_TOKEN_BUDGET and TARGET_MAX_TRAINING_TOKEN separately." >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE != ANALYSIS_EVAL_PROMPTS || EVAL_BATCH_SIZE <= 0 ||
      DRAFT_TOKEN_BUDGET <= 0 || TARGET_MAX_TRAINING_TOKEN <= 0 )); then
  echo "Require TRAIN_BATCH_SIZE=ANALYSIS_EVAL_PROMPTS and positive eval batch/budgets." >&2
  exit 2
fi
if (( DRAFT_UPDATE_STEPS <= 0 )); then
  echo "DRAFT_UPDATE_STEPS must be positive." >&2
  exit 2
fi
if (( NUM_EPOCHS <= 0 )); then
  echo "NUM_EPOCHS must be positive." >&2
  exit 2
fi
if [[ "$SMOKE_TEST" != "true" ]] && (( TOTAL_POLICY_STEPS > 0 )) && [[ -n "$ANALYSIS_BOUNDARIES" ]]; then
  IFS=',' read -ra boundary_values <<< "$ANALYSIS_BOUNDARIES"
  for boundary_value in "${boundary_values[@]}"; do
    if (( boundary_value >= TOTAL_POLICY_STEPS )); then
      echo "TOTAL_POLICY_STEPS must exceed each ANALYSIS_BOUNDARIES value: boundary $boundary_value needs a following real GRPO update." >&2
      exit 2
    fi
  done
fi
if [[ "$DRAFT_INITIALIZATION_MODE" == "pretrained" && -z "$DRAFT_CHECKPOINT" ]]; then
  echo "DRAFT_INITIALIZATION_MODE=pretrained requires DRAFT_CHECKPOINT. Set DRAFT_INITIALIZATION_MODE=random explicitly for one-time config-compatible random initialization." >&2
  exit 2
fi
if [[ "$DRAFT_INITIALIZATION_MODE" != "pretrained" && "$DRAFT_INITIALIZATION_MODE" != "random" ]]; then
  echo "DRAFT_INITIALIZATION_MODE must be pretrained or random." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT
export PYTHONPATH="$SPECFORGE_DIR:$SCRIPT_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/target_lora" "$OUTPUT_DIR/draft" "$OUTPUT_DIR/statistics" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/analysis"

validation=(
  "$PYTHON_BIN" "$SCRIPT_DIR/policy_lag_analysis.py"
  --mode validate --output-dir "$OUTPUT_DIR/analysis"
  --model-dir "$TARGET_MODEL_PATH" --target-adapter "$TARGET_ADAPTER_PATH"
  --draft-checkpoint "$DRAFT_CHECKPOINT" --draft-config "$DRAFT_CONFIG"
  --draft-initialization-mode "$DRAFT_INITIALIZATION_MODE"
  --vocab-mapping "$VOCAB_MAPPING" --dataset-path "$DATASET_PATH"
  --eval-dataset-path "$EVAL_DATASET_PATH"
  --reflex-backend "$REFLEX_BACKEND"
)

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'Validation:'; printf ' %q' "${validation[@]}"; printf '\n'
else
  "${validation[@]}"
fi

if [[ "$SMOKE_TEST" == "true" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/policy_lag_analysis.py" --mode smoke --output-dir "$OUTPUT_DIR/analysis/plumbing_smoke"
  TOTAL_POLICY_STEPS=2
  ANALYSIS_BOUNDARIES=1
  ANALYSIS_INTERVAL=0
  MAX_TRAIN_SAMPLES=2
  TRAIN_BATCH_SIZE=1
  ANALYSIS_EVAL_PROMPTS=1
  GRADIENT_ACCUMULATION=1
  RESPONSES_PER_PROMPT=2
  MAX_LENGTH="${SMOKE_MAX_LENGTH:-128}"
  MAX_PROMPT_LENGTH="${SMOKE_MAX_PROMPT_LENGTH:-96}"
  DRAFT_TOKEN_BUDGET="${SMOKE_DRAFT_TOKEN_BUDGET:-32}"
  TARGET_MAX_TRAINING_TOKEN="${SMOKE_TARGET_MAX_TRAINING_TOKEN:-32}"
  BOOTSTRAP_SAMPLES=100
fi

resume_checkpoint="$TARGET_RESUME_CHECKPOINT"
if [[ "$RESUME" == "true" && -z "$resume_checkpoint" && -f "$OUTPUT_DIR/checkpoints/latest.pt" ]]; then
  resume_checkpoint="$OUTPUT_DIR/checkpoints/latest.pt"
fi

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/grpo_speculative.py"
  --model_dir "$TARGET_MODEL_PATH"
  --load_lora_path "$TARGET_ADAPTER_PATH"
  --resume_checkpoint "$resume_checkpoint"
  --adapter_path "$DRAFT_CHECKPOINT"
  --draft_backend eagle3
  --draft_config "$DRAFT_CONFIG"
  --draft_initialization_mode "$DRAFT_INITIALIZATION_MODE"
  --vocab_mapping "$VOCAB_MAPPING"
  --eagle_ttt_length 7
  --dtype "$MODEL_DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --model_type qwen2
  --train_option "$TRAIN_OPTION"
  --dataset_path "$DATASET_PATH"
  --eval_dataset_path "$EVAL_DATASET_PATH"
  --train_split "$TRAIN_SPLIT"
  --train_data_fraction "$TRAIN_DATA_FRACTION"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --train_subset_seed "$TRACE_SEED"
  --version_name policy_lag
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_epochs "$NUM_EPOCHS"
  --sample_num 100
  --accumulation_steps "$GRADIENT_ACCUMULATION"
  --draft_accumulation_steps "$DRAFT_GRADIENT_ACCUMULATION"
  --target_lr "$TARGET_LR"
  --draft_lr "$DRAFT_LR"
  --is_train_draft True
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_length "$MAX_LENGTH"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_training_token "$TARGET_MAX_TRAINING_TOKEN"
  --max_training_padding_gap 4096
  --grpo_iteration_num 1
  --repeated_generate_nums "$RESPONSES_PER_PROMPT"
  --beta 0.04 --epsilon 0.1
  --verification_capacity "$VERIFICATION_CAPACITY"
  --max_verification_num "$MAX_VERIFICATION_NUM"
  --max_draft_token_length "$MAX_DRAFT_TOKEN_LENGTH"
  --max_draft_k "$MAX_DRAFT_K"
  --min_draft_token_length "$MIN_DRAFT_TOKEN_LENGTH"
  --draft_token_length_c "$DRAFT_TOKEN_LENGTH_C"
  --statistical_time "$STATISTICAL_TIME"
  --num_workers "$NUM_WORKERS" --persistent_workers false
  --log_file "$OUTPUT_DIR/logs/train.jsonl"
  --summary_file "$OUTPUT_DIR/logs/summary.json"
  --saved_model_dir "$OUTPUT_DIR/target_lora"
  --saved_draft_model_dir "$OUTPUT_DIR/draft"
  --saved_statistics_dir "$OUTPUT_DIR/statistics"
  --checkpoint_dir "$OUTPUT_DIR/checkpoints"
  --save_checkpoint_steps 1 --keep_last_checkpoints 3
  --seed "$TRACE_SEED" --max_grpo_steps "$TOTAL_POLICY_STEPS"
  --policy_lag_output_dir "$OUTPUT_DIR/analysis"
  --analysis_boundaries "$ANALYSIS_BOUNDARIES"
  --analysis_interval "$ANALYSIS_INTERVAL"
  --analysis_eval_prompts "$ANALYSIS_EVAL_PROMPTS"
  --analysis_eval_batch_size "$EVAL_BATCH_SIZE"
  --analysis_training_token_budget "$DRAFT_TOKEN_BUDGET"
  --analysis_draft_update_steps "$DRAFT_UPDATE_STEPS"
  --analysis_bootstrap_samples "$BOOTSTRAP_SAMPLES"
  --analysis_resume "$ANALYSIS_RESUME"
  --analysis_keep_branch_checkpoints "$ANALYSIS_KEEP_BRANCH_CHECKPOINTS"
  --reflex_backend "$REFLEX_BACKEND"
  --reflex_feedback_scope "$REFLEX_FEEDBACK_SCOPE"
  --reflex_proposal_strategy "$REFLEX_PROPOSAL_STRATEGY"
  --reflex_correction_strategy "$REFLEX_CORRECTION_STRATEGY"
  --reflex_feedback_strategy "$REFLEX_FEEDBACK_STRATEGY"
  --reflex_feature_strategy "$REFLEX_FEATURE_STRATEGY"
  --reflex_update_stream "$REFLEX_UPDATE_STREAM"
  --reflex_feature_dim "$REFLEX_FEATURE_DIM"
  --reflex_lr "$REFLEX_LR"
  --reflex_weight_decay "$REFLEX_WEIGHT_DECAY"
)

printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" != "true" ]]; then
  "${cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/logs/console.log"
fi
