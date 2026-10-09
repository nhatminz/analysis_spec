#!/usr/bin/env bash
# Actual production shape, real checkpoints/data/rewards, one real boundary.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_DIR" PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_DIR/.cache/inductor}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -c 'from motivation.compatibility import validate_cuda_runtime; print(validate_cuda_runtime(require_b200=True))'
: "${MODEL:?Set MODEL to the absolute existing Qwen2.5-3B-Instruct directory}"
: "${DRAFT_CHECKPOINT:?Set DRAFT_CHECKPOINT to the absolute existing SpecNaacl pretrained checkpoint}"
: "${TRAIN_DATASET_PATH:?Set TRAIN_DATASET_PATH to the official SimpleLR train.parquet}"
: "${TEST_DATASET_PATH:?Set TEST_DATASET_PATH to the official SimpleLR test.parquet}"
for path in "$MODEL" "$DRAFT_CHECKPOINT" "$TRAIN_DATASET_PATH" "$TEST_DATASET_PATH"; do
  [[ "$path" == /* && -e "$path" ]] || { echo "ERROR: absolute existing input required: $path" >&2; exit 2; }
done
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/b200_execution_validation}"
mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" -m pytest -q "$PROJECT_DIR/tests" >"$OUTPUT_DIR/pytest.log" 2>&1
common=(--require-b200 --validation-run --train-steps 1 --eval-steps 1 --confirmation-steps '' --zero-drift-control-steps 1)
# No --smoke: preserve 8x8 per learner, 2048 total length and 16 held-out prompts.
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/run_policy_lag_motivation.sh" "${common[@]}" --mode validate
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/run_policy_lag_motivation.sh" "${common[@]}"
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/run_policy_lag_motivation.sh" "${common[@]}" --resume auto
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/run_policy_lag_motivation.sh" --mode replot
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/run_policy_lag_motivation.sh" --mode report
"$PYTHON_BIN" "$PROJECT_DIR/scripts/verify_execution.py" --output-dir "$OUTPUT_DIR" --require-b200
