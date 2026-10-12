#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
export TEST_DATASET_PATH="${TEST_DATASET_PATH:-/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" run_rq2_repeated_misses.py all \
  --root "${ANALYSIS_ROOT:-$PWD/outputs/e2_e6_fastgrpo}" \
  --steps 30 100 200 --bootstrap-samples 2000 --wall-minutes 150 \
  --require-b200 "$@"
