#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
export TEST_DATASET_PATH="${TEST_DATASET_PATH:-/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet}"
mode="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${PYTHON_BIN:-python3}" run_rq2_aal_progress.py "$mode" --require-b200 "$@"
