#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export METHOD=fastgrpo
exec "${PYTHON_BIN:-python3}" "$PROJECT_DIR/run_e2_e6.py" "$@"
