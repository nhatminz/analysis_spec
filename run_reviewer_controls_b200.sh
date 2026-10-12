#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
# Task 1 never loads a model; hide CUDA as an additional execution guard.
export CUDA_VISIBLE_DEVICES=""
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" run_reviewer_controls.py \
  --root "${ANALYSIS_ROOT:-$PWD/outputs/e2_e6_fastgrpo}" \
  --steps 30 100 200 --bootstrap-samples 2000 \
  --permutations "${PERMUTATIONS:-500}" --seed 2026 "$@"
