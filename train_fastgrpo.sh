#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible default: train Qwen2.5-7B FastGRPO.
MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-7B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"
