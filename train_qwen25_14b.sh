#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="qwen25_14b"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-14B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"
