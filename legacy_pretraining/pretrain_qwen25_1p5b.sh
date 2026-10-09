#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="qwen25_1p5b"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-1.5B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/pretrain_model.sh"
