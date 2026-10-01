#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="llama31_8b"
MODEL="${MODEL:-/workspace/storage-shared/models/Llama-3.1-8B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"
