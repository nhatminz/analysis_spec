#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PRETRAIN="${RUN_PRETRAIN:-true}"

if [[ "$RUN_PRETRAIN" == "true" ]]; then
  bash "$SCRIPT_DIR/pretrain_eagle3_sharegpt_b200.sh"
fi

if [[ "${PREPARE_ONLY:-false}" == "true" ]]; then
  exit 0
fi

exec bash "$SCRIPT_DIR/run_policy_lag_analysis_b200.sh"
