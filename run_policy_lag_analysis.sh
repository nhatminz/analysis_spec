#!/usr/bin/env bash
# Retired experiment name forwards only to SimpleLR A1/A2.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_DIR/run_policy_lag_motivation.sh" "$@"
