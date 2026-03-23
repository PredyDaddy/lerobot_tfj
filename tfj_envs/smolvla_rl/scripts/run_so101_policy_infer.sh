#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SAVE_DATASET="${SAVE_DATASET:-false}"
export DATASET_VIDEO="${DATASET_VIDEO:-false}"
export CLEAR_DATASET_ROOT="${CLEAR_DATASET_ROOT:-false}"

exec "${SCRIPT_DIR}/run_so101_policy_record.sh" "$@"

