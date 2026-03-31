#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-full}"
exec "${SCRIPT_DIR}/scripts/start_act_distill_train_nohup.sh" "$@"
