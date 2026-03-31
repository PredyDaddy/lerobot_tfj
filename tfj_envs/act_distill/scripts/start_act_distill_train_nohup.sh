#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs}"
MODE="${MODE:-full}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/act_distill_train_${MODE}_${TIMESTAMP}.log"

nohup "${SCRIPT_DIR}/launch_act_distill_train.sh" "$@" >"${LOG_PATH}" 2>&1 &
PID=$!

echo "Started ACT distillation training in background."
echo "mode: ${MODE}"
echo "pid: ${PID}"
echo "log: ${LOG_PATH}"
