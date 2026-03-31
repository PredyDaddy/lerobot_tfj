#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR=${LOG_DIR:-outputs/logs}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-${LOG_DIR}/smolvla_grasp_block_in_bin1_${TIMESTAMP}.log}
PID_FILE=${PID_FILE:-${LOG_FILE%.log}.pid}

mkdir -p "${LOG_DIR}"

nohup bash scripts/train_smolvla_grasp_block_in_bin1.sh >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started SmolVLA training in background."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Watch logs: tail -f ${LOG_FILE}"
echo "Stop training: kill ${PID}"
