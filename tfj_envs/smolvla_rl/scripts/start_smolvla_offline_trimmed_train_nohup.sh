#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}.train.log}"
MONITOR_LOG="${MONITOR_LOG:-${REPO_ROOT}/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}.monitor.log}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${REPO_ROOT}/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}.train.pid}"
MONITOR_PID_FILE="${MONITOR_PID_FILE:-${REPO_ROOT}/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}.monitor.pid}"

mkdir -p "${REPO_ROOT}/outputs/train" "${REPO_ROOT}/outputs/logs"

nohup env \
  RUN_TAG="${RUN_TAG}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_FILE="${TRAIN_LOG}" \
  "${SCRIPT_DIR}/launch_smolvla_offline_trimmed_train.sh" \
  > /dev/null 2>&1 &
TRAIN_PID="$!"
printf "%s\n" "${TRAIN_PID}" > "${TRAIN_PID_FILE}"

nohup "${SCRIPT_DIR}/monitor_training_process.sh" \
  "${TRAIN_PID}" \
  "${TRAIN_LOG}" \
  "${MONITOR_LOG}" \
  "${MONITOR_INTERVAL}" \
  > /dev/null 2>&1 &
MONITOR_PID="$!"
printf "%s\n" "${MONITOR_PID}" > "${MONITOR_PID_FILE}"

printf "RUN_TAG=%s\n" "${RUN_TAG}"
printf "TRAIN_PID=%s\n" "${TRAIN_PID}"
printf "MONITOR_PID=%s\n" "${MONITOR_PID}"
printf "OUTPUT_DIR=%s\n" "${OUTPUT_DIR}"
printf "TRAIN_LOG=%s\n" "${TRAIN_LOG}"
printf "MONITOR_LOG=%s\n" "${MONITOR_LOG}"
printf "TRAIN_PID_FILE=%s\n" "${TRAIN_PID_FILE}"
printf "MONITOR_PID_FILE=%s\n" "${MONITOR_PID_FILE}"
