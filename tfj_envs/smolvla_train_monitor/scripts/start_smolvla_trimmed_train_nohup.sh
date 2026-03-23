#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/tfj/lerobot_tfj"
SCRIPT_DIR="${REPO_ROOT}/tfj_envs/smolvla_train_monitor/scripts"

STEPS="${1:-10000}"
BATCH_SIZE="${2:-32}"
NUM_WORKERS="${3:-4}"
MONITOR_INTERVAL="${4:-60}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="smolvla_grasp_block_in_bin1_trimmed_static_tail_${TS}"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/${RUN_NAME}"
TRAIN_LOG="${REPO_ROOT}/outputs/logs/${RUN_NAME}.train.log"
MONITOR_LOG="${REPO_ROOT}/outputs/logs/${RUN_NAME}.monitor.log"
TRAIN_PID_FILE="${REPO_ROOT}/outputs/logs/${RUN_NAME}.train.pid"
MONITOR_PID_FILE="${REPO_ROOT}/outputs/logs/${RUN_NAME}.monitor.pid"

mkdir -p "${REPO_ROOT}/outputs/train" "${REPO_ROOT}/outputs/logs"

nohup "${SCRIPT_DIR}/launch_smolvla_trimmed_train.sh" \
  "${OUTPUT_DIR}" \
  "${TRAIN_LOG}" \
  "${STEPS}" \
  "${BATCH_SIZE}" \
  "${NUM_WORKERS}" \
  >/dev/null 2>&1 &
TRAIN_PID="$!"
printf "%s\n" "${TRAIN_PID}" > "${TRAIN_PID_FILE}"

nohup "${SCRIPT_DIR}/monitor_train_process.sh" \
  "${TRAIN_PID}" \
  "${TRAIN_LOG}" \
  "${MONITOR_LOG}" \
  "${MONITOR_INTERVAL}" \
  >/dev/null 2>&1 &
MONITOR_PID="$!"
printf "%s\n" "${MONITOR_PID}" > "${MONITOR_PID_FILE}"

printf "RUN_NAME=%s\n" "${RUN_NAME}"
printf "TRAIN_PID=%s\n" "${TRAIN_PID}"
printf "MONITOR_PID=%s\n" "${MONITOR_PID}"
printf "OUTPUT_DIR=%s\n" "${OUTPUT_DIR}"
printf "TRAIN_LOG=%s\n" "${TRAIN_LOG}"
printf "MONITOR_LOG=%s\n" "${MONITOR_LOG}"
printf "TRAIN_PID_FILE=%s\n" "${TRAIN_PID_FILE}"
printf "MONITOR_PID_FILE=%s\n" "${MONITOR_PID_FILE}"
