#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <env_type> <env_task> [extra draccus args...]" >&2
  exit 2
fi

ENV_TYPE="$1"
ENV_TASK="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.train.log}"
MONITOR_LOG="${MONITOR_LOG:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.monitor.log}"
WRAPPER_LOG="${WRAPPER_LOG:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.wrapper.log}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.train.pid}"
MONITOR_PID_FILE="${MONITOR_PID_FILE:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.monitor.pid}"

mkdir -p "${REPO_ROOT}/outputs/train" "${REPO_ROOT}/outputs/logs"

nohup env \
  RUN_TAG="${RUN_TAG}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_FILE="${TRAIN_LOG}" \
  WRAPPER_LOG="${WRAPPER_LOG}" \
  "${SCRIPT_DIR}/launch_smolvla_hybrid_train.sh" "${ENV_TYPE}" "${ENV_TASK}" "$@" \
  > "${WRAPPER_LOG}" 2>&1 &
TRAIN_PID="$!"
printf "%s\n" "${TRAIN_PID}" > "${TRAIN_PID_FILE}"

# Give launcher a moment to fail fast with a clear wrapper log if config is invalid.
sleep 2
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
  printf "ERROR: training launcher exited early. Check logs:\n" >&2
  printf "  TRAIN_LOG=%s\n" "${TRAIN_LOG}" >&2
  printf "  WRAPPER_LOG=%s\n" "${WRAPPER_LOG}" >&2
  if [[ -f "${WRAPPER_LOG}" ]]; then
    tail -n 30 "${WRAPPER_LOG}" >&2 || true
  fi
  exit 1
fi

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
printf "WRAPPER_LOG=%s\n" "${WRAPPER_LOG}"
printf "TRAIN_PID_FILE=%s\n" "${TRAIN_PID_FILE}"
printf "MONITOR_PID_FILE=%s\n" "${MONITOR_PID_FILE}"
