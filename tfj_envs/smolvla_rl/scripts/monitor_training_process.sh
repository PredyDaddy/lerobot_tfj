#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <train_pid> <train_log> <monitor_log> [interval_seconds]" >&2
  exit 2
fi

TRAIN_PID="$1"
TRAIN_LOG="$2"
MONITOR_LOG="$3"
INTERVAL_SECONDS="${4:-60}"

mkdir -p "$(dirname "${MONITOR_LOG}")"

log_line() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$1" >>"${MONITOR_LOG}"
}

latest_signal_line() {
  if [[ -f "${TRAIN_LOG}" ]]; then
    rg -n "step=|step:|Checkpoint policy after step|End of training|Traceback|RuntimeError|ERROR|Exception" \
      "${TRAIN_LOG}" 2>/dev/null | tail -n 1 | sed 's/[[:space:]]\+/ /g'
  fi
}

latest_tail_line() {
  if [[ -f "${TRAIN_LOG}" ]]; then
    tail -n 20 "${TRAIN_LOG}" 2>/dev/null | tail -n 1 | sed 's/[[:space:]]\+/ /g'
  fi
}

log_line "monitor_start pid=${TRAIN_PID} train_log=${TRAIN_LOG}"

while true; do
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    proc="$(ps -p "${TRAIN_PID}" -o etime=,%cpu=,%mem= --no-headers | xargs || true)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 | xargs || true)"
    last="$(latest_signal_line)"
    log_line "alive pid=${TRAIN_PID} proc=\"${proc}\" gpu=\"${gpu}\" latest=\"${last}\""
    sleep "${INTERVAL_SECONDS}"
  else
    last="$(latest_signal_line)"
    tail_last="$(latest_tail_line)"
    log_line "finished pid=${TRAIN_PID} latest=\"${last}\" tail=\"${tail_last}\""
    break
  fi
done
