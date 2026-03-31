#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/one_click_common.sh"

groot_trt_require_conda

POLICY_PATH="${1:-}"
RUN_DIR="${2:-}"
OUT_DIR="${3:-}"
if [[ -z "${POLICY_PATH}" || -z "${RUN_DIR}" ]]; then
  groot_trt_print_usage_header "$(basename "$0")" "POLICY_PATH RUN_DIR [OUT_DIR]"
  echo "Run the local GROOT TRT smoke/runtime script against an existing RUN_DIR." >&2
  exit 1
fi

RUN_DIR="$(cd -- "${RUN_DIR}" && pwd)"
GROOT_TRT_TMPDIR="$(groot_trt_resolve_tmpdir "${RUN_DIR}")"
ENGINE_DIR="${ENGINE_DIR:-${RUN_DIR}/gr00t_engine_api_trt1013}"
SOURCE="${SOURCE:-random}"
NUM_STEPS="${NUM_STEPS:-4}"
SEED="${SEED:-1234}"
TASK="${TASK:-Perform the task.}"
ROBOT_TYPE="${ROBOT_TYPE:-}"
REFRESH_OBS_PER_STEP="${REFRESH_OBS_PER_STEP:-0}"

if [[ "${SOURCE}" != "random" ]]; then
  echo "[ERR] Current local one-click run only supports SOURCE=random." >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${RUN_DIR}/local_run_$(date +%Y%m%d_%H%M%S)"
fi

RUN_ARGS=(
  "${SCRIPT_DIR}/run_groot_infer_trt_local.py"
  "--policy-path" "${POLICY_PATH}"
  "--engine-dir" "${ENGINE_DIR}"
  "--out-dir" "${OUT_DIR}"
  "--source" "${SOURCE}"
  "--num-steps" "${NUM_STEPS}"
  "--seed" "${SEED}"
  "--task" "${TASK}"
  "--device" "cuda"
)
if [[ -n "${ROBOT_TYPE}" ]]; then
  RUN_ARGS+=("--robot-type" "${ROBOT_TYPE}")
fi
if [[ "${REFRESH_OBS_PER_STEP}" == "1" ]]; then
  RUN_ARGS+=("--refresh-observation-per-step")
fi

groot_trt_conda_python_with_trt "${RUN_ARGS[@]}"

echo "[OK] GROOT local TRT run finished."
echo "OUT_DIR=${OUT_DIR}"
echo "REPORT=${OUT_DIR}/run_report.json"
echo "TMPDIR=${GROOT_TRT_TMPDIR}"
