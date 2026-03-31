#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/one_click_common.sh"

POLICY_PATH="${1:-${POLICY_PATH:-${GROOT_TRT_REPO_ROOT}/tmp/train/groot_grasp/checkpoints/010000}}"
RUN_DIR="${2:-${RUN_DIR:-${GROOT_TRT_REPO_ROOT}/outputs/trt/groot_self_run_20260311_161210}}"
ENGINE_DIR="${3:-${ENGINE_DIR:-${RUN_DIR}/gr00t_engine_api_trt1013}}"

shift $(( $# >= 3 ? 3 : $# ))

groot_trt_require_conda

TMPDIR_RESOLVED="$(groot_trt_resolve_tmpdir "${RUN_DIR}")"

groot_trt_conda_python_with_trt \
  "${SCRIPT_DIR}/run_groot_trt_infer_so101.py" \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}" \
  --engine-dir "${ENGINE_DIR}" \
  --tmpdir "${TMPDIR_RESOLVED}" \
  "$@"
