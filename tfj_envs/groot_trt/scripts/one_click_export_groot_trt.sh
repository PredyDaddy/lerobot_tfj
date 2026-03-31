#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/one_click_common.sh"

groot_trt_require_conda

POLICY_PATH="${1:-}"
RUN_DIR="${2:-}"
if [[ -z "${POLICY_PATH}" ]]; then
  groot_trt_print_usage_header "$(basename "$0")" "POLICY_PATH [RUN_DIR]"
  echo "Export and build the 7 GROOT TRT artifacts into RUN_DIR." >&2
  exit 1
fi

VIDEO_VIEWS="${VIDEO_VIEWS:-2}"
read -r DEFAULT_SEQ_LEN DEFAULT_MIN_SEQ_LEN DEFAULT_OPT_SEQ_LEN DEFAULT_MAX_SEQ_LEN <<<"$(groot_trt_default_profile "${VIDEO_VIEWS}")"
SEQ_LEN="${SEQ_LEN:-${DEFAULT_SEQ_LEN}}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-${DEFAULT_MIN_SEQ_LEN}}"
OPT_SEQ_LEN="${OPT_SEQ_LEN:-${DEFAULT_OPT_SEQ_LEN}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-${DEFAULT_MAX_SEQ_LEN}}"

STATE_HORIZON="${STATE_HORIZON:-1}"
OPSET="${OPSET:-19}"
MAX_BATCH="${MAX_BATCH:-2}"
OPT_BATCH="${OPT_BATCH:-1}"
VIT_OPT_BATCH="${VIT_OPT_BATCH:-${VIDEO_VIEWS}}"
WORKSPACE_GB="${WORKSPACE_GB:-8.0}"
STRICT_LOAD="${STRICT_LOAD:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_ID="${RUN_ID:-groot_oneclick_export_$(date +%Y%m%d_%H%M%S)}"
  RUN_DIR="${GROOT_TRT_REPO_ROOT}/outputs/trt/${RUN_ID}"
fi

mkdir -p "${RUN_DIR}"
GROOT_TRT_TMPDIR="$(groot_trt_resolve_tmpdir "${RUN_DIR}")"

STEP1_ARGS=(
  "${SCRIPT_DIR}/step1_safetensors_to_torch.py"
  "--policy-path" "${POLICY_PATH}"
  "--run-dir" "${RUN_DIR}"
  "--tensorrt-py-dir" "${GROOT_TRT_TENSORRT_PY_DIR}"
  "--tmpdir" "${GROOT_TRT_TMPDIR}"
)
if [[ "${STRICT_LOAD}" == "1" ]]; then
  STEP1_ARGS+=("--strict")
fi

STEP2_ARGS=(
  "${SCRIPT_DIR}/step2_export_onnx.py"
  "--policy-path" "${POLICY_PATH}"
  "--run-dir" "${RUN_DIR}"
  "--conda-env" "${GROOT_TRT_CONDA_ENV}"
  "--device" "cuda"
  "--seq-len" "${SEQ_LEN}"
  "--video-views" "${VIDEO_VIEWS}"
  "--state-horizon" "${STATE_HORIZON}"
  "--opset" "${OPSET}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  STEP2_ARGS+=("--skip-existing")
fi

STEP4_ARGS=(
  "${SCRIPT_DIR}/step4_build_engines.py"
  "--run-dir" "${RUN_DIR}"
  "--conda-env" "${GROOT_TRT_CONDA_ENV}"
  "--video-views" "${VIDEO_VIEWS}"
  "--max-batch" "${MAX_BATCH}"
  "--vit-opt-batch" "${VIT_OPT_BATCH}"
  "--opt-batch" "${OPT_BATCH}"
  "--min-seq-len" "${MIN_SEQ_LEN}"
  "--opt-seq-len" "${OPT_SEQ_LEN}"
  "--max-seq-len" "${MAX_SEQ_LEN}"
  "--workspace-gb" "${WORKSPACE_GB}"
  "--tensorrt-py-dir" "${GROOT_TRT_TENSORRT_PY_DIR}"
  "--tmpdir" "${GROOT_TRT_TMPDIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  STEP4_ARGS+=("--skip-existing")
fi

groot_trt_conda_python "${STEP1_ARGS[@]}"
groot_trt_conda_python "${STEP2_ARGS[@]}"
groot_trt_conda_python "${STEP4_ARGS[@]}"

echo "[OK] GROOT export/build finished."
echo "RUN_DIR=${RUN_DIR}"
echo "STAGE1=${RUN_DIR}/stage1_safetensors_to_torch.json"
echo "STAGE2=${RUN_DIR}/stage2_export_onnx.json"
echo "STAGE4=${RUN_DIR}/stage4_build_engines.json"
echo "ONNX_DIR=${RUN_DIR}/gr00t_onnx"
echo "ENGINE_DIR=${RUN_DIR}/gr00t_engine_api_trt1013"
echo "TMPDIR=${GROOT_TRT_TMPDIR}"
