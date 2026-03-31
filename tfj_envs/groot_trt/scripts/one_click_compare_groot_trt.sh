#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/one_click_common.sh"

groot_trt_require_conda

POLICY_PATH="${1:-}"
RUN_DIR="${2:-}"
if [[ -z "${POLICY_PATH}" || -z "${RUN_DIR}" ]]; then
  groot_trt_print_usage_header "$(basename "$0")" "POLICY_PATH RUN_DIR"
  echo "Compare safetensors-loaded PyTorch vs ONNX vs TensorRT against an existing RUN_DIR." >&2
  exit 1
fi

RUN_DIR="$(cd -- "${RUN_DIR}" && pwd)"
GROOT_TRT_TMPDIR="$(groot_trt_resolve_tmpdir "${RUN_DIR}")"
SEED="${SEED:-20260303}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MIN_LLM_FROM_VIT_COSINE="${MIN_LLM_FROM_VIT_COSINE:-}"
MIN_DENOISING_COSINE="${MIN_DENOISING_COSINE:-}"

ARGS=(
  "${SCRIPT_DIR}/compare_safetensor_onnx_trt.py"
  "--policy-path" "${POLICY_PATH}"
  "--run-dir" "${RUN_DIR}"
  "--conda-env" "${GROOT_TRT_CONDA_ENV}"
  "--device" "cuda"
  "--seed" "${SEED}"
  "--tensorrt-py-dir" "${GROOT_TRT_TENSORRT_PY_DIR}"
  "--tmpdir" "${GROOT_TRT_TMPDIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=("--skip-existing")
fi
if [[ -n "${MIN_LLM_FROM_VIT_COSINE}" ]]; then
  ARGS+=("--min-llm-from-vit-cosine" "${MIN_LLM_FROM_VIT_COSINE}")
fi
if [[ -n "${MIN_DENOISING_COSINE}" ]]; then
  ARGS+=("--min-denoising-cosine" "${MIN_DENOISING_COSINE}")
fi

groot_trt_conda_python_with_trt "${ARGS[@]}"
