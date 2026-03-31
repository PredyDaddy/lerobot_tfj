#!/usr/bin/env bash

set -euo pipefail

GROOT_TRT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROOT_TRT_REPO_ROOT="$(cd -- "${GROOT_TRT_SCRIPT_DIR}/../../.." && pwd)"

GROOT_TRT_CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
GROOT_TRT_TENSORRT_PY_DIR="${TENSORRT_PY_DIR:-}"
GROOT_TRT_TMPDIR="${TMPDIR:-}"

groot_trt_print_usage_header() {
  local script_name="$1"
  local usage_tail="${2:-POLICY_PATH [RUN_DIR] [OUT_DIR]}"
  echo "Usage: ${script_name} ${usage_tail}" >&2
}

groot_trt_require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda is not available in PATH." >&2
    exit 1
  fi
}

groot_trt_default_profile() {
  local video_views="$1"
  if [[ "${video_views}" == "2" ]]; then
    echo "568 80 568 600"
  elif [[ "${video_views}" == "1" ]]; then
    echo "296 80 296 300"
  else
    echo "[ERR] Unsupported VIDEO_VIEWS=${video_views}. Expected 1 or 2." >&2
    exit 1
  fi
}

groot_trt_conda_python() {
  conda run --no-capture-output -n "${GROOT_TRT_CONDA_ENV}" python "$@"
}

groot_trt_conda_python_with_trt() {
  local env_args=()
  if [[ -n "${GROOT_TRT_TMPDIR}" ]]; then
    env_args+=("TMPDIR=${GROOT_TRT_TMPDIR}")
  fi
  if [[ -n "${GROOT_TRT_TENSORRT_PY_DIR}" ]]; then
    env_args+=("TENSORRT_PY_DIR=${GROOT_TRT_TENSORRT_PY_DIR}")
  fi
  conda run --no-capture-output -n "${GROOT_TRT_CONDA_ENV}" env "${env_args[@]}" python "$@"
}

groot_trt_resolve_tmpdir() {
  local run_dir="$1"
  if [[ -n "${GROOT_TRT_TMPDIR}" ]]; then
    mkdir -p "${GROOT_TRT_TMPDIR}"
    echo "${GROOT_TRT_TMPDIR}"
    return 0
  fi
  local fallback="${run_dir}/.tmp"
  mkdir -p "${fallback}"
  echo "${fallback}"
}
