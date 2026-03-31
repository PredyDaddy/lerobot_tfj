#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

normalize_bool() {
  local value="${1:-false}"
  case "${value,,}" in
    1|true|yes|y|on) echo "true" ;;
    0|false|no|n|off|"") echo "false" ;;
    *)
      echo "Unsupported boolean value: ${value}" >&2
      exit 2
      ;;
  esac
}

python_supports_geom_grasp() {
  local python_bin="$1"
  [[ -n "${python_bin}" ]] || return 1
  [[ -x "${python_bin}" ]] || return 1

  "${python_bin}" - <<'PY' >/dev/null 2>&1
required = [
    "pyrealsense2",
    "placo",
    "draccus",
    "scservo_sdk",
    "serial",
    "deepdiff",
    "datasets",
    "accelerate",
]
for module_name in required:
    __import__(module_name)
PY
}

resolve_geom_grasp_python_bin() {
  local requested="${PYTHON_BIN:-}"
  local -a candidates=()

  if [[ -n "${requested}" ]]; then
    candidates+=("${requested}")
  fi

  candidates+=(
    "/home/cqy/miniconda3/envs/robot_vision/bin/python"
    "/home/cqy/miniconda3/bin/python"
    "$(command -v python3 2>/dev/null || true)"
    "$(command -v python 2>/dev/null || true)"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if python_supports_geom_grasp "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  printf '%s\n' "python"
  return 0
}

has_graphical_display() {
  [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" || -n "${WAYLAND_SOCKET:-}" ]]
}

append_if_set() {
  local flag="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    CMD+=("${flag}=${value}")
  fi
}

TASK_TEXT="${TASK_TEXT:-${TASK:-Grasp the block from the table}}"
SAFETY_PROFILE="$(printf '%s' "${SAFETY_PROFILE:-default}" | tr '[:upper:]' '[:lower:]')"
PYTHON_BIN="$(resolve_geom_grasp_python_bin)"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
REALSENSE_SERIAL="${REALSENSE_SERIAL:-}"
CAMERA_TO_BASE_XYZ="${CAMERA_TO_BASE_XYZ:-}"
CAMERA_TO_BASE_RPY_DEG="${CAMERA_TO_BASE_RPY_DEG:-}"
WORKSPACE_MIN_XYZ="${WORKSPACE_MIN_XYZ:-}"
WORKSPACE_MAX_XYZ="${WORKSPACE_MAX_XYZ:-}"
DEPTH_MIN_M="${DEPTH_MIN_M:-}"
DEPTH_MAX_M="${DEPTH_MAX_M:-}"
FOREGROUND_MIN_HEIGHT_M="${FOREGROUND_MIN_HEIGHT_M:-}"
PERCEPTION_CONSISTENCY_FRAMES="${PERCEPTION_CONSISTENCY_FRAMES:-}"
PERCEPTION_MIN_CONSISTENT_DETECTIONS="${PERCEPTION_MIN_CONSISTENT_DETECTIONS:-}"
PERCEPTION_POSITION_STD_MAX_M="${PERCEPTION_POSITION_STD_MAX_M:-}"
PERCEPTION_MASK_AREA_REL_STD_MAX="${PERCEPTION_MASK_AREA_REL_STD_MAX:-}"
TOP_SURFACE_PERCENTILE="${TOP_SURFACE_PERCENTILE:-}"
TOP_SURFACE_BAND_M="${TOP_SURFACE_BAND_M:-}"
GRASPABLE_MIN_HEIGHT_M="${GRASPABLE_MIN_HEIGHT_M:-}"
GRASPABLE_MAX_HEIGHT_M="${GRASPABLE_MAX_HEIGHT_M:-}"
GRASPABLE_MIN_MASK_AREA_PX="${GRASPABLE_MIN_MASK_AREA_PX:-}"
GRASPABLE_MAX_MASK_AREA_PX="${GRASPABLE_MAX_MASK_AREA_PX:-}"
PREGRASP_OFFSET_M="${PREGRASP_OFFSET_M:-}"
GRASP_Z_OFFSET_M="${GRASP_Z_OFFSET_M:-}"
GRASP_RPY_DEG="${GRASP_RPY_DEG:-}"
LIFT_OFFSET_M="${LIFT_OFFSET_M:-}"
GRIPPER_OPEN_POS="${GRIPPER_OPEN_POS:-}"
GRIPPER_CLOSE_POS="${GRIPPER_CLOSE_POS:-}"
MOVE_SLEEP_S="${MOVE_SLEEP_S:-}"
SETTLE_S="${SETTLE_S:-}"
VERIFICATION_WARMUP_FRAMES="${VERIFICATION_WARMUP_FRAMES:-}"
VERIFICATION_POSITION_TOL_M="${VERIFICATION_POSITION_TOL_M:-}"
VERIFICATION_HEIGHT_TOL_M="${VERIFICATION_HEIGHT_TOL_M:-}"
ROBOT_MAX_RELATIVE_TARGET="${ROBOT_MAX_RELATIVE_TARGET:-}"
DRY_RUN="$(normalize_bool "${DRY_RUN:-true}")"

if [[ -v DISPLAY_DATA ]]; then
  DISPLAY_DATA="$(normalize_bool "${DISPLAY_DATA}")"
elif has_graphical_display; then
  DISPLAY_DATA="true"
else
  DISPLAY_DATA="false"
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/lerobot_run_so101_geom_grasp.py"
  --task="${TASK_TEXT}"
  --robot_port="${ROBOT_PORT}"
  --robot_id="${ROBOT_ID}"
  --robot_calib_dir="${ROBOT_CALIB_DIR}"
  --camera_width="${CAMERA_WIDTH}"
  --camera_height="${CAMERA_HEIGHT}"
  --camera_fps="${CAMERA_FPS}"
  --safety_profile="${SAFETY_PROFILE}"
  --display_data="${DISPLAY_DATA}"
  --dry_run="${DRY_RUN}"
)

append_if_set --realsense_serial "${REALSENSE_SERIAL}"
append_if_set --camera_to_base_xyz "${CAMERA_TO_BASE_XYZ}"
append_if_set --camera_to_base_rpy_deg "${CAMERA_TO_BASE_RPY_DEG}"
append_if_set --workspace_min_xyz "${WORKSPACE_MIN_XYZ}"
append_if_set --workspace_max_xyz "${WORKSPACE_MAX_XYZ}"
append_if_set --depth_min_m "${DEPTH_MIN_M}"
append_if_set --depth_max_m "${DEPTH_MAX_M}"
append_if_set --foreground_min_height_m "${FOREGROUND_MIN_HEIGHT_M}"
append_if_set --perception_consistency_frames "${PERCEPTION_CONSISTENCY_FRAMES}"
append_if_set --perception_min_consistent_detections "${PERCEPTION_MIN_CONSISTENT_DETECTIONS}"
append_if_set --perception_position_std_max_m "${PERCEPTION_POSITION_STD_MAX_M}"
append_if_set --perception_mask_area_rel_std_max "${PERCEPTION_MASK_AREA_REL_STD_MAX}"
append_if_set --top_surface_percentile "${TOP_SURFACE_PERCENTILE}"
append_if_set --top_surface_band_m "${TOP_SURFACE_BAND_M}"
append_if_set --graspable_min_height_m "${GRASPABLE_MIN_HEIGHT_M}"
append_if_set --graspable_max_height_m "${GRASPABLE_MAX_HEIGHT_M}"
append_if_set --graspable_min_mask_area_px "${GRASPABLE_MIN_MASK_AREA_PX}"
append_if_set --graspable_max_mask_area_px "${GRASPABLE_MAX_MASK_AREA_PX}"
append_if_set --pregrasp_offset_m "${PREGRASP_OFFSET_M}"
append_if_set --grasp_z_offset_m "${GRASP_Z_OFFSET_M}"
append_if_set --grasp_rpy_deg "${GRASP_RPY_DEG}"
append_if_set --lift_offset_m "${LIFT_OFFSET_M}"
append_if_set --gripper_open_pos "${GRIPPER_OPEN_POS}"
append_if_set --gripper_close_pos "${GRIPPER_CLOSE_POS}"
append_if_set --move_sleep_s "${MOVE_SLEEP_S}"
append_if_set --settle_s "${SETTLE_S}"
append_if_set --verification_warmup_frames "${VERIFICATION_WARMUP_FRAMES}"
append_if_set --verification_position_tol_m "${VERIFICATION_POSITION_TOL_M}"
append_if_set --verification_height_tol_m "${VERIFICATION_HEIGHT_TOL_M}"
append_if_set --robot_max_relative_target "${ROBOT_MAX_RELATIVE_TARGET}"

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Backend: geom_grasp"
echo "Entry point: ${REPO_ROOT}/src/lerobot/scripts/lerobot_run_so101_geom_grasp.py"
echo "Task text: ${TASK_TEXT}"
echo "Robot ID: ${ROBOT_ID}"
echo "Safety profile: ${SAFETY_PROFILE}"
echo "Python bin: ${PYTHON_BIN}"
echo "Display data: ${DISPLAY_DATA}"
echo "Dry run: ${DRY_RUN}"
if [[ -n "${REALSENSE_SERIAL}" ]]; then
  echo "RealSense serial: ${REALSENSE_SERIAL}"
fi
printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
