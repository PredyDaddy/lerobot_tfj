#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TFJ_ENVS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${TFJ_ENVS_ROOT}/so101_control_pause_20260319}"
ENTRYPOINT="${ENTRYPOINT:-${RUNTIME_ROOT}/src/lerobot/scripts/lerobot_run_so101_pickplace.py}"
cd "${PROJECT_ROOT}"

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

has_graphical_display() {
  [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" || -n "${WAYLAND_SOCKET:-}" ]]
}

draccus_quote_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '"%s"' "${value}"
}

resolve_python_bin() {
  local candidate
  for candidate in \
    "${PYTHON_BIN:-}" \
    "/home/cqy/miniconda3/bin/python" \
    "/home/cqy/miniconda3/envs/gr00t/bin/python" \
    "/home/cqy/miniconda3/envs/lerobot/bin/python" \
    "/home/cqy/miniconda3/envs/lerobot_flex/bin/python"
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  echo "No usable python interpreter found for GROOT runtime." >&2
  exit 127
}

resolve_policy_path() {
  local candidate="$1"
  if [[ -f "${candidate}/config.json" && -f "${candidate}/model.safetensors" ]]; then
    printf '%s' "${candidate}"
    return
  fi
  if [[ -f "${candidate}/pretrained_model/config.json" && -f "${candidate}/pretrained_model/model.safetensors" ]]; then
    printf '%s' "${candidate}/pretrained_model"
    return
  fi
  if [[ -f "${candidate}/checkpoints/last/pretrained_model/config.json" && -f "${candidate}/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    printf '%s' "${candidate}/checkpoints/last/pretrained_model"
    return
  fi
  if [[ -f "${candidate}/checkpoints/last/config.json" && -f "${candidate}/checkpoints/last/model.safetensors" ]]; then
    printf '%s' "${candidate}/checkpoints/last"
    return
  fi
  printf '%s' "${candidate}"
}

find_latest_stage2_rl_policy_path() {
  local train_root="${PROJECT_ROOT}/outputs/train"
  local latest_run=""

  if [[ -d "${train_root}" ]]; then
    latest_run="$(
      find "${train_root}" -maxdepth 1 -type d -name 'groot_offline_rl_stage2_run_*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR == 1 {print $2}'
    )"
  fi

  if [[ -n "${latest_run}" ]]; then
    local latest_pretrained="${latest_run}/checkpoints/last/pretrained_model"
    if [[ -f "${latest_pretrained}/config.json" && -f "${latest_pretrained}/model.safetensors" ]]; then
      printf '%s' "${latest_pretrained}"
      return
    fi
  fi

  printf '%s' "/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model"
}

default_safe_policy_path() {
  printf '%s' "/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model"
}

TASK_TEXT="${TASK_TEXT:-${TASK:-Pick up the block with the GROOT policy}}"
INTENT_JSON="${INTENT_JSON:-${TASK_INTENT_JSON:-${INTENT:-}}}"
TASK_INTENT_JSON="${TASK_INTENT_JSON:-${INTENT_JSON}}"
SAFETY_PROFILE="$(printf '%s' "${SAFETY_PROFILE:-${SAFETY:-default}}" | tr '[:upper:]' '[:lower:]')"
EVENTS_JSONL_PATH="${EVENTS_JSONL_PATH:-${EVENTS_PATH:-}}"
EVENTS_PATH="${EVENTS_PATH:-${EVENTS_JSONL_PATH}}"
PREFER_STAGE2_RL="$(normalize_bool "${PREFER_STAGE2_RL:-false}")"
PREFLIGHT_ONLY="$(normalize_bool "${PREFLIGHT_ONLY:-${DRY_RUN:-false}}")"
if [[ -v DISPLAY_DATA ]]; then
  DISPLAY_DATA="$(normalize_bool "${DISPLAY_DATA}")"
elif has_graphical_display; then
  DISPLAY_DATA="true"
else
  DISPLAY_DATA="false"
fi
PLAY_SOUNDS="$(normalize_bool "${PLAY_SOUNDS:-false}")"

PYTHON_BIN="$(resolve_python_bin)"
if [[ ! -f "${ENTRYPOINT}" ]]; then
  echo "SO101 runtime entrypoint not found: ${ENTRYPOINT}" >&2
  echo "Checked runtime root: ${RUNTIME_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${RUNTIME_ROOT}/src:${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SAFE_POLICY_PATH="$(default_safe_policy_path)"
STAGE2_RL_POLICY_PATH="$(find_latest_stage2_rl_policy_path)"
if [[ "${PREFER_STAGE2_RL}" == "true" ]]; then
  DEFAULT_POLICY_PATH="${STAGE2_RL_POLICY_PATH}"
  POLICY_SELECTION_MODE="stage2_rl"
else
  DEFAULT_POLICY_PATH="${SAFE_POLICY_PATH}"
  POLICY_SELECTION_MODE="stage1_safe"
fi
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
POLICY_PATH="$(resolve_policy_path "${POLICY_PATH}")"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
ROBOT_MAX_RELATIVE_TARGET="${ROBOT_MAX_RELATIVE_TARGET:-}"

TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-4}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_WARMUP_S="${CAMERA_WARMUP_S:-1}"

FPS="${FPS:-${DATASET_FPS:-${CAMERA_FPS}}}"
RUN_TIME_S="${RUN_TIME_S:-${EPISODE_TIME_S:-300}}"

CMD=(
  "${PYTHON_BIN}" "${ENTRYPOINT}"
  --policy.path="${POLICY_PATH}"
  --policy.device="${POLICY_DEVICE}"
  --robot_port="${ROBOT_PORT}"
  --robot_id="${ROBOT_ID}"
  --robot_calibration_dir="${ROBOT_CALIB_DIR}"
  --top_camera_index="${TOP_CAMERA_INDEX}"
  --wrist_camera_index="${WRIST_CAMERA_INDEX}"
  --camera_width="${CAMERA_WIDTH}"
  --camera_height="${CAMERA_HEIGHT}"
  --camera_fps="${CAMERA_FPS}"
  --camera_warmup_s="${CAMERA_WARMUP_S}"
  --task="${TASK_TEXT}"
  --enable_perception_bridge=true
  --fps="${FPS}"
  --run_time_s="${RUN_TIME_S}"
  --display_data="${DISPLAY_DATA}"
  --play_sounds="${PLAY_SOUNDS}"
  "--safety_profile=$(draccus_quote_string "${SAFETY_PROFILE}")"
)

if [[ -n "${INTENT_JSON}" ]]; then
  CMD+=("--intent_json=$(draccus_quote_string "${INTENT_JSON}")")
fi

if [[ -n "${EVENTS_JSONL_PATH}" ]]; then
  CMD+=(--events_jsonl_path="${EVENTS_JSONL_PATH}")
fi

if [[ -n "${ROBOT_MAX_RELATIVE_TARGET}" ]]; then
  CMD+=(--robot_max_relative_target="${ROBOT_MAX_RELATIVE_TARGET}")
fi

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  CMD+=(--dry_run=true)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Backend: groot"
echo "Bundle root: ${BUNDLE_ROOT}"
echo "Project root: ${PROJECT_ROOT}"
echo "Runtime root: ${RUNTIME_ROOT}"
echo "Direct entrypoint: ${ENTRYPOINT}"
echo "Policy selection mode: ${POLICY_SELECTION_MODE}"
echo "Stage1 safe policy source: ${SAFE_POLICY_PATH}"
echo "Latest stage2 RL policy source: ${STAGE2_RL_POLICY_PATH}"
echo "Default/final policy source: ${DEFAULT_POLICY_PATH}"
echo "Robot ID: ${ROBOT_ID}"
echo "Task text: ${TASK_TEXT}"
echo "Safety profile: ${SAFETY_PROFILE}"
echo "Display data: ${DISPLAY_DATA}"
echo "Run time (s): ${RUN_TIME_S}"
echo "FPS: ${FPS}"
if [[ -n "${EVENTS_JSONL_PATH}" ]]; then
  echo "Events JSONL path: ${EVENTS_JSONL_PATH}"
else
  echo "Events JSONL path: <disabled>"
fi
echo "Using policy: ${POLICY_PATH}"
printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
