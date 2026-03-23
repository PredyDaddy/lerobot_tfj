#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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

POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model}"
POLICY_PATH="$(resolve_policy_path "${POLICY_PATH}")"
if [[ ! -f "${POLICY_PATH}/config.json" || ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  LATEST_POLICY_CONFIG="$(find "${REPO_ROOT}/outputs/train" -path '*/checkpoints/*/pretrained_model/config.json' | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_POLICY_CONFIG}" ]]; then
    POLICY_PATH="$(dirname "${LATEST_POLICY_CONFIG}")"
  fi
fi
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

TASK_TEXT="${TASK_TEXT:-Put the block in the bin}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_smolvla_rl_so101}"
DATASET_ROOT="${DATASET_ROOT:-./outputs/eval_smolvla_rl_so101}"
DATASET_FPS="${DATASET_FPS:-30}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
RESET_TIME_S="${RESET_TIME_S:-15}"
DATASET_VIDEO="$(normalize_bool "${DATASET_VIDEO:-false}")"
CLEAR_DATASET_ROOT="$(normalize_bool "${CLEAR_DATASET_ROOT:-false}")"
SAVE_DATASET="$(normalize_bool "${SAVE_DATASET:-false}")"

if [[ "${SAVE_DATASET}" == "false" ]]; then
  DATASET_VIDEO="false"
  CLEAR_DATASET_ROOT="false"
fi

DISPLAY_DATA="$(normalize_bool "${DISPLAY_DATA:-false}")"
PLAY_SOUNDS="$(normalize_bool "${PLAY_SOUNDS:-false}")"

LEADER_PORT="${LEADER_PORT:-}"
LEADER_ID="${LEADER_ID:-so101_leader}"
LEADER_CALIB_DIR="${LEADER_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tfj_envs/smolvla_rl/scripts/lerobot_record_so101_policy.py"
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
  --dataset_repo_id="${DATASET_REPO_ID}"
  --dataset_root="${DATASET_ROOT}"
  --dataset_fps="${DATASET_FPS}"
  --num_episodes="${NUM_EPISODES}"
  --episode_time_s="${EPISODE_TIME_S}"
  --reset_time_s="${RESET_TIME_S}"
  --dataset_video="${DATASET_VIDEO}"
  --display_data="${DISPLAY_DATA}"
  --play_sounds="${PLAY_SOUNDS}"
  --clear_dataset_root="${CLEAR_DATASET_ROOT}"
  --save_dataset="${SAVE_DATASET}"
)

if [[ -n "${ROBOT_MAX_RELATIVE_TARGET}" ]]; then
  CMD+=(--robot_max_relative_target="${ROBOT_MAX_RELATIVE_TARGET}")
fi

if [[ -n "${LEADER_PORT}" ]]; then
  CMD+=(
    --leader_port="${LEADER_PORT}"
    --leader_id="${LEADER_ID}"
    --leader_calibration_dir="${LEADER_CALIB_DIR}"
  )
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
