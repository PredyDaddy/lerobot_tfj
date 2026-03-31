#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/cqy/miniconda3/envs/lerobot_flex/bin/python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PI05_LOCAL_TOKENIZER_PATH="${PI05_LOCAL_TOKENIZER_PATH:-/home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224}"

POLICY_PATH="${POLICY_PATH:-/data/tfj/lerobot_tfj/pi_model/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-float32}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-1}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-10}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_so101}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
ROBOT_MAX_RELATIVE_TARGET="${ROBOT_MAX_RELATIVE_TARGET:-0.5}"

TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-4}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-5}"

TASK_TEXT="${TASK_TEXT:-grasp block in bin}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_pi05_so101_debug}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/lerobot_record_pi05}"
DATASET_FPS="${DATASET_FPS:-5}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-0}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"
DATASET_VIDEO="${DATASET_VIDEO:-false}"
DATASET_PUSH_TO_HUB="${DATASET_PUSH_TO_HUB:-false}"
DATASET_NUM_IMAGE_WRITER_THREADS="${DATASET_NUM_IMAGE_WRITER_THREADS:-1}"

CAMERAS=$(cat <<EOF
{
  top: {type: opencv, index_or_path: ${TOP_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}},
  wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}
}
EOF
)

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/lerobot_record.py"
  --robot.type=so101_follower
  --robot.id="${ROBOT_ID}"
  --robot.port="${ROBOT_PORT}"
  --robot.calibration_dir="${ROBOT_CALIB_DIR}"
  --robot.max_relative_target="${ROBOT_MAX_RELATIVE_TARGET}"
  --robot.cameras="${CAMERAS}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.single_task="${TASK_TEXT}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.fps="${DATASET_FPS}"
  --dataset.episode_time_s="${EPISODE_TIME_S}"
  --dataset.reset_time_s="${RESET_TIME_S}"
  --dataset.num_episodes="${NUM_EPISODES}"
  --dataset.video="${DATASET_VIDEO}"
  --dataset.push_to_hub="${DATASET_PUSH_TO_HUB}"
  --dataset.num_image_writer_threads_per_camera="${DATASET_NUM_IMAGE_WRITER_THREADS}"
  --policy.path="${POLICY_PATH}"
  --policy.device="${POLICY_DEVICE}"
  --policy.use_amp="${POLICY_USE_AMP}"
  --policy.dtype="${POLICY_DTYPE}"
  --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
  --policy.num_inference_steps="${POLICY_NUM_INFERENCE_STEPS}"
  --display_data="${DISPLAY_DATA}"
  --play_sounds="${PLAY_SOUNDS}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Using python: ${PYTHON_BIN}"
echo "Using lerobot from: ${PYTHONPATH%%:*}"
echo "Task text: ${TASK_TEXT}"
echo "Robot max_relative_target: ${ROBOT_MAX_RELATIVE_TARGET}"
printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
