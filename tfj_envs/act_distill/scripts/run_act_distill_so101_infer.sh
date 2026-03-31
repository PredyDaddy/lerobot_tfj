#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# POLICY_PATH="${POLICY_PATH:-/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740}"
POLICY_PATH="${POLICY_PATH:-/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

ROBOT_ID="${ROBOT_ID:-my_so101}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
ROBOT_MAX_RELATIVE_TARGET="${ROBOT_MAX_RELATIVE_TARGET:-8}"

TOP_CAM_INDEX="${TOP_CAM_INDEX:-4}"
WRIST_CAM_INDEX="${WRIST_CAM_INDEX:-6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_WARMUP_S="${CAMERA_WARMUP_S:-1}"

FPS="${FPS:-30}"
RUN_TIME_S="${RUN_TIME_S:-300}"
TASK_TEXT="${TASK_TEXT:-Put the block in the bin}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tfj_envs/act_distill/scripts/lerobot_run_act_so101.py"
  --policy-path="${POLICY_PATH}"
  --policy-device="${POLICY_DEVICE}"
  --robot-id="${ROBOT_ID}"
  --robot-port="${ROBOT_PORT}"
  --robot-calibration-dir="${ROBOT_CALIB_DIR}"
  --robot-max-relative-target="${ROBOT_MAX_RELATIVE_TARGET}"
  --top-cam-index="${TOP_CAM_INDEX}"
  --wrist-cam-index="${WRIST_CAM_INDEX}"
  --camera-width="${CAMERA_WIDTH}"
  --camera-height="${CAMERA_HEIGHT}"
  --camera-fps="${CAMERA_FPS}"
  --camera-warmup-s="${CAMERA_WARMUP_S}"
  --fps="${FPS}"
  --run-time-s="${RUN_TIME_S}"
  --task="${TASK_TEXT}"
)

if [[ "${DISPLAY_DATA}" == "true" ]]; then
  CMD+=(--display-data)
else
  CMD+=(--no-display-data)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
