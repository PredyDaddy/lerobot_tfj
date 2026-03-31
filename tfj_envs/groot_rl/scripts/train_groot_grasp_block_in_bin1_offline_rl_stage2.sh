#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

resolve_python_bin() {
  local candidate
  for candidate in \
    "/home/cqy/miniconda3/envs/gr00t/bin/python" \
    "/home/cqy/miniconda3/envs/lerobot/bin/python" \
    "/home/cqy/miniconda3/envs/lerobot_flex/bin/python"
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  command -v python
}

print_usage() {
  cat <<'EOF'
Dataset-only offline RL stage-2 launcher for GR00T on grasp_block_in_bin1.

Examples
  bash scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh
  PREFLIGHT_ONLY=1 bash scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh
  RESUME=1 bash scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh

Defaults
  POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model
  DATASET_ROOT=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1
  DATASET_REPO_ID=admin123/grasp_block_in_bin1
  DATASET_VIDEO_BACKEND=pyav
  OUTPUT_DIR=outputs/train/groot_offline_rl_stage2_<timestamp>
  NUM_WORKERS=0
  OFFLINE_REPLAY_VALUE_TARGET_MODE=monte_carlo

Notes
  - This stage does not use an online env.
  - Reward is synthesized from demo position only.
  - NUM_WORKERS is forced to 0 because replay sampling decodes videos in the main process.
  - Resume path must point to:
    OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json
EOF
}

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "${path}" ]]; then
    echo "${description} not found: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${description} not found: ${path}" >&2
    exit 1
  fi
}

print_command() {
  local -a cmd=("$@")
  printf 'Resolved command:\n'
  printf ' %q' "${cmd[@]}"
  printf '\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
POLICY_PATH="${POLICY_PATH:-/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/groot_offline_rl_stage2_${RUN_TAG}}"
JOB_NAME="${JOB_NAME:-groot_offline_rl_stage2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-8}"
OFFLINE_LOSS_WEIGHT="${OFFLINE_LOSS_WEIGHT:-1.0}"
ONLINE_FLOW_LOSS_WEIGHT="${ONLINE_FLOW_LOSS_WEIGHT:-0.5}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-1.0}"
DISCOUNT="${DISCOUNT:-0.99}"
OFFLINE_REPLAY_TRANSITION_STRIDE="${OFFLINE_REPLAY_TRANSITION_STRIDE:-1}"
OFFLINE_REPLAY_VALUE_TARGET_MODE="${OFFLINE_REPLAY_VALUE_TARGET_MODE:-monte_carlo}"
OFFLINE_REPLAY_TERMINAL_REWARD="${OFFLINE_REPLAY_TERMINAL_REWARD:-1.0}"
OFFLINE_REPLAY_STEP_REWARD="${OFFLINE_REPLAY_STEP_REWARD:-0.0}"
PREFLIGHT_ONLY="$(normalize_bool "${PREFLIGHT_ONLY:-false}")"
DRY_RUN="$(normalize_bool "${DRY_RUN:-false}")"
RESUME="$(normalize_bool "${RESUME:-false}")"

if [[ "${DRY_RUN}" == "true" ]]; then
  PREFLIGHT_ONLY="true"
fi

RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DATASET_ROOT
export DATASET_VIDEO_BACKEND

require_dir "${POLICY_PATH}" "Policy path"
require_file "${POLICY_PATH}/config.json" "Policy config"
require_file "${POLICY_PATH}/model.safetensors" "Policy weights"
require_dir "${DATASET_ROOT}" "Dataset root"
require_file "${DATASET_ROOT}/meta/info.json" "Dataset metadata"
require_dir "${DATASET_ROOT}/videos" "Dataset videos"

if [[ "${NUM_WORKERS}" != "0" ]]; then
  echo "NUM_WORKERS must be 0 for dataset-only offline RL stage 2." >&2
  exit 2
fi

if [[ "${RESUME}" != "true" && -e "${OUTPUT_DIR}" ]]; then
  echo "Output directory already exists: ${OUTPUT_DIR}" >&2
  echo "Use RESUME=1 with ${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" >&2
  exit 1
fi

if [[ "${RESUME}" == "true" ]]; then
  require_file "${RESUME_CONFIG_PATH}" "Resume config"
fi

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import av

dataset_root = Path(os.environ["DATASET_ROOT"])
video_files = sorted(path for path in (dataset_root / "videos").rglob("*") if path.is_file())
if not video_files:
    raise SystemExit(f"No video files found under {dataset_root / 'videos'}")

sample_video = video_files[0]
with av.open(str(sample_video)) as container:
    stream = next((stream for stream in container.streams if stream.type == "video"), None)
    if stream is None:
        raise SystemExit(f"No video stream found in {sample_video}")
    decoded = False
    for _ in container.decode(stream):
        decoded = True
        break
    if not decoded:
        raise SystemExit(f"PyAV opened {sample_video} but did not decode a frame")

print(f"PyAV decode preflight OK: {sample_video}")
PY

TRAIN_CMD=(
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train_groot_hybrid
  --policy.path="${POLICY_PATH}"
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${DATASET_VIDEO_BACKEND}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --num_workers=0
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint=true
  --eval_freq=0
  --log_freq="${LOG_FREQ}"
  --wandb.enable=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --collector.chunks_per_step=0
  --collector.warmup_chunks=0
  --replay_buffer.online_batch_size="${ONLINE_BATCH_SIZE}"
  --losses.offline_loss_weight="${OFFLINE_LOSS_WEIGHT}"
  --losses.online_flow_loss_weight="${ONLINE_FLOW_LOSS_WEIGHT}"
  --losses.value_loss_weight="${VALUE_LOSS_WEIGHT}"
  --losses.discount="${DISCOUNT}"
  --offline_replay.enabled=true
  --offline_replay.transition_stride="${OFFLINE_REPLAY_TRANSITION_STRIDE}"
  --offline_replay.value_target_mode="${OFFLINE_REPLAY_VALUE_TARGET_MODE}"
  --offline_replay.terminal_reward="${OFFLINE_REPLAY_TERMINAL_REWARD}"
  --offline_replay.step_reward="${OFFLINE_REPLAY_STEP_REWARD}"
  --offline_replay.success_value=true
)

RESUME_CMD=(
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train_groot_hybrid
  --resume=true
  --config_path="${RESUME_CONFIG_PATH}"
)

if [[ "$#" -gt 0 ]]; then
  if [[ "${RESUME}" == "true" ]]; then
    RESUME_CMD+=("$@")
  else
    TRAIN_CMD+=("$@")
  fi
fi

echo "GROOT dataset-only offline RL stage-2 preflight passed."
echo "Policy path: ${POLICY_PATH}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset video backend: ${DATASET_VIDEO_BACKEND}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Resume config path: ${RESUME_CONFIG_PATH}"

if [[ "${RESUME}" == "true" ]]; then
  print_command "${RESUME_CMD[@]}"
else
  print_command "${TRAIN_CMD[@]}"
fi

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  exit 0
fi

if [[ "${RESUME}" == "true" ]]; then
  exec "${RESUME_CMD[@]}"
fi

exec "${TRAIN_CMD[@]}"
