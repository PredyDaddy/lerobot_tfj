#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <env_type> <env_task> [extra draccus args...]" >&2
  echo "Example: $0 libero libero_object --env.obs_type=pixels_agent_pos" >&2
  exit 2
fi

ENV_TYPE="$1"
ENV_TASK="$2"
shift 2

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1_trimmed_static_tail}"
DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

HF_CACHE_DIR="${HF_CACHE_DIR:-/home/cqy/.cache/huggingface/hub}"
SMOLVLM2_PATH="${SMOLVLM2_PATH:-$(find "${HF_CACHE_DIR}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
WARMUP_CHUNKS="${WARMUP_CHUNKS:-0}"
CHUNKS_PER_STEP="${CHUNKS_PER_STEP:-1}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-4096}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-16}"

OFFLINE_LOSS_WEIGHT="${OFFLINE_LOSS_WEIGHT:-1.0}"
ONLINE_FLOW_LOSS_WEIGHT="${ONLINE_FLOW_LOSS_WEIGHT:-0.3}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-1.0}"
DISCOUNT="${DISCOUNT:-0.99}"
ADV_TEMPERATURE="${ADV_TEMPERATURE:-1.0}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_${ENV_TYPE}_${RUN_TAG}.train.log}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG_FILE}")"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Pretrained policy path does not exist: ${POLICY_PATH}" >&2
  exit 1
fi

if [[ -z "${SMOLVLM2_PATH}" || ! -d "${SMOLVLM2_PATH}" ]]; then
  echo "Local SmolVLM2 backbone not found under ${HF_CACHE_DIR}." >&2
  exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/lerobot_train_smolvla_hybrid.py"
  --policy.path="${POLICY_PATH}"
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --policy.empty_cameras=1
  --policy.vlm_model_name="${SMOLVLM2_PATH}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --env.type="${ENV_TYPE}"
  --env.task="${ENV_TASK}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --num_workers="${NUM_WORKERS}"
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint=true
  --eval_freq=0
  --log_freq="${LOG_FREQ}"
  --wandb.enable=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="smolvla_hybrid_${ENV_TYPE}"
  --collector.n_envs=1
  --collector.use_async_envs=false
  --collector.chunks_per_step="${CHUNKS_PER_STEP}"
  --collector.warmup_chunks="${WARMUP_CHUNKS}"
  --replay_buffer.capacity="${REPLAY_CAPACITY}"
  --replay_buffer.online_batch_size="${ONLINE_BATCH_SIZE}"
  --losses.offline_loss_weight="${OFFLINE_LOSS_WEIGHT}"
  --losses.online_flow_loss_weight="${ONLINE_FLOW_LOSS_WEIGHT}"
  --losses.value_loss_weight="${VALUE_LOSS_WEIGHT}"
  --losses.discount="${DISCOUNT}"
  --losses.advantage_temperature="${ADV_TEMPERATURE}"
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}" >"${LOG_FILE}" 2>&1
