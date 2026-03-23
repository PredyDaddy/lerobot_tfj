#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1_trimmed_static_tail}"
DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

HF_CACHE_DIR="${HF_CACHE_DIR:-/home/cqy/.cache/huggingface/hub}"
SMOLVLA_BASE_PATH="${SMOLVLA_BASE_PATH:-$(find "${HF_CACHE_DIR}/models--lerobot--smolvla_base/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"
SMOLVLM2_PATH="${SMOLVLM2_PATH:-$(find "${HF_CACHE_DIR}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_${RUN_TAG}.train.log}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG_FILE}")"

if [[ -z "${SMOLVLA_BASE_PATH}" || ! -d "${SMOLVLA_BASE_PATH}" ]]; then
  echo "Local SmolVLA base checkpoint not found under ${HF_CACHE_DIR}." >&2
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
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

${PYTHON_BIN} -c 'import importlib.util as u, sys; missing=[m for m in ["transformers","accelerate","safetensors","num2words"] if u.find_spec(m) is None]; sys.exit("Missing packages: " + ", ".join(missing) + ". Run: pip install -e \".[smolvla]\"") if missing else None'

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/lerobot_train.py"
  --policy.path="${SMOLVLA_BASE_PATH}"
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --policy.empty_cameras=1
  --policy.vlm_model_name="${SMOLVLM2_PATH}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --num_workers="${NUM_WORKERS}"
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint=true
  --eval_freq=0
  --log_freq=50
  --wandb.enable=false
  --output_dir="${OUTPUT_DIR}"
  --job_name=smolvla_grasp_block_in_bin1_trimmed_static_tail
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}" >"${LOG_FILE}" 2>&1
