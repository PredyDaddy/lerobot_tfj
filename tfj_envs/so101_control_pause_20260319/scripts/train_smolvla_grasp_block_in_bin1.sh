#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${DATASET_PATH:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/train/smolvla_grasp_block_in_bin2}
BATCH_SIZE=${BATCH_SIZE:-32}
STEPS=${STEPS:-18000}
SAVE_FREQ=${SAVE_FREQ:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}
PYTHON_BIN=${PYTHON_BIN:-python}
HF_CACHE_DIR=${HF_CACHE_DIR:-/home/cqy/.cache/huggingface/hub}
SMOLVLA_BASE_PATH=${SMOLVLA_BASE_PATH:-$(find "${HF_CACHE_DIR}/models--lerobot--smolvla_base/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}
SMOLVLM2_PATH=${SMOLVLM2_PATH:-$(find "${HF_CACHE_DIR}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}

cd /data/tfj/lerobot_tfj

if [[ -z "${SMOLVLA_BASE_PATH}" || ! -d "${SMOLVLA_BASE_PATH}" ]]; then
  echo "Local SmolVLA base checkpoint not found under ${HF_CACHE_DIR}." >&2
  exit 1
fi

if [[ -z "${SMOLVLM2_PATH}" || ! -d "${SMOLVLM2_PATH}" ]]; then
  echo "Local SmolVLM2 backbone not found under ${HF_CACHE_DIR}." >&2
  exit 1
fi

echo "Using local SmolVLA base: ${SMOLVLA_BASE_PATH}"
echo "Using local SmolVLM2 backbone: ${SMOLVLM2_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

${PYTHON_BIN} -c 'import importlib.util as u, sys; missing=[m for m in ["transformers","accelerate","safetensors","num2words"] if u.find_spec(m) is None]; sys.exit("Missing packages: " + ", ".join(missing) + ". Run: pip install -e \".[smolvla]\"") if missing else None'

lerobot-train \
  --policy.path=${SMOLVLA_BASE_PATH} \
  --policy.device=${DEVICE} \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.vlm_model_name=${SMOLVLM2_PATH} \
  --dataset.repo_id=${DATASET_PATH} \
  --batch_size=${BATCH_SIZE} \
  --steps=${STEPS} \
  --num_workers=${NUM_WORKERS} \
  --save_freq=${SAVE_FREQ} \
  --save_checkpoint=true \
  --eval_freq=0 \
  --log_freq=50 \
  --wandb.enable=false \
  --output_dir=${OUTPUT_DIR} \
  --job_name=smolvla_grasp_block_in_bin1 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
