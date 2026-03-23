#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 <output_dir> <train_log> [steps] [batch_size] [num_workers]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
TRAIN_LOG="$2"
STEPS="${3:-10000}"
BATCH_SIZE="${4:-32}"
NUM_WORKERS="${5:-4}"

mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$OUTPUT_DIR")"
cd /data/tfj/lerobot_tfj

exec env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/data/tfj/lerobot_tfj/src \
  python /data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_train.py \
    --policy.path=/home/cqy/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/4d2f2b37fa245361ef1efe6d91ce96b8bd4af511 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --policy.vlm_model_name=/home/cqy/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
    --dataset.repo_id=admin123/grasp_block_in_bin1_trimmed_static_tail \
    --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail \
    --dataset.video_backend=pyav \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers="$NUM_WORKERS" \
    --save_freq=2000 \
    --save_checkpoint=true \
    --eval_freq=0 \
    --log_freq=50 \
    --wandb.enable=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name=smolvla_grasp_block_in_bin1_trimmed_static_tail \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    >"$TRAIN_LOG" 2>&1
