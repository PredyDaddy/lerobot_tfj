#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

BASE_CONFIG=${BASE_CONFIG:-/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model/train_config.json}
TEACHER_PATH=${TEACHER_PATH:-/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last}
OUTPUT_DIR=${OUTPUT_DIR:-/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740}

"${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
  --config_path="${BASE_CONFIG}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_grasp_block_in_bin1_stage2 \
  --batch_size=8 \
  --steps=18000 \
  --num_workers=4 \
  --log_freq=20 \
  --save_freq=2000 \
  --wandb.enable=false \
  --dataset.video_backend=pyav \
  --policy.kd=true \
  --policy.teacher_policy_path="${TEACHER_PATH}" \
  --policy.kd_weight=1.0 \
  --policy.kd_overlap_steps=100 \
  --policy.kd_temporal_decay=0.0 \
  --policy.kd_prefix_weight=1.0 \
  --policy.kd_tail_weight=1.0 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 \
  --policy.n_decoder_layers=1 \
  --policy.latent_dim=16 \
  --policy.n_vae_encoder_layers=2 \
  --policy.decoder_kd.enabled=true \
  --policy.decoder_kd.require_action_kd=true \
  --policy.decoder_kd.peak_weight=0.10 \
  --policy.decoder_kd.loss_type=smooth_l1 \
  --policy.decoder_kd.smooth_l1_beta=1.0 \
  --policy.decoder_kd.latent_mode=zero \
  --policy.decoder_kd.overlap_steps=100 \
  --policy.decoder_kd.start_step=1000 \
  --policy.decoder_kd.ramp_steps=2000 \
  --policy.decoder_kd.anneal_start_step=12000 \
  --policy.decoder_kd.end_step=18000 \
  --policy.decoder_kd.enable_noise_gate=true \
  --policy.decoder_kd.enable_grad_gate=true \
  --policy.decoder_kd.log_grad_ratio=true
