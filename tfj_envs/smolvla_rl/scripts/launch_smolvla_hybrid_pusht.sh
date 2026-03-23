#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/cqy/miniconda3/envs/lerobot/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/datasets/pusht_teacher_rl_filtered_${RUN_TAG}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/pusht_teacher_rl_filtered_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smolvla_hybrid_pusht_${RUN_TAG}}"

TARGET_EPISODES="${TARGET_EPISODES:-128}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-400}"
MIN_SUM_REWARD="${MIN_SUM_REWARD:-5.0}"

STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-50}"
WARMUP_CHUNKS="${WARMUP_CHUNKS:-32}"
CHUNKS_PER_STEP="${CHUNKS_PER_STEP:-1}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-4096}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-16}"
TASK="${TASK:-PushT-v0}"
DEVICE="${DEVICE:-cuda}"

COLLECT_LOG="${COLLECT_LOG:-${REPO_ROOT}/outputs/logs/pusht_collect_${RUN_TAG}.log}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/outputs/logs/smolvla_hybrid_pusht_${RUN_TAG}.train.log}"
mkdir -p "$(dirname "${COLLECT_LOG}")" "$(dirname "${TRAIN_LOG}")" "$(dirname "${DATASET_ROOT}")" "$(dirname "${OUTPUT_DIR}")"

COLLECT_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tfj_envs/smolvla_rl/scripts/pusht_hybrid_workflow.py"
  --log-file="${COLLECT_LOG}"
  collect
  --dataset-root="${DATASET_ROOT}"
  --dataset-repo-id="${DATASET_REPO_ID}"
  --task="${TASK}"
  --device="${DEVICE}"
  --target-episodes="${TARGET_EPISODES}"
  --max-attempts="${MAX_ATTEMPTS}"
  --min-sum-reward="${MIN_SUM_REWARD}"
  --overwrite
)

TRAIN_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tfj_envs/smolvla_rl/scripts/pusht_hybrid_workflow.py"
  --log-file="${TRAIN_LOG}"
  train
  --dataset-root="${DATASET_ROOT}"
  --dataset-repo-id="${DATASET_REPO_ID}"
  --output-dir="${OUTPUT_DIR}"
  --task="${TASK}"
  --device="${DEVICE}"
  --steps="${STEPS}"
  --batch-size="${BATCH_SIZE}"
  --num-workers="${NUM_WORKERS}"
  --save-freq="${SAVE_FREQ}"
  --log-freq="${LOG_FREQ}"
  --warmup-chunks="${WARMUP_CHUNKS}"
  --chunks-per-step="${CHUNKS_PER_STEP}"
  --replay-capacity="${REPLAY_CAPACITY}"
  --online-batch-size="${ONLINE_BATCH_SIZE}"
)

printf 'Collect command:\n'
printf ' %q' "${COLLECT_CMD[@]}"
printf '\n'

"${COLLECT_CMD[@]}"

printf 'Train command:\n'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

exec "${TRAIN_CMD[@]}"
