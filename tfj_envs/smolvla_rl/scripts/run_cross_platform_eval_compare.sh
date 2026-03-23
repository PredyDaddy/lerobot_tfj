#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/tfj/lerobot_tfj"
PYTHON_BIN="/home/cqy/miniconda3/envs/lerobot/bin/python"
EVAL_SCRIPT="${ROOT_DIR}/tfj_envs/smolvla_rl/scripts/eval_smolvla_cross_env.py"
OUTPUT_DIR="${ROOT_DIR}/outputs/eval_compare"

OFFLINE_POLICY="${ROOT_DIR}/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model"
RL_POLICY="${ROOT_DIR}/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "${OUTPUT_DIR}"

run_eval() {
    local policy_path="$1"
    local env_type="$2"
    local env_task="$3"
    local state_dim="$4"
    local action_dim="$5"
    local n_episodes="$6"
    local batch_size="$7"
    local output_subdir="$8"

    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --policy-path "${policy_path}" \
        --env-type "${env_type}" \
        --env-task "${env_task}" \
        --state-dim "${state_dim}" \
        --action-dim "${action_dim}" \
        --n-episodes "${n_episodes}" \
        --batch-size "${batch_size}" \
        --output-dir "${OUTPUT_DIR}/${output_subdir}"
}

echo "[1/6] Aloha offline"
run_eval "${OFFLINE_POLICY}" aloha AlohaInsertion-v0 6 6 20 5 smolvla_offline_aloha_20ep

echo "[2/6] Aloha RL"
run_eval "${RL_POLICY}" aloha AlohaInsertion-v0 6 6 20 5 smolvla_rl_aloha_20ep

echo "[3/6] PushT offline"
run_eval "${OFFLINE_POLICY}" pusht PushT-v0 2 2 20 5 smolvla_offline_pusht_20ep

echo "[4/6] PushT RL"
run_eval "${RL_POLICY}" pusht PushT-v0 2 2 20 5 smolvla_rl_pusht_20ep

echo "[5/6] MetaWorld offline"
run_eval "${OFFLINE_POLICY}" metaworld push-v3 4 4 20 5 smolvla_offline_metaworld_push_20ep

echo "[6/6] MetaWorld RL"
run_eval "${RL_POLICY}" metaworld push-v3 4 4 20 5 smolvla_rl_metaworld_push_20ep

echo "Cross-platform evaluation completed."
