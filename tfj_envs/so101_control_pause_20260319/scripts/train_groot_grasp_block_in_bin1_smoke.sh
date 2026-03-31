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

print_usage() {
  cat <<'EOF'
Warm start smoke script for GR00T on grasp_block_in_bin1.

Examples
  bash scripts/train_groot_grasp_block_in_bin1_smoke.sh
  PREFLIGHT_ONLY=1 bash scripts/train_groot_grasp_block_in_bin1_smoke.sh
  RESUME=1 bash scripts/train_groot_grasp_block_in_bin1_smoke.sh
  RESUME=1 RESUME_CONFIG_PATH=outputs/groot_warm_start_smoke/checkpoints/last/pretrained_model/train_config.json \
    bash scripts/train_groot_grasp_block_in_bin1_smoke.sh

Key defaults
  DATASET_ROOT=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1
  DATASET_REPO_ID=admin123/grasp_block_in_bin1
  DATASET_VIDEO_BACKEND=pyav
  BASE_MODEL_PATH=/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B
  OUTPUT_DIR=outputs/groot_warm_start_smoke
  BATCH_SIZE=1
  STEPS=10
  SAVE_FREQ=5
  NUM_WORKERS=0
  DEVICE=cuda

Resume note
  The correct resume config path is:
  OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json
  Do not point resume to checkpoints/last/train_config.json.
  If resume still fails after using the correct path, the current repo may still
  hit an upstream GR00T processor override mismatch. That is separate from the
  path fix in this script.

Useful env flags
  PREFLIGHT_ONLY=1  Run all checks and print the resolved command without training.
  DRY_RUN=1         Alias of PREFLIGHT_ONLY for command preview.
  RESUME=1          Resume from RESUME_CONFIG_PATH instead of starting a new run.
  PYTHON_BIN=...    Override interpreter selection. If unset, the script prefers
                    /home/cqy/miniconda3/envs/gr00t/bin/python on this machine.
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

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  echo "Unable to find a usable python interpreter. Set PYTHON_BIN explicitly." >&2
  exit 1
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

DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1}"
DATASET_VIDEO_BACKEND="pyav"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B}"
TOKENIZER_ASSETS_PATH="${TOKENIZER_ASSETS_PATH:-/home/cqy/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groot_warm_start_smoke}"
JOB_NAME="${JOB_NAME:-groot_grasp_block_in_bin1_warm_start_smoke}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-10}"
SAVE_FREQ="${SAVE_FREQ:-5}"
LOG_FREQ="${LOG_FREQ:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
RESUME="$(normalize_bool "${RESUME:-false}")"
PREFLIGHT_ONLY="$(normalize_bool "${PREFLIGHT_ONLY:-false}")"
DRY_RUN="$(normalize_bool "${DRY_RUN:-false}")"

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
export DATASET_ROOT
export DEVICE

BASE_MODEL_REALPATH="$(readlink -f "${BASE_MODEL_PATH}" 2>/dev/null || printf '%s' "${BASE_MODEL_PATH}")"
OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"

require_dir "${DATASET_ROOT}" "Dataset root"
require_file "${DATASET_ROOT}/meta/info.json" "Dataset metadata"
require_dir "${DATASET_ROOT}/videos" "Dataset videos directory"

require_dir "${BASE_MODEL_REALPATH}" "GR00T base model directory"
require_file "${BASE_MODEL_REALPATH}/config.json" "GR00T base model config"
if [[ ! -f "${BASE_MODEL_REALPATH}/model.safetensors.index.json" && ! -f "${BASE_MODEL_REALPATH}/model.safetensors" ]]; then
  echo "GR00T base model weights not found under ${BASE_MODEL_REALPATH}" >&2
  exit 1
fi

require_dir "${TOKENIZER_ASSETS_PATH}" "Local Eagle tokenizer assets cache"
require_file "${TOKENIZER_ASSETS_PATH}/config.json" "Local Eagle tokenizer config"
require_file "${TOKENIZER_ASSETS_PATH}/vocab.json" "Local Eagle tokenizer vocab"
require_file "${TOKENIZER_ASSETS_PATH}/merges.txt" "Local Eagle tokenizer merges"

mkdir -p "${OUTPUT_PARENT}"
if [[ ! -w "${OUTPUT_PARENT}" ]]; then
  echo "Output parent directory is not writable: ${OUTPUT_PARENT}" >&2
  exit 1
fi

if [[ "${RESUME}" == "true" ]]; then
  require_file "${RESUME_CONFIG_PATH}" "Resume config"
else
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output directory already exists: ${OUTPUT_DIR}" >&2
    echo "Use RESUME=1 with ${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" >&2
    exit 1
  fi
fi

TRAIN_CMD=()
TRAIN_CMD=("${PYTHON_BIN}" -m lerobot.scripts.lerobot_train)

"${PYTHON_BIN}" - <<'PY'
import importlib.util as import_util
import os
import sys

required = {
    "torch": "torch",
    "accelerate": "accelerate",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "datasets": "datasets",
    "draccus": "draccus",
    "av": "av",
    "PIL": "PIL",
    "peft": "peft",
    "timm": "timm",
    "tree": "tree",
}
missing = [package for package, module in required.items() if import_util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + ". Install the GR00T stack, for example: pip install -e '.[groot]'"
    )

if os.environ.get("DEVICE", "cuda") == "cuda":
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("DEVICE=cuda but torch.cuda.is_available() is false.")

if import_util.find_spec("flash_attn") is None:
    print(
        "Warning: flash_attn is not installed; GR00T will fall back to the non-FlashAttention path.",
        file=sys.stderr,
    )
PY

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
    video_streams = [stream for stream in container.streams if stream.type == "video"]
    if not video_streams:
        raise SystemExit(f"No video stream found in {sample_video}")

    decoded = False
    for _ in container.decode(video_streams[0]):
        decoded = True
        break

    if not decoded:
        raise SystemExit(f"PyAV opened {sample_video} but did not decode a frame")

print(f"PyAV decode preflight OK: {sample_video}")
PY

FRESH_CMD=(
  "${TRAIN_CMD[@]}"
  --policy.type=groot
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --policy.base_model_path="${BASE_MODEL_PATH}"
  --policy.chunk_size=16
  --policy.n_action_steps=16
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend="${DATASET_VIDEO_BACKEND}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --num_workers="${NUM_WORKERS}"
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint=true
  --eval_freq=0
  --log_freq="${LOG_FREQ}"
  --wandb.enable=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
)

RESUME_CMD=(
  "${TRAIN_CMD[@]}"
  --resume=true
  --config_path="${RESUME_CONFIG_PATH}"
)

if [[ "$#" -gt 0 ]]; then
  if [[ "${RESUME}" == "true" ]]; then
    RESUME_CMD+=("$@")
  else
    FRESH_CMD+=("$@")
  fi
fi

echo "GR00T warm start smoke preflight passed."
echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset repo id: ${DATASET_REPO_ID}"
echo "Dataset video backend: ${DATASET_VIDEO_BACKEND}"
echo "Python interpreter: ${PYTHON_BIN}"
echo "Base model path: ${BASE_MODEL_PATH}"
echo "Resolved base model path: ${BASE_MODEL_REALPATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Resume config path: ${RESUME_CONFIG_PATH}"
echo "Resume path must be: checkpoints/last/pretrained_model/train_config.json"

if [[ "${RESUME}" == "true" ]]; then
  echo "Note: this script uses the correct resume config path." >&2
  echo "If resume still fails, the remaining issue is likely the upstream GR00T processor override mismatch, not the path." >&2
  print_command "${RESUME_CMD[@]}"
else
  print_command "${FRESH_CMD[@]}"
  printf 'Correct resume command after a smoke run:\n'
  printf ' %q' "${RESUME_CMD[@]}"
  printf '\n'
fi

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  exit 0
fi

if [[ "${RESUME}" == "true" ]]; then
  exec "${RESUME_CMD[@]}"
fi

exec "${FRESH_CMD[@]}"
