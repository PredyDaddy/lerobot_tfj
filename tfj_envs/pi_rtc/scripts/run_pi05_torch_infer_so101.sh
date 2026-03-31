#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

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

resolve_policy_path() {
  local candidate="$1"
  if [[ -f "${candidate}/config.json" && -f "${candidate}/model.safetensors" ]]; then
    printf '%s' "${candidate}"
    return
  fi
  if [[ -f "${candidate}/pretrained_model/config.json" && -f "${candidate}/pretrained_model/model.safetensors" ]]; then
    printf '%s' "${candidate}/pretrained_model"
    return
  fi
  printf '%s' "${candidate}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PALIGEMMA_LOCAL_TOKENIZER_PATH="${PALIGEMMA_LOCAL_TOKENIZER_PATH:-/data/modelscope/hub/models/google/paligemma-3b-pt-224}"

ROBOT_ID="${ROBOT_ID:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"

TOP_CAM_INDEX="${TOP_CAM_INDEX:-4}"
WRIST_CAM_INDEX="${WRIST_CAM_INDEX:-6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

POLICY_PATH="${POLICY_PATH:-/data/tfj/lerobot_tfj/pi_model}"
POLICY_PATH="$(resolve_policy_path "${POLICY_PATH}")"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-15}"
POLICY_TEMPORAL_ENSEMBLE_COEFF="${POLICY_TEMPORAL_ENSEMBLE_COEFF:-}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-}"

TRT_PATH="${TRT_PATH:-}"
TRT_METADATA_PATH="${TRT_METADATA_PATH:-}"
TRT_DEVICE="${TRT_DEVICE:-cuda:0}"

TASK_TEXT="${TASK_TEXT:-grasp block in bin}"
RUN_TIME_S="${RUN_TIME_S:-300}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-}"
SYNC_REFILL_TIMEOUT_S="${SYNC_REFILL_TIMEOUT_S:-}"
RTC_ENABLE="$(normalize_bool "${RTC_ENABLE:-true}")"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-}"
RTC_DEBUG="$(normalize_bool "${RTC_DEBUG:-false}")"
RTC_DEBUG_MAXLEN="${RTC_DEBUG_MAXLEN:-}"

SKIP_CAMERA_PREFLIGHT="$(normalize_bool "${SKIP_CAMERA_PREFLIGHT:-false}")"
SKIP_TRT_PREFLIGHT="$(normalize_bool "${SKIP_TRT_PREFLIGHT:-false}")"
PREFLIGHT_ONLY="$(normalize_bool "${PREFLIGHT_ONLY:-false}")"
DRY_RUN="$(normalize_bool "${DRY_RUN:-false}")"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tfj_envs/pi_rtc/scripts/run_pi05_torch_infer_so101.py"
  --robot-id="${ROBOT_ID}"
  --robot-port="${ROBOT_PORT}"
  --robot-calibration-dir="${ROBOT_CALIB_DIR}"
  --top-cam-index="${TOP_CAM_INDEX}"
  --wrist-cam-index="${WRIST_CAM_INDEX}"
  --camera-width="${CAMERA_WIDTH}"
  --camera-height="${CAMERA_HEIGHT}"
  --camera-fps="${CAMERA_FPS}"
  --policy-path="${POLICY_PATH}"
  --policy-device="${POLICY_DEVICE}"
  --trt-device="${TRT_DEVICE}"
  --task="${TASK_TEXT}"
  --run-time-s="${RUN_TIME_S}"
  --log-interval="${LOG_INTERVAL}"
)

if [[ -n "${POLICY_N_ACTION_STEPS}" ]]; then
  CMD+=(--policy-n-action-steps="${POLICY_N_ACTION_STEPS}")
fi

if [[ -n "${POLICY_NUM_INFERENCE_STEPS}" ]]; then
  CMD+=(--policy-num-inference-steps="${POLICY_NUM_INFERENCE_STEPS}")
fi

if [[ -n "${POLICY_TEMPORAL_ENSEMBLE_COEFF}" ]]; then
  CMD+=(--policy-temporal-ensemble-coeff="${POLICY_TEMPORAL_ENSEMBLE_COEFF}")
fi

if [[ -n "${TRT_PATH}" ]]; then
  CMD+=(--trt-path="${TRT_PATH}")
fi

if [[ -n "${TRT_METADATA_PATH}" ]]; then
  CMD+=(--trt-metadata-path="${TRT_METADATA_PATH}")
fi

if [[ -n "${PREFETCH_THRESHOLD}" ]]; then
  CMD+=(--prefetch-threshold="${PREFETCH_THRESHOLD}")
fi

if [[ -n "${SYNC_REFILL_TIMEOUT_S}" ]]; then
  CMD+=(--sync-refill-timeout-s="${SYNC_REFILL_TIMEOUT_S}")
fi

if [[ "${RTC_ENABLE}" == "true" ]]; then
  CMD+=(--rtc-enable)
fi

if [[ -n "${RTC_EXECUTION_HORIZON}" ]]; then
  CMD+=(--rtc-execution-horizon="${RTC_EXECUTION_HORIZON}")
fi

if [[ -n "${RTC_MAX_GUIDANCE_WEIGHT}" ]]; then
  CMD+=(--rtc-max-guidance-weight="${RTC_MAX_GUIDANCE_WEIGHT}")
fi

if [[ -n "${RTC_PREFIX_ATTENTION_SCHEDULE}" ]]; then
  CMD+=(--rtc-prefix-attention-schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}")
fi

if [[ "${RTC_DEBUG}" == "true" ]]; then
  CMD+=(--rtc-debug)
fi

if [[ -n "${RTC_DEBUG_MAXLEN}" ]]; then
  CMD+=(--rtc-debug-maxlen="${RTC_DEBUG_MAXLEN}")
fi

if [[ "${SKIP_CAMERA_PREFLIGHT}" == "true" ]]; then
  CMD+=(--skip-camera-preflight)
fi

if [[ "${SKIP_TRT_PREFLIGHT}" == "true" ]]; then
  CMD+=(--skip-trt-preflight)
fi

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  CMD+=(--preflight-only)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Using PI05 torch SO101 launcher"
echo "Policy path: ${POLICY_PATH}"
echo "Robot port: ${ROBOT_PORT}"
echo "Task: ${TASK_TEXT}"
echo "RTC defaults: enable=${RTC_ENABLE}, execution_horizon=${RTC_EXECUTION_HORIZON}, n_action_steps=${POLICY_N_ACTION_STEPS}"
printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
