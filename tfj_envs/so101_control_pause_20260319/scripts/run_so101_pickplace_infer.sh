#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

normalize_backend() {
  local backend
  backend="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${backend}" in
    pi|pi0.5) echo "pi05" ;;
    act_distill) echo "act" ;;
    policy) echo "policy_record" ;;
    geom|geom_grasp) echo "geom_grasp" ;;
    groot|smolvla|pi05|act|policy_record|geom_grasp) echo "${backend}" ;;
    *)
      echo "Unsupported BACKEND: $1" >&2
      echo "Supported BACKEND values: groot, smolvla, pi05, act, policy_record, geom_grasp" >&2
      exit 2
      ;;
  esac
}

require_arg_value() {
  local flag="$1"
  if [[ $# -lt 2 ]]; then
    echo "Missing value for ${flag}" >&2
    exit 2
  fi
}

BACKEND_RAW="${BACKEND:-groot}"
TASK_TEXT="${TASK_TEXT:-${TASK:-}}"
INTENT_JSON="${INTENT_JSON:-${TASK_INTENT_JSON:-${INTENT:-}}}"
SAFETY_PROFILE="${SAFETY_PROFILE:-${SAFETY:-}}"
EVENTS_JSONL_PATH="${EVENTS_JSONL_PATH:-${EVENTS_PATH:-}}"
CLEAR_DATASET_ROOT="${CLEAR_DATASET_ROOT:-${CLEAR_DATASET:-false}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend=*)
      BACKEND_RAW="${1#*=}"
      ;;
    --backend)
      require_arg_value "$1" "$@"
      shift
      BACKEND_RAW="$1"
      ;;
    --task=*|--task_text=*|--instruction=*)
      TASK_TEXT="${1#*=}"
      ;;
    --task|--task_text|--instruction)
      require_arg_value "$1" "$@"
      shift
      TASK_TEXT="$1"
      ;;
    --robot_id=*)
      ROBOT_ID="${1#*=}"
      ;;
    --robot_id)
      require_arg_value "$1" "$@"
      shift
      ROBOT_ID="$1"
      ;;
    --intent_json=*|--task_intent_json=*|--intent=*)
      INTENT_JSON="${1#*=}"
      ;;
    --intent_json|--task_intent_json|--intent)
      require_arg_value "$1" "$@"
      shift
      INTENT_JSON="$1"
      ;;
    --safety_profile=*|--safety=*|--guard_profile=*)
      SAFETY_PROFILE="${1#*=}"
      ;;
    --safety_profile|--safety|--guard_profile)
      require_arg_value "$1" "$@"
      shift
      SAFETY_PROFILE="$1"
      ;;
    --events_jsonl_path=*|--events_path=*)
      EVENTS_JSONL_PATH="${1#*=}"
      ;;
    --events_jsonl_path|--events_path)
      require_arg_value "$1" "$@"
      shift
      EVENTS_JSONL_PATH="$1"
      ;;
    --clear_dataset_root=*)
      CLEAR_DATASET_ROOT="$(normalize_bool "${1#*=}")"
      ;;
    --clear_dataset=*)
      CLEAR_DATASET_ROOT="$(normalize_bool "${1#*=}")"
      ;;
    --clear_dataset_root|--clear_dataset)
      if [[ $# -ge 2 && "${2}" != --* ]]; then
        shift
        CLEAR_DATASET_ROOT="$(normalize_bool "$1")"
      else
        CLEAR_DATASET_ROOT="true"
      fi
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      ;;
  esac
  shift
done
set -- "${FORWARDED_ARGS[@]}"

BACKEND="$(normalize_backend "${BACKEND_RAW}")"
if [[ -z "${TASK_TEXT}" ]]; then
  TASK_TEXT="Put the block in the bin"
fi
TASK_INTENT_JSON="${INTENT_JSON}"
SAFETY_PROFILE="$(printf '%s' "${SAFETY_PROFILE:-default}" | tr '[:upper:]' '[:lower:]')"
DEFAULT_EVENTS_JSONL_PATH=""
if [[ -n "${OPENCLAW_JOB_DIR:-}" ]]; then
  DEFAULT_EVENTS_JSONL_PATH="${OPENCLAW_JOB_DIR}/events.jsonl"
fi
if [[ -z "${EVENTS_JSONL_PATH}" && -n "${DEFAULT_EVENTS_JSONL_PATH}" ]]; then
  EVENTS_JSONL_PATH="${DEFAULT_EVENTS_JSONL_PATH}"
fi
EVENTS_PATH="${EVENTS_JSONL_PATH}"
CLEAR_DATASET_ROOT="$(normalize_bool "${CLEAR_DATASET_ROOT}")"

export BACKEND TASK_TEXT INTENT_JSON TASK_INTENT_JSON SAFETY_PROFILE EVENTS_JSONL_PATH EVENTS_PATH CLEAR_DATASET_ROOT ROBOT_ID

echo "Unified SO101 router"
echo "  backend=${BACKEND}"
echo "  robot_id=${ROBOT_ID}"
echo "  clear_dataset_root=${CLEAR_DATASET_ROOT}"
if [[ -n "${EVENTS_JSONL_PATH}" ]]; then
  echo "  events_jsonl_path=${EVENTS_JSONL_PATH}"
fi
if [[ -n "${SAFETY_PROFILE}" ]]; then
  echo "  safety_profile=${SAFETY_PROFILE}"
fi

case "${BACKEND}" in
  groot)
    exec "${SCRIPT_DIR}/run_groot_so101_infer.sh" "$@"
    ;;
  smolvla)
    exec "${SCRIPT_DIR}/run_smolvla_so101_infer.sh" "$@"
    ;;
  pi05)
    exec "${SCRIPT_DIR}/run_pi05_so101_infer.sh" "$@"
    ;;
  act)
    exec "${SCRIPT_DIR}/run_act_distill_so101_infer.sh" "$@"
    ;;
  policy_record)
    exec "${SCRIPT_DIR}/run_so101_policy_record.sh" "$@"
    ;;
  geom_grasp)
    exec "${SCRIPT_DIR}/run_geom_grasp_so101_infer.sh" "$@"
    ;;
esac
