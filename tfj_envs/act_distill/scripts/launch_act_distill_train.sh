#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-full}"

case "${MODE}" in
  smoke)
    exec "${SCRIPT_DIR}/train_act_distill_smoke.sh" "$@"
    ;;
  full)
    exec "${SCRIPT_DIR}/train_act_distill_full.sh" "$@"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use MODE=smoke or MODE=full." >&2
    exit 1
    ;;
esac
