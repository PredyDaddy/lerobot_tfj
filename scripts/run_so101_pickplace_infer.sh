#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/../tfj_envs/groot_rl/scripts/run_so101_pickplace_infer.sh"

exec "${TARGET_SCRIPT}" "$@"
