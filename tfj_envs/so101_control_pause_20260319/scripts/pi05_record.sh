#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK_TEXT="${1:-Clean the desk}"

export TASK_TEXT

exec "${REPO_ROOT}/scripts/run_pi05_so101_builtin_record.sh"
