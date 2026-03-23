#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/launch_smolvla_hybrid_train.sh" aloha AlohaBlockInBin-v0 "$@"
