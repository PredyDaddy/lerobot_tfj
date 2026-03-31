#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OPENCLAW_GROOT_HOST="${OPENCLAW_GROOT_HOST:-127.0.0.1}"
OPENCLAW_GROOT_PORT="${OPENCLAW_GROOT_PORT:-8765}"

exec "${PYTHON_BIN}" scripts/openclaw_groot_server.py
