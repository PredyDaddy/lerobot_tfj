#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

python - <<'PY'
from pathlib import Path
import json

import pyarrow.parquet as pq

root = Path("/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail")
root = Path(__import__("os").environ.get("DATASET_ROOT", str(root)))

info_path = root / "meta" / "info.json"
tasks_path = root / "meta" / "tasks.parquet"

print("=== info.json ===")
print(json.dumps(json.loads(info_path.read_text()), indent=2, ensure_ascii=False))
print()
print("=== tasks.parquet ===")
table = pq.read_table(tasks_path)
print(table.to_pydict())
PY
