from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ACT_TRT_DIR = SCRIPT_DIR.parent
DOCS_DIR = ACT_TRT_DIR / "docs"
TFJ_ENVS_DIR = ACT_TRT_DIR.parent
REPO_ROOT = TFJ_ENVS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
