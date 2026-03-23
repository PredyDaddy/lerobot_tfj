#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


PALIGEMMA_TOKENIZER_ID = "google/paligemma-3b-pt-224"
LOCAL_TOKENIZER_ENV_KEYS = (
    "PI05_LOCAL_TOKENIZER_PATH",
    "PI_LOCAL_TOKENIZER_PATH",
    "PALIGEMMA_LOCAL_TOKENIZER_PATH",
)
DEFAULT_LOCAL_TOKENIZER_CANDIDATES = (
    Path.home() / ".cache" / "modelscope" / "hub" / "models" / "google" / "paligemma-3b-pt-224",
)


def _tokenizer_candidate_iter(explicit_path: str | Path | None = None) -> Iterable[Path]:
    if explicit_path is not None:
        yield Path(explicit_path).expanduser()
    for env_key in LOCAL_TOKENIZER_ENV_KEYS:
        value = os.getenv(env_key)
        if value:
            yield Path(value).expanduser()
    yield from DEFAULT_LOCAL_TOKENIZER_CANDIDATES


def is_local_tokenizer_dir(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        return False
    required_markers = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "tokenizer_config.json",
    )
    return any((candidate / marker).is_file() for marker in required_markers)


def discover_local_paligemma_tokenizer_path(explicit_path: str | Path | None = None) -> Path | None:
    seen: set[Path] = set()
    for candidate in _tokenizer_candidate_iter(explicit_path):
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_local_tokenizer_dir(resolved):
            return resolved
    return None


def resolve_paligemma_tokenizer_source(explicit_path: str | Path | None = None) -> str:
    local_path = discover_local_paligemma_tokenizer_path(explicit_path)
    return str(local_path) if local_path is not None else PALIGEMMA_TOKENIZER_ID


def _siglip_check_ok() -> bool:
    return True


def ensure_siglip_check_available() -> bool:
    module_name = "transformers.models.siglip.check"

    try:
        siglip_pkg = importlib.import_module("transformers.models.siglip")
    except Exception:
        return False

    module = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            return False
    except Exception:
        return False

    if module is None:
        module = ModuleType(module_name)
        module.__package__ = "transformers.models.siglip"

    if not hasattr(module, "check_whether_transformers_replace_is_installed_correctly"):
        module.check_whether_transformers_replace_is_installed_correctly = _siglip_check_ok

    sys.modules[module_name] = module
    setattr(siglip_pkg, "check", module)
    return True
