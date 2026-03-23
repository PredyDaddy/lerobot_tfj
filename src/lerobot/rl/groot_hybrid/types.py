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

import json
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from torch import Tensor


RawObservation: TypeAlias = dict[str, Any]
RawObservationBatch: TypeAlias = dict[str, Any]
SerializableMetadata: TypeAlias = dict[str, Any]

GROOT_CHUNK_TRANSITION_FIELDS = (
    "observation",
    "action_chunk",
    "reward",
    "next_observation",
    "done",
    "success",
    "bootstrap_discount",
    "metadata",
)


def ensure_serializable_metadata(metadata: SerializableMetadata | None) -> SerializableMetadata:
    """Validate that chunk metadata stays JSON-serializable and cheap to checkpoint."""
    validated = dict(metadata) if metadata is not None else {}
    try:
        json.dumps(validated)
    except TypeError as exc:
        raise TypeError(
            "GROOT chunk metadata must only contain JSON-serializable values."
        ) from exc
    return validated


@dataclass
class GrootChunkTransition:
    observation: RawObservation
    action_chunk: Tensor
    reward: Tensor
    next_observation: RawObservation
    done: Tensor
    success: Tensor
    bootstrap_discount: Tensor
    metadata: SerializableMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = ensure_serializable_metadata(self.metadata)


@dataclass
class GrootChunkBatch:
    observation: RawObservationBatch
    action_chunk: Tensor
    reward: Tensor
    next_observation: RawObservationBatch
    done: Tensor
    success: Tensor
    bootstrap_discount: Tensor
    metadata: list[SerializableMetadata] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.metadata = [ensure_serializable_metadata(item) for item in self.metadata]

