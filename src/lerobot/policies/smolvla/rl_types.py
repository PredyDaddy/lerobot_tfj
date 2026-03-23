#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class SmolVLAPrefixContext:
    embeddings: Tensor
    pad_masks: Tensor
    att_masks: Tensor
    hidden_states: Tensor
    pooled_features: Tensor
    past_key_values: Any | None = None


@dataclass
class SmolVLAActionChunkPrediction:
    actions: Tensor
    value: Tensor
    noise: Tensor
    prefix_features: Tensor
    predicted_flow: Tensor | None = None
