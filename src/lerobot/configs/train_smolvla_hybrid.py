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

from dataclasses import dataclass, field

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@dataclass
class SmolVLAHybridCollectorConfig:
    n_envs: int = 1
    use_async_envs: bool = False
    chunks_per_step: int = 1
    warmup_chunks: int = 0
    max_steps_per_chunk: int | None = None


@dataclass
class SmolVLAHybridReplayBufferConfig:
    capacity: int = 4096
    online_batch_size: int = 16


@dataclass
class SmolVLAHybridLossConfig:
    offline_loss_weight: float = 1.0
    online_flow_loss_weight: float = 0.3
    value_loss_weight: float = 1.0
    discount: float = 0.99
    advantage_temperature: float = 1.0
    normalize_advantage: bool = True
    advantage_clip_min: float = -5.0
    advantage_clip_max: float = 5.0
    max_advantage_weight: float = 20.0


@dataclass(kw_only=True)
class TrainSmolVLAHybridConfig(TrainPipelineConfig):
    policy: SmolVLAConfig = field(default_factory=lambda: SmolVLAConfig(push_to_hub=False))
    eval_freq: int = 0
    collector: SmolVLAHybridCollectorConfig = field(default_factory=SmolVLAHybridCollectorConfig)
    replay_buffer: SmolVLAHybridReplayBufferConfig = field(default_factory=SmolVLAHybridReplayBufferConfig)
    losses: SmolVLAHybridLossConfig = field(default_factory=SmolVLAHybridLossConfig)

    def validate(self) -> None:
        super().validate()

        if self.policy.type != "smolvla":
            raise ValueError(
                f"`TrainSmolVLAHybridConfig` expects a `smolvla` policy, got {self.policy.type!r}."
            )
        if self.env is None:
            raise ValueError("SmolVLA hybrid training requires `env` to be configured.")
        if self.collector.n_envs != 1:
            raise NotImplementedError(
                "The first SmolVLA hybrid trainer version only supports `collector.n_envs == 1`."
            )
        if self.collector.chunks_per_step <= 0:
            raise ValueError(
                f"`collector.chunks_per_step` must be > 0, got {self.collector.chunks_per_step}."
            )
        if self.replay_buffer.capacity <= 0:
            raise ValueError(
                f"`replay_buffer.capacity` must be > 0, got {self.replay_buffer.capacity}."
            )
        if self.replay_buffer.online_batch_size <= 0:
            raise ValueError(
                "`replay_buffer.online_batch_size` must be > 0, "
                f"got {self.replay_buffer.online_batch_size}."
            )
