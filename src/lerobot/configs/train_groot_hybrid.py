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

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.groot.configuration_groot import GROOT_ACTION_CHUNK_SIZE, GrootConfig


@dataclass
class GrootHybridCollectorConfig:
    n_envs: int = 1
    use_async_envs: bool = False
    chunks_per_step: int = 1
    warmup_chunks: int = 0
    max_steps_per_chunk: int | None = None


@dataclass
class GrootHybridReplayBufferConfig:
    capacity: int = 4096
    online_batch_size: int = 16


@dataclass
class GrootHybridValueConfig:
    hidden_dim: int = 512
    num_layers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class GrootHybridLossConfig:
    offline_loss_weight: float = 1.0
    online_flow_loss_weight: float = 0.3
    value_loss_weight: float = 1.0
    discount: float = 0.99
    use_advantage_weighting: bool = True
    advantage_temperature: float = 1.0
    normalize_advantage: bool = True
    advantage_clip_min: float = -5.0
    advantage_clip_max: float = 5.0
    max_advantage_weight: float = 20.0


@dataclass
class GrootOfflineReplayConfig:
    enabled: bool = False
    transition_stride: int = 1
    value_target_mode: str = "monte_carlo"
    terminal_reward: float = 1.0
    step_reward: float = 0.0
    success_value: bool = True
    action_pad_mode: str = "repeat_last"


@dataclass(kw_only=True)
class TrainGrootHybridConfig(TrainPipelineConfig):
    policy: GrootConfig = field(default_factory=lambda: GrootConfig(push_to_hub=False))
    eval_freq: int = 0
    collector: GrootHybridCollectorConfig = field(default_factory=GrootHybridCollectorConfig)
    replay_buffer: GrootHybridReplayBufferConfig = field(default_factory=GrootHybridReplayBufferConfig)
    value: GrootHybridValueConfig = field(default_factory=GrootHybridValueConfig)
    losses: GrootHybridLossConfig = field(default_factory=GrootHybridLossConfig)
    offline_replay: GrootOfflineReplayConfig = field(default_factory=GrootOfflineReplayConfig)

    def validate(self) -> None:
        if (
            self.policy.type == "groot"
            and self.policy.push_to_hub
            and self.policy.repo_id is None
            and parser.parse_arg("policy.push_to_hub") is None
            and parser.parse_arg("config_path") is None
            and parser.get_path_arg("policy") is None
        ):
            self.policy.push_to_hub = False

        super().validate()

        if self.policy.type != "groot":
            raise ValueError(
                f"`TrainGrootHybridConfig` expects a `groot` policy, got {self.policy.type!r}."
            )
        if self.offline_replay.enabled:
            if self.env is not None:
                raise ValueError(
                    "Dataset-only GROOT offline RL expects `env=None`; unset env when "
                    "`offline_replay.enabled=true`."
                )
            if self.num_workers != 0:
                raise ValueError(
                    "`offline_replay.enabled=true` requires `num_workers=0` because replay "
                    "sampling decodes dataset videos in the main process."
                )
        elif self.env is None:
            raise ValueError(
                "Groot hybrid training requires `env` to be configured unless "
                "`offline_replay.enabled=true`."
            )
        if self.eval_freq < 0:
            raise ValueError(f"`eval_freq` must be >= 0, got {self.eval_freq}.")
        if self.eval_freq > 0:
            raise NotImplementedError("The initial Groot hybrid trainer does not implement evaluation yet.")
        if self.collector.n_envs != 1:
            raise NotImplementedError(
                "The first Groot hybrid trainer version only supports `collector.n_envs == 1`."
            )
        if self.collector.chunks_per_step < 0:
            raise ValueError(
                f"`collector.chunks_per_step` must be >= 0, got {self.collector.chunks_per_step}."
            )
        if self.collector.warmup_chunks < 0:
            raise ValueError(
                f"`collector.warmup_chunks` must be >= 0, got {self.collector.warmup_chunks}."
            )
        if self.offline_replay.enabled:
            if self.collector.chunks_per_step != 0:
                raise ValueError(
                    "`collector.chunks_per_step` must be 0 when `offline_replay.enabled=true`, "
                    f"got {self.collector.chunks_per_step}."
                )
            if self.collector.warmup_chunks != 0:
                raise ValueError(
                    "`collector.warmup_chunks` must be 0 when `offline_replay.enabled=true`, "
                    f"got {self.collector.warmup_chunks}."
                )
            if self.offline_replay.transition_stride <= 0:
                raise ValueError(
                    "`offline_replay.transition_stride` must be > 0, "
                    f"got {self.offline_replay.transition_stride}."
                )
            if self.offline_replay.value_target_mode not in {"n_step", "monte_carlo"}:
                raise ValueError(
                    "`offline_replay.value_target_mode` must be one of "
                    "{'n_step', 'monte_carlo'}, "
                    f"got {self.offline_replay.value_target_mode!r}."
                )
            if self.offline_replay.action_pad_mode != "repeat_last":
                raise ValueError(
                    "Only `offline_replay.action_pad_mode='repeat_last'` is currently supported, "
                    f"got {self.offline_replay.action_pad_mode!r}."
                )
        elif self.collector.chunks_per_step == 0:
            raise ValueError(
                "`collector.chunks_per_step` must be > 0 for online Groot hybrid training."
            )
        if self.collector.max_steps_per_chunk is not None:
            if self.collector.max_steps_per_chunk <= 0:
                raise ValueError(
                    "`collector.max_steps_per_chunk` must be > 0 when provided, "
                    f"got {self.collector.max_steps_per_chunk}."
                )
            max_supported_chunk_steps = int(
                getattr(self.policy, "action_chunk_size", GROOT_ACTION_CHUNK_SIZE)
            )
            if self.collector.max_steps_per_chunk > max_supported_chunk_steps:
                raise ValueError(
                    "`collector.max_steps_per_chunk` cannot exceed the Groot action chunk size "
                    f"({max_supported_chunk_steps}), got {self.collector.max_steps_per_chunk}."
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
        if self.value.hidden_dim <= 0:
            raise ValueError(f"`value.hidden_dim` must be > 0, got {self.value.hidden_dim}.")
        if self.value.num_layers <= 0:
            raise ValueError(f"`value.num_layers` must be > 0, got {self.value.num_layers}.")
        if self.value.learning_rate <= 0:
            raise ValueError(
                f"`value.learning_rate` must be > 0, got {self.value.learning_rate}."
            )
        if self.value.weight_decay < 0:
            raise ValueError(
                f"`value.weight_decay` must be >= 0, got {self.value.weight_decay}."
            )
        if self.losses.offline_loss_weight < 0:
            raise ValueError(
                f"`losses.offline_loss_weight` must be >= 0, got {self.losses.offline_loss_weight}."
            )
        if self.losses.online_flow_loss_weight < 0:
            raise ValueError(
                "`losses.online_flow_loss_weight` must be >= 0, "
                f"got {self.losses.online_flow_loss_weight}."
            )
        if self.losses.value_loss_weight < 0:
            raise ValueError(
                f"`losses.value_loss_weight` must be >= 0, got {self.losses.value_loss_weight}."
            )
        if (
            self.losses.offline_loss_weight == 0
            and self.losses.online_flow_loss_weight == 0
            and self.losses.value_loss_weight == 0
        ):
            raise ValueError("At least one Groot hybrid loss weight must be > 0.")
        if not 0.0 <= self.losses.discount <= 1.0:
            raise ValueError(f"`losses.discount` must be within [0, 1], got {self.losses.discount}.")
        if self.losses.advantage_temperature <= 0:
            raise ValueError(
                "`losses.advantage_temperature` must be > 0, "
                f"got {self.losses.advantage_temperature}."
            )
        if self.losses.advantage_clip_min > self.losses.advantage_clip_max:
            raise ValueError(
                "`losses.advantage_clip_min` must be <= `losses.advantage_clip_max`, "
                f"got {self.losses.advantage_clip_min} > {self.losses.advantage_clip_max}."
            )
        if self.losses.max_advantage_weight <= 0:
            raise ValueError(
                "`losses.max_advantage_weight` must be > 0, "
                f"got {self.losses.max_advantage_weight}."
            )
