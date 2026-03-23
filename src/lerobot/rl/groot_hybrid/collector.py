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

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from lerobot.envs.utils import preprocess_observation
from lerobot.rl.groot_hybrid.buffer import GrootHybridTransition, RawObservation


def _identity(value: Any) -> Any:
    return value


def _to_scalar_float(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected a scalar reward, got shape {np.asarray(value).shape}.")
    return float(array[0])


def _to_scalar_bool(value: Any) -> bool:
    array = np.asarray(value, dtype=np.bool_).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected a scalar boolean, got shape {np.asarray(value).shape}.")
    return bool(array[0])


def _extract_success(info: Mapping[str, Any]) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, Mapping) and "is_success" in final_info:
            return _to_scalar_bool(final_info["is_success"])
        if isinstance(final_info, (list, tuple)) and final_info:
            first_item = final_info[0]
            if isinstance(first_item, Mapping) and "is_success" in first_item:
                return _to_scalar_bool(first_item["is_success"])

    if "is_success" in info:
        return _to_scalar_bool(info["is_success"])
    if "success" in info:
        return _to_scalar_bool(info["success"])
    return False


def default_observation_builder(raw_observation: RawObservation, task: str) -> dict[str, Any]:
    observation = preprocess_observation(raw_observation)
    observation["task"] = [task]
    return observation


class RolloutAdapter(Protocol):
    num_envs: int

    def reset(self, seed: int | None = None) -> tuple[RawObservation, dict[str, Any]]: ...

    def step(self, action: np.ndarray) -> tuple[RawObservation, Any, Any, Any, dict[str, Any]]: ...

    def get_task(self) -> str: ...


class ChunkPolicy(Protocol):
    training: bool

    def eval(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def reset(self) -> None: ...

    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor: ...


class GrootHybridCollector:
    def __init__(
        self,
        rollout: RolloutAdapter,
        policy: ChunkPolicy,
        *,
        discount: float,
        max_steps_per_chunk: int | None = None,
        observation_builder: Callable[[RawObservation, str], dict[str, Any]] | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        postprocessor: Callable[[Tensor], Tensor] | None = None,
    ):
        if rollout.num_envs != 1:
            raise NotImplementedError("The initial Groot hybrid collector only supports `num_envs == 1`.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f"`discount` must be within [0, 1], got {discount}.")
        if max_steps_per_chunk is not None and max_steps_per_chunk <= 0:
            raise ValueError(
                f"`max_steps_per_chunk` must be > 0 when provided, got {max_steps_per_chunk}."
            )
        if max_steps_per_chunk is not None and max_steps_per_chunk > 16:
            raise ValueError("Groot hybrid collector only supports `max_steps_per_chunk <= 16`.")

        self.rollout = rollout
        self.policy = policy
        self.discount = discount
        self.max_steps_per_chunk = max_steps_per_chunk
        self.observation_builder = observation_builder or default_observation_builder
        self.preprocessor = preprocessor or _identity
        self.postprocessor = postprocessor or _identity
        self._current_observation: RawObservation | None = None

    def reset(self, seed: int | None = None) -> RawObservation:
        self.policy.reset()
        self._current_observation, _ = self.rollout.reset(seed=seed)
        return self._current_observation

    def _make_policy_input(self, raw_observation: RawObservation) -> dict[str, Any]:
        observation = self.observation_builder(raw_observation, self.rollout.get_task())
        return self.preprocessor(observation)

    def collect(self, num_chunks: int) -> list[GrootHybridTransition]:
        if num_chunks <= 0:
            return []

        if self._current_observation is None:
            self.reset()

        if self._current_observation is None:
            raise RuntimeError("Collector failed to initialize its first observation.")

        was_training = getattr(self.policy, "training", False)
        transitions: list[GrootHybridTransition] = []

        for _ in range(num_chunks):
            raw_observation = self._current_observation
            policy_input = self._make_policy_input(raw_observation)

            with torch.inference_mode():
                predicted_action_chunk = self.policy.predict_action_chunk(policy_input)

            env_action_chunk = self.postprocessor(predicted_action_chunk)
            if not isinstance(env_action_chunk, torch.Tensor):
                raise TypeError(f"Expected postprocessed action chunk tensor, got {type(env_action_chunk)}.")
            if env_action_chunk.ndim != 3 or env_action_chunk.shape[0] != 1:
                raise ValueError(
                    "Expected action chunk shape `(1, steps, action_dim)` "
                    f"for single-env rollout, got {tuple(env_action_chunk.shape)}."
                )

            rollout_horizon = env_action_chunk.shape[1]
            if self.max_steps_per_chunk is not None:
                rollout_horizon = min(rollout_horizon, self.max_steps_per_chunk)
            if rollout_horizon <= 0:
                raise RuntimeError("Policy returned an empty action chunk.")

            discounted_reward = 0.0
            bootstrap_discount = 1.0
            done = False
            success = False
            terminated = False
            truncated = False
            next_observation = raw_observation
            executed_steps = 0

            for step_idx in range(rollout_horizon):
                step_action = env_action_chunk[:, step_idx].detach().to(device="cpu").numpy()
                next_observation, reward, terminated_flag, truncated_flag, info = self.rollout.step(step_action)

                terminated = _to_scalar_bool(terminated_flag)
                truncated = _to_scalar_bool(truncated_flag)
                done = terminated or truncated
                success = success or _extract_success(info)

                discounted_reward += bootstrap_discount * _to_scalar_float(reward)
                bootstrap_discount *= self.discount
                executed_steps += 1

                if done:
                    break

            # Keep the full predicted chunk in replay and track the executed prefix in metadata.
            transitions.append(
                GrootHybridTransition(
                    observation=raw_observation,
                    action_chunk=predicted_action_chunk[0].detach().to(device="cpu"),
                    reward=torch.tensor(discounted_reward, dtype=torch.float32),
                    next_observation=next_observation,
                    done=torch.tensor(done, dtype=torch.bool),
                    success=torch.tensor(success, dtype=torch.bool),
                    bootstrap_discount=torch.tensor(
                        0.0 if done else bootstrap_discount,
                        dtype=torch.float32,
                    ),
                    metadata={
                        "task": self.rollout.get_task(),
                        "executed_steps": executed_steps,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )
            )

            if done:
                self.reset()
            else:
                self._current_observation = next_observation

        if was_training:
            self.policy.train()

        return transitions
