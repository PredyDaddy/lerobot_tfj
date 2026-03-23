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

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.rl.smolvla_hybrid.buffer import SmolVLAChunkTransition
from lerobot.utils.constants import ACTION


def resolve_single_vector_env(envs: dict[str, dict[int, gym.vector.VectorEnv]]) -> gym.vector.VectorEnv:
    flattened = [(suite_name, task_id, env) for suite_name, suite_envs in envs.items() for task_id, env in suite_envs.items()]
    if len(flattened) != 1:
        raise NotImplementedError(
            "The initial SmolVLA hybrid collector only supports a single suite/task environment."
        )

    return flattened[0][2]


class SmolVLAChunkCollector:
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        policy: SmolVLAPolicy,
        env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        *,
        discount: float,
        max_steps_per_chunk: int | None = None,
    ):
        if env.num_envs != 1:
            raise NotImplementedError(
                "The initial SmolVLA hybrid collector only supports `num_envs == 1`."
            )

        self.env = env
        self.policy = policy
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.discount = discount
        self.max_steps_per_chunk = max_steps_per_chunk
        self._current_observation: dict[str, np.ndarray] | None = None

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.policy.reset()
        seeds = [seed] if seed is not None else None
        self._current_observation, _ = self.env.reset(seed=seeds)
        return self._current_observation

    def _make_policy_input(self, raw_observation: dict[str, np.ndarray]) -> dict[str, Any]:
        observation = preprocess_observation(raw_observation)
        observation = add_envs_task(self.env, observation)
        observation = self.env_preprocessor(observation)
        return self.preprocessor(observation)

    def _extract_tensor_observation(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != ACTION
        }

    def _single_env_observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value[0].detach().cpu() for key, value in observation.items()}

    def _postprocess_action_chunk(self, action_chunk: Tensor) -> Tensor:
        action = self.postprocessor(action_chunk)
        action_transition = self.env_postprocessor({ACTION: action})
        env_action = action_transition[ACTION]
        if not isinstance(env_action, torch.Tensor):
            raise TypeError(f"Expected env action tensor, got {type(env_action)}")
        return env_action

    def collect(self, num_chunks: int) -> list[SmolVLAChunkTransition]:
        if num_chunks <= 0:
            return []

        if self._current_observation is None:
            self.reset()

        if self._current_observation is None:
            raise RuntimeError("Collector failed to initialize its first observation.")

        was_training = self.policy.training
        transitions: list[SmolVLAChunkTransition] = []

        for _ in range(num_chunks):
            policy_input = self._make_policy_input(self._current_observation)
            observation_tensors = self._extract_tensor_observation(policy_input)

            with torch.inference_mode():
                prediction = self.policy.predict_action_chunk_with_info(policy_input)

            env_action_chunk = self._postprocess_action_chunk(prediction.actions)
            rollout_horizon = min(self.policy.config.n_action_steps, env_action_chunk.shape[1])
            if self.max_steps_per_chunk is not None:
                rollout_horizon = min(rollout_horizon, self.max_steps_per_chunk)

            discounted_reward = 0.0
            bootstrap_discount = 1.0
            done = False
            next_observation = self._current_observation

            for step_idx in range(rollout_horizon):
                action_np = env_action_chunk[:, step_idx].detach().cpu().numpy()
                next_observation, reward, terminated, truncated, _ = self.env.step(action_np)

                discounted_reward += bootstrap_discount * float(reward[0])
                bootstrap_discount *= self.discount
                done = bool(terminated[0] or truncated[0])

                if done:
                    self.policy.reset()
                    break

            next_policy_input = self._make_policy_input(next_observation)
            next_observation_tensors = self._extract_tensor_observation(next_policy_input)
            transitions.append(
                SmolVLAChunkTransition(
                    observation=self._single_env_observation(observation_tensors),
                    action=prediction.actions[0].detach().cpu(),
                    reward=torch.tensor(discounted_reward, dtype=torch.float32),
                    next_observation=self._single_env_observation(next_observation_tensors),
                    done=torch.tensor(done, dtype=torch.bool),
                    bootstrap_discount=torch.tensor(
                        0.0 if done else bootstrap_discount,
                        dtype=torch.float32,
                    ),
                )
            )
            self._current_observation = next_observation

        if was_training:
            self.policy.train()

        return transitions
