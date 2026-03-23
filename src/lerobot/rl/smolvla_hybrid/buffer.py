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

import random
from dataclasses import dataclass

import torch
from torch import Tensor


ObservationBatch = dict[str, Tensor]


@dataclass
class SmolVLAChunkTransition:
    observation: ObservationBatch
    action: Tensor
    reward: Tensor
    next_observation: ObservationBatch
    done: Tensor
    bootstrap_discount: Tensor


@dataclass
class SmolVLAChunkBatch:
    observation: ObservationBatch
    action: Tensor
    reward: Tensor
    next_observation: ObservationBatch
    done: Tensor
    bootstrap_discount: Tensor


def _to_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu")


def _to_cpu_observation(observation: ObservationBatch) -> ObservationBatch:
    return {key: _to_cpu_tensor(value) for key, value in observation.items()}


class SmolVLAChunkReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"`capacity` must be > 0, got {capacity}.")

        self.capacity = capacity
        self._storage: list[SmolVLAChunkTransition] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: SmolVLAChunkTransition) -> None:
        stored_transition = SmolVLAChunkTransition(
            observation=_to_cpu_observation(transition.observation),
            action=_to_cpu_tensor(transition.action),
            reward=_to_cpu_tensor(transition.reward),
            next_observation=_to_cpu_observation(transition.next_observation),
            done=_to_cpu_tensor(transition.done),
            bootstrap_discount=_to_cpu_tensor(transition.bootstrap_discount),
        )

        if len(self._storage) < self.capacity:
            self._storage.append(stored_transition)
        else:
            self._storage[self._position] = stored_transition

        self._position = (self._position + 1) % self.capacity

    def extend(self, transitions: list[SmolVLAChunkTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, device: torch.device | str) -> SmolVLAChunkBatch:
        if not self._storage:
            raise RuntimeError("Cannot sample from an empty SmolVLAChunkReplayBuffer.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")

        actual_batch_size = min(batch_size, len(self._storage))
        indices = random.sample(range(len(self._storage)), k=actual_batch_size)
        batch = [self._storage[index] for index in indices]

        observation = {
            key: torch.stack([transition.observation[key] for transition in batch], dim=0).to(device)
            for key in batch[0].observation
        }
        next_observation = {
            key: torch.stack([transition.next_observation[key] for transition in batch], dim=0).to(device)
            for key in batch[0].next_observation
        }

        return SmolVLAChunkBatch(
            observation=observation,
            action=torch.stack([transition.action for transition in batch], dim=0).to(device),
            reward=torch.stack([transition.reward for transition in batch], dim=0).to(device).view(-1),
            next_observation=next_observation,
            done=torch.stack([transition.done for transition in batch], dim=0).to(device).view(-1).bool(),
            bootstrap_discount=torch.stack(
                [transition.bootstrap_discount for transition in batch], dim=0
            ).to(device).view(-1),
        )
