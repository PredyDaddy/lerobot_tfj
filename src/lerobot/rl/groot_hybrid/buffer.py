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
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


RawObservation = dict[str, Any]
ObservationTransform = Callable[[list[RawObservation]], Any]


@dataclass
class GrootHybridTransition:
    observation: RawObservation
    action_chunk: Tensor
    reward: Tensor
    next_observation: RawObservation
    done: Tensor
    success: Tensor
    bootstrap_discount: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrootHybridBatch:
    observation: Any
    action_chunk: Tensor
    reward: Tensor
    next_observation: Any
    done: Tensor
    success: Tensor
    bootstrap_discount: Tensor
    metadata: list[dict[str, Any]]


def _clone_raw_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_raw_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_raw_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_raw_value(item) for item in value]
    return deepcopy(value)


def _clone_observation(observation: RawObservation) -> RawObservation:
    return {key: _clone_raw_value(value) for key, value in observation.items()}


def _to_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu").clone()


def _to_scalar_tensor(value: Tensor, *, dtype: torch.dtype) -> Tensor:
    return _to_cpu_tensor(value).to(dtype=dtype).reshape(())


def _transition_to_state(transition: GrootHybridTransition) -> dict[str, Any]:
    return {
        "observation": _clone_observation(transition.observation),
        "action_chunk": _to_cpu_tensor(transition.action_chunk),
        "reward": _to_scalar_tensor(transition.reward, dtype=torch.float32),
        "next_observation": _clone_observation(transition.next_observation),
        "done": _to_scalar_tensor(transition.done, dtype=torch.bool),
        "success": _to_scalar_tensor(transition.success, dtype=torch.bool),
        "bootstrap_discount": _to_scalar_tensor(transition.bootstrap_discount, dtype=torch.float32),
        "metadata": _clone_raw_value(transition.metadata),
    }


def _transition_from_state(state: Mapping[str, Any]) -> GrootHybridTransition:
    return GrootHybridTransition(
        observation=_clone_observation(state["observation"]),
        action_chunk=_to_cpu_tensor(state["action_chunk"]),
        reward=_to_scalar_tensor(state["reward"], dtype=torch.float32),
        next_observation=_clone_observation(state["next_observation"]),
        done=_to_scalar_tensor(state["done"], dtype=torch.bool),
        success=_to_scalar_tensor(state["success"], dtype=torch.bool),
        bootstrap_discount=_to_scalar_tensor(state["bootstrap_discount"], dtype=torch.float32),
        metadata=_clone_raw_value(state.get("metadata", {})),
    )


def _load_state_file(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _collate_raw_values(values: list[Any]) -> Any:
    if not values:
        return []

    first = values[0]

    if isinstance(first, torch.Tensor):
        return torch.stack([value.detach().to(device="cpu") for value in values], dim=0)

    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)

    if isinstance(first, Mapping):
        return {
            key: _collate_raw_values([value[key] for value in values])
            for key in first
        }

    if isinstance(first, tuple):
        return tuple(_collate_raw_values([value[idx] for value in values]) for idx in range(len(first)))

    if isinstance(first, list) and all(len(value) == len(first) for value in values):
        return [_collate_raw_values([value[idx] for value in values]) for idx in range(len(first))]

    if isinstance(first, np.generic | bool | int | float):
        return np.asarray(values)

    return deepcopy(values)


def _stack_action_chunks(action_chunks: list[Tensor]) -> Tensor:
    normalized = [_to_cpu_tensor(action_chunk) for action_chunk in action_chunks]
    shapes = {tuple(action_chunk.shape) for action_chunk in normalized}
    if len(shapes) == 1:
        return torch.stack(normalized, dim=0)

    if not normalized or any(action_chunk.ndim == 0 for action_chunk in normalized):
        raise ValueError("Cannot pad scalar action chunks.")

    tail_shape = normalized[0].shape[1:]
    if any(action_chunk.shape[1:] != tail_shape for action_chunk in normalized):
        raise ValueError("Action chunk shapes are incompatible for padding.")

    max_steps = max(action_chunk.shape[0] for action_chunk in normalized)
    padded = torch.zeros(
        (len(normalized), max_steps, *tail_shape),
        dtype=normalized[0].dtype,
    )
    for index, action_chunk in enumerate(normalized):
        padded[index, : action_chunk.shape[0]] = action_chunk

    return padded


def _move_nested_tensors(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_nested_tensors(nested_value, device) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    return value


class GrootHybridReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"`capacity` must be > 0, got {capacity}.")

        self.capacity = capacity
        self._storage: list[GrootHybridTransition] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: GrootHybridTransition) -> None:
        stored_transition = _transition_from_state(_transition_to_state(transition))

        if len(self._storage) < self.capacity:
            self._storage.append(stored_transition)
        else:
            self._storage[self._position] = stored_transition

        self._position = (self._position + 1) % self.capacity

    def extend(self, transitions: Sequence[GrootHybridTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def as_list(self) -> list[GrootHybridTransition]:
        return [_transition_from_state(_transition_to_state(transition)) for transition in self._storage]

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        observation_transform: ObservationTransform | None = None,
        next_observation_transform: ObservationTransform | None = None,
    ) -> GrootHybridBatch:
        if not self._storage:
            raise RuntimeError("Cannot sample from an empty GrootHybridReplayBuffer.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")

        actual_batch_size = min(batch_size, len(self._storage))
        indices = random.sample(range(len(self._storage)), k=actual_batch_size)
        batch = [self._storage[index] for index in indices]

        raw_observations = [_clone_observation(transition.observation) for transition in batch]
        raw_next_observations = [_clone_observation(transition.next_observation) for transition in batch]

        observation = (
            observation_transform(raw_observations)
            if observation_transform is not None
            else _collate_raw_values(raw_observations)
        )
        next_observation = (
            next_observation_transform(raw_next_observations)
            if next_observation_transform is not None
            else _collate_raw_values(raw_next_observations)
        )

        return GrootHybridBatch(
            observation=_move_nested_tensors(observation, device),
            action_chunk=_stack_action_chunks([transition.action_chunk for transition in batch]).to(device),
            reward=torch.stack([transition.reward for transition in batch], dim=0).to(device).view(-1),
            next_observation=_move_nested_tensors(next_observation, device),
            done=torch.stack([transition.done for transition in batch], dim=0).to(device).view(-1).bool(),
            success=torch.stack([transition.success for transition in batch], dim=0).to(device).view(-1).bool(),
            bootstrap_discount=torch.stack(
                [transition.bootstrap_discount for transition in batch],
                dim=0,
            ).to(device).view(-1),
            metadata=[_clone_raw_value(transition.metadata) for transition in batch],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "position": self._position,
            "storage": [_transition_to_state(transition) for transition in self._storage],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        capacity = int(state_dict["capacity"])
        position = int(state_dict["position"])
        storage = [_transition_from_state(item) for item in state_dict["storage"]]

        if capacity <= 0:
            raise ValueError(f"`capacity` must be > 0, got {capacity}.")
        if len(storage) > capacity:
            raise ValueError(
                f"State dict contains {len(storage)} transitions, which exceeds capacity {capacity}."
            )
        if position < 0 or position >= capacity:
            raise ValueError(f"`position` must be within [0, {capacity}), got {position}.")

        self.capacity = capacity
        self._position = position
        self._storage = storage

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GrootHybridReplayBuffer:
        state_dict = _load_state_file(Path(path))
        buffer = cls(capacity=int(state_dict["capacity"]))
        buffer.load_state_dict(state_dict)
        return buffer
