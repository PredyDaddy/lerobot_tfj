#!/usr/bin/env python

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.rl.groot_hybrid.buffer import GrootHybridBatch

RawObservation = dict[str, Any]
ObservationTransform = Callable[[list[RawObservation]], Any]


@dataclass
class GrootOfflineTransition:
    start_index: int
    next_index: int
    executed_steps: int
    reward: float
    bootstrap_discount: float
    done: bool
    success: bool
    episode_index: int


def _as_action_tensor(action: Any) -> Tensor:
    if isinstance(action, torch.Tensor):
        return action.detach().to(device="cpu").clone().float().view(-1)
    if isinstance(action, np.ndarray):
        return torch.from_numpy(action).detach().to(device="cpu").clone().float().view(-1)
    return torch.as_tensor(action, dtype=torch.float32).detach().to(device="cpu").clone().view(-1)


def _discounted_step_sum(step_reward: float, num_steps: int, discount: float) -> float:
    if num_steps <= 0 or step_reward == 0.0:
        return 0.0
    if discount == 1.0:
        return step_reward * num_steps
    return step_reward * (1.0 - discount**num_steps) / (1.0 - discount)


def _build_transitions(
    episode_indices: list[int],
    *,
    action_chunk_size: int,
    transition_stride: int,
    discount: float,
    value_target_mode: str,
    terminal_reward: float,
    step_reward: float,
    success_value: bool,
) -> list[GrootOfflineTransition]:
    transitions: list[GrootOfflineTransition] = []
    num_frames = len(episode_indices)
    start = 0

    while start < num_frames:
        episode_index = episode_indices[start]
        end = start
        while end < num_frames and episode_indices[end] == episode_index:
            end += 1

        # A transition needs at least one future observation.
        for frame_index in range(start, max(start, end - 1), transition_stride):
            remaining_steps = end - 1 - frame_index
            if remaining_steps <= 0:
                continue

            executed_steps = min(action_chunk_size, remaining_steps)
            next_index = frame_index + executed_steps
            done = next_index >= end - 1

            if value_target_mode == "monte_carlo":
                reward = _discounted_step_sum(step_reward, remaining_steps, discount)
                reward += terminal_reward * (discount ** (remaining_steps - 1))
                bootstrap_discount = 0.0
            else:
                reward = _discounted_step_sum(step_reward, executed_steps, discount)
                if done:
                    reward += terminal_reward * (discount ** (executed_steps - 1))
                bootstrap_discount = 0.0 if done else discount**executed_steps

            transitions.append(
                GrootOfflineTransition(
                    start_index=frame_index,
                    next_index=next_index,
                    executed_steps=executed_steps,
                    reward=float(reward),
                    bootstrap_discount=float(bootstrap_discount),
                    done=done,
                    success=bool(success_value),
                    episode_index=int(episode_index),
                )
            )

        start = end

    return transitions


def _load_state_file(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GrootOfflineDatasetReplayBuffer:
    def __init__(
        self,
        dataset: Any,
        *,
        action_chunk_size: int,
        transition_stride: int,
        discount: float,
        value_target_mode: str,
        terminal_reward: float,
        step_reward: float,
        success_value: bool,
        action_pad_mode: str = "repeat_last",
    ) -> None:
        if action_chunk_size <= 0:
            raise ValueError(f"`action_chunk_size` must be > 0, got {action_chunk_size}.")
        if transition_stride <= 0:
            raise ValueError(f"`transition_stride` must be > 0, got {transition_stride}.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f"`discount` must be within [0, 1], got {discount}.")
        if value_target_mode not in {"n_step", "monte_carlo"}:
            raise ValueError(
                "`value_target_mode` must be one of {'n_step', 'monte_carlo'}, "
                f"got {value_target_mode!r}."
            )
        if action_pad_mode != "repeat_last":
            raise ValueError(
                "Only `action_pad_mode='repeat_last'` is currently supported, "
                f"got {action_pad_mode!r}."
            )
        if not hasattr(dataset, "hf_dataset"):
            raise TypeError("Dataset-backed replay expects a dataset exposing `hf_dataset`.")

        episode_indices = [int(value) for value in dataset.hf_dataset["episode_index"]]
        if len(episode_indices) < 2:
            raise ValueError("Dataset-backed replay requires at least two frames.")

        self.dataset = dataset
        self.action_chunk_size = action_chunk_size
        self.transition_stride = transition_stride
        self.discount = discount
        self.value_target_mode = value_target_mode
        self.terminal_reward = terminal_reward
        self.step_reward = step_reward
        self.success_value = success_value
        self.action_pad_mode = action_pad_mode
        self._actions = [_as_action_tensor(action) for action in dataset.hf_dataset["action"]]
        self._transitions = _build_transitions(
            episode_indices,
            action_chunk_size=action_chunk_size,
            transition_stride=transition_stride,
            discount=discount,
            value_target_mode=value_target_mode,
            terminal_reward=terminal_reward,
            step_reward=step_reward,
            success_value=success_value,
        )
        self.capacity = len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Any) -> None:
        del transition
        raise NotImplementedError("Dataset-backed replay does not support appending transitions.")

    def extend(self, transitions: Any) -> None:
        del transitions
        raise NotImplementedError("Dataset-backed replay does not support appending transitions.")

    def _build_action_chunk(self, transition: GrootOfflineTransition) -> Tensor:
        actions = self._actions[
            transition.start_index : transition.start_index + transition.executed_steps
        ]
        if not actions:
            raise RuntimeError("Offline replay transition has no actions to build a chunk from.")

        chunk = torch.stack(actions, dim=0)
        if chunk.shape[0] < self.action_chunk_size:
            pad = chunk[-1:].repeat(self.action_chunk_size - chunk.shape[0], 1)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk

    def _get_dataset_item(self, index: int) -> RawObservation:
        item = self.dataset[index]
        if not isinstance(item, Mapping):
            raise TypeError(f"Expected dataset item to be a mapping, got {type(item)}.")
        return dict(item)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        observation_transform: ObservationTransform | None = None,
        next_observation_transform: ObservationTransform | None = None,
    ) -> GrootHybridBatch:
        if not self._transitions:
            raise RuntimeError("Cannot sample from an empty dataset-backed replay source.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")

        actual_batch_size = min(batch_size, len(self._transitions))
        batch_indices = random.sample(range(len(self._transitions)), k=actual_batch_size)
        transitions = [self._transitions[index] for index in batch_indices]

        observations = [self._get_dataset_item(transition.start_index) for transition in transitions]
        next_observations = [self._get_dataset_item(transition.next_index) for transition in transitions]

        observation = (
            observation_transform(observations) if observation_transform is not None else observations
        )
        next_observation = (
            next_observation_transform(next_observations)
            if next_observation_transform is not None
            else next_observations
        )

        return GrootHybridBatch(
            observation=observation,
            action_chunk=torch.stack(
                [self._build_action_chunk(transition) for transition in transitions],
                dim=0,
            ).to(device),
            reward=torch.tensor(
                [transition.reward for transition in transitions],
                dtype=torch.float32,
                device=device,
            ),
            next_observation=next_observation,
            done=torch.tensor(
                [transition.done for transition in transitions],
                dtype=torch.bool,
                device=device,
            ),
            success=torch.tensor(
                [transition.success for transition in transitions],
                dtype=torch.bool,
                device=device,
            ),
            bootstrap_discount=torch.tensor(
                [transition.bootstrap_discount for transition in transitions],
                dtype=torch.float32,
                device=device,
            ),
            metadata=[
                {
                    "episode_index": transition.episode_index,
                    "dataset_index": transition.start_index,
                    "next_dataset_index": transition.next_index,
                    "executed_steps": transition.executed_steps,
                    "value_target_mode": self.value_target_mode,
                }
                for transition in transitions
            ],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "action_chunk_size": self.action_chunk_size,
            "transition_stride": self.transition_stride,
            "discount": self.discount,
            "value_target_mode": self.value_target_mode,
            "terminal_reward": self.terminal_reward,
            "step_reward": self.step_reward,
            "success_value": self.success_value,
            "action_pad_mode": self.action_pad_mode,
            "transitions": [asdict(transition) for transition in self._transitions],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        transitions = state_dict.get("transitions")
        if transitions is None:
            raise KeyError("Dataset-backed replay state must include `transitions`.")

        self.capacity = int(state_dict.get("capacity", len(transitions)))
        self.action_chunk_size = int(state_dict.get("action_chunk_size", self.action_chunk_size))
        self.transition_stride = int(state_dict.get("transition_stride", self.transition_stride))
        self.discount = float(state_dict.get("discount", self.discount))
        self.value_target_mode = str(state_dict.get("value_target_mode", self.value_target_mode))
        self.terminal_reward = float(state_dict.get("terminal_reward", self.terminal_reward))
        self.step_reward = float(state_dict.get("step_reward", self.step_reward))
        self.success_value = bool(state_dict.get("success_value", self.success_value))
        self.action_pad_mode = str(state_dict.get("action_pad_mode", self.action_pad_mode))
        self._transitions = [GrootOfflineTransition(**transition) for transition in transitions]

    def save(self, path: str | Path) -> Path:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path, dataset: Any) -> "GrootOfflineDatasetReplayBuffer":
        state = _load_state_file(Path(path))
        replay = cls(
            dataset=dataset,
            action_chunk_size=int(state["action_chunk_size"]),
            transition_stride=int(state["transition_stride"]),
            discount=float(state["discount"]),
            value_target_mode=str(state["value_target_mode"]),
            terminal_reward=float(state["terminal_reward"]),
            step_reward=float(state["step_reward"]),
            success_value=bool(state["success_value"]),
            action_pad_mode=str(state.get("action_pad_mode", "repeat_last")),
        )
        replay.load_state_dict(state)
        return replay
