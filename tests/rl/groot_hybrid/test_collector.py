#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lerobot.rl.groot_hybrid import GrootHybridCollector


@dataclass
class FakeStep:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FakeRolloutAdapter:
    num_envs = 1

    def __init__(self, steps: list[FakeStep], *, task: str = "pick block") -> None:
        self.steps = steps
        self.task = task
        self.reset_calls = 0
        self.step_index = 0
        self.received_actions: list[np.ndarray] = []

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        del seed
        self.reset_calls += 1
        self.step_index = 0
        return (
            {"agent_pos": np.asarray([0.0, float(self.reset_calls)], dtype=np.float32)},
            {"reset_calls": self.reset_calls},
        )

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self.received_actions.append(action.copy())
        scripted_step = self.steps[self.step_index]
        self.step_index += 1
        return (
            scripted_step.observation,
            scripted_step.reward,
            scripted_step.terminated,
            scripted_step.truncated,
            scripted_step.info,
        )

    def get_task(self) -> str:
        return self.task


class MockChunkPolicy:
    def __init__(self, action_chunk: torch.Tensor) -> None:
        self._action_chunk = action_chunk
        self.training = True
        self.reset_calls = 0
        self.last_batch: dict | None = None

    def eval(self) -> MockChunkPolicy:
        self.training = False
        return self

    def train(self, mode: bool = True) -> MockChunkPolicy:
        self.training = mode
        return self

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.last_batch = batch
        return self._action_chunk.clone()


def test_collector_collects_reward_done_success_and_bootstrap_discount() -> None:
    rollout = FakeRolloutAdapter(
        steps=[
            FakeStep(
                observation={"agent_pos": np.asarray([1.0, 1.0], dtype=np.float32)},
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
            ),
            FakeStep(
                observation={"agent_pos": np.asarray([2.0, 2.0], dtype=np.float32)},
                reward=2.0,
                terminated=True,
                truncated=False,
                info={"final_info": {"is_success": True}},
            ),
        ]
    )
    policy = MockChunkPolicy(
        action_chunk=torch.tensor(
            [[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]],
            dtype=torch.float32,
        )
    )
    collector = GrootHybridCollector(rollout, policy, discount=0.5)

    transitions = collector.collect(1)

    assert len(transitions) == 1
    transition = transitions[0]
    assert torch.isclose(transition.reward, torch.tensor(2.0))
    assert bool(transition.done.item()) is True
    assert bool(transition.success.item()) is True
    assert torch.isclose(transition.bootstrap_discount, torch.tensor(0.0))
    assert transition.action_chunk.shape == (3, 2)
    assert transition.metadata["executed_steps"] == 2
    assert transition.metadata["terminated"] is True
    assert transition.metadata["truncated"] is False
    assert transition.metadata["task"] == "pick block"
    assert len(rollout.received_actions) == 2
    assert policy.last_batch is not None
    assert "observation.state" in policy.last_batch
    assert policy.last_batch["task"] == ["pick block"]
    assert policy.reset_calls == 2
    assert policy.training is True


def test_collector_respects_max_steps_per_chunk_and_top_level_success() -> None:
    rollout = FakeRolloutAdapter(
        steps=[
            FakeStep(
                observation={"agent_pos": np.asarray([1.0, 0.0], dtype=np.float32)},
                reward=1.0,
                terminated=False,
                truncated=False,
                info={"success": False},
            ),
            FakeStep(
                observation={"agent_pos": np.asarray([2.0, 0.0], dtype=np.float32)},
                reward=3.0,
                terminated=False,
                truncated=False,
                info={"success": True},
            ),
            FakeStep(
                observation={"agent_pos": np.asarray([3.0, 0.0], dtype=np.float32)},
                reward=100.0,
                terminated=False,
                truncated=False,
                info={"success": True},
            ),
        ]
    )
    policy = MockChunkPolicy(
        action_chunk=torch.tensor(
            [[[0.0], [1.0], [2.0], [3.0]]],
            dtype=torch.float32,
        )
    )
    collector = GrootHybridCollector(rollout, policy, discount=0.9, max_steps_per_chunk=2)

    transition = collector.collect(1)[0]

    assert torch.isclose(transition.reward, torch.tensor(3.7))
    assert bool(transition.done.item()) is False
    assert bool(transition.success.item()) is True
    assert torch.isclose(transition.bootstrap_discount, torch.tensor(0.81))
    assert transition.metadata["executed_steps"] == 2
    assert len(rollout.received_actions) == 2
    assert rollout.reset_calls == 1


def test_collector_rejects_max_steps_per_chunk_above_16() -> None:
    rollout = FakeRolloutAdapter(steps=[])
    policy = MockChunkPolicy(action_chunk=torch.zeros(1, 1, 1))

    with pytest.raises(ValueError, match="max_steps_per_chunk <= 16"):
        GrootHybridCollector(rollout, policy, discount=0.99, max_steps_per_chunk=17)
