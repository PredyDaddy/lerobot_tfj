#!/usr/bin/env python

from __future__ import annotations

import random

import numpy as np
import torch

from lerobot.rl.groot_hybrid import GrootHybridReplayBuffer, GrootHybridTransition


def _make_transition(index: int) -> GrootHybridTransition:
    observation = {
        "agent_pos": np.asarray([index, index + 0.5], dtype=np.float32),
        "pixels": {
            "front": np.full((2, 2, 3), fill_value=index, dtype=np.uint8),
        },
    }
    next_observation = {
        "agent_pos": np.asarray([index + 1, index + 1.5], dtype=np.float32),
        "pixels": {
            "front": np.full((2, 2, 3), fill_value=index + 1, dtype=np.uint8),
        },
    }
    return GrootHybridTransition(
        observation=observation,
        action_chunk=torch.full((2, 3), fill_value=float(index), dtype=torch.float32),
        reward=torch.tensor(float(index + 1), dtype=torch.float32),
        next_observation=next_observation,
        done=torch.tensor(index % 2 == 0, dtype=torch.bool),
        success=torch.tensor(index == 1, dtype=torch.bool),
        bootstrap_discount=torch.tensor(0.25 * (index + 1), dtype=torch.float32),
        metadata={"episode": index, "executed_steps": 2},
    )


def test_replay_buffer_sample_supports_raw_preprocess() -> None:
    random.seed(0)
    replay_buffer = GrootHybridReplayBuffer(capacity=4)
    replay_buffer.extend([_make_transition(0), _make_transition(1)])

    def observation_transform(observations):
        return {
            "state": torch.tensor(
                np.stack([observation["agent_pos"] for observation in observations], axis=0),
                dtype=torch.float32,
            )
        }

    batch = replay_buffer.sample(
        batch_size=2,
        device="cpu",
        observation_transform=observation_transform,
        next_observation_transform=observation_transform,
    )

    assert batch.observation["state"].shape == (2, 2)
    assert batch.next_observation["state"].shape == (2, 2)
    assert batch.action_chunk.shape == (2, 2, 3)
    assert batch.reward.shape == (2,)
    assert batch.done.dtype == torch.bool
    assert batch.success.dtype == torch.bool
    assert batch.bootstrap_discount.shape == (2,)
    assert {metadata["episode"] for metadata in batch.metadata} == {0, 1}


def test_replay_buffer_save_and_load_roundtrip(tmp_path) -> None:
    replay_buffer = GrootHybridReplayBuffer(capacity=3)
    original_transitions = [_make_transition(0), _make_transition(1)]
    replay_buffer.extend(original_transitions)

    save_path = tmp_path / "groot_hybrid_buffer.pt"
    replay_buffer.save(save_path)
    loaded_buffer = GrootHybridReplayBuffer.load(save_path)

    assert loaded_buffer.capacity == 3
    assert len(loaded_buffer) == 2

    restored_transitions = loaded_buffer.as_list()
    assert len(restored_transitions) == 2
    assert np.array_equal(
        restored_transitions[0].observation["agent_pos"],
        original_transitions[0].observation["agent_pos"],
    )
    assert torch.equal(restored_transitions[1].action_chunk, original_transitions[1].action_chunk)
    assert torch.equal(restored_transitions[1].reward, original_transitions[1].reward)
    assert torch.equal(restored_transitions[1].done, original_transitions[1].done)
    assert torch.equal(restored_transitions[1].success, original_transitions[1].success)
    assert torch.equal(
        restored_transitions[1].bootstrap_discount,
        original_transitions[1].bootstrap_discount,
    )
    assert restored_transitions[1].metadata == original_transitions[1].metadata
