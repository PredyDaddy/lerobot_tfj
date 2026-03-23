#!/usr/bin/env python

import pytest
import torch

from lerobot.rl.groot_hybrid.types import (
    GROOT_CHUNK_TRANSITION_FIELDS,
    GrootChunkBatch,
    GrootChunkTransition,
)


def test_groot_chunk_transition_schema_is_frozen():
    assert tuple(GrootChunkTransition.__dataclass_fields__) == GROOT_CHUNK_TRANSITION_FIELDS


def test_groot_chunk_transition_accepts_bootstrap_discount_and_serializable_metadata():
    transition = GrootChunkTransition(
        observation={"observation.state": torch.zeros(4)},
        action_chunk=torch.zeros(2, 4),
        reward=torch.tensor(1.25),
        next_observation={"observation.state": torch.ones(4)},
        done=torch.tensor(False),
        success=torch.tensor(True),
        bootstrap_discount=torch.tensor(0.81),
        metadata={"executed_steps": 2, "terminated": False},
    )

    assert torch.equal(transition.bootstrap_discount, torch.tensor(0.81))
    assert transition.metadata == {"executed_steps": 2, "terminated": False}


def test_groot_chunk_batch_validates_each_metadata_entry():
    batch = GrootChunkBatch(
        observation={"observation.state": torch.zeros(2, 4)},
        action_chunk=torch.zeros(2, 3, 4),
        reward=torch.zeros(2),
        next_observation={"observation.state": torch.ones(2, 4)},
        done=torch.zeros(2, dtype=torch.bool),
        success=torch.ones(2, dtype=torch.bool),
        bootstrap_discount=torch.full((2,), 0.9),
        metadata=[{"episode_return": 3.5}, {"executed_steps": 3}],
    )

    assert batch.metadata == [{"episode_return": 3.5}, {"executed_steps": 3}]


def test_groot_chunk_metadata_rejects_non_serializable_values():
    with pytest.raises(TypeError, match="JSON-serializable"):
        GrootChunkTransition(
            observation={},
            action_chunk=torch.zeros(1, 1),
            reward=torch.tensor(0.0),
            next_observation={},
            done=torch.tensor(False),
            success=torch.tensor(False),
            bootstrap_discount=torch.tensor(1.0),
            metadata={"bad": torch.tensor(1.0)},
        )
