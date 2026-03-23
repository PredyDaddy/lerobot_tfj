#!/usr/bin/env python

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import datasets
import torch

from lerobot.rl.groot_hybrid.offline_replay import GrootOfflineDatasetReplayBuffer
from lerobot.rl.groot_hybrid.trainer import train


class _FakeHybridPolicy(torch.nn.Module):
    def __init__(self, *, action_dim: int = 2, chunk_size: int = 2) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, action_dim)
        self.value_head = torch.nn.Linear(2, 1)
        self.chunk_size = chunk_size

    def reset(self) -> None:
        return None

    def _state_tensor(self, batch_or_observation) -> torch.Tensor:
        state = batch_or_observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        return state.float()

    def _predict_chunk(self, batch_or_observation, chunk_size: int | None = None) -> torch.Tensor:
        state = self._state_tensor(batch_or_observation)
        base_action = self.actor(state)
        return base_action.unsqueeze(1).repeat(1, chunk_size or self.chunk_size, 1)

    def forward(self, batch):
        predicted_chunk = self._predict_chunk(batch, chunk_size=batch["action_chunk"].shape[1])
        loss = torch.nn.functional.mse_loss(predicted_chunk, batch["action_chunk"])
        return loss, {"offline_recon_loss": loss.detach().item()}

    def forward_action_chunk(self, batch=None, *, hybrid_context=None, action_chunk=None, **kwargs):
        del hybrid_context, kwargs
        if batch is None:
            raise ValueError("_FakeHybridPolicy.forward_action_chunk requires `batch`.")
        target_actions = action_chunk if action_chunk is not None else batch["action"]
        predicted_chunk = self._predict_chunk(batch, chunk_size=target_actions.shape[1])
        loss = torch.nn.functional.mse_loss(predicted_chunk, target_actions)
        return {"loss": loss, "online_recon_loss": loss.detach()}

    def predict_action_chunk(self, batch):
        return self._predict_chunk(batch)

    def predict_value(self, observation):
        state = self._state_tensor(observation)
        return self.value_head(state).squeeze(-1)


def _make_offline_batch(value: float) -> dict[str, torch.Tensor]:
    observation = torch.tensor([[value, value + 0.5], [value + 1.0, value + 1.5]], dtype=torch.float32)
    action_chunk = torch.stack(
        [
            observation * 0.25,
            observation * -0.5,
        ],
        dim=1,
    )
    return {
        "observation.state": observation,
        "action_chunk": action_chunk,
    }


class _FakeDataset:
    def __init__(self) -> None:
        self.items = [
            {
                "observation.state": torch.tensor([0.0, 0.5], dtype=torch.float32),
                "observation.images.top": torch.zeros((3, 2, 2), dtype=torch.uint8),
                "observation.images.wrist": torch.zeros((3, 2, 2), dtype=torch.uint8),
                "task": "stack blocks",
            },
            {
                "observation.state": torch.tensor([1.0, 1.5], dtype=torch.float32),
                "observation.images.top": torch.ones((3, 2, 2), dtype=torch.uint8),
                "observation.images.wrist": torch.ones((3, 2, 2), dtype=torch.uint8),
                "task": "stack blocks",
            },
            {
                "observation.state": torch.tensor([2.0, 2.5], dtype=torch.float32),
                "observation.images.top": torch.full((3, 2, 2), 2, dtype=torch.uint8),
                "observation.images.wrist": torch.full((3, 2, 2), 2, dtype=torch.uint8),
                "task": "stack blocks",
            },
            {
                "observation.state": torch.tensor([10.0, 10.5], dtype=torch.float32),
                "observation.images.top": torch.full((3, 2, 2), 3, dtype=torch.uint8),
                "observation.images.wrist": torch.full((3, 2, 2), 3, dtype=torch.uint8),
                "task": "stack blocks",
            },
            {
                "observation.state": torch.tensor([11.0, 11.5], dtype=torch.float32),
                "observation.images.top": torch.full((3, 2, 2), 4, dtype=torch.uint8),
                "observation.images.wrist": torch.full((3, 2, 2), 4, dtype=torch.uint8),
                "task": "stack blocks",
            },
            {
                "observation.state": torch.tensor([12.0, 12.5], dtype=torch.float32),
                "observation.images.top": torch.full((3, 2, 2), 5, dtype=torch.uint8),
                "observation.images.wrist": torch.full((3, 2, 2), 5, dtype=torch.uint8),
                "task": "stack blocks",
            },
        ]
        self.hf_dataset = datasets.Dataset.from_dict(
            {
                "episode_index": [0, 0, 0, 1, 1, 1],
                "action": [
                    [0.0, 0.1],
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                    [4.0, 4.1],
                    [5.0, 5.1],
                ],
            }
        ).with_format("torch")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {key: value for key, value in item.items()}


def _observation_transform(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.stack(
            [item["observation.state"] for item in items],
            dim=0,
        )
    }


def test_offline_dataset_replay_buffer_builds_mc_targets_and_fixed_chunks() -> None:
    random.seed(0)
    dataset = _FakeDataset()
    replay_buffer = GrootOfflineDatasetReplayBuffer(
        dataset=dataset,
        action_chunk_size=4,
        transition_stride=1,
        discount=0.5,
        value_target_mode="monte_carlo",
        terminal_reward=1.0,
        step_reward=0.0,
        success_value=True,
    )

    assert len(replay_buffer) == 4

    state = replay_buffer.state_dict()
    assert state["transitions"][0]["reward"] == 0.5
    assert state["transitions"][0]["bootstrap_discount"] == 0.0
    assert state["transitions"][1]["reward"] == 1.0

    batch = replay_buffer.sample(
        batch_size=2,
        device="cpu",
        observation_transform=_observation_transform,
        next_observation_transform=_observation_transform,
    )

    assert batch.observation["observation.state"].shape == (2, 2)
    assert batch.next_observation["observation.state"].shape == (2, 2)
    assert batch.action_chunk.shape == (2, 4, 2)
    assert batch.done.dtype == torch.bool
    assert batch.success.dtype == torch.bool
    assert all(metadata["value_target_mode"] == "monte_carlo" for metadata in batch.metadata)


def test_train_entrypoint_supports_dataset_only_offline_rl_stage() -> None:
    torch.manual_seed(0)
    dataset = _FakeDataset()
    policy = _FakeHybridPolicy(action_dim=2, chunk_size=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    def preprocessor(batch):
        state = batch["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return {"observation.state": state.float()}

    cfg = SimpleNamespace(
        steps=1,
        seed=123,
        output_dir=None,
        env=None,
        offline_replay=SimpleNamespace(
            enabled=True,
            transition_stride=1,
            value_target_mode="monte_carlo",
            terminal_reward=1.0,
            step_reward=0.0,
            success_value=True,
            action_pad_mode="repeat_last",
        ),
        collector=SimpleNamespace(chunks_per_step=0, warmup_chunks=0),
        replay_buffer=SimpleNamespace(capacity=4, online_batch_size=2),
        losses=SimpleNamespace(
            discount=0.9,
            offline_loss_weight=1.0,
            online_flow_loss_weight=0.4,
            value_loss_weight=0.6,
            use_advantage_weighting=True,
            normalize_advantage=True,
            advantage_temperature=1.0,
            advantage_clip_min=-5.0,
            advantage_clip_max=5.0,
            max_advantage_weight=20.0,
        ),
        optimizer=SimpleNamespace(grad_clip_norm=1.0),
        _train_components={
            "dataset": dataset,
            "policy": policy,
            "optimizer": optimizer,
            "offline_data": itertools.cycle([_make_offline_batch(0.0), _make_offline_batch(1.0)]),
            "preprocessor": preprocessor,
            "postprocessor": lambda action_chunk: action_chunk,
            "num_steps": 1,
        },
    )

    trainer = train(cfg)

    assert isinstance(trainer.replay_buffer, GrootOfflineDatasetReplayBuffer)
    assert trainer.global_step == 1
    assert trainer.last_metrics["buffer_size"] == float(len(trainer.replay_buffer))
    assert trainer.last_metrics["collected_chunks"] == 0.0
    assert trainer.last_metrics["warmup_chunks"] == 0.0
