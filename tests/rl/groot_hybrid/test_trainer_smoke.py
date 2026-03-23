#!/usr/bin/env python

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lerobot.rl.groot_hybrid.trainer as trainer_module
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.groot_hybrid import GrootHybridCollector, GrootHybridReplayBuffer
from lerobot.rl.groot_hybrid.trainer import GrootHybridTrainer, train


@dataclass
class FakeStep:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FakeRolloutAdapter:
    num_envs = 1

    def __init__(self, steps: list[FakeStep], *, task: str = "stack blocks") -> None:
        self.steps = steps
        self.task = task
        self.reset_calls = 0
        self.step_index = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        del seed
        self.reset_calls += 1
        self.step_index = 0
        return (
            {"agent_pos": np.asarray([0.0, float(self.reset_calls)], dtype=np.float32)},
            {},
        )

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        del action
        scripted_step = self.steps[self.step_index % len(self.steps)]
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


class FakeHybridPolicy(torch.nn.Module):
    def __init__(self, *, action_dim: int = 2, chunk_size: int = 2) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, action_dim)
        self.value_head = torch.nn.Linear(2, 1)
        self.chunk_size = chunk_size
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

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
            raise ValueError("FakeHybridPolicy.forward_action_chunk requires `batch`.")
        target_actions = action_chunk if action_chunk is not None else batch["action"]
        predicted_chunk = self._predict_chunk(batch, chunk_size=target_actions.shape[1])
        loss = torch.nn.functional.mse_loss(predicted_chunk, target_actions)
        return {"loss": loss, "online_recon_loss": loss.detach()}

    def predict_action_chunk(self, batch):
        return self._predict_chunk(batch)

    def predict_value(self, observation):
        state = self._state_tensor(observation)
        return self.value_head(state).squeeze(-1)


def _make_cfg(*, online_flow_loss_weight: float, value_loss_weight: float) -> SimpleNamespace:
    return SimpleNamespace(
        steps=2,
        collector=SimpleNamespace(chunks_per_step=1, warmup_chunks=1),
        replay_buffer=SimpleNamespace(online_batch_size=2),
        losses=SimpleNamespace(
            offline_loss_weight=1.0,
            online_flow_loss_weight=online_flow_loss_weight,
            value_loss_weight=value_loss_weight,
            use_advantage_weighting=True,
            normalize_advantage=True,
            advantage_temperature=1.0,
            advantage_clip_min=-5.0,
            advantage_clip_max=5.0,
            max_advantage_weight=20.0,
        ),
        optimizer=SimpleNamespace(grad_clip_norm=1.0),
    )


def _make_observation_transform(observations):
    return {
        "observation.state": torch.tensor(
            np.stack([observation["agent_pos"] for observation in observations], axis=0),
            dtype=torch.float32,
        )
    }


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


def _make_offline_sample(value: float) -> dict[str, torch.Tensor]:
    observation = torch.tensor([value, value + 0.5], dtype=torch.float32)
    action_chunk = torch.stack(
        [
            observation * 0.25,
            observation * -0.5,
        ],
        dim=0,
    )
    return {
        "observation.state": observation,
        "action_chunk": action_chunk,
    }


def _make_rollout() -> FakeRolloutAdapter:
    return FakeRolloutAdapter(
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
                reward=0.5,
                terminated=True,
                truncated=False,
                info={"final_info": {"is_success": True}},
            ),
        ]
    )


def test_groot_hybrid_trainer_smoke_supports_collect_train_save_and_resume(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = _make_cfg(online_flow_loss_weight=0.4, value_loss_weight=0.6)
    policy = FakeHybridPolicy()
    collector = GrootHybridCollector(_make_rollout(), policy, discount=0.9)
    replay_buffer = GrootHybridReplayBuffer(capacity=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    offline_data = itertools.cycle([_make_offline_batch(0.0), _make_offline_batch(1.0)])

    trainer = GrootHybridTrainer(
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        collector=collector,
        replay_buffer=replay_buffer,
        offline_data=offline_data,
        device="cpu",
        online_observation_transform=_make_observation_transform,
        online_next_observation_transform=_make_observation_transform,
    )

    initial_state = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    step_metrics = trainer.step()

    assert trainer.global_step == 1
    assert len(replay_buffer) == 2
    assert step_metrics["warmup_chunks"] == 1.0
    assert step_metrics["collected_chunks"] == 1.0
    assert step_metrics["online_policy_loss"] >= 0.0
    assert step_metrics["value_loss_total"] >= 0.0
    assert "done_ratio" in step_metrics
    assert any(
        not torch.allclose(policy.state_dict()[name], initial_state[name])
        for name in initial_state
    )
    assert optimizer.state_dict()["state"]

    checkpoint_path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    assert checkpoint_path.is_file()

    resumed_policy = FakeHybridPolicy()
    resumed_collector = GrootHybridCollector(_make_rollout(), resumed_policy, discount=0.9)
    resumed_buffer = GrootHybridReplayBuffer(capacity=1)
    resumed_optimizer = torch.optim.Adam(resumed_policy.parameters(), lr=0.05)
    resumed_trainer = GrootHybridTrainer(
        cfg=cfg,
        policy=resumed_policy,
        optimizer=resumed_optimizer,
        collector=resumed_collector,
        replay_buffer=resumed_buffer,
        offline_data=itertools.cycle([_make_offline_batch(0.0), _make_offline_batch(1.0)]),
        device="cpu",
        online_observation_transform=_make_observation_transform,
        online_next_observation_transform=_make_observation_transform,
    )

    resumed_trainer.load_checkpoint(checkpoint_path)

    assert resumed_trainer.global_step == 1
    assert len(resumed_buffer) == 2
    assert resumed_buffer.capacity == 8
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(resumed_policy.state_dict()[name], tensor)
    assert resumed_optimizer.state_dict()["state"]

    resumed_metrics = resumed_trainer.step()

    assert resumed_trainer.global_step == 2
    assert len(resumed_buffer) == 3
    assert resumed_metrics["warmup_chunks"] == 0.0


def test_groot_hybrid_trainer_skips_online_and_value_losses_when_weights_are_zero() -> None:
    torch.manual_seed(0)
    cfg = _make_cfg(online_flow_loss_weight=0.0, value_loss_weight=0.0)
    policy = FakeHybridPolicy()
    trainer = GrootHybridTrainer(
        cfg=cfg,
        policy=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.01),
        collector=GrootHybridCollector(_make_rollout(), policy, discount=0.9),
        replay_buffer=GrootHybridReplayBuffer(capacity=4),
        offline_data=itertools.cycle([_make_offline_batch(0.0)]),
        device="cpu",
        online_observation_transform=_make_observation_transform,
        online_next_observation_transform=_make_observation_transform,
    )

    metrics = trainer.step()

    assert trainer.global_step == 1
    assert metrics["online_policy_loss"] == 0.0
    assert metrics["value_loss_total"] == 0.0
    assert len(trainer.replay_buffer) == 2


def test_groot_hybrid_train_entrypoint_smoke_wires_validation_and_minimal_components(tmp_path) -> None:
    torch.manual_seed(0)

    class ValidatingConfig(SimpleNamespace):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.validated = False

        def validate(self) -> None:
            self.validated = True

    policy = FakeHybridPolicy()
    replay_buffer = GrootHybridReplayBuffer(capacity=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    collector = GrootHybridCollector(_make_rollout(), policy, discount=0.9)

    cfg = ValidatingConfig(
        steps=1,
        seed=123,
        batch_size=2,
        num_workers=0,
        output_dir=tmp_path / "train-output",
        collector=SimpleNamespace(chunks_per_step=1, warmup_chunks=1),
        replay_buffer=SimpleNamespace(capacity=4, online_batch_size=2),
        losses=SimpleNamespace(
            discount=0.9,
            offline_loss_weight=1.0,
            online_flow_loss_weight=0.0,
            value_loss_weight=0.0,
        ),
        optimizer=SimpleNamespace(grad_clip_norm=1.0),
        _train_components={
            "dataset": [_make_offline_sample(0.0), _make_offline_sample(1.0)],
            "policy": policy,
            "optimizer": optimizer,
            "collector": collector,
            "replay_buffer": replay_buffer,
            "preprocessor": lambda batch: batch,
            "num_steps": 1,
        },
    )

    trainer = train(cfg)

    assert cfg.validated is True
    assert cfg.output_dir.is_dir()
    assert trainer.global_step == 1
    assert trainer.last_metrics["warmup_chunks"] == 1.0
    assert trainer.last_metrics["collected_chunks"] == 1.0
    assert len(replay_buffer) == 2
    assert optimizer.state_dict()["state"]
    assert policy.reset_calls >= 1


def test_robot_rollout_adapter_reset_and_step_follow_processed_transition_contract() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.last_seed = None

        def reset(self, *, seed=None):
            self.last_seed = seed
            return {"agent_pos": np.asarray([0.5, 1.5], dtype=np.float32)}, {"from_env": True}

    class FakeProcessor:
        def __init__(self, fn):
            self.fn = fn
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def __call__(self, transition):
            return self.fn(transition)

    def env_processor_fn(transition):
        raw_obs = transition[TransitionKey.OBSERVATION]
        assert isinstance(raw_obs, dict)
        return create_transition(
            observation={
                "observation.state": torch.tensor(
                    [[raw_obs["agent_pos"][0], raw_obs["agent_pos"][1]]], dtype=torch.float32
                )
            },
            info=transition[TransitionKey.INFO],
        )

    def step_transition_fn(*, env, transition, action, env_processor, action_processor):
        del env, env_processor, action_processor
        assert isinstance(action, torch.Tensor)
        assert tuple(action.shape) == (1, 2)
        assert "observation.state" in transition[TransitionKey.OBSERVATION]
        return create_transition(
            observation={"observation.state": torch.tensor([[3.0, 4.0]], dtype=torch.float32)},
            reward=torch.tensor(1.25, dtype=torch.float32),
            done=torch.tensor(True),
            truncated=torch.tensor(False),
            info={"success": True},
        )

    env = FakeEnv()
    env_processor = FakeProcessor(env_processor_fn)
    action_processor = FakeProcessor(lambda transition: transition)
    rollout = trainer_module._RobotTransitionRolloutAdapter(
        env,
        env_processor=env_processor,
        action_processor=action_processor,
        step_transition_fn=step_transition_fn,
        task="robot stack",
    )

    reset_observation, reset_info = rollout.reset(seed=123)
    assert env.last_seed == 123
    assert env_processor.reset_calls == 1
    assert action_processor.reset_calls == 1
    assert "observation.state" in reset_observation
    assert reset_info == {"from_env": True}

    next_observation, reward, terminated, truncated, info = rollout.step(
        np.asarray([[0.1, -0.2]], dtype=np.float32)
    )
    assert "observation.state" in next_observation
    assert reward == 1.25
    assert terminated is True
    assert truncated is False
    assert info == {"success": True}
    assert rollout.get_task() == "robot stack"


def test_train_entrypoint_uses_gym_manipulator_robot_rollout_path(tmp_path, monkeypatch) -> None:
    torch.manual_seed(0)

    class ValidatingConfig(SimpleNamespace):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.validated = False

        def validate(self) -> None:
            self.validated = True

    class FakeEnv:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self, *, seed=None):
            del seed
            self.reset_calls += 1
            return {"agent_pos": np.asarray([0.0, 1.0], dtype=np.float32)}, {}

    class FakeProcessor:
        def __init__(self, fn):
            self.fn = fn
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def __call__(self, transition):
            return self.fn(transition)

    fake_env = FakeEnv()

    def env_processor_fn(transition):
        raw_obs = transition[TransitionKey.OBSERVATION]
        return create_transition(
            observation={
                "observation.state": torch.tensor(
                    [[raw_obs["agent_pos"][0], raw_obs["agent_pos"][1]]], dtype=torch.float32
                )
            },
            info=transition[TransitionKey.INFO],
        )

    def step_transition_fn(*, env, transition, action, env_processor, action_processor):
        del env, env_processor, action_processor
        assert isinstance(action, torch.Tensor)
        assert "observation.state" in transition[TransitionKey.OBSERVATION]
        fake_env.step_calls += 1
        return create_transition(
            observation={
                "observation.state": torch.tensor(
                    [[float(fake_env.step_calls), float(fake_env.step_calls) + 0.5]],
                    dtype=torch.float32,
                )
            },
            reward=1.0,
            done=True,
            truncated=False,
            info={"is_success": True},
        )

    fake_rollout = trainer_module._RobotTransitionRolloutAdapter(
        fake_env,
        env_processor=FakeProcessor(env_processor_fn),
        action_processor=FakeProcessor(lambda transition: transition),
        step_transition_fn=step_transition_fn,
        task="robot task",
    )
    call_record: dict[str, str] = {}

    def fake_build_gym_manipulator_rollout(*, cfg, device):
        del cfg
        call_record["device"] = str(device)
        return fake_rollout, object(), object()

    monkeypatch.setattr(
        trainer_module,
        "_build_gym_manipulator_rollout",
        fake_build_gym_manipulator_rollout,
    )
    monkeypatch.setattr(
        trainer_module,
        "make_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("make_env must not be called for gym_manipulator branch")
        ),
    )

    policy = FakeHybridPolicy()
    replay_buffer = GrootHybridReplayBuffer(capacity=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    def preprocessor(batch):
        state = batch["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return {"observation.state": state.float()}

    cfg = ValidatingConfig(
        steps=1,
        seed=123,
        output_dir=tmp_path / "train-output-robot",
        env=SimpleNamespace(type="gym_manipulator", task="stack"),
        collector=SimpleNamespace(chunks_per_step=1, warmup_chunks=1),
        replay_buffer=SimpleNamespace(capacity=4, online_batch_size=2),
        losses=SimpleNamespace(
            discount=0.9,
            offline_loss_weight=1.0,
            online_flow_loss_weight=0.0,
            value_loss_weight=0.0,
        ),
        optimizer=SimpleNamespace(grad_clip_norm=1.0),
        _train_components={
            "policy": policy,
            "optimizer": optimizer,
            "replay_buffer": replay_buffer,
            "offline_data": itertools.cycle([_make_offline_batch(0.0), _make_offline_batch(1.0)]),
            "preprocessor": preprocessor,
            "postprocessor": lambda action_chunk: action_chunk,
            "num_steps": 1,
        },
    )

    trainer = train(cfg)

    assert cfg.validated is True
    assert call_record["device"] == "cpu"
    assert trainer.collector.rollout is fake_rollout
    assert fake_env.reset_calls >= 1
    assert len(replay_buffer) == 2

    built_observation = trainer.collector.observation_builder(
        {"observation.state": torch.tensor([[5.0, 6.0]], dtype=torch.float32)},
        "",
    )
    assert built_observation["task"] == ["robot task"]


def test_build_gym_manipulator_rollout_cleans_up_resources_when_make_processors_fails(monkeypatch) -> None:
    import lerobot.rl.gym_manipulator as gym_manipulator_module

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeTeleop:
        def __init__(self) -> None:
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    fake_env = FakeEnv()
    fake_teleop = FakeTeleop()
    expected_error = RuntimeError("processor-build-failed")

    monkeypatch.setattr(
        gym_manipulator_module,
        "make_robot_env",
        lambda cfg: (fake_env, fake_teleop),
    )

    def _raise_make_processors(*args, **kwargs):
        del args, kwargs
        raise expected_error

    monkeypatch.setattr(
        gym_manipulator_module,
        "make_processors",
        _raise_make_processors,
    )

    cfg = SimpleNamespace(env=SimpleNamespace(type="gym_manipulator", task="stack"))
    with pytest.raises(RuntimeError, match="processor-build-failed"):
        trainer_module._build_gym_manipulator_rollout(cfg=cfg, device=torch.device("cpu"))

    assert fake_env.closed is True
    assert fake_teleop.disconnected is True
