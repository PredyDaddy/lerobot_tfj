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
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.groot_hybrid.buffer import GrootHybridBatch, GrootHybridReplayBuffer
from lerobot.rl.groot_hybrid.collector import GrootHybridCollector, default_observation_builder
from lerobot.rl.groot_hybrid.losses import compute_offline_loss, compute_online_losses
from lerobot.rl.groot_hybrid.offline_replay import GrootOfflineDatasetReplayBuffer
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint as save_run_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.constants import ACTION


OfflineLossFn = Callable[[Any, Any], tuple[Tensor, dict[str, float]]]
OnlineLossFn = Callable[[Any, GrootHybridBatch, Any], tuple[Tensor, Tensor, dict[str, float]]]


class _IdentityProcessor:
    def __call__(self, value: Any) -> Any:
        return value


class _NullCollector:
    def reset(self, seed: int | None = None) -> None:
        del seed
        return None

    def collect(self, num_chunks: int) -> list[Any]:
        del num_chunks
        return []


class _SingleVectorEnvRolloutAdapter:
    def __init__(self, env: Any, *, task: str = "") -> None:
        self.env = env
        self._task = task

    @property
    def num_envs(self) -> int:
        return int(getattr(self.env, "num_envs", 1))

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        seeds = None if seed is None else [seed]
        return self.env.reset(seed=seeds)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        return self.env.step(action)

    def get_task(self) -> str:
        if self._task:
            return self._task

        if not hasattr(self.env, "call"):
            return ""

        for attr_name in ("task_description", "task"):
            try:
                values = self.env.call(attr_name)
            except Exception:
                continue

            if isinstance(values, tuple):
                values = list(values)
            if isinstance(values, list) and values and isinstance(values[0], str):
                return values[0]

        return ""


class _RobotTransitionRolloutAdapter:
    def __init__(
        self,
        env: Any,
        *,
        env_processor: Any,
        action_processor: Any,
        step_transition_fn: Callable[..., Any],
        task: str = "",
    ) -> None:
        self.env = env
        self._env_processor = env_processor
        self._action_processor = action_processor
        self._step_transition_fn = step_transition_fn
        self._task = task
        self._transition: dict[str, Any] | None = None

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_result: Any
        try:
            reset_result = self.env.reset(seed=seed)
        except TypeError:
            reset_result = self.env.reset()

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation, info = reset_result, {}

        if not isinstance(info, Mapping):
            info = {}

        if hasattr(self._env_processor, "reset"):
            self._env_processor.reset()
        if hasattr(self._action_processor, "reset"):
            self._action_processor.reset()

        transition = create_transition(observation=observation, info=dict(info))
        transition = self._env_processor(transition)
        self._transition = transition

        processed_observation = transition.get(TransitionKey.OBSERVATION)
        processed_info = transition.get(TransitionKey.INFO, {})
        if not isinstance(processed_observation, Mapping):
            raise TypeError(
                "Robot rollout adapter expected processed observation mapping after env processor."
            )
        if not isinstance(processed_info, Mapping):
            processed_info = {}
        return dict(processed_observation), dict(processed_info)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        if self._transition is None:
            raise RuntimeError("Robot rollout adapter must be reset before stepping.")

        action_tensor = torch.as_tensor(action)
        new_transition = self._step_transition_fn(
            env=self.env,
            transition=self._transition,
            action=action_tensor,
            env_processor=self._env_processor,
            action_processor=self._action_processor,
        )
        self._transition = new_transition

        processed_observation = new_transition.get(TransitionKey.OBSERVATION)
        info = new_transition.get(TransitionKey.INFO, {})
        if not isinstance(processed_observation, Mapping):
            raise TypeError(
                "Robot rollout adapter expected processed observation mapping after env step."
            )
        if not isinstance(info, Mapping):
            info = {}

        reward = _to_python_scalar(new_transition.get(TransitionKey.REWARD, 0.0))
        terminated = bool(_to_python_scalar(new_transition.get(TransitionKey.DONE, False)))
        truncated = bool(_to_python_scalar(new_transition.get(TransitionKey.TRUNCATED, False)))
        return dict(processed_observation), reward, terminated, truncated, dict(info)

    def get_task(self) -> str:
        return self._task


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    current = cfg
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue

        if not hasattr(current, part):
            return default
        current = getattr(current, part)

    return current


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(nested_value, device) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}.")
        return value.detach().to(device="cpu").item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape {value.shape}.")
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_task(task: Any) -> str:
    if task is None:
        return ""
    return str(task)


def _is_gym_manipulator_env_cfg(env_cfg: Any) -> bool:
    if env_cfg is None:
        return False
    if isinstance(env_cfg, HILSerlRobotEnvConfig):
        return True
    return str(getattr(env_cfg, "type", "")) == "gym_manipulator"


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random_state = state.get("python")
    if random_state is not None:
        random.setstate(random_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _load_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_train_components(cfg: Any) -> Mapping[str, Any]:
    # `_train_components` keeps smoke tests local to this module without forcing
    # changes in config/factory code paths.
    for attr_name in ("_train_components", "train_components"):
        components = getattr(cfg, attr_name, None)
        if components is None:
            continue
        if not isinstance(components, Mapping):
            raise TypeError(f"`{attr_name}` must be a mapping, got {type(components)}.")
        return components
    return {}


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_single_vector_env(envs: Mapping[str, Any]) -> Any:
    flattened = [env for suite_envs in envs.values() for env in suite_envs.values()]
    if len(flattened) != 1:
        raise NotImplementedError(
            "The initial Groot hybrid train entrypoint only supports a single suite/task environment."
        )
    return flattened[0]


def _collate_values(values: list[Any]) -> Any:
    if not values:
        return []

    first = values[0]

    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    if isinstance(first, Mapping):
        return {key: _collate_values([value[key] for value in values]) for key in first}
    if isinstance(first, tuple):
        return tuple(_collate_values([value[idx] for value in values]) for idx in range(len(first)))
    if isinstance(first, list) and all(len(value) == len(first) for value in values):
        return [_collate_values([value[idx] for value in values]) for idx in range(len(first))]
    if isinstance(first, np.generic | bool | int | float):
        return np.asarray(values)
    return list(values)


def _squeeze_singleton_env_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == 1:
            return value.squeeze(0)
        return value
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == 1:
            return value.squeeze(0)
        return value
    if isinstance(value, Mapping):
        return {key: _squeeze_singleton_env_dim(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_squeeze_singleton_env_dim(item) for item in value)
    if isinstance(value, list):
        if len(value) == 1:
            return _squeeze_singleton_env_dim(value[0])
        return [_squeeze_singleton_env_dim(item) for item in value]
    return value


def _make_dataloader(cfg: Any, dataset: Any, device: torch.device) -> torch.utils.data.DataLoader:
    if hasattr(dataset, "__len__") and len(dataset) == 0:
        raise ValueError("Offline dataset is empty.")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "batch_size", 1)),
        num_workers=int(_cfg_get(cfg, "num_workers", 0)),
        shuffle=not bool(_cfg_get(cfg, "dataset.streaming", False)),
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if int(_cfg_get(cfg, "num_workers", 0)) > 0 else None,
    )


def _make_single_observation_builder(
    *,
    env_preprocessor: Callable[[dict[str, Any]], Any],
    task: str,
) -> Callable[[dict[str, Any]], Any]:
    def build(raw_observation: dict[str, Any]) -> Any:
        observation = default_observation_builder(raw_observation, task)
        return env_preprocessor(observation)

    return build


def _make_collector_observation_builder(
    *,
    env_preprocessor: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any], str], Any]:
    def build(raw_observation: dict[str, Any], task: str) -> Any:
        observation = default_observation_builder(raw_observation, task)
        return env_preprocessor(observation)

    return build


def _make_robot_single_observation_builder(*, task: str) -> Callable[[dict[str, Any]], Any]:
    def build(processed_observation: dict[str, Any]) -> Any:
        if not isinstance(processed_observation, Mapping):
            raise TypeError(
                "Robot single observation builder expected processed observation mapping."
            )
        observation = dict(processed_observation)
        observation["task"] = [task]
        return observation

    return build


def _make_robot_collector_observation_builder(
    *,
    task: str,
) -> Callable[[dict[str, Any], str], Any]:
    def build(processed_observation: dict[str, Any], collector_task: str) -> Any:
        if not isinstance(processed_observation, Mapping):
            raise TypeError(
                "Robot collector observation builder expected processed observation mapping."
            )
        observation = dict(processed_observation)
        resolved_task = collector_task or task
        observation["task"] = [resolved_task]
        return observation

    return build


def _make_online_observation_transform(
    *,
    observation_builder: Callable[[dict[str, Any]], Any],
    preprocessor: Callable[[Any], Any],
) -> Callable[[list[dict[str, Any]]], Any]:
    def transform(raw_observations: list[dict[str, Any]]) -> Any:
        batched_observation = _collate_values(
            [_squeeze_singleton_env_dim(observation_builder(observation)) for observation in raw_observations]
        )
        return preprocessor(batched_observation)

    return transform


def _extract_dataset_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key == "task" or key.startswith("observation.")
    }


def _make_dataset_observation_transform(
    *,
    preprocessor: Callable[[Any], Any],
) -> Callable[[list[dict[str, Any]]], Any]:
    def transform(dataset_items: list[dict[str, Any]]) -> Any:
        raw_batch = _collate_values([_extract_dataset_observation(item) for item in dataset_items])
        return preprocessor(raw_batch)

    return transform


def _make_collector_postprocessor(
    *,
    policy_postprocessor: Callable[[Any], Any],
    env_postprocessor: Callable[[Any], Any],
) -> Callable[[Tensor], Tensor]:
    def postprocess(action_chunk: Tensor) -> Tensor:
        policy_action = policy_postprocessor(action_chunk)
        env_transition = env_postprocessor({ACTION: policy_action})
        if isinstance(env_transition, Mapping):
            env_action = env_transition.get(ACTION, policy_action)
        else:
            env_action = env_transition
        if not isinstance(env_action, torch.Tensor):
            env_action = torch.as_tensor(env_action)
        return env_action

    return postprocess


def _build_gym_manipulator_rollout(
    *,
    cfg: Any,
    device: torch.device,
) -> tuple[_RobotTransitionRolloutAdapter, Any, Any]:
    from lerobot.rl.gym_manipulator import (
        make_processors,
        make_robot_env,
        step_env_and_process_transition,
    )

    env_cfg = _cfg_get(cfg, "env", None)
    if env_cfg is None:
        raise ValueError("Expected `cfg.env` for gym_manipulator rollout.")

    online_env, teleop_device = make_robot_env(cfg=env_cfg)
    try:
        env_processor, action_processor = make_processors(
            online_env,
            teleop_device,
            env_cfg,
            str(device),
        )
    except Exception:
        _close_robot_runtime(online_env, teleop_device)
        raise

    rollout = _RobotTransitionRolloutAdapter(
        online_env,
        env_processor=env_processor,
        action_processor=action_processor,
        step_transition_fn=step_env_and_process_transition,
        task=_normalize_task(_cfg_get(cfg, "env.task", "")),
    )
    return rollout, online_env, teleop_device


def _normalize_pretrained_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _ensure_full_chunk_postprocessor(postprocessor: Any) -> Any:
    for step in getattr(postprocessor, "steps", []):
        if hasattr(step, "output_mode"):
            step.output_mode = "full_chunk"
    return postprocessor


def _infer_policy_device(policy: Any) -> torch.device:
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        return torch.device("cpu")

    first_parameter = next(iter(parameters()), None)
    if first_parameter is None:
        return torch.device("cpu")
    return first_parameter.device


def _load_trainer_checkpoint_if_requested(
    trainer: "GrootHybridTrainer",
    cfg: Any,
    components: Mapping[str, Any],
) -> None:
    checkpoint = components.get("checkpoint_path")
    if checkpoint is None and bool(_cfg_get(cfg, "resume", False)):
        checkpoint = _cfg_get(cfg, "checkpoint_path", None)
    if checkpoint is None:
        return

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return
    trainer.load_checkpoint(checkpoint_path)


def _close_robot_runtime(env: Any | None, teleop_device: Any | None) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass
    if teleop_device is not None and hasattr(teleop_device, "disconnect"):
        try:
            teleop_device.disconnect()
        except Exception:
            pass


class GrootHybridTrainer:
    def __init__(
        self,
        *,
        cfg: Any,
        policy: Any,
        optimizer: torch.optim.Optimizer,
        collector: Any,
        replay_buffer: Any,
        offline_data: Iterable[Any] | Iterator[Any],
        device: torch.device | str | None = None,
        lr_scheduler: Any | None = None,
        offline_batch_transform: Callable[[Any], Any] | None = None,
        online_observation_transform: Callable[[list[dict[str, Any]]], Any] | None = None,
        online_next_observation_transform: Callable[[list[dict[str, Any]]], Any] | None = None,
        offline_loss_fn: OfflineLossFn = compute_offline_loss,
        online_loss_fn: OnlineLossFn = compute_online_losses,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.optimizer = optimizer
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.offline_data = offline_data
        self.lr_scheduler = lr_scheduler
        self.offline_batch_transform = offline_batch_transform
        self.online_observation_transform = online_observation_transform
        self.online_next_observation_transform = online_next_observation_transform
        self.offline_loss_fn = offline_loss_fn
        self.online_loss_fn = online_loss_fn

        if device is None:
            try:
                device = next(policy.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        self.global_step = 0
        self.last_metrics: dict[str, float] = {}
        self._warmed_up = False
        self._offline_iterator: Iterator[Any] | None = None

    def _next_offline_batch(self) -> Any:
        if self._offline_iterator is None:
            self._offline_iterator = iter(self.offline_data)

        try:
            batch = next(self._offline_iterator)
        except StopIteration:
            self._offline_iterator = iter(self.offline_data)
            batch = next(self._offline_iterator)

        if self.offline_batch_transform is not None:
            batch = self.offline_batch_transform(batch)
        else:
            batch = _move_to_device(batch, self.device)
        return batch

    def _warmup_if_needed(self) -> int:
        if self._warmed_up:
            return 0

        warmup_chunks = int(_cfg_get(self.cfg, "collector.warmup_chunks", 0))
        self._warmed_up = True
        if warmup_chunks <= 0:
            return 0

        transitions = self.collector.collect(warmup_chunks)
        self.replay_buffer.extend(transitions)
        return len(transitions)

    def _collect_online_transitions(self) -> int:
        chunks_per_step = int(_cfg_get(self.cfg, "collector.chunks_per_step", 1))
        if chunks_per_step <= 0:
            return 0

        transitions = self.collector.collect(chunks_per_step)
        self.replay_buffer.extend(transitions)
        return len(transitions)

    def _sample_online_batch(self) -> GrootHybridBatch:
        batch_size = int(_cfg_get(self.cfg, "replay_buffer.online_batch_size", 1))
        return self.replay_buffer.sample(
            batch_size,
            device=self.device,
            observation_transform=self.online_observation_transform,
            next_observation_transform=self.online_next_observation_transform,
        )

    def _compute_grad_norm(self) -> float:
        parameters = [parameter for parameter in self.policy.parameters() if parameter.grad is not None]
        if not parameters:
            return 0.0

        grad_clip_norm = float(_cfg_get(self.cfg, "optimizer.grad_clip_norm", 0.0))
        max_norm = grad_clip_norm if grad_clip_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)
        return grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

    def step(self) -> dict[str, float]:
        self.policy.train()

        warmup_chunks = self._warmup_if_needed()
        collected_chunks = self._collect_online_transitions()
        offline_batch = self._next_offline_batch()

        self.optimizer.zero_grad(set_to_none=True)

        offline_loss, offline_metrics = self.offline_loss_fn(self.policy, offline_batch)
        total_loss = float(_cfg_get(self.cfg, "losses.offline_loss_weight", 1.0)) * offline_loss

        online_policy_loss = torch.zeros((), dtype=offline_loss.dtype, device=offline_loss.device)
        value_loss = torch.zeros((), dtype=offline_loss.dtype, device=offline_loss.device)
        online_metrics: dict[str, float] = {}

        online_flow_loss_weight = float(_cfg_get(self.cfg, "losses.online_flow_loss_weight", 0.0))
        value_loss_weight = float(_cfg_get(self.cfg, "losses.value_loss_weight", 0.0))
        should_compute_online = (
            len(self.replay_buffer) > 0 and (online_flow_loss_weight > 0.0 or value_loss_weight > 0.0)
        )

        if should_compute_online:
            online_batch = self._sample_online_batch()
            online_policy_loss, value_loss, online_metrics = self.online_loss_fn(
                self.policy,
                online_batch,
                _cfg_get(self.cfg, "losses", self.cfg),
            )
            total_loss = (
                total_loss
                + online_flow_loss_weight * online_policy_loss
                + value_loss_weight * value_loss
            )

        total_loss.backward()
        grad_norm = self._compute_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1
        self.last_metrics = {
            "loss": total_loss.detach().item(),
            "offline_loss": offline_loss.detach().item(),
            "online_policy_loss": online_policy_loss.detach().item(),
            "value_loss_total": value_loss.detach().item(),
            "grad_norm": grad_norm,
            "buffer_size": float(len(self.replay_buffer)),
            "collected_chunks": float(collected_chunks),
            "warmup_chunks": float(warmup_chunks),
            **offline_metrics,
            **online_metrics,
        }
        return dict(self.last_metrics)

    def run(self, num_steps: int | None = None) -> list[dict[str, float]]:
        if num_steps is None:
            target_steps = int(_cfg_get(self.cfg, "steps", self.global_step))
            num_steps = max(target_steps - self.global_step, 0)

        metrics: list[dict[str, float]] = []
        for _ in range(num_steps):
            metrics.append(self.step())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "trainer": {
                "warmed_up": self._warmed_up,
                "last_metrics": dict(self.last_metrics),
            },
            "rng_state": _capture_rng_state(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool = True) -> None:
        self.policy.load_state_dict(state_dict["policy"], strict=strict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

        scheduler_state = state_dict.get("lr_scheduler")
        if self.lr_scheduler is not None and scheduler_state is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)

        replay_buffer_state = state_dict.get("replay_buffer")
        if replay_buffer_state is not None:
            self.replay_buffer.load_state_dict(replay_buffer_state)

        trainer_state = state_dict.get("trainer", {})
        self.global_step = int(state_dict.get("global_step", 0))
        self._warmed_up = bool(trainer_state.get("warmed_up", False))
        self.last_metrics = {
            str(key): float(value)
            for key, value in trainer_state.get("last_metrics", {}).items()
        }
        _restore_rng_state(state_dict.get("rng_state"))

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        if checkpoint_path.suffix == "":
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_path / "trainer.pt"
        else:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "trainer.pt"

        state_dict = _load_checkpoint_file(checkpoint_path)
        self.load_state_dict(state_dict, strict=strict)


def train(cfg: Any) -> GrootHybridTrainer:
    validate = getattr(cfg, "validate", None)
    if callable(validate):
        validate()

    _set_seed(_cfg_get(cfg, "seed", None))

    output_dir = _cfg_get(cfg, "output_dir", None)
    if isinstance(output_dir, Path):
        output_dir.mkdir(parents=True, exist_ok=True)

    components = _resolve_train_components(cfg)
    offline_replay_enabled = bool(_cfg_get(cfg, "offline_replay.enabled", False))

    dataset = components.get("dataset")
    if dataset is None and "offline_data" not in components:
        dataset = make_dataset(cfg)

    policy = components.get("policy")
    if policy is None:
        if dataset is None or not hasattr(dataset, "meta"):
            raise ValueError("`train(cfg)` needs a dataset with `meta` to build the Groot policy.")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=_cfg_get(cfg, "rename_map", None),
        )

    device = torch.device(
        components.get("device")
        or _cfg_get(cfg, "policy.device", None)
        or _infer_policy_device(policy)
        or "cpu"
    )
    if hasattr(policy, "to"):
        policy.to(device)

    preprocessor = components.get("preprocessor")
    postprocessor = components.get("postprocessor")
    if preprocessor is None or postprocessor is None:
        if not hasattr(cfg, "policy"):
            preprocessor = preprocessor or _IdentityProcessor()
            postprocessor = postprocessor or _IdentityProcessor()
        else:
            dataset_stats = None if dataset is None or not hasattr(dataset, "meta") else dataset.meta.stats
            built_preprocessor, built_postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=_normalize_pretrained_path(_cfg_get(cfg, "policy.pretrained_path", None)),
                dataset_stats=dataset_stats,
                postprocessor_overrides={
                    "groot_action_unpack_unnormalize_v1": {
                        "output_mode": "full_chunk",
                    }
                },
            )
            preprocessor = preprocessor or built_preprocessor
            postprocessor = postprocessor or _ensure_full_chunk_postprocessor(built_postprocessor)

    env_preprocessor = components.get("env_preprocessor")
    env_postprocessor = components.get("env_postprocessor")
    if env_preprocessor is None or env_postprocessor is None:
        if offline_replay_enabled or _cfg_get(cfg, "env", None) is None:
            env_preprocessor = env_preprocessor or _IdentityProcessor()
            env_postprocessor = env_postprocessor or _IdentityProcessor()
        else:
            built_env_preprocessor, built_env_postprocessor = make_env_pre_post_processors(cfg.env)
            env_preprocessor = env_preprocessor or built_env_preprocessor
            env_postprocessor = env_postprocessor or built_env_postprocessor

    optimizer = components.get("optimizer")
    lr_scheduler = components.get("lr_scheduler")
    if optimizer is None:
        optimizer, built_lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        if lr_scheduler is None:
            lr_scheduler = built_lr_scheduler

    replay_buffer = components.get("replay_buffer")
    if replay_buffer is None:
        if offline_replay_enabled:
            if dataset is None:
                raise ValueError(
                    "Dataset-only offline replay requires `dataset` to be available."
                )
            replay_buffer = GrootOfflineDatasetReplayBuffer(
                dataset=dataset,
                action_chunk_size=int(
                    getattr(
                        _cfg_get(cfg, "policy", None),
                        "n_action_steps_effective",
                        getattr(_cfg_get(cfg, "policy", None), "n_action_steps", 16),
                    )
                ),
                transition_stride=int(_cfg_get(cfg, "offline_replay.transition_stride", 1)),
                discount=float(_cfg_get(cfg, "losses.discount", 0.99)),
                value_target_mode=str(_cfg_get(cfg, "offline_replay.value_target_mode", "monte_carlo")),
                terminal_reward=float(_cfg_get(cfg, "offline_replay.terminal_reward", 1.0)),
                step_reward=float(_cfg_get(cfg, "offline_replay.step_reward", 0.0)),
                success_value=bool(_cfg_get(cfg, "offline_replay.success_value", True)),
                action_pad_mode=str(_cfg_get(cfg, "offline_replay.action_pad_mode", "repeat_last")),
            )
        else:
            replay_buffer = GrootHybridReplayBuffer(int(_cfg_get(cfg, "replay_buffer.capacity", 4096)))

    built_envs = None
    built_robot_env = None
    built_teleop_device = None
    collector = components.get("collector")
    if collector is None:
        if offline_replay_enabled:
            collector = _NullCollector()
        else:
            default_collector_observation_builder = components.get("observation_builder")
            rollout = components.get("rollout")
            if rollout is None:
                if _cfg_get(cfg, "env", None) is None:
                    raise ValueError("`train(cfg)` needs `cfg.env` or an injected rollout/collector.")
                if _is_gym_manipulator_env_cfg(_cfg_get(cfg, "env", None)):
                    rollout, built_robot_env, built_teleop_device = _build_gym_manipulator_rollout(
                        cfg=cfg,
                        device=device,
                    )
                else:
                    built_envs = make_env(
                        cfg.env,
                        n_envs=int(_cfg_get(cfg, "collector.n_envs", 1)),
                        use_async_envs=bool(_cfg_get(cfg, "collector.use_async_envs", False)),
                    )
                    rollout = _SingleVectorEnvRolloutAdapter(
                        _resolve_single_vector_env(built_envs),
                        task=_normalize_task(_cfg_get(cfg, "env.task", "")),
                    )

            if default_collector_observation_builder is None:
                if isinstance(rollout, _RobotTransitionRolloutAdapter):
                    default_collector_observation_builder = _make_robot_collector_observation_builder(
                        task=rollout.get_task(),
                    )
                else:
                    default_collector_observation_builder = _make_collector_observation_builder(
                        env_preprocessor=env_preprocessor,
                    )

            collector = GrootHybridCollector(
                rollout=rollout,
                policy=policy,
                discount=float(_cfg_get(cfg, "losses.discount", 0.99)),
                max_steps_per_chunk=_cfg_get(cfg, "collector.max_steps_per_chunk", None),
                observation_builder=default_collector_observation_builder,
                preprocessor=preprocessor,
                postprocessor=components.get("collector_postprocessor")
                or _make_collector_postprocessor(
                    policy_postprocessor=postprocessor,
                    env_postprocessor=env_postprocessor,
                ),
            )

    if hasattr(collector, "reset"):
        collector.reset(seed=_cfg_get(cfg, "seed", None))

    offline_data = components.get("offline_data")
    offline_batch_transform = components.get("offline_batch_transform")
    if offline_batch_transform is None and offline_data is None:
        offline_batch_transform = preprocessor
    if offline_data is None:
        if dataset is None:
            raise ValueError("`train(cfg)` needs either an offline dataset or injected `offline_data`.")
        offline_data = cycle(_make_dataloader(cfg, dataset, device))

    online_observation_transform = components.get("online_observation_transform")
    online_next_observation_transform = components.get("online_next_observation_transform")
    if online_observation_transform is None or online_next_observation_transform is None:
        if offline_replay_enabled:
            default_observation_transform = _make_dataset_observation_transform(
                preprocessor=preprocessor,
            )
            online_observation_transform = online_observation_transform or default_observation_transform
            online_next_observation_transform = (
                online_next_observation_transform or default_observation_transform
            )
        else:
            rollout = getattr(collector, "rollout", None)
            task = ""
            if rollout is not None and hasattr(rollout, "get_task"):
                try:
                    task = str(rollout.get_task())
                except Exception:
                    task = ""
            if not task:
                task = _normalize_task(_cfg_get(cfg, "env.task", ""))

            default_single_observation_builder = components.get("single_observation_builder")
            if default_single_observation_builder is None:
                if isinstance(rollout, _RobotTransitionRolloutAdapter):
                    default_single_observation_builder = _make_robot_single_observation_builder(task=task)
                else:
                    default_single_observation_builder = _make_single_observation_builder(
                        env_preprocessor=env_preprocessor,
                        task=task,
                    )

            default_observation_transform = _make_online_observation_transform(
                observation_builder=default_single_observation_builder,
                preprocessor=preprocessor,
            )
            online_observation_transform = online_observation_transform or default_observation_transform
            online_next_observation_transform = (
                online_next_observation_transform or default_observation_transform
            )

    trainer = GrootHybridTrainer(
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        collector=collector,
        replay_buffer=replay_buffer,
        offline_data=offline_data,
        device=device,
        lr_scheduler=lr_scheduler,
        offline_batch_transform=offline_batch_transform,
        online_observation_transform=online_observation_transform,
        online_next_observation_transform=online_next_observation_transform,
    )

    _load_trainer_checkpoint_if_requested(trainer, cfg, components)

    total_steps = int(components.get("num_steps", _cfg_get(cfg, "steps", 0)))
    try:
        steps_to_run = max(total_steps - trainer.global_step, 0)
        for _ in range(steps_to_run):
            trainer.step()

            should_save = bool(_cfg_get(cfg, "save_checkpoint", False)) and isinstance(output_dir, Path)
            if not should_save:
                continue

            save_freq = int(_cfg_get(cfg, "save_freq", 0))
            is_last_step = trainer.global_step == total_steps
            is_save_step = save_freq > 0 and trainer.global_step % save_freq == 0
            if not (is_last_step or is_save_step):
                continue

            checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, trainer.global_step)
            save_run_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=trainer.global_step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            trainer.save_checkpoint(checkpoint_dir)
            update_last_checkpoint(checkpoint_dir)
    finally:
        if built_envs is not None:
            close_envs(built_envs)
        if built_robot_env is not None or built_teleop_device is not None:
            _close_robot_runtime(built_robot_env, built_teleop_device)

    return trainer
