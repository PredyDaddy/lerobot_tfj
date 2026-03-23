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

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.rl.groot_hybrid.buffer import GrootHybridBatch


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().to(device="cpu").item())
    return None


def _extract_loss_and_metrics(outputs: Any) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, torch.Tensor):
        return outputs, {}

    if isinstance(outputs, tuple):
        if len(outputs) != 2:
            raise TypeError(
                "Expected `policy.forward(...)` to return a loss tensor or `(loss, metrics)` tuple."
            )
        loss, metrics = outputs
    elif isinstance(outputs, Mapping):
        if "loss" not in outputs:
            raise KeyError("Expected mapping outputs to include a `loss` entry.")
        loss = outputs["loss"]
        metrics = {key: value for key, value in outputs.items() if key != "loss"}
    else:
        raise TypeError(f"Unsupported loss output type: {type(outputs)}")

    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"Expected loss to be a torch.Tensor, got {type(loss)}.")

    scalar_metrics: dict[str, float] = {}
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            maybe_scalar = _as_float(value)
            if maybe_scalar is not None:
                scalar_metrics[str(key)] = maybe_scalar

    return loss, scalar_metrics


def _resolve_value_fn(
    policy: Any,
    value_fn: Callable[[Any], torch.Tensor] | None = None,
) -> Callable[[Any], torch.Tensor] | None:
    if value_fn is not None:
        return value_fn

    for candidate_name in ("predict_value", "get_value", "value"):
        candidate = getattr(policy, candidate_name, None)
        if callable(candidate):
            return candidate

    return None


def _align_action_chunks(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted_actions.ndim != 3 or target_actions.ndim != 3:
        raise ValueError(
            "Expected both predicted and target action chunks to have shape `(batch, steps, dim)`."
        )

    batch_size = min(predicted_actions.shape[0], target_actions.shape[0])
    num_steps = min(predicted_actions.shape[1], target_actions.shape[1])
    action_dim = min(predicted_actions.shape[2], target_actions.shape[2])
    if batch_size <= 0 or num_steps <= 0 or action_dim <= 0:
        raise ValueError(
            "Predicted and target action chunks must have at least one batch element, step, and dim."
        )

    return (
        predicted_actions[:batch_size, :num_steps, :action_dim],
        target_actions[:batch_size, :num_steps, :action_dim],
    )


def _make_action_batch(online_batch: GrootHybridBatch) -> dict[str, Any] | None:
    if not isinstance(online_batch.observation, Mapping):
        return None

    batch = dict(online_batch.observation)
    batch["action"] = online_batch.action_chunk
    batch["action_mask"] = torch.ones_like(online_batch.action_chunk, dtype=torch.bool)
    return batch


def _compute_advantage_weights(
    advantages: torch.Tensor,
    loss_cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalize_advantage = bool(getattr(loss_cfg, "normalize_advantage", True))
    advantage_temperature = float(getattr(loss_cfg, "advantage_temperature", 1.0))
    advantage_clip_min = float(getattr(loss_cfg, "advantage_clip_min", -5.0))
    advantage_clip_max = float(getattr(loss_cfg, "advantage_clip_max", 5.0))
    max_advantage_weight = float(getattr(loss_cfg, "max_advantage_weight", 20.0))

    normalized_advantages = advantages
    if normalize_advantage and advantages.numel() > 1:
        normalized_advantages = (
            normalized_advantages - normalized_advantages.mean()
        ) / (normalized_advantages.std(unbiased=False) + 1e-6)

    clipped_advantages = normalized_advantages.clamp(advantage_clip_min, advantage_clip_max)
    scaled_advantages = clipped_advantages / max(advantage_temperature, 1e-6)
    weights = torch.exp(scaled_advantages).clamp(max=max_advantage_weight)

    return weights, {
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(unbiased=False).item() if advantages.numel() > 1 else 0.0,
        "weight_mean": weights.mean().item(),
        "weight_max": weights.max().item(),
    }


def compute_offline_loss(policy: Any, offline_batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
    return _extract_loss_and_metrics(policy.forward(offline_batch))


def compute_online_flow_loss(
    policy: Any,
    online_batch: GrootHybridBatch,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    custom_flow_loss = getattr(policy, "compute_online_flow_loss", None)
    if callable(custom_flow_loss):
        flow_loss, metrics = _extract_loss_and_metrics(custom_flow_loss(online_batch))
        metrics.setdefault("online_flow_loss", flow_loss.detach().item())
        return flow_loss, metrics, None

    forward_action_chunk = getattr(policy, "forward_action_chunk", None)
    action_batch = _make_action_batch(online_batch)
    if callable(forward_action_chunk) and action_batch is not None:
        flow_outputs = forward_action_chunk(batch=action_batch)
        flow_loss, metrics = _extract_loss_and_metrics(flow_outputs)
        metrics.setdefault("online_flow_loss", flow_loss.detach().item())
        return flow_loss, metrics, None

    predict_action_chunk = getattr(policy, "predict_action_chunk", None)
    if callable(predict_action_chunk):
        predicted_actions = predict_action_chunk(online_batch.observation)
        aligned_predictions, aligned_targets = _align_action_chunks(predicted_actions, online_batch.action_chunk)
        per_sample_loss = F.mse_loss(aligned_predictions, aligned_targets, reduction="none").mean(dim=(1, 2))
        return per_sample_loss.mean(), {"online_flow_loss": per_sample_loss.mean().item()}, per_sample_loss

    raise AttributeError(
        "Policy does not expose `compute_online_flow_loss`, `forward_action_chunk`, or `predict_action_chunk`."
    )


def compute_online_value_loss(
    policy: Any,
    online_batch: GrootHybridBatch,
    value_fn: Callable[[Any], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    resolved_value_fn = _resolve_value_fn(policy, value_fn)
    zero = torch.zeros((), dtype=online_batch.reward.dtype, device=online_batch.reward.device)
    if resolved_value_fn is None:
        return zero, {"value_loss": 0.0, "done_ratio": online_batch.done.float().mean().item()}, None

    values = resolved_value_fn(online_batch.observation).to(dtype=online_batch.reward.dtype).view(-1)
    with torch.no_grad():
        next_values = resolved_value_fn(online_batch.next_observation).to(dtype=online_batch.reward.dtype).view(-1)
        value_targets = online_batch.reward + online_batch.bootstrap_discount * next_values

    value_loss = F.mse_loss(values, value_targets)
    advantages = value_targets - values.detach()
    return value_loss, {
        "value_loss": value_loss.detach().item(),
        "value_mean": values.detach().mean().item(),
        "target_value_mean": value_targets.detach().mean().item(),
        "done_ratio": online_batch.done.float().mean().item(),
    }, advantages


def compute_online_losses(
    policy: Any,
    online_batch: GrootHybridBatch,
    loss_cfg: Any,
    *,
    value_fn: Callable[[Any], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    policy_loss, flow_metrics, per_sample_flow_loss = compute_online_flow_loss(policy, online_batch)
    value_loss, value_metrics, advantages = compute_online_value_loss(policy, online_batch, value_fn=value_fn)

    use_advantage_weighting = bool(getattr(loss_cfg, "use_advantage_weighting", True))
    weighting_metrics = {"advantage_weighting_applied": 0.0}

    if use_advantage_weighting and per_sample_flow_loss is not None and advantages is not None:
        weights, weighting_metrics = _compute_advantage_weights(advantages, loss_cfg)
        policy_loss = (weights * per_sample_flow_loss).mean()
        flow_metrics["online_flow_loss"] = policy_loss.detach().item()
        weighting_metrics["advantage_weighting_applied"] = 1.0

    metrics = {
        **flow_metrics,
        **value_metrics,
        **weighting_metrics,
    }
    return policy_loss, value_loss, metrics
