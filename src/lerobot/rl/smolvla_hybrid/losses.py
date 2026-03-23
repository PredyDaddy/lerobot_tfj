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

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rl.smolvla_hybrid.buffer import SmolVLAChunkBatch


def compute_online_losses(
    policy: SmolVLAPolicy,
    online_batch: SmolVLAChunkBatch,
    loss_cfg,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    actions = online_batch.action
    target_action_dim = int(policy.config.max_action_dim)
    current_action_dim = int(actions.shape[-1])
    if current_action_dim < target_action_dim:
        pad_dim = target_action_dim - current_action_dim
        actions = F.pad(actions, (0, pad_dim))
    elif current_action_dim > target_action_dim:
        actions = actions[..., :target_action_dim]

    noise = policy.model.sample_noise(actions.shape, actions.device)
    timestep = policy.model.sample_time(actions.shape[0], actions.device)
    timestep_expanded = timestep[:, None, None]
    noisy_actions = timestep_expanded * noise + (1 - timestep_expanded) * actions
    target_flow = noise - actions

    predicted_flow = policy.compute_fm_score(online_batch.observation, noisy_actions, timestep)
    per_sample_flow_loss = F.mse_loss(target_flow, predicted_flow, reduction="none").mean(dim=(1, 2))

    values = policy.get_value(online_batch.observation)
    with torch.no_grad():
        next_values = policy.get_value(online_batch.next_observation)
        value_targets = online_batch.reward + online_batch.bootstrap_discount * next_values

    advantages = value_targets - values.detach()
    if loss_cfg.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    clipped_advantages = advantages.clamp(loss_cfg.advantage_clip_min, loss_cfg.advantage_clip_max)
    scaled_advantages = clipped_advantages / max(loss_cfg.advantage_temperature, 1e-6)
    weights = torch.exp(scaled_advantages).clamp(max=loss_cfg.max_advantage_weight)

    policy_loss = (weights * per_sample_flow_loss).mean()
    value_loss = F.mse_loss(values, value_targets)

    return policy_loss, value_loss, {
        "online_flow_loss": per_sample_flow_loss.mean().item(),
        "value_loss": value_loss.item(),
        "value_mean": values.mean().item(),
        "target_value_mean": value_targets.mean().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(unbiased=False).item() if advantages.numel() > 1 else 0.0,
        "weight_mean": weights.mean().item(),
        "weight_max": weights.max().item(),
        "done_ratio": online_batch.done.float().mean().item(),
        "fm_time_mean": timestep.mean().item(),
    }
