# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
else:
    PretrainedConfig = object
    BatchFeature = None

from lerobot.policies.groot.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer

BACKBONE_FEATURES_KEY = "backbone_features"
BACKBONE_ATTENTION_MASK_KEY = "backbone_attention_mask"
STATE_FEATURES_KEY = "state_features"
FUTURE_TOKENS_KEY = "future_tokens"
EMBODIMENT_ID_KEY = "embodiment_id"
ACTION_MASK_KEY = "action_mask"
NOISY_ACTION_KEY = "noisy_action"
PREDICTED_VELOCITY_KEY = "pred_velocity"
TARGET_VELOCITY_KEY = "target_velocity"
TIMESTEP_BUCKET_KEY = "timestep_bucket"


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        b, t, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maximum Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        processed_backbone_output = BatchFeature(data=dict(backbone_output))
        backbone_features = processed_backbone_output[BACKBONE_FEATURES_KEY]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        processed_backbone_output[BACKBONE_FEATURES_KEY] = backbone_features
        return processed_backbone_output

    def _expand_batch_feature(self, batch_feature: BatchFeature) -> BatchFeature:
        if self.config.expand_batch is None:
            return BatchFeature(data=dict(batch_feature))

        expanded_data = {}
        for key, value in batch_feature.items():
            if not isinstance(value, Tensor):
                expanded_data[key] = value
                continue
            factors = (self.config.expand_batch, *([1] * (value.ndim - 1)))
            expanded_data[key] = value.repeat(*factors)
        return BatchFeature(data=expanded_data)

    def build_context(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        processed_backbone_output = self.process_backbone_output(backbone_output)
        processed_backbone_output = self._expand_batch_feature(processed_backbone_output)
        expanded_action_input = self._expand_batch_feature(action_input)

        backbone_features = processed_backbone_output[BACKBONE_FEATURES_KEY]
        embodiment_id = expanded_action_input[EMBODIMENT_ID_KEY]
        state_features = self.state_encoder(expanded_action_input["state"], embodiment_id)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(backbone_features.shape[0], -1, -1)

        context_data = {
            BACKBONE_FEATURES_KEY: backbone_features,
            STATE_FEATURES_KEY: state_features,
            FUTURE_TOKENS_KEY: future_tokens,
            EMBODIMENT_ID_KEY: embodiment_id,
        }
        backbone_attention_mask = processed_backbone_output.get(BACKBONE_ATTENTION_MASK_KEY)
        if backbone_attention_mask is not None:
            context_data[BACKBONE_ATTENTION_MASK_KEY] = backbone_attention_mask
        return BatchFeature(data=context_data)

    def _normalize_action_mask(self, action_mask: Tensor, action_shape: tuple[int, int, int]) -> Tensor:
        if action_mask.ndim == 2:
            action_mask = action_mask.unsqueeze(-1)
        if action_mask.ndim != 3:
            raise ValueError(
                "action_mask must be rank-2 or rank-3 with shape compatible with actions. "
                f"Got ndim={action_mask.ndim} and shape={tuple(action_mask.shape)}."
            )
        if action_mask.shape[0] != action_shape[0] or action_mask.shape[1] != action_shape[1]:
            raise ValueError(
                "action_mask batch/time dimensions must match actions before chunk normalization. "
                f"Got action_mask shape={tuple(action_mask.shape)} and actions shape={action_shape}."
            )
        if action_mask.shape[2] == 1 and action_shape[2] != 1:
            action_mask = action_mask.expand(-1, -1, action_shape[2])
        elif action_mask.shape[2] != action_shape[2]:
            raise ValueError(
                "action_mask feature dimension must either match actions or be singleton for broadcasting. "
                f"Got action_mask shape={tuple(action_mask.shape)} and actions shape={action_shape}."
            )
        return action_mask

    def normalize_action_chunk(
        self,
        actions: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        elif actions.ndim != 3:
            raise ValueError(
                "actions must be rank-2 or rank-3 tensors shaped as (B, D) or (B, T, D). "
                f"Got ndim={actions.ndim} and shape={tuple(actions.shape)}."
            )
        if actions.shape[1] == 0 or actions.shape[2] == 0:
            raise ValueError(f"actions must have non-zero chunk and feature dimensions. Got shape={tuple(actions.shape)}.")

        batch_size, chunk_size, action_dim = actions.shape
        if action_mask is None:
            action_mask = torch.ones(batch_size, chunk_size, action_dim, dtype=torch.bool, device=actions.device)
        else:
            action_mask = self._normalize_action_mask(action_mask.to(device=actions.device), tuple(actions.shape))

        if chunk_size < self.action_horizon:
            pad_steps = self.action_horizon - chunk_size
            last_action = actions[:, -1:, :]
            actions = torch.cat([actions, last_action.expand(-1, pad_steps, -1)], dim=1)
            action_mask = torch.cat(
                [action_mask, action_mask.new_zeros(batch_size, pad_steps, action_mask.shape[2])],
                dim=1,
            )
        elif chunk_size > self.action_horizon:
            actions = actions[:, : self.action_horizon]
            action_mask = action_mask[:, : self.action_horizon]

        if action_dim < self.action_dim:
            pad_dims = self.action_dim - action_dim
            actions = torch.cat(
                [actions, actions.new_zeros(batch_size, actions.shape[1], pad_dims)],
                dim=2,
            )
            action_mask = torch.cat(
                [action_mask, action_mask.new_zeros(batch_size, action_mask.shape[1], pad_dims)],
                dim=2,
            )
        elif action_dim > self.action_dim:
            actions = actions[:, :, : self.action_dim]
            action_mask = action_mask[:, :, : self.action_dim]

        return actions, action_mask

    def _resolve_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        timesteps: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if timesteps is None:
            continuous_timesteps = self.sample_time(batch_size, device=device, dtype=dtype)
        else:
            timesteps = timesteps.to(device=device)
            if timesteps.numel() == 1:
                first_timestep = timesteps.reshape(1).expand(batch_size)
            else:
                if timesteps.shape[0] != batch_size:
                    raise ValueError(
                        "timesteps batch dimension must match actions. "
                        f"Got timesteps shape={tuple(timesteps.shape)} and batch_size={batch_size}."
                    )
                first_timestep = timesteps.reshape(batch_size, -1)[:, 0]
            if torch.is_floating_point(first_timestep):
                continuous_timesteps = first_timestep.to(dtype=dtype).clamp(min=0.0, max=1.0)
            else:
                discrete_timesteps = first_timestep.to(dtype=torch.long)
                discrete_timesteps = discrete_timesteps.clamp(min=0, max=self.num_timestep_buckets - 1)
                continuous_timesteps = discrete_timesteps.to(dtype=dtype) / float(self.num_timestep_buckets)

        discrete_timesteps = (continuous_timesteps * self.num_timestep_buckets).long()
        discrete_timesteps = discrete_timesteps.clamp(min=0, max=self.num_timestep_buckets - 1)
        continuous_timesteps = continuous_timesteps.clamp(min=0.0, max=1.0)
        return continuous_timesteps[:, None, None], discrete_timesteps

    def _encode_action_features(
        self,
        actions: Tensor,
        timestep_buckets: Tensor,
        embodiment_id: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        action_features = self.action_encoder(actions, timestep_buckets, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        return action_features

    def forward_chunk(
        self,
        context: BatchFeature,
        *,
        actions: Tensor,
        action_mask: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> BatchFeature:
        actions, action_mask = self.normalize_action_chunk(actions, action_mask)
        if noise is None:
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        elif noise.shape != actions.shape:
            raise ValueError(f"noise must match normalized action shape {tuple(actions.shape)}, got {tuple(noise.shape)}.")
        else:
            noise = noise.to(device=actions.device, dtype=actions.dtype)

        continuous_timesteps, timestep_buckets = self._resolve_timesteps(
            actions.shape[0],
            device=actions.device,
            dtype=actions.dtype,
            timesteps=timesteps,
        )
        noisy_trajectory = (1 - continuous_timesteps) * noise + continuous_timesteps * actions
        target_velocity = actions - noise

        action_features = self._encode_action_features(
            noisy_trajectory,
            timestep_buckets,
            context[EMBODIMENT_ID_KEY],
            device=context[BACKBONE_FEATURES_KEY].device,
        )
        sa_embs = torch.cat(
            (context[STATE_FEATURES_KEY], context[FUTURE_TOKENS_KEY], action_features),
            dim=1,
        )
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=context[BACKBONE_FEATURES_KEY],
            encoder_attention_mask=context.get(BACKBONE_ATTENTION_MASK_KEY),
            timestep=timestep_buckets,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, context[EMBODIMENT_ID_KEY])
        pred_velocity = pred[:, -actions.shape[1] :]

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum().clamp(min=1)
        return BatchFeature(
            data={
                "loss": loss,
                PREDICTED_VELOCITY_KEY: pred_velocity,
                TARGET_VELOCITY_KEY: target_velocity,
                NOISY_ACTION_KEY: noisy_trajectory,
                ACTION_MASK_KEY: action_mask,
                TIMESTEP_BUCKET_KEY: timestep_buckets,
            }
        )

    @torch.no_grad()
    def sample_actions_from_context(
        self,
        context: BatchFeature,
        *,
        noise: Tensor | None = None,
        num_inference_timesteps: int | None = None,
    ) -> BatchFeature:
        backbone_features = context[BACKBONE_FEATURES_KEY]
        batch_size = backbone_features.shape[0]
        device = backbone_features.device

        if noise is None:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=backbone_features.dtype,
                device=device,
            )
        else:
            actions, _ = self.normalize_action_chunk(noise.to(device=device, dtype=backbone_features.dtype))

        num_steps = num_inference_timesteps or self.num_inference_timesteps
        if num_steps is None or num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}.")
        dt = 1.0 / num_steps

        pred_velocity = None
        for timestep_index in range(num_steps):
            t_cont = timestep_index / float(num_steps)
            timestep_bucket = int(t_cont * self.num_timestep_buckets)
            timestep_bucket = min(timestep_bucket, self.num_timestep_buckets - 1)
            timestep_tensor = torch.full(size=(batch_size,), fill_value=timestep_bucket, device=device)
            action_features = self._encode_action_features(
                actions,
                timestep_tensor,
                context[EMBODIMENT_ID_KEY],
                device=device,
            )
            sa_embs = torch.cat(
                (context[STATE_FEATURES_KEY], context[FUTURE_TOKENS_KEY], action_features),
                dim=1,
            )

            # Preserve the warm-start inference path by matching the original call shape,
            # which omitted encoder_attention_mask for denoising.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=backbone_features,
                timestep=timestep_tensor,
            )
            pred = self.action_decoder(model_output, context[EMBODIMENT_ID_KEY])
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        output_dict = {"action_pred": actions}
        if pred_velocity is not None:
            output_dict[PREDICTED_VELOCITY_KEY] = pred_velocity
        return BatchFeature(data=output_dict)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        context = self.build_context(backbone_output, action_input)
        return self.forward_chunk(
            context,
            actions=action_input["action"],
            action_mask=action_input.get(ACTION_MASK_KEY),
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        context = self.build_context(backbone_output, action_input)
        return self.sample_actions_from_context(context)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
