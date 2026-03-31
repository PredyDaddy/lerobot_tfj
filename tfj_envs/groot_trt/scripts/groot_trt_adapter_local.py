#!/usr/bin/env python

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from common import validate_engine_dir
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from trt_runtime_local import TrtSession


def _resolve_num_patches(backbone: Any) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model"):
        return int(vision_model.vision_model.embeddings.num_patches)
    return int(vision_model.embeddings.num_patches)


class TrtGrootPolicyAdapter(PreTrainedPolicy):
    config_class = GrootConfig
    name = "groot_trt"

    def __init__(
        self,
        config: GrootConfig,
        *,
        base_policy: PreTrainedPolicy,
        engine_dir: str | Path,
        trt_device: str = "cuda:0",
    ) -> None:
        super().__init__(config)
        if not str(trt_device).startswith("cuda"):
            raise ValueError("TrtGrootPolicyAdapter requires a CUDA device.")

        self.config = config
        self.device = torch.device(trt_device)

        self.base_policy = base_policy.to(self.device)
        self.base_policy.eval()
        self._groot_model = self.base_policy._groot_model
        self.backbone = self._groot_model.backbone
        self.action_head = self._groot_model.action_head
        self.backbone.eval()
        self.action_head.eval()

        self.engine_dir = Path(engine_dir).expanduser().resolve()
        validate_engine_dir(self.engine_dir)
        self.sessions = {
            "vit": TrtSession(self.engine_dir / "vit_fp16.engine", device=str(self.device)),
            "llm": TrtSession(self.engine_dir / "llm_fp16.engine", device=str(self.device)),
            "vlln": TrtSession(self.engine_dir / "vlln_vl_self_attention.engine", device=str(self.device)),
            "state_encoder": TrtSession(self.engine_dir / "state_encoder.engine", device=str(self.device)),
            "action_encoder": TrtSession(self.engine_dir / "action_encoder.engine", device=str(self.device)),
            "dit": TrtSession(self.engine_dir / "DiT_fp16.engine", device=str(self.device)),
            "action_decoder": TrtSession(self.engine_dir / "action_decoder.engine", device=str(self.device)),
        }

        self.num_patches = _resolve_num_patches(self.backbone)
        self.image_token_index = int(self.backbone.eagle_model.image_token_index)
        self.use_pixel_shuffle = bool(getattr(self.backbone.eagle_model, "use_pixel_shuffle", False))
        self.downsample_ratio = float(getattr(self.backbone.eagle_model, "downsample_ratio", 1.0))
        self.original_action_dim = int(self.config.output_features["action"].shape[0])
        self.action_horizon = int(self.action_head.config.action_horizon)
        self.model_action_dim = int(self.action_head.config.action_dim)
        self.num_inference_timesteps = int(self.action_head.num_inference_timesteps)
        self.num_timestep_buckets = int(self.action_head.num_timestep_buckets)
        self.add_pos_embed = bool(getattr(self.action_head.config, "add_pos_embed", False))

        if self.num_inference_timesteps <= 0:
            raise ValueError(f"Invalid num_inference_timesteps: {self.num_inference_timesteps}")

        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("TrtGrootPolicyAdapter does not support training forward().")

    def describe_engines(self) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [meta.__dict__ for meta in session.describe()]
            for name, session in self.sessions.items()
        }

    def _require_tensor(self, batch: dict[str, Tensor], key: str) -> Tensor:
        if key not in batch:
            raise KeyError(f"Missing runtime batch key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Runtime batch key `{key}` must be a torch.Tensor, got {type(value)}")
        return value

    def _prepare_runtime_inputs(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pixel_values = self._require_tensor(batch, "eagle_pixel_values").to(
            device=self.device, dtype=torch.float16
        )
        input_ids = self._require_tensor(batch, "eagle_input_ids").to(device=self.device, dtype=torch.int64)
        attention_mask = self._require_tensor(batch, "eagle_attention_mask").to(
            device=self.device, dtype=torch.int64
        )
        state = self._require_tensor(batch, "state").to(device=self.device, dtype=torch.float16)
        embodiment_id = self._require_tensor(batch, "embodiment_id").to(device=self.device, dtype=torch.int64)

        return (
            pixel_values.contiguous(),
            input_ids.contiguous(),
            attention_mask.contiguous(),
            state.contiguous(),
            embodiment_id.contiguous(),
        )

    def _build_position_ids(self, batch_size: int) -> Tensor:
        return torch.arange(self.num_patches, device=self.device, dtype=torch.int64).unsqueeze(0).expand(
            batch_size, -1
        ).contiguous()

    def _postprocess_vit(self, vit_embeds: Tensor) -> Tensor:
        vit_embeds = vit_embeds.to(
            device=self.device,
            dtype=next(self.backbone.eagle_model.mlp1.parameters()).dtype,
        )
        if self.use_pixel_shuffle:
            num_tokens = int(vit_embeds.shape[1])
            grid = int(round(math.sqrt(num_tokens)))
            if grid * grid != num_tokens:
                raise ValueError(
                    f"ViT token count {num_tokens} is not a square grid required by pixel shuffle."
                )
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], grid, grid, vit_embeds.shape[-1])
            vit_embeds = self.backbone.eagle_model.pixel_shuffle(
                vit_embeds,
                scale_factor=self.downsample_ratio,
            )
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.backbone.eagle_model.mlp1(vit_embeds)
        return vit_embeds.to(dtype=torch.float16).contiguous()

    def _build_inputs_embeds_from_vit(self, input_ids: Tensor, vit_embeds: Tensor) -> Tensor:
        embedding_layer = self.backbone.eagle_model.language_model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids).to(device=self.device)
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        flat_embeds = inputs_embeds.reshape(batch_size * seq_len, hidden_dim)
        flat_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_ids == self.image_token_index
        num_selected = int(selected.sum().item())
        if num_selected == 0:
            raise RuntimeError(
                "No image-token slots were found in eagle_input_ids. "
                "The prompt/template no longer matches the exported TRT boundary."
            )

        flat_vit = vit_embeds.reshape(-1, hidden_dim).to(dtype=flat_embeds.dtype)
        if flat_vit.shape[0] < num_selected:
            raise RuntimeError(
                "Not enough ViT embeddings to replace image-token slots: "
                f"selected={num_selected}, vit_tokens={flat_vit.shape[0]}"
            )

        flat_embeds[selected] = flat_embeds[selected] * 0.0 + flat_vit[:num_selected]
        inputs_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_dim)
        return inputs_embeds.to(dtype=torch.float16).contiguous()

    def _future_tokens(self, batch_size: int, dtype: torch.dtype) -> Tensor:
        return self.action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=self.device,
            dtype=dtype,
        ).contiguous()

    def _apply_position_embedding(self, action_features: Tensor) -> Tensor:
        if not self.add_pos_embed:
            return action_features
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
        pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0).to(dtype=action_features.dtype)
        return (action_features + pos_embs).contiguous()

    def _init_actions(self, batch_size: int, noise: Tensor | None) -> Tensor:
        if noise is not None:
            if not isinstance(noise, torch.Tensor):
                raise TypeError(f"`noise` must be a torch.Tensor, got {type(noise)}")
            expected = (batch_size, self.action_horizon, self.model_action_dim)
            if tuple(noise.shape) != expected:
                raise ValueError(f"`noise` shape must be {expected}, got {tuple(noise.shape)}")
            return noise.to(device=self.device, dtype=torch.float16).contiguous()

        return torch.randn(
            size=(batch_size, self.action_horizon, self.model_action_dim),
            dtype=torch.float16,
            device=self.device,
        ).contiguous()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        pixel_values, input_ids, attention_mask, state, embodiment_id = self._prepare_runtime_inputs(batch)

        position_ids = self._build_position_ids(int(pixel_values.shape[0]))
        vit_embeds = self.sessions["vit"].run(
            {
                "pixel_values": pixel_values,
                "position_ids": position_ids,
            }
        )["vit_embeds"]
        vit_embeds = self._postprocess_vit(vit_embeds)

        inputs_embeds = self._build_inputs_embeds_from_vit(input_ids, vit_embeds)
        backbone_features = self.sessions["llm"].run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
        )["embeddings"].to(dtype=torch.float16)

        vl_embs = self.sessions["vlln"].run({"backbone_features": backbone_features})["output"].to(
            dtype=torch.float16
        )
        state_features = self.sessions["state_encoder"].run(
            {
                "state": state,
                "embodiment_id": embodiment_id,
            }
        )["output"].to(dtype=torch.float16)

        batch_size = int(attention_mask.shape[0])
        actions = self._init_actions(batch_size, kwargs.get("noise"))
        future_tokens = self._future_tokens(batch_size, dtype=torch.float16)
        dt = 1.0 / float(self.num_inference_timesteps)

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                dtype=torch.int64,
                device=self.device,
            )
            action_features = self.sessions["action_encoder"].run(
                {
                    "actions": actions,
                    "timesteps_tensor": timesteps_tensor,
                    "embodiment_id": embodiment_id,
                }
            )["output"].to(dtype=torch.float16)
            action_features = self._apply_position_embedding(action_features)

            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).contiguous()
            model_output = self.sessions["dit"].run(
                {
                    "sa_embs": sa_embs,
                    "vl_embs": vl_embs,
                    "timesteps_tensor": timesteps_tensor,
                }
            )["output"].to(dtype=torch.float16)
            pred = self.sessions["action_decoder"].run(
                {
                    "model_output": model_output,
                    "embodiment_id": embodiment_id,
                }
            )["output"].to(dtype=torch.float16)

            pred_velocity = pred[:, -self.action_horizon :, :]
            actions = (actions + dt * pred_velocity).contiguous()

        return actions[:, :, : self.original_action_dim].to(dtype=torch.float32).contiguous()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
