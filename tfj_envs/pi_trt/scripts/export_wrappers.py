#!/usr/bin/env python3

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn


def _policy_compute_dtype(policy: Any) -> torch.dtype:
    return policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype


def _autocast_context(enabled: bool = True) -> Any:
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@contextmanager
def _force_pi05_cpu_sincos_float32() -> Any:
    from lerobot.policies.pi05 import modeling_pi05

    original_get_safe_dtype = modeling_pi05.get_safe_dtype

    def _patched_get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
        if device_type == "cpu" and target_dtype == torch.float64:
            return torch.float32
        return original_get_safe_dtype(target_dtype, device_type)

    modeling_pi05.get_safe_dtype = _patched_get_safe_dtype
    try:
        yield
    finally:
        modeling_pi05.get_safe_dtype = original_get_safe_dtype


def describe_execution_mode(policy: Any, *, use_autocast: bool) -> dict[str, Any]:
    parameter = next(policy.parameters())
    policy_device = parameter.device
    autocast_active = bool(use_autocast and policy_device.type == "cuda" and torch.cuda.is_available())
    return {
        "policy_device": str(policy_device),
        "policy_parameter_dtype": str(parameter.dtype).replace("torch.", ""),
        "compute_weight_dtype": str(_policy_compute_dtype(policy)).replace("torch.", ""),
        "use_autocast": bool(use_autocast),
        "autocast_active": autocast_active,
        "autocast_dtype": "bfloat16" if autocast_active else None,
    }


def prepare_policy_for_export(policy: Any) -> None:
    paligemma_with_expert = policy.model.paligemma_with_expert
    paligemma = paligemma_with_expert.paligemma

    # Force exporter-friendly attention kernels. The default SDPA path produces
    # ComplexDouble graph values in this environment during legacy ONNX export.
    paligemma.model.vision_tower.vision_model.config._attn_implementation = "eager"
    paligemma.language_model.config._attn_implementation = "eager"
    paligemma.config.vision_config._attn_implementation = "eager"
    paligemma.config.text_config._attn_implementation = "eager"
    paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"


def _make_att_2d_masks(policy: Any, pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    helper = getattr(policy, "modeling_make_att_2d_masks", None)
    if helper is None:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        helper = make_att_2d_masks
    if att_masks.dtype == torch.bool:
        att_masks = att_masks.to(dtype=torch.int64)
    return helper(pad_masks, att_masks)


def cache_tensor_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"past_key_values.layer_{layer_idx:02d}.key")
        names.append(f"past_key_values.layer_{layer_idx:02d}.value")
    return names


class Pi05VisionEncoderExportWrapper(nn.Module):
    def __init__(self, policy: Any, *, use_autocast: bool = True):
        super().__init__()
        self.policy = policy
        self.use_autocast = bool(use_autocast)
        prepare_policy_for_export(self.policy)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with _autocast_context(self.use_autocast):
            image_embs = self.policy.model.paligemma_with_expert.embed_image(image)
        return image_embs.float()


class Pi05PrefixCacheExportWrapper(nn.Module):
    def __init__(self, policy: Any, *, num_layers: int, use_autocast: bool = True):
        super().__init__()
        self.policy = policy
        self.num_layers = int(num_layers)
        self.use_autocast = bool(use_autocast)
        prepare_policy_for_export(self.policy)

    def forward(
        self,
        image_embs_top: torch.Tensor,
        image_embs_wrist: torch.Tensor,
        image_mask_top: torch.Tensor,
        image_mask_wrist: torch.Tensor,
        tokens: torch.Tensor,
        token_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        compute_dtype = _policy_compute_dtype(self.policy)
        image_embs_top = image_embs_top.to(dtype=compute_dtype)
        image_embs_wrist = image_embs_wrist.to(dtype=compute_dtype)
        image_mask_top = image_mask_top.to(dtype=torch.bool)
        image_mask_wrist = image_mask_wrist.to(dtype=torch.bool)
        token_attention_mask = token_attention_mask.to(dtype=torch.bool)

        with _autocast_context(self.use_autocast):
            lang_emb = self.policy.model.paligemma_with_expert.embed_language_tokens(tokens)
            hidden_size = int(self.policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
            lang_emb = lang_emb * math.sqrt(hidden_size)

            batch_size, top_tokens = image_embs_top.shape[:2]
            wrist_tokens = image_embs_wrist.shape[1]
            prefix_embs = torch.cat([image_embs_top, image_embs_wrist, lang_emb], dim=1)
            prefix_pad_masks = torch.cat(
                [
                    image_mask_top[:, None].expand(batch_size, top_tokens),
                    image_mask_wrist[:, None].expand(batch_size, wrist_tokens),
                    token_attention_mask,
                ],
                dim=1,
            )
            prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
            prefix_att_2d_masks = self.policy.model._prepare_attention_masks_4d(
                _make_att_2d_masks(self.policy, prefix_pad_masks, prefix_att_masks)
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1
            _, past_key_values = self.policy.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        outputs: list[torch.Tensor] = [prefix_pad_masks.to(dtype=torch.int32)]
        for layer_idx in range(self.num_layers):
            key, value = past_key_values[layer_idx]
            outputs.append(key.float())
            outputs.append(value.float())
        return tuple(outputs)


class Pi05DenoiseStepExportWrapper(nn.Module):
    def __init__(self, policy: Any, *, num_layers: int, dynamic_cache_cls: Any, use_autocast: bool = True):
        super().__init__()
        self.policy = policy
        self.num_layers = int(num_layers)
        self.dynamic_cache_cls = dynamic_cache_cls
        self.use_autocast = bool(use_autocast)
        prepare_policy_for_export(self.policy)

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> torch.Tensor:
        compute_dtype = _policy_compute_dtype(self.policy)
        prefix_pad_masks = prefix_pad_masks.to(dtype=torch.bool)
        legacy_cache = []
        for layer_idx in range(self.num_layers):
            key = past_key_values_flat[layer_idx * 2].to(dtype=compute_dtype)
            value = past_key_values_flat[layer_idx * 2 + 1].to(dtype=compute_dtype)
            legacy_cache.append((key, value))
        past_key_values = self.dynamic_cache_cls.from_legacy_cache(tuple(legacy_cache))

        with _force_pi05_cpu_sincos_float32(), _autocast_context(self.use_autocast):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.policy.model.embed_suffix(
                x_t,
                timestep,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = _make_att_2d_masks(self.policy, suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks.to(dtype=torch.int64), dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1

            full_att_2d_masks_4d = self.policy.model._prepare_attention_masks_4d(full_att_2d_masks)
            self.policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.policy.model.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.policy.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.policy.model.action_out_proj(suffix_out)
        return v_t.float()
