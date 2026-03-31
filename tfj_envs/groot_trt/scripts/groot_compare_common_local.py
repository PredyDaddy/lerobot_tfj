#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from common import load_policy, resolve_policy_dir


EMBODIMENT_TAG_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}


class VisionModelForOnnx(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        vit_embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        dummy = position_ids.to(dtype=vit_embeds.dtype).sum() * 0.0
        return vit_embeds + dummy


class LanguageModelForOnnx(torch.nn.Module):
    def __init__(
        self,
        language_model: torch.nn.Module,
        eagle_linear: torch.nn.Module,
        select_layer: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.eagle_linear = eagle_linear
        self.select_layer = int(select_layer)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        neg_inf = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=neg_inf,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        valid_tokens = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
        causal_mask = causal_mask + (1.0 - valid_tokens) * neg_inf
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        embeddings = outputs.hidden_states[self.select_layer]
        return self.eagle_linear(embeddings)


class VLLNVLSelfAttention(torch.nn.Module):
    def __init__(self, vlln: torch.nn.Module, vl_self_attention: torch.nn.Module) -> None:
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))


class DiTForOnnx(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, sa_embs: torch.Tensor, vl_embs: torch.Tensor, timesteps_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
            return_all_hidden_states=False,
        )


@dataclass
class CompareContext:
    policy_dir: Path
    device: torch.device
    policy: Any
    backbone: Any
    action_head: Any
    vit_wrapper: VisionModelForOnnx
    llm_wrapper: LanguageModelForOnnx
    vlln_wrapper: VLLNVLSelfAttention
    dit_wrapper: DiTForOnnx
    num_patches: int
    image_token_index: int
    language_hidden_size: int
    original_action_dim: int
    action_horizon: int
    model_action_dim: int
    max_state_dim: int
    num_target_vision_tokens: int
    num_inference_timesteps: int
    num_timestep_buckets: int
    add_pos_embed: bool
    embodiment_id_value: int


def _resolve_num_patches(backbone: Any) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model"):
        return int(vision_model.vision_model.embeddings.num_patches)
    return int(vision_model.embeddings.num_patches)


def load_compare_context(policy_path: str | Path, device: str = "cuda") -> CompareContext:
    policy_dir = resolve_policy_dir(policy_path)
    _, _, policy = load_policy(policy_dir, device=device, strict=False)
    backbone = policy._groot_model.backbone
    action_head = policy._groot_model.action_head
    backbone.eval()
    action_head.eval()

    if hasattr(backbone.eagle_model.vision_model, "config"):
        backbone.eagle_model.vision_model.config._attn_implementation = "eager"
    if hasattr(backbone.eagle_model.language_model, "config"):
        backbone.eagle_model.language_model.config._attn_implementation = "eager"

    device_obj = torch.device(device)
    vit_wrapper = VisionModelForOnnx(backbone.eagle_model.vision_model).to(device=device_obj, dtype=torch.float16)
    llm_wrapper = LanguageModelForOnnx(
        backbone.eagle_model.language_model,
        backbone.eagle_linear,
        backbone.select_layer,
    ).to(device=device_obj, dtype=torch.float16)
    action_head = action_head.to(device=device_obj, dtype=torch.float16)
    vlln_wrapper = VLLNVLSelfAttention(action_head.vlln, action_head.vl_self_attention).to(
        device=device_obj, dtype=torch.float16
    )
    dit_wrapper = DiTForOnnx(action_head.model).to(device=device_obj, dtype=torch.float16)

    for module in (vit_wrapper, llm_wrapper, vlln_wrapper, dit_wrapper):
        module.eval()

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id_value = int(EMBODIMENT_TAG_MAPPING.get(embodiment_tag, 0))

    return CompareContext(
        policy_dir=policy_dir,
        device=device_obj,
        policy=policy,
        backbone=backbone,
        action_head=action_head,
        vit_wrapper=vit_wrapper,
        llm_wrapper=llm_wrapper,
        vlln_wrapper=vlln_wrapper,
        dit_wrapper=dit_wrapper,
        num_patches=_resolve_num_patches(backbone),
        image_token_index=int(backbone.eagle_model.image_token_index),
        language_hidden_size=int(backbone.eagle_model.language_model.config.hidden_size),
        original_action_dim=int(policy.config.output_features["action"].shape[0]),
        action_horizon=int(action_head.config.action_horizon),
        model_action_dim=int(action_head.config.action_dim),
        max_state_dim=int(action_head.config.max_state_dim),
        num_target_vision_tokens=int(action_head.config.num_target_vision_tokens),
        num_inference_timesteps=int(action_head.num_inference_timesteps),
        num_timestep_buckets=int(action_head.num_timestep_buckets),
        add_pos_embed=bool(getattr(action_head.config, "add_pos_embed", False)),
        embodiment_id_value=embodiment_id_value,
    )


def build_synthetic_inputs(
    ctx: CompareContext,
    *,
    seq_len: int,
    video_views: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if video_views <= 0:
        raise ValueError(f"video_views must be > 0, got {video_views}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(seed)

    pixel_values = torch.randn(
        (video_views, 3, 224, 224),
        dtype=torch.float16,
        device=ctx.device,
        generator=generator,
    ).contiguous()
    position_ids = torch.arange(ctx.num_patches, dtype=torch.int64, device=ctx.device).expand(video_views, -1).contiguous()

    text_token_id = 0 if ctx.image_token_index != 0 else 1
    image_token_slots = video_views * ctx.num_patches
    if image_token_slots > seq_len:
        raise ValueError(
            f"seq_len={seq_len} cannot hold image_token_slots={image_token_slots} for video_views={video_views}"
        )
    input_ids = torch.full((1, seq_len), fill_value=text_token_id, dtype=torch.int64, device=ctx.device)
    input_ids[:, :image_token_slots] = ctx.image_token_index
    attention_mask = torch.ones((1, seq_len), dtype=torch.int64, device=ctx.device)
    llm_direct_inputs_embeds = torch.randn(
        (1, seq_len, ctx.language_hidden_size),
        dtype=torch.float16,
        device=ctx.device,
        generator=generator,
    ).contiguous()

    state = torch.randn(
        (1, 1, ctx.max_state_dim),
        dtype=torch.float16,
        device=ctx.device,
        generator=generator,
    ).contiguous()
    embodiment_id = torch.full((1,), ctx.embodiment_id_value, dtype=torch.int64, device=ctx.device).contiguous()
    timestep0 = torch.zeros((1,), dtype=torch.int64, device=ctx.device).contiguous()
    initial_actions = torch.randn(
        (1, ctx.action_horizon, ctx.model_action_dim),
        dtype=torch.float16,
        device=ctx.device,
        generator=generator,
    ).contiguous()

    return {
        "pixel_values": pixel_values,
        "position_ids": position_ids,
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "llm_direct_inputs_embeds": llm_direct_inputs_embeds,
        "state": state,
        "embodiment_id": embodiment_id,
        "timestep0": timestep0,
        "initial_actions": initial_actions,
    }


def postprocess_vit(backbone: Any, vit_embeds: torch.Tensor, device: torch.device) -> torch.Tensor:
    vit_embeds = vit_embeds.to(
        device=device,
        dtype=next(backbone.eagle_model.mlp1.parameters()).dtype,
    )
    if bool(getattr(backbone.eagle_model, "use_pixel_shuffle", False)):
        num_tokens = int(vit_embeds.shape[1])
        grid = int(round(math.sqrt(num_tokens)))
        if grid * grid != num_tokens:
            raise ValueError(f"ViT token count {num_tokens} is not a square grid required by pixel shuffle.")
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], grid, grid, vit_embeds.shape[-1])
        vit_embeds = backbone.eagle_model.pixel_shuffle(
            vit_embeds,
            scale_factor=float(getattr(backbone.eagle_model, "downsample_ratio", 1.0)),
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = backbone.eagle_model.mlp1(vit_embeds)
    return vit_embeds.to(dtype=torch.float16).contiguous()


def build_inputs_embeds_from_vit(
    backbone: Any,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    embedding_layer = backbone.eagle_model.language_model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).to(device=device)
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(batch_size * seq_len, hidden_dim)
    flat_ids = input_ids.reshape(batch_size * seq_len)
    selected = flat_ids == int(backbone.eagle_model.image_token_index)
    num_selected = int(selected.sum().item())
    if num_selected == 0:
        raise RuntimeError("No image-token positions were found in synthetic input_ids.")
    flat_vit = vit_embeds.reshape(-1, hidden_dim).to(dtype=flat_embeds.dtype)
    if flat_vit.shape[0] < num_selected:
        raise RuntimeError(
            f"Not enough ViT embeddings for image-token replacement: have {flat_vit.shape[0]}, need {num_selected}"
        )
    flat_embeds[selected] = flat_embeds[selected] * 0.0 + flat_vit[:num_selected]
    return flat_embeds.reshape(batch_size, seq_len, hidden_dim).to(dtype=torch.float16).contiguous()


def apply_position_embedding(action_head: Any, action_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not bool(getattr(action_head.config, "add_pos_embed", False)):
        return action_features.contiguous()
    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
    pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0).to(dtype=action_features.dtype)
    return (action_features + pos_embs).contiguous()


def future_tokens(action_head: Any, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(
        device=device,
        dtype=dtype,
    ).contiguous()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def compute_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"Shape mismatch for metrics: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}"
        )
    ref = reference.detach().to(device="cpu", dtype=torch.float32)
    cand = candidate.detach().to(device="cpu", dtype=torch.float32)
    diff = cand - ref
    ref_flat = ref.reshape(-1).to(dtype=torch.float64)
    cand_flat = cand.reshape(-1).to(dtype=torch.float64)
    ref_norm = ref_flat.norm()
    cand_norm = cand_flat.norm()
    if float(ref_norm) == 0.0 or float(cand_norm) == 0.0:
        cosine = 1.0 if torch.equal(ref_flat, cand_flat) else 0.0
    else:
        cosine = float(torch.dot(ref_flat, cand_flat) / (ref_norm * cand_norm))
    return {
        "shape_ref": list(ref.shape),
        "shape_pred": list(cand.shape),
        "cosine": cosine,
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
    }


def build_runner_dict(ctx: CompareContext) -> dict[str, Callable[..., torch.Tensor]]:
    return {
        "vit": lambda pixel_values, position_ids: ctx.vit_wrapper(pixel_values, position_ids),
        "llm": lambda inputs_embeds, attention_mask: ctx.llm_wrapper(inputs_embeds, attention_mask),
        "vlln": lambda backbone_features: ctx.vlln_wrapper(backbone_features),
        "state_encoder": lambda state, embodiment_id: ctx.action_head.state_encoder(state, embodiment_id),
        "action_encoder": lambda actions, timesteps_tensor, embodiment_id: ctx.action_head.action_encoder(
            actions, timesteps_tensor, embodiment_id
        ),
        "dit": lambda sa_embs, vl_embs, timesteps_tensor: ctx.dit_wrapper(sa_embs, vl_embs, timesteps_tensor),
        "action_decoder": lambda model_output, embodiment_id: ctx.action_head.action_decoder(model_output, embodiment_id),
    }


@torch.inference_mode()
def run_compare_pipeline(
    ctx: CompareContext,
    runners: dict[str, Callable[..., torch.Tensor]],
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    vit_module = runners["vit"](inputs["pixel_values"], inputs["position_ids"]).to(
        device=ctx.device,
        dtype=torch.float16,
    )
    vit_postprocess = postprocess_vit(ctx.backbone, vit_module, ctx.device)
    inputs_embeds = build_inputs_embeds_from_vit(
        ctx.backbone,
        inputs["input_ids"],
        vit_postprocess,
        device=ctx.device,
    )
    llm_output = runners["llm"](inputs_embeds, inputs["attention_mask"]).to(device=ctx.device, dtype=torch.float16)
    llm_direct = runners["llm"](inputs["llm_direct_inputs_embeds"], inputs["attention_mask"]).to(
        device=ctx.device,
        dtype=torch.float16,
    )
    vl_embs = runners["vlln"](llm_output).to(device=ctx.device, dtype=torch.float16)
    state_features = runners["state_encoder"](inputs["state"], inputs["embodiment_id"]).to(
        device=ctx.device,
        dtype=torch.float16,
    )

    action_encoder_module = runners["action_encoder"](
        inputs["initial_actions"],
        inputs["timestep0"],
        inputs["embodiment_id"],
    ).to(device=ctx.device, dtype=torch.float16)
    action_features_t0 = apply_position_embedding(ctx.action_head, action_encoder_module, ctx.device)
    future_toks = future_tokens(ctx.action_head, batch_size=1, device=ctx.device, dtype=torch.float16)
    sa_embs_t0 = torch.cat((state_features, future_toks, action_features_t0), dim=1).contiguous()
    dit_module = runners["dit"](sa_embs_t0, vl_embs, inputs["timestep0"]).to(device=ctx.device, dtype=torch.float16)
    action_decoder_module = runners["action_decoder"](dit_module, inputs["embodiment_id"]).to(
        device=ctx.device,
        dtype=torch.float16,
    )

    actions = inputs["initial_actions"].clone().to(device=ctx.device, dtype=torch.float16).contiguous()
    dt = 1.0 / float(ctx.num_inference_timesteps)
    for step in range(ctx.num_inference_timesteps):
        t_cont = step / float(ctx.num_inference_timesteps)
        timestep_value = int(t_cont * ctx.num_timestep_buckets)
        timesteps_tensor = torch.full((1,), timestep_value, dtype=torch.int64, device=ctx.device)
        action_features = runners["action_encoder"](actions, timesteps_tensor, inputs["embodiment_id"]).to(
            device=ctx.device,
            dtype=torch.float16,
        )
        action_features = apply_position_embedding(ctx.action_head, action_features, ctx.device)
        sa_embs = torch.cat((state_features, future_toks, action_features), dim=1).contiguous()
        model_output = runners["dit"](sa_embs, vl_embs, timesteps_tensor).to(
            device=ctx.device,
            dtype=torch.float16,
        )
        pred = runners["action_decoder"](model_output, inputs["embodiment_id"]).to(
            device=ctx.device,
            dtype=torch.float16,
        )
        pred_velocity = pred[:, -ctx.action_horizon :, :]
        actions = (actions + dt * pred_velocity).contiguous()

    return {
        "vit": vit_module,
        "vit_postprocess": vit_postprocess,
        "llm_from_vit_pipeline": llm_output,
        "llm_direct": llm_direct,
        "action_vlln_vl_self_attention": vl_embs,
        "action_state_encoder": state_features,
        "action_action_encoder": action_encoder_module,
        "action_dit": dit_module,
        "action_decoder": action_decoder_module,
        "action_denoising_pipeline": actions[:, :, : ctx.original_action_dim].to(dtype=torch.float32).contiguous(),
    }


def compare_outputs(
    reference_outputs: dict[str, torch.Tensor],
    candidate_outputs: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    keys = sorted(reference_outputs.keys())
    if sorted(candidate_outputs.keys()) != keys:
        raise ValueError(
            f"Output key mismatch.\nreference={sorted(reference_outputs.keys())}\ncandidate={sorted(candidate_outputs.keys())}"
        )
    return {key: compute_metrics(reference_outputs[key], candidate_outputs[key]) for key in keys}
