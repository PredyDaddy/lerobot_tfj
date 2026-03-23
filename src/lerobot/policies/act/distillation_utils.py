#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy


KD_COMPARISON_SPACE_NORMALIZED_ACTION = "normalized_action_space"
ACT_DECODER_FEATURE_SPACE_V1 = "act_decoder_out_v1"


@dataclass(frozen=True)
class KDProcessorCompatibilityReport:
    compatible: bool
    comparison_space: str = KD_COMPARISON_SPACE_NORMALIZED_ACTION
    reason: str | None = None

    def require_compatible(self) -> None:
        if self.comparison_space != KD_COMPARISON_SPACE_NORMALIZED_ACTION:
            raise ValueError(
                "Stage 1 ACT KD only supports `normalized_action_space` comparison. "
                f"Got `{self.comparison_space}`."
            )
        if not self.compatible:
            message = self.reason or "Teacher and student ACT processors are not compatible for KD."
            raise ValueError(message)


@dataclass(frozen=True)
class ACTTeacherBundle:
    policy: ACTPolicy
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
    processor_compatibility: KDProcessorCompatibilityReport | None = None
    comparison_space: str = KD_COMPARISON_SPACE_NORMALIZED_ACTION
    resolved_policy_path: Path | None = None


@dataclass(frozen=True)
class KDLossBreakdown:
    loss: Tensor
    masked_mean: Tensor
    valid_ratio: Tensor
    prefix_l1_loss: Tensor
    tail_l1_loss: Tensor
    overlap_steps: int


@dataclass(frozen=True)
class KDSegmentLayout:
    overlap_steps: int
    prefix_steps: int
    tail_steps: int


@dataclass(frozen=True)
class ACTDecoderFeatureOutput:
    decoder_out: Tensor
    feature_space: Literal["act_decoder_out_v1"] = ACT_DECODER_FEATURE_SPACE_V1
    latent_mode: Literal["posterior", "zero"] = "zero"
    chunk_size: int = 0
    feature_dim: int = 0

    def __post_init__(self) -> None:
        if self.feature_space != ACT_DECODER_FEATURE_SPACE_V1:
            raise ValueError(
                "ACT decoder feature outputs must use feature_space="
                f"`{ACT_DECODER_FEATURE_SPACE_V1}`. Got `{self.feature_space}`."
            )
        if self.decoder_out.ndim != 3:
            raise ValueError(
                "ACT decoder features must be rank-3 tensors with shape (B, chunk_size, dim_model). "
                f"Got ndim={self.decoder_out.ndim} and shape={tuple(self.decoder_out.shape)}."
            )
        if self.chunk_size != self.decoder_out.shape[1]:
            raise ValueError(
                "ACT decoder feature metadata chunk_size must match decoder_out.shape[1]. "
                f"Got chunk_size={self.chunk_size} and shape[1]={self.decoder_out.shape[1]}."
            )
        if self.feature_dim != self.decoder_out.shape[2]:
            raise ValueError(
                "ACT decoder feature metadata feature_dim must match decoder_out.shape[2]. "
                f"Got feature_dim={self.feature_dim} and shape[2]={self.decoder_out.shape[2]}."
            )


@dataclass(frozen=True)
class ACTForwardWithFeaturesOutput:
    actions: Tensor
    mu: Tensor | None
    log_sigma_x2: Tensor | None
    decoder_features: ACTDecoderFeatureOutput | None

    def __post_init__(self) -> None:
        if self.actions.ndim != 3:
            raise ValueError(
                "ACT forward outputs must expose actions with shape (B, chunk_size, action_dim). "
                f"Got ndim={self.actions.ndim} and shape={tuple(self.actions.shape)}."
            )
        if (self.mu is None) != (self.log_sigma_x2 is None):
            raise ValueError("`mu` and `log_sigma_x2` must either both be set or both be None.")
        if self.decoder_features is not None and self.decoder_features.chunk_size != self.actions.shape[1]:
            raise ValueError(
                "ACT forward feature outputs must share the same chunk_size as actions. "
                f"Got feature chunk_size={self.decoder_features.chunk_size} and actions shape[1]={self.actions.shape[1]}."
            )


@dataclass(frozen=True)
class DecoderKDLossBreakdown:
    raw_loss: Tensor
    weighted_loss: Tensor
    valid_ratio: Tensor
    prefix_loss: Tensor
    tail_loss: Tensor
    overlap_steps: int


@dataclass(frozen=True)
class DecoderKDRatioBreakdown:
    weighted_to_bc_ratio: Tensor
    weighted_to_action_kd_ratio: Tensor
    prefix_to_tail_ratio: Tensor
    noise_to_signal_ratio: Tensor


@dataclass(frozen=True)
class DecoderKDGateBreakdown:
    scheduler_weight: Tensor
    gate_multiplier: Tensor
    effective_weight: Tensor
    noise_gate_blocked: bool
    grad_gate_blocked: bool
    grad_ratios_available: bool
    decoder_to_bc_grad_ratio: Tensor
    decoder_to_action_kd_grad_ratio: Tensor
    decoder_to_behavior_grad_ratio: Tensor


def get_kd_segment_layout(
    *,
    overlap_steps: int,
    n_action_steps: int,
    kd_prefix_weight: float,
    kd_tail_weight: float,
) -> KDSegmentLayout:
    if overlap_steps <= 0:
        raise ValueError("KD overlap must be strictly positive.")

    prefix_steps = min(overlap_steps, n_action_steps)
    tail_steps = max(0, overlap_steps - prefix_steps)

    if kd_prefix_weight == 0 and kd_tail_weight > 0 and tail_steps == 0:
        raise ValueError(
            "Tail-only ACT KD requires at least one tail step beyond `n_action_steps`, "
            f"but the resolved overlap is {overlap_steps} and `n_action_steps` is {n_action_steps}."
        )

    return KDSegmentLayout(
        overlap_steps=overlap_steps,
        prefix_steps=prefix_steps,
        tail_steps=tail_steps,
    )


def masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape != values.shape:
        mask = torch.broadcast_to(mask, values.shape)

    valid_ratio = mask.mean()
    combined_weights = mask

    if weights is not None:
        weights = weights.to(device=values.device, dtype=values.dtype)
        if weights.shape != values.shape:
            weights = torch.broadcast_to(weights, values.shape)
        combined_weights = combined_weights * weights

    denominator = combined_weights.sum()
    if denominator.item() == 0:
        return values.new_zeros(()), valid_ratio

    return (values * combined_weights).sum() / denominator, valid_ratio


def get_kd_temporal_weights(
    overlap_steps: int,
    temporal_decay: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if overlap_steps <= 0:
        raise ValueError("KD overlap must be strictly positive.")
    if temporal_decay < 0:
        raise ValueError(f"`temporal_decay` must be non-negative. Got {temporal_decay}.")
    if temporal_decay == 0:
        return torch.ones(overlap_steps, device=device, dtype=dtype)

    positions = torch.arange(overlap_steps, device=device, dtype=dtype)
    weights = torch.exp(-temporal_decay * positions)
    return weights / weights.mean()


def get_kd_segment_weights(
    *,
    overlap_steps: int,
    n_action_steps: int,
    kd_prefix_weight: float,
    kd_tail_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    segment_layout = get_kd_segment_layout(
        overlap_steps=overlap_steps,
        n_action_steps=n_action_steps,
        kd_prefix_weight=kd_prefix_weight,
        kd_tail_weight=kd_tail_weight,
    )
    weights = torch.full(
        (overlap_steps,),
        fill_value=kd_tail_weight,
        device=device,
        dtype=dtype,
    )
    weights[: segment_layout.prefix_steps] = kd_prefix_weight
    return weights


def safe_ratio(
    numerator: Tensor | float | int | None,
    denominator: Tensor | float | int | None,
    *,
    reference: Tensor | None = None,
    eps: float = 1e-8,
    zero_denominator: Literal["zero", "infinity"] = "zero",
) -> Tensor:
    if reference is None:
        if isinstance(numerator, Tensor):
            reference = numerator
        elif isinstance(denominator, Tensor):
            reference = denominator
        else:
            reference = torch.tensor(0.0, dtype=torch.float32)

    ratio = reference.new_zeros(())
    if numerator is None or denominator is None:
        return ratio

    numerator_t = torch.as_tensor(numerator, device=reference.device, dtype=reference.dtype)
    denominator_t = torch.as_tensor(denominator, device=reference.device, dtype=reference.dtype)
    if denominator_t.abs().item() <= eps:
        if zero_denominator == "infinity" and numerator_t.abs().item() > eps:
            return reference.new_full((), float("inf"))
        return ratio
    return numerator_t / denominator_t


def _resolve_hidden_overlap(
    student_decoder_out: Tensor,
    teacher_decoder_out: Tensor,
    *,
    overlap_steps: int | None,
) -> int:
    resolved_overlap = min(student_decoder_out.shape[1], teacher_decoder_out.shape[1])
    if overlap_steps is not None:
        if overlap_steps <= 0:
            raise ValueError(f"`overlap_steps` must be strictly positive when provided. Got {overlap_steps}.")
        resolved_overlap = min(resolved_overlap, overlap_steps)
    if resolved_overlap <= 0:
        raise ValueError("Decoder KD overlap is empty. Check feature chunk sizes and overlap_steps.")
    return resolved_overlap


def _resolve_hidden_mask(
    *,
    feature_error: Tensor,
    action_is_pad: Tensor | None,
    overlap_steps: int,
) -> Tensor:
    if action_is_pad is None:
        return torch.ones_like(feature_error)
    return (~action_is_pad[:, :overlap_steps]).unsqueeze(-1)


def _compute_hidden_elementwise_loss(
    student_decoder_out: Tensor,
    teacher_decoder_out: Tensor,
    *,
    loss_type: Literal["smooth_l1", "mse"],
    smooth_l1_beta: float,
) -> Tensor:
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(
            student_decoder_out,
            teacher_decoder_out,
            beta=smooth_l1_beta,
            reduction="none",
        )
    if loss_type == "mse":
        return F.mse_loss(student_decoder_out, teacher_decoder_out, reduction="none")
    raise ValueError(f"Unsupported decoder KD loss_type `{loss_type}`.")


def compute_decoder_kd_loss(
    *,
    student_decoder_out: Tensor,
    teacher_decoder_out: Tensor,
    n_action_steps: int,
    action_is_pad: Tensor | None = None,
    overlap_steps: int | None = None,
    temporal_decay: float = 0.0,
    prefix_weight: float = 1.0,
    tail_weight: float = 1.0,
    loss_type: Literal["smooth_l1", "mse"] = "smooth_l1",
    smooth_l1_beta: float = 1.0,
) -> DecoderKDLossBreakdown:
    resolved_overlap = _resolve_hidden_overlap(
        student_decoder_out,
        teacher_decoder_out,
        overlap_steps=overlap_steps,
    )
    elementwise_loss = _compute_hidden_elementwise_loss(
        student_decoder_out[:, :resolved_overlap],
        teacher_decoder_out[:, :resolved_overlap],
        loss_type=loss_type,
        smooth_l1_beta=smooth_l1_beta,
    )
    mask = _resolve_hidden_mask(
        feature_error=elementwise_loss,
        action_is_pad=action_is_pad,
        overlap_steps=resolved_overlap,
    )
    temporal_weights = get_kd_temporal_weights(
        resolved_overlap,
        temporal_decay,
        device=elementwise_loss.device,
        dtype=elementwise_loss.dtype,
    )
    segment_weights = get_kd_segment_weights(
        overlap_steps=resolved_overlap,
        n_action_steps=n_action_steps,
        kd_prefix_weight=prefix_weight,
        kd_tail_weight=tail_weight,
        device=elementwise_loss.device,
        dtype=elementwise_loss.dtype,
    )
    loss_weights = temporal_weights.view(1, resolved_overlap, 1) * segment_weights.view(1, resolved_overlap, 1)
    weighted_loss, _ = masked_mean(elementwise_loss, mask, weights=loss_weights)
    raw_loss, valid_ratio = masked_mean(elementwise_loss, mask)

    segment_layout = get_kd_segment_layout(
        overlap_steps=resolved_overlap,
        n_action_steps=n_action_steps,
        kd_prefix_weight=prefix_weight,
        kd_tail_weight=tail_weight,
    )
    prefix_loss, _ = masked_mean(
        elementwise_loss[:, : segment_layout.prefix_steps],
        mask[:, : segment_layout.prefix_steps],
    )
    if segment_layout.tail_steps > 0:
        tail_loss, _ = masked_mean(
            elementwise_loss[:, segment_layout.prefix_steps :],
            mask[:, segment_layout.prefix_steps :],
        )
    else:
        tail_loss = elementwise_loss.new_zeros(())

    return DecoderKDLossBreakdown(
        raw_loss=raw_loss,
        weighted_loss=weighted_loss,
        valid_ratio=valid_ratio,
        prefix_loss=prefix_loss,
        tail_loss=tail_loss,
        overlap_steps=resolved_overlap,
    )


def compute_noise_to_signal_ratio(
    *,
    student_train_decoder_out: Tensor,
    student_eval_decoder_out: Tensor,
    teacher_eval_decoder_out: Tensor,
    n_action_steps: int,
    action_is_pad: Tensor | None = None,
    overlap_steps: int | None = None,
    loss_type: Literal["smooth_l1", "mse"] = "smooth_l1",
    smooth_l1_beta: float = 1.0,
) -> Tensor:
    train_eval_gap = compute_decoder_kd_loss(
        student_decoder_out=student_train_decoder_out,
        teacher_decoder_out=student_eval_decoder_out,
        n_action_steps=n_action_steps,
        action_is_pad=action_is_pad,
        overlap_steps=overlap_steps,
        loss_type=loss_type,
        smooth_l1_beta=smooth_l1_beta,
    ).raw_loss
    eval_teacher_gap = compute_decoder_kd_loss(
        student_decoder_out=student_eval_decoder_out,
        teacher_decoder_out=teacher_eval_decoder_out,
        n_action_steps=n_action_steps,
        action_is_pad=action_is_pad,
        overlap_steps=overlap_steps,
        loss_type=loss_type,
        smooth_l1_beta=smooth_l1_beta,
    ).raw_loss
    return safe_ratio(
        train_eval_gap,
        eval_teacher_gap,
        reference=train_eval_gap,
        zero_denominator="infinity",
    )


def compute_decoder_kd_ratios(
    *,
    weighted_decoder_kd_loss: Tensor | float,
    bc_loss: Tensor | float | None,
    action_kd_loss: Tensor | float | None = None,
    prefix_loss: Tensor | float | None = None,
    tail_loss: Tensor | float | None = None,
    noise_to_signal_ratio: Tensor | float | None = None,
) -> DecoderKDRatioBreakdown:
    reference = torch.as_tensor(weighted_decoder_kd_loss, dtype=torch.float32)
    return DecoderKDRatioBreakdown(
        weighted_to_bc_ratio=safe_ratio(weighted_decoder_kd_loss, bc_loss, reference=reference),
        weighted_to_action_kd_ratio=safe_ratio(weighted_decoder_kd_loss, action_kd_loss, reference=reference),
        prefix_to_tail_ratio=safe_ratio(prefix_loss, tail_loss, reference=reference),
        noise_to_signal_ratio=(
            torch.as_tensor(noise_to_signal_ratio, device=reference.device, dtype=reference.dtype)
            if noise_to_signal_ratio is not None
            else reference.new_zeros(())
        ),
    )


def compute_decoder_kd_gate(
    *,
    scheduler_weight: Tensor | float,
    noise_to_signal_ratio: Tensor | float | None = None,
    decoder_to_bc_grad_ratio: Tensor | float | None = None,
    decoder_to_action_kd_grad_ratio: Tensor | float | None = None,
    decoder_to_behavior_grad_ratio: Tensor | float | None = None,
    enable_noise_gate: bool = True,
    enable_grad_gate: bool = True,
) -> DecoderKDGateBreakdown:
    scheduler_weight_t = torch.as_tensor(scheduler_weight, dtype=torch.float32)
    noise_ratio_t = (
        torch.as_tensor(noise_to_signal_ratio, device=scheduler_weight_t.device, dtype=scheduler_weight_t.dtype)
        if noise_to_signal_ratio is not None
        else scheduler_weight_t.new_zeros(())
    )
    bc_grad_ratio_t = (
        torch.as_tensor(decoder_to_bc_grad_ratio, device=scheduler_weight_t.device, dtype=scheduler_weight_t.dtype)
        if decoder_to_bc_grad_ratio is not None
        else scheduler_weight_t.new_full((), float("nan"))
    )
    action_grad_ratio_t = (
        torch.as_tensor(
            decoder_to_action_kd_grad_ratio,
            device=scheduler_weight_t.device,
            dtype=scheduler_weight_t.dtype,
        )
        if decoder_to_action_kd_grad_ratio is not None
        else scheduler_weight_t.new_full((), float("nan"))
    )
    behavior_grad_ratio_t = (
        torch.as_tensor(
            decoder_to_behavior_grad_ratio,
            device=scheduler_weight_t.device,
            dtype=scheduler_weight_t.dtype,
        )
        if decoder_to_behavior_grad_ratio is not None
        else scheduler_weight_t.new_full((), float("nan"))
    )
    grad_ratios_available = decoder_to_bc_grad_ratio is not None and decoder_to_behavior_grad_ratio is not None

    noise_gate_blocked = bool(enable_noise_gate and noise_ratio_t.item() >= 1.0)
    grad_gate_blocked = bool(
        enable_grad_gate
        and (
            not grad_ratios_available
            or bc_grad_ratio_t.item() > 1.0
            or behavior_grad_ratio_t.item() >= 1.0
        )
    )
    gate_multiplier = scheduler_weight_t.new_zeros(()) if (noise_gate_blocked or grad_gate_blocked) else scheduler_weight_t.new_ones(())
    effective_weight = scheduler_weight_t * gate_multiplier

    return DecoderKDGateBreakdown(
        scheduler_weight=scheduler_weight_t,
        gate_multiplier=gate_multiplier,
        effective_weight=effective_weight,
        noise_gate_blocked=noise_gate_blocked,
        grad_gate_blocked=grad_gate_blocked,
        grad_ratios_available=grad_ratios_available,
        decoder_to_bc_grad_ratio=bc_grad_ratio_t,
        decoder_to_action_kd_grad_ratio=action_grad_ratio_t,
        decoder_to_behavior_grad_ratio=behavior_grad_ratio_t,
    )


def load_act_teacher_bundle(
    *,
    student_policy: "ACTPolicy",
    student_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    teacher_pretrained_path: str | Path,
) -> ACTTeacherBundle:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import assert_act_kd_processor_compatibility

    teacher_policy = ACTPolicy.from_pretrained(teacher_pretrained_path)
    teacher_policy.requires_grad_(False)
    teacher_policy.eval()
    teacher_preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=teacher_pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    processor_compatibility = assert_act_kd_processor_compatibility(
        student_config=student_policy.config,
        student_preprocessor=student_preprocessor,
        teacher_config=teacher_policy.config,
        teacher_preprocessor=teacher_preprocessor,
    )
    return ACTTeacherBundle(
        policy=teacher_policy,
        preprocessor=teacher_preprocessor,
        processor_compatibility=processor_compatibility,
        resolved_policy_path=Path(teacher_pretrained_path),
    )
