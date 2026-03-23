#!/usr/bin/env python

import math

import pytest
import torch

from lerobot.policies.act.distillation_utils import (
    ACT_DECODER_FEATURE_SPACE_V1,
    ACTDecoderFeatureOutput,
    ACTForwardWithFeaturesOutput,
    compute_decoder_kd_gate,
    compute_decoder_kd_loss,
    compute_decoder_kd_ratios,
    compute_noise_to_signal_ratio,
    get_kd_temporal_weights,
)


def test_act_decoder_feature_output_validates_canonical_shape_and_metadata():
    decoder_out = torch.zeros(2, 3, 4)
    output = ACTDecoderFeatureOutput(
        decoder_out=decoder_out,
        feature_space=ACT_DECODER_FEATURE_SPACE_V1,
        latent_mode="zero",
        chunk_size=3,
        feature_dim=4,
    )

    assert output.decoder_out.shape == (2, 3, 4)
    assert output.feature_space == ACT_DECODER_FEATURE_SPACE_V1
    assert output.latent_mode == "zero"

    with pytest.raises(ValueError, match="feature_dim"):
        ACTDecoderFeatureOutput(
            decoder_out=decoder_out,
            feature_space=ACT_DECODER_FEATURE_SPACE_V1,
            latent_mode="zero",
            chunk_size=3,
            feature_dim=5,
        )


def test_act_forward_with_features_output_validates_matching_chunk_size():
    decoder_features = ACTDecoderFeatureOutput(
        decoder_out=torch.zeros(2, 3, 4),
        feature_space=ACT_DECODER_FEATURE_SPACE_V1,
        latent_mode="posterior",
        chunk_size=3,
        feature_dim=4,
    )
    output = ACTForwardWithFeaturesOutput(
        actions=torch.zeros(2, 3, 2),
        mu=torch.zeros(2, 8),
        log_sigma_x2=torch.zeros(2, 8),
        decoder_features=decoder_features,
    )

    assert output.actions.shape == (2, 3, 2)
    assert output.decoder_features is decoder_features

    with pytest.raises(ValueError, match="chunk_size"):
        ACTForwardWithFeaturesOutput(
            actions=torch.zeros(2, 2, 2),
            mu=None,
            log_sigma_x2=None,
            decoder_features=decoder_features,
        )


def test_get_kd_temporal_weights_normalizes_to_unit_mean():
    weights = get_kd_temporal_weights(
        3,
        math.log(2.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert weights.shape == (3,)
    assert weights[0] > weights[-1]
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_compute_decoder_kd_loss_applies_mask_overlap_and_prefix_tail_weighting():
    student = torch.zeros(1, 3, 1)
    teacher = torch.tensor([[[2.0], [1.0], [1.0]]])
    action_is_pad = torch.tensor([[False, True, False]])

    breakdown = compute_decoder_kd_loss(
        student_decoder_out=student,
        teacher_decoder_out=teacher,
        n_action_steps=1,
        action_is_pad=action_is_pad,
        overlap_steps=3,
        temporal_decay=0.0,
        prefix_weight=2.0,
        tail_weight=1.0,
        loss_type="mse",
    )

    assert breakdown.overlap_steps == 3
    assert breakdown.raw_loss.item() == pytest.approx(2.5)
    assert breakdown.weighted_loss.item() == pytest.approx(3.0)
    assert breakdown.prefix_loss.item() == pytest.approx(4.0)
    assert breakdown.tail_loss.item() == pytest.approx(1.0)
    assert breakdown.valid_ratio.item() == pytest.approx(2.0 / 3.0)


def test_compute_noise_to_signal_ratio_uses_masked_overlap_gap():
    ratio = compute_noise_to_signal_ratio(
        student_train_decoder_out=torch.tensor([[[2.0], [0.0]]]),
        student_eval_decoder_out=torch.tensor([[[1.0], [0.0]]]),
        teacher_eval_decoder_out=torch.tensor([[[0.0], [0.0]]]),
        n_action_steps=1,
        overlap_steps=2,
        loss_type="mse",
    )

    assert ratio.item() == pytest.approx(1.0)


def test_compute_noise_to_signal_ratio_returns_inf_when_teacher_signal_is_zero_but_noise_remains():
    ratio = compute_noise_to_signal_ratio(
        student_train_decoder_out=torch.tensor([[[1.0], [0.0]]]),
        student_eval_decoder_out=torch.tensor([[[0.0], [0.0]]]),
        teacher_eval_decoder_out=torch.tensor([[[0.0], [0.0]]]),
        n_action_steps=1,
        overlap_steps=2,
        loss_type="mse",
    )

    assert torch.isinf(ratio)
    assert ratio.item() > 0


def test_compute_decoder_kd_ratios_and_gate_helpers():
    ratios = compute_decoder_kd_ratios(
        weighted_decoder_kd_loss=torch.tensor(2.0),
        bc_loss=torch.tensor(4.0),
        action_kd_loss=torch.tensor(1.0),
        prefix_loss=torch.tensor(3.0),
        tail_loss=torch.tensor(1.0),
        noise_to_signal_ratio=torch.tensor(0.25),
    )

    assert ratios.weighted_to_bc_ratio.item() == pytest.approx(0.5)
    assert ratios.weighted_to_action_kd_ratio.item() == pytest.approx(2.0)
    assert ratios.prefix_to_tail_ratio.item() == pytest.approx(3.0)
    assert ratios.noise_to_signal_ratio.item() == pytest.approx(0.25)

    noise_blocked = compute_decoder_kd_gate(
        scheduler_weight=0.3,
        noise_to_signal_ratio=1.2,
        decoder_to_bc_grad_ratio=0.2,
        decoder_to_behavior_grad_ratio=0.2,
    )
    assert noise_blocked.grad_ratios_available is True
    assert noise_blocked.noise_gate_blocked is True
    assert noise_blocked.grad_gate_blocked is False
    assert noise_blocked.effective_weight.item() == pytest.approx(0.0)

    grad_blocked = compute_decoder_kd_gate(
        scheduler_weight=0.3,
        noise_to_signal_ratio=0.5,
        decoder_to_bc_grad_ratio=1.1,
        decoder_to_action_kd_grad_ratio=0.2,
        decoder_to_behavior_grad_ratio=0.8,
    )
    assert grad_blocked.grad_ratios_available is True
    assert grad_blocked.noise_gate_blocked is False
    assert grad_blocked.grad_gate_blocked is True
    assert grad_blocked.effective_weight.item() == pytest.approx(0.0)


def test_compute_decoder_kd_gate_blocks_when_grad_gate_is_enabled_without_real_ratios():
    grad_blocked = compute_decoder_kd_gate(
        scheduler_weight=0.3,
        noise_to_signal_ratio=0.2,
        enable_noise_gate=True,
        enable_grad_gate=True,
    )

    assert grad_blocked.grad_ratios_available is False
    assert grad_blocked.noise_gate_blocked is False
    assert grad_blocked.grad_gate_blocked is True
    assert grad_blocked.effective_weight.item() == pytest.approx(0.0)
    assert torch.isnan(grad_blocked.decoder_to_bc_grad_ratio)
    assert torch.isnan(grad_blocked.decoder_to_behavior_grad_ratio)
