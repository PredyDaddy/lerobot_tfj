#!/usr/bin/env python

import math
from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.distillation_utils import (
    ACTDecoderFeatureOutput,
    ACTForwardWithFeaturesOutput,
    ACTTeacherBundle,
    KDProcessorCompatibilityReport,
    compute_decoder_kd_loss,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


INPUT_FEATURES = {
    OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
}
OUTPUT_FEATURES = {
    ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
}


def _make_config(
    *,
    noise_gate: bool = True,
    grad_gate: bool = False,
    start_step: int = 0,
    ramp_steps: int = 0,
) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        use_vae=False,
        pretrained_backbone_weights=None,
        chunk_size=3,
        n_action_steps=2,
        dim_model=4,
        n_heads=4,
        dim_feedforward=16,
        n_encoder_layers=1,
        n_decoder_layers=1,
        kd=True,
        teacher_policy_path=Path("/tmp/fake-teacher"),
        kd_weight=1.5,
        kd_overlap_steps=3,
        kd_prefix_weight=1.0,
        kd_tail_weight=1.0,
        decoder_kd={
            "enabled": True,
            "peak_weight": 0.4,
            "loss_type": "mse",
            "start_step": start_step,
            "ramp_steps": ramp_steps,
            "enable_noise_gate": noise_gate,
            "enable_grad_gate": grad_gate,
        },
    )


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_ENV_STATE: torch.zeros(1, 4),
        ACTION: torch.tensor([[[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]], dtype=torch.float32),
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
    }


class _FakeTeacherPolicy:
    def __init__(self, config: ACTConfig, *, teacher_actions: torch.Tensor, teacher_decoder_out: torch.Tensor):
        self.config = ACTConfig(
            device="cpu",
            push_to_hub=False,
            input_features=INPUT_FEATURES,
            output_features=OUTPUT_FEATURES,
            use_vae=False,
            pretrained_backbone_weights=None,
            chunk_size=3,
            n_action_steps=2,
            dim_model=4,
            n_heads=4,
            dim_feedforward=16,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        self._teacher_actions = teacher_actions
        self._teacher_decoder_out = teacher_decoder_out

    def requires_grad_(self, _requires_grad):
        return self

    def eval(self):
        return self

    def to(self, _device):
        return self

    def predict_action_chunk(self, _batch):
        return self._teacher_actions

    def get_decoder_features(self, _batch, *, latent_mode="zero"):
        return ACTDecoderFeatureOutput(
            decoder_out=self._teacher_decoder_out,
            latent_mode=latent_mode,
            chunk_size=self._teacher_decoder_out.shape[1],
            feature_dim=self._teacher_decoder_out.shape[2],
        )


def _attach_teacher(policy: ACTPolicy, *, teacher_actions: torch.Tensor, teacher_decoder_out: torch.Tensor) -> None:
    teacher_policy = _FakeTeacherPolicy(
        policy.config,
        teacher_actions=teacher_actions,
        teacher_decoder_out=teacher_decoder_out,
    )
    policy.attach_teacher_bundle(
        ACTTeacherBundle(
            policy=teacher_policy,
            processor_compatibility=KDProcessorCompatibilityReport(compatible=True),
        )
    )


def test_policy_forward_adds_decoder_kd_metrics_without_polluting_phase_one_keys(monkeypatch):
    config = _make_config(noise_gate=True, grad_gate=False)
    policy = ACTPolicy(config)
    batch = _make_batch()

    student_actions = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]], dtype=torch.float32)
    teacher_actions = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)
    student_train_decoder_out = torch.tensor(
        [[[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32
    )
    student_eval_decoder_out = torch.tensor(
        [[[0.5, 0.5, 0.5, 0.5], [0.75, 0.75, 0.75, 0.75], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32
    )
    teacher_decoder_out = torch.zeros_like(student_train_decoder_out)

    _attach_teacher(policy, teacher_actions=teacher_actions, teacher_decoder_out=teacher_decoder_out)

    monkeypatch.setattr(policy.model, "forward", lambda _batch: (student_actions, (None, None)))

    def fake_forward_with_features(_batch, *, latent_mode="auto", return_decoder_out=False):
        decoder_out = student_train_decoder_out if policy.model.training else student_eval_decoder_out
        decoder_features = None
        if return_decoder_out:
            decoder_features = ACTDecoderFeatureOutput(
                decoder_out=decoder_out,
                latent_mode="zero" if latent_mode != "posterior" else "posterior",
                chunk_size=decoder_out.shape[1],
                feature_dim=decoder_out.shape[2],
            )
        return ACTForwardWithFeaturesOutput(
            actions=student_actions,
            mu=None,
            log_sigma_x2=None,
            decoder_features=decoder_features,
        )

    monkeypatch.setattr(policy.model, "forward_with_features", fake_forward_with_features)

    loss, loss_dict = policy.forward(batch)

    bc_loss = (torch.abs(batch[ACTION] - student_actions) * ~batch["action_is_pad"].unsqueeze(-1)).mean()
    action_kd_loss = torch.abs(student_actions - teacher_actions).mean()
    decoder_breakdown = compute_decoder_kd_loss(
        student_decoder_out=student_train_decoder_out,
        teacher_decoder_out=teacher_decoder_out,
        n_action_steps=policy.config.n_action_steps,
        action_is_pad=batch["action_is_pad"],
        overlap_steps=policy.config.decoder_kd.overlap_steps,
        temporal_decay=policy.config.kd_temporal_decay,
        prefix_weight=policy.config.kd_prefix_weight,
        tail_weight=policy.config.kd_tail_weight,
        loss_type=policy.config.decoder_kd.loss_type,
        smooth_l1_beta=policy.config.decoder_kd.smooth_l1_beta,
    )
    expected_loss = bc_loss + policy.config.kd_weight * action_kd_loss + 0.4 * decoder_breakdown.weighted_loss

    torch.testing.assert_close(loss, expected_loss)
    assert loss_dict["kd_weighted_l1_loss"] == pytest.approx(action_kd_loss.item())
    assert loss_dict["decoder_kd_weighted_loss"] == pytest.approx(decoder_breakdown.weighted_loss.item())
    assert loss_dict["decoder_kd_effective_weight"] == pytest.approx(0.4)
    assert loss_dict["decoder_kd_noise_gate_blocked"] == pytest.approx(0.0)
    assert loss_dict["decoder_kd_grad_ratio_available"] == pytest.approx(0.0)
    assert "kd_l1_loss" in loss_dict
    assert "decoder_kd_loss" in loss_dict
    assert "decoder_kd_weighted_to_bc_ratio" in loss_dict
    assert "decoder_kd_weighted_to_action_kd_ratio" in loss_dict


def test_policy_forward_blocks_decoder_kd_when_noise_gate_trips(monkeypatch):
    config = _make_config(noise_gate=True, grad_gate=False)
    policy = ACTPolicy(config)
    batch = _make_batch()

    student_actions = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]], dtype=torch.float32)
    teacher_actions = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)
    student_train_decoder_out = torch.ones(1, 3, 4, dtype=torch.float32)
    student_eval_decoder_out = torch.zeros(1, 3, 4, dtype=torch.float32)
    teacher_decoder_out = torch.full((1, 3, 4), 0.25, dtype=torch.float32)

    _attach_teacher(policy, teacher_actions=teacher_actions, teacher_decoder_out=teacher_decoder_out)
    monkeypatch.setattr(policy.model, "forward", lambda _batch: (student_actions, (None, None)))

    def fake_forward_with_features(_batch, *, latent_mode="auto", return_decoder_out=False):
        decoder_out = student_train_decoder_out if policy.model.training else student_eval_decoder_out
        decoder_features = None
        if return_decoder_out:
            decoder_features = ACTDecoderFeatureOutput(
                decoder_out=decoder_out,
                latent_mode="zero",
                chunk_size=decoder_out.shape[1],
                feature_dim=decoder_out.shape[2],
            )
        return ACTForwardWithFeaturesOutput(
            actions=student_actions,
            mu=None,
            log_sigma_x2=None,
            decoder_features=decoder_features,
        )

    monkeypatch.setattr(policy.model, "forward_with_features", fake_forward_with_features)

    loss, loss_dict = policy.forward(batch)

    bc_loss = (torch.abs(batch[ACTION] - student_actions) * ~batch["action_is_pad"].unsqueeze(-1)).mean()
    action_kd_loss = torch.abs(student_actions - teacher_actions).mean()
    expected_loss = bc_loss + policy.config.kd_weight * action_kd_loss

    torch.testing.assert_close(loss, expected_loss)
    assert loss_dict["decoder_kd_noise_gate_blocked"] == pytest.approx(1.0)
    assert loss_dict["decoder_kd_effective_weight"] == pytest.approx(0.0)
    assert loss_dict["noise_to_signal_ratio"] >= 1.0


def test_policy_forward_reports_real_grad_ratios_when_grad_gate_is_enabled():
    torch.manual_seed(0)
    config = _make_config(noise_gate=False, grad_gate=True)
    policy = ACTPolicy(config)
    batch = _make_batch()

    teacher_actions = torch.zeros(1, 3, 2, dtype=torch.float32)
    teacher_decoder_out = torch.zeros(1, 3, 4, dtype=torch.float32)
    _attach_teacher(policy, teacher_actions=teacher_actions, teacher_decoder_out=teacher_decoder_out)

    loss, loss_dict = policy.forward(batch)

    assert loss.requires_grad is True
    assert loss_dict["decoder_kd_grad_ratio_available"] == pytest.approx(1.0)
    assert math.isnan(loss_dict["decoder_grad_to_bc_grad_ratio"]) is False
    assert math.isnan(loss_dict["decoder_grad_to_behavior_grad_ratio"]) is False
    assert loss_dict["decoder_grad_to_bc_grad_ratio"] >= 0.0
    assert loss_dict["decoder_grad_to_behavior_grad_ratio"] >= 0.0
    assert policy._get_decoder_kd_step() == 1


def test_policy_forward_blocks_decoder_kd_when_grad_gate_ratio_exceeds_threshold(monkeypatch):
    config = _make_config(noise_gate=False, grad_gate=True)
    policy = ACTPolicy(config)
    batch = _make_batch()

    student_actions = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]], dtype=torch.float32)
    teacher_actions = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)
    student_decoder_out = torch.zeros(1, 3, 4, dtype=torch.float32)
    teacher_decoder_out = torch.ones(1, 3, 4, dtype=torch.float32)

    _attach_teacher(policy, teacher_actions=teacher_actions, teacher_decoder_out=teacher_decoder_out)
    monkeypatch.setattr(policy.model, "forward", lambda _batch: (student_actions, (None, None)))

    def fake_forward_with_features(_batch, *, latent_mode="auto", return_decoder_out=False):
        decoder_features = None
        if return_decoder_out:
            decoder_features = ACTDecoderFeatureOutput(
                decoder_out=student_decoder_out,
                latent_mode="zero",
                chunk_size=student_decoder_out.shape[1],
                feature_dim=student_decoder_out.shape[2],
            )
        return ACTForwardWithFeaturesOutput(
            actions=student_actions,
            mu=None,
            log_sigma_x2=None,
            decoder_features=decoder_features,
        )

    monkeypatch.setattr(policy.model, "forward_with_features", fake_forward_with_features)
    monkeypatch.setattr(
        policy,
        "_compute_decoder_kd_grad_ratios",
        lambda **_kwargs: (torch.tensor(1.2), torch.tensor(0.4), torch.tensor(1.0)),
    )

    loss, loss_dict = policy.forward(batch)

    bc_loss = (torch.abs(batch[ACTION] - student_actions) * ~batch["action_is_pad"].unsqueeze(-1)).mean()
    action_kd_loss = torch.abs(student_actions - teacher_actions).mean()
    expected_loss = bc_loss + policy.config.kd_weight * action_kd_loss

    torch.testing.assert_close(loss, expected_loss)
    assert loss_dict["decoder_kd_grad_ratio_available"] == pytest.approx(1.0)
    assert loss_dict["decoder_kd_grad_gate_blocked"] == pytest.approx(1.0)
    assert loss_dict["decoder_kd_effective_weight"] == pytest.approx(0.0)
    assert loss_dict["decoder_grad_to_bc_grad_ratio"] == pytest.approx(1.2)
    assert loss_dict["decoder_grad_to_behavior_grad_ratio"] == pytest.approx(1.0)


def test_decoder_kd_scheduler_step_round_trips_through_state_dict():
    config = _make_config(noise_gate=False, grad_gate=False, start_step=2, ramp_steps=2)
    policy = ACTPolicy(config)

    for _ in range(3):
        policy._advance_decoder_kd_step()

    assert policy._get_decoder_kd_step() == 3
    assert policy._get_decoder_kd_scheduler_weight(policy._get_decoder_kd_step()) == pytest.approx(0.4)

    restored = ACTPolicy(config)
    load_result = restored.load_state_dict(policy.state_dict(), strict=True)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert restored._get_decoder_kd_step() == 3
    assert restored._get_decoder_kd_scheduler_weight(restored._get_decoder_kd_step()) == pytest.approx(0.4)
