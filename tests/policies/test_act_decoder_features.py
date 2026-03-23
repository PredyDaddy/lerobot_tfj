#!/usr/bin/env python

from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.distillation_utils import ACTForwardWithFeaturesOutput
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


INPUT_FEATURES = {
    OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
}
OUTPUT_FEATURES = {
    ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
}


def _make_config(*, use_vae: bool = False) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        use_vae=use_vae,
        pretrained_backbone_weights=None,
        chunk_size=3,
        n_action_steps=2,
        dim_model=8,
        n_heads=4,
        dim_feedforward=16,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )


def _make_obs_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_ENV_STATE: torch.randn(batch_size, 4),
    }


def _make_posterior_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    batch = _make_obs_batch(batch_size=batch_size)
    batch[ACTION] = torch.randn(batch_size, 3, 2)
    batch["action_is_pad"] = torch.zeros(batch_size, 3, dtype=torch.bool)
    return batch


def test_forward_with_features_is_opt_in_and_preserves_forward_contract():
    torch.manual_seed(0)
    model = ACT(_make_config(use_vae=False))
    model.eval()
    batch = _make_obs_batch()

    actions, (mu, log_sigma_x2) = model(batch)
    forward_output = model.forward_with_features(batch, return_decoder_out=False)

    assert isinstance(forward_output, ACTForwardWithFeaturesOutput)
    torch.testing.assert_close(forward_output.actions, actions)
    assert mu is None and log_sigma_x2 is None
    assert forward_output.mu is None and forward_output.log_sigma_x2 is None
    assert forward_output.decoder_features is None


def test_forward_with_features_zero_mode_returns_canonical_decoder_out_in_training_without_actions():
    torch.manual_seed(0)
    model = ACT(_make_config(use_vae=True))
    model.train()
    batch = _make_obs_batch()

    forward_output = model.forward_with_features(
        batch,
        latent_mode="zero",
        return_decoder_out=True,
    )

    assert forward_output.mu is None
    assert forward_output.log_sigma_x2 is None
    assert forward_output.decoder_features is not None
    assert forward_output.decoder_features.latent_mode == "zero"
    assert forward_output.decoder_features.decoder_out.shape == (2, 3, 8)
    assert forward_output.decoder_features.chunk_size == 3
    assert forward_output.decoder_features.feature_dim == 8


def test_forward_with_features_posterior_mode_returns_latent_metadata():
    torch.manual_seed(0)
    model = ACT(_make_config(use_vae=True))
    model.eval()
    batch = _make_posterior_batch()

    forward_output = model.forward_with_features(
        batch,
        latent_mode="posterior",
        return_decoder_out=True,
    )

    assert forward_output.mu is not None
    assert forward_output.log_sigma_x2 is not None
    assert forward_output.mu.shape == (2, model.config.latent_dim)
    assert forward_output.log_sigma_x2.shape == (2, model.config.latent_dim)
    assert forward_output.decoder_features is not None
    assert forward_output.decoder_features.latent_mode == "posterior"
    assert forward_output.decoder_features.decoder_out.shape == (2, 3, 8)


def test_policy_get_decoder_features_reads_zero_latent_eval_features():
    torch.manual_seed(0)
    policy = ACTPolicy(_make_config(use_vae=True))
    batch = _make_obs_batch(batch_size=1)

    policy.eval()
    expected = policy.model.forward_with_features(
        batch,
        latent_mode="zero",
        return_decoder_out=True,
    ).decoder_features
    assert expected is not None

    policy.train()
    decoder_features = policy.get_decoder_features(batch, latent_mode="zero")

    assert policy.training is False
    assert decoder_features.latent_mode == "zero"
    assert decoder_features.decoder_out.requires_grad is False
    torch.testing.assert_close(decoder_features.decoder_out, expected.decoder_out)


def test_forward_with_features_posterior_mode_requires_ground_truth_actions():
    model = ACT(_make_config(use_vae=True))
    batch = _make_obs_batch(batch_size=1)

    with pytest.raises(ValueError, match="ground-truth actions"):
        model.forward_with_features(batch, latent_mode="posterior", return_decoder_out=True)
