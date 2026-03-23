#!/usr/bin/env python

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers.feature_extraction_utils import BatchFeature

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot.action_head.flow_matching_action_head import FlowmatchingActionHead
from lerobot.policies.groot.configuration_groot import GROOT_ACTION_CHUNK_SIZE, GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy


def _make_uninitialized_action_head(*, action_horizon: int = 16, action_dim: int = 6) -> FlowmatchingActionHead:
    head = object.__new__(FlowmatchingActionHead)
    nn.Module.__init__(head)
    head.action_horizon = action_horizon
    head.action_dim = action_dim
    return head


def test_groot_config_exposes_fixed_public_chunk_size():
    config = GrootConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
        chunk_size=4,
        n_action_steps=12,
        use_bf16=False,
    )

    assert config.action_chunk_size == GROOT_ACTION_CHUNK_SIZE
    assert config.n_action_steps_effective == 12
    assert config.action_delta_indices == list(range(GROOT_ACTION_CHUNK_SIZE))


def test_flowmatching_action_head_normalize_action_chunk_uses_fixed_model_shape():
    head = _make_uninitialized_action_head(action_horizon=GROOT_ACTION_CHUNK_SIZE, action_dim=6)
    short_actions = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    normalized_actions, action_mask = head.normalize_action_chunk(short_actions)

    assert normalized_actions.shape == (2, GROOT_ACTION_CHUNK_SIZE, 6)
    assert action_mask.shape == (2, GROOT_ACTION_CHUNK_SIZE, 6)
    torch.testing.assert_close(normalized_actions[:, :4, :3], short_actions)
    torch.testing.assert_close(normalized_actions[:, 4:, :3], short_actions[:, -1:, :3].expand(-1, 12, -1))
    assert torch.equal(action_mask[:, :4, :3], torch.ones(2, 4, 3, dtype=torch.bool))
    assert torch.equal(action_mask[:, 4:, :3], torch.zeros(2, 12, 3, dtype=torch.bool))
    assert torch.equal(action_mask[:, :, 3:], torch.zeros(2, GROOT_ACTION_CHUNK_SIZE, 3, dtype=torch.bool))

    long_actions = torch.arange(2 * 20 * 8, dtype=torch.float32).view(2, 20, 8)
    normalized_long_actions, long_mask = head.normalize_action_chunk(long_actions)

    assert normalized_long_actions.shape == (2, GROOT_ACTION_CHUNK_SIZE, 6)
    assert long_mask.shape == (2, GROOT_ACTION_CHUNK_SIZE, 6)
    torch.testing.assert_close(normalized_long_actions, long_actions[:, :GROOT_ACTION_CHUNK_SIZE, :6])
    assert torch.equal(long_mask, torch.ones_like(long_mask, dtype=torch.bool))


def test_flowmatching_action_head_legacy_entrypoints_delegate_to_hybrid_helpers(monkeypatch):
    head = _make_uninitialized_action_head(action_horizon=GROOT_ACTION_CHUNK_SIZE, action_dim=4)
    backbone_output = BatchFeature(data={"backbone_features": torch.randn(1, 3, 8)})
    action_input = BatchFeature(
        data={
            "action": torch.randn(1, 5, 4),
            "action_mask": torch.ones(1, 5, 4, dtype=torch.bool),
        }
    )
    context = BatchFeature(data={"shared": torch.randn(1, 2, 3)})
    calls = {}

    def fake_build_context(backbone_output_arg, action_input_arg):
        calls["build_context"] = (backbone_output_arg, action_input_arg)
        return context

    def fake_forward_chunk(context_arg, *, actions, action_mask=None, noise=None, timesteps=None):
        calls["forward_chunk"] = {
            "context": context_arg,
            "actions": actions,
            "action_mask": action_mask,
            "noise": noise,
            "timesteps": timesteps,
        }
        return BatchFeature(data={"loss": torch.tensor(1.25)})

    def fake_sample_actions_from_context(context_arg, *, noise=None, num_inference_timesteps=None):
        calls["sample_actions_from_context"] = {
            "context": context_arg,
            "noise": noise,
            "num_inference_timesteps": num_inference_timesteps,
        }
        return BatchFeature(data={"action_pred": torch.ones(1, GROOT_ACTION_CHUNK_SIZE, 4)})

    monkeypatch.setattr(head, "build_context", fake_build_context)
    monkeypatch.setattr(head, "forward_chunk", fake_forward_chunk)
    monkeypatch.setattr(head, "sample_actions_from_context", fake_sample_actions_from_context)

    training_outputs = head.forward(backbone_output, action_input)
    inference_outputs = head.get_action(backbone_output, action_input)

    assert training_outputs["loss"].item() == pytest.approx(1.25)
    assert calls["build_context"] == (backbone_output, action_input)
    assert calls["forward_chunk"]["context"] is context
    assert calls["forward_chunk"]["actions"] is action_input["action"]
    assert calls["forward_chunk"]["action_mask"] is action_input["action_mask"]
    assert calls["sample_actions_from_context"]["context"] is context
    assert inference_outputs["action_pred"].shape == (1, GROOT_ACTION_CHUNK_SIZE, 4)


def test_groot_policy_hybrid_interfaces_filter_inputs_and_trim_action_dim():
    class FakeGrootModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.seen_context_inputs = None
            self.seen_predict_context = None

        def get_hybrid_context(self, inputs):
            self.seen_context_inputs = inputs
            return {"shared_context": torch.ones(2, 3)}

        def predict_action_chunk_from_context(self, hybrid_context, *, noise=None, num_inference_timesteps=None):
            self.seen_predict_context = {
                "hybrid_context": hybrid_context,
                "noise": noise,
                "num_inference_timesteps": num_inference_timesteps,
            }
            return {
                "action_pred": torch.arange(2 * GROOT_ACTION_CHUNK_SIZE * 6, dtype=torch.float32).view(
                    2, GROOT_ACTION_CHUNK_SIZE, 6
                )
            }

    policy = object.__new__(GrootPolicy)
    nn.Module.__init__(policy)
    policy.config = type(
        "Config",
        (),
        {
            "use_bf16": False,
            "n_action_steps_effective": 3,
            "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        },
    )()
    policy._groot_model = FakeGrootModel()
    policy.reset()

    batch = {
        "state": torch.randn(2, 1, 64),
        "state_mask": torch.ones(2, 1, 64, dtype=torch.bool),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "action": torch.randn(2, 5, 4),
        "action_mask": torch.ones(2, 5, 4, dtype=torch.bool),
        "eagle_attention_mask": torch.ones(2, 8, dtype=torch.long),
        "info": {"ignore": True},
        "next.state": torch.zeros(2, 1, 64),
    }

    hybrid_context = policy.get_hybrid_context(batch)
    predicted_actions = policy.predict_action_chunk_from_context(hybrid_context)
    selected_action = policy.select_action(batch)

    assert set(policy._groot_model.seen_context_inputs) == {
        "state",
        "state_mask",
        "embodiment_id",
        "eagle_attention_mask",
    }
    torch.testing.assert_close(
        policy._groot_model.seen_predict_context["hybrid_context"]["shared_context"],
        hybrid_context["shared_context"],
    )
    assert predicted_actions.shape == (2, GROOT_ACTION_CHUNK_SIZE, 4)
    torch.testing.assert_close(selected_action, predicted_actions[:, 0])
    assert len(policy._action_queue) == 2


def test_groot_policy_exposes_value_interfaces_via_hybrid_context():
    class FakeGrootModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.seen_context_inputs = None
            self.seen_value_context = None

        def get_hybrid_context(self, inputs):
            self.seen_context_inputs = inputs
            return {"shared_context": torch.arange(6, dtype=torch.float32).view(2, 3)}

        def get_value_from_hybrid_context(self, hybrid_context):
            self.seen_value_context = hybrid_context
            return hybrid_context["shared_context"].sum(dim=-1)

    policy = object.__new__(GrootPolicy)
    nn.Module.__init__(policy)
    policy.config = type(
        "Config",
        (),
        {
            "use_bf16": False,
            "n_action_steps_effective": 1,
            "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        },
    )()
    policy._groot_model = FakeGrootModel()
    policy.reset()

    batch = {
        "state": torch.randn(2, 1, 64),
        "state_mask": torch.ones(2, 1, 64, dtype=torch.bool),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "eagle_attention_mask": torch.ones(2, 8, dtype=torch.long),
        "info": {"ignore": True},
        "next.state": torch.zeros(2, 1, 64),
    }

    hybrid_context = policy.get_hybrid_context(batch)
    values_from_context = policy.get_value_from_hybrid_context(hybrid_context)

    assert set(policy._groot_model.seen_context_inputs) == {
        "state",
        "state_mask",
        "embodiment_id",
        "eagle_attention_mask",
    }
    assert policy._groot_model.seen_value_context is hybrid_context
    assert values_from_context.shape == (2,)
    torch.testing.assert_close(values_from_context, torch.tensor([3.0, 12.0]))
    predicted_values = policy.predict_value(batch)
    torch.testing.assert_close(predicted_values, values_from_context)
    torch.testing.assert_close(policy.get_value(batch), values_from_context)
    torch.testing.assert_close(policy.value(batch), values_from_context)
