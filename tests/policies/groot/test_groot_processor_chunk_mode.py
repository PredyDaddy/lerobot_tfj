#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK,
    GrootActionUnpackUnnormalizeStep,
    GrootPackInputsStep,
    build_groot_sample_time_batch,
    make_groot_pre_post_processors,
)
from lerobot.processor.converters import create_transition


def _make_groot_config() -> GrootConfig:
    return GrootConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )


def _make_stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "observation.state": {
            "min": torch.tensor([-1.0, -2.0, -3.0, -4.0]),
            "max": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        },
        "action": {
            "min": torch.tensor([1.0, 10.0, 100.0]),
            "max": torch.tensor([3.0, 14.0, 106.0]),
        },
    }


def _assert_stats_match(
    actual: dict[str, dict[str, torch.Tensor]] | None,
    expected: dict[str, dict[str, torch.Tensor]],
) -> None:
    assert actual is not None
    assert actual.keys() == expected.keys()
    for key, sub_stats in expected.items():
        assert actual[key].keys() == sub_stats.keys()
        for stat_name, tensor in sub_stats.items():
            assert torch.equal(torch.as_tensor(actual[key][stat_name]), tensor)


def test_groot_postprocessor_full_chunk_mode_unnormalizes_every_timestep():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=3,
        normalize_min_max=True,
        stats=_make_stats(),
        output_mode=GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK,
    )

    transition = create_transition(
        action=torch.tensor(
            [[[-1.0, 0.0, 1.0, 99.0], [0.0, -1.0, 0.0, 42.0]]],
            dtype=torch.float32,
        )
    )

    result = step(transition)

    assert result["action"].shape == (1, 2, 3)
    expected = torch.tensor([[[1.0, 12.0, 106.0], [2.0, 10.0, 103.0]]], dtype=torch.float32)
    assert torch.allclose(result["action"], expected)


def test_groot_postprocessor_defaults_to_last_step_output():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=3,
        normalize_min_max=False,
    )
    transition = create_transition(action=torch.arange(8, dtype=torch.float32).view(1, 2, 4))

    result = step(transition)

    assert result["action"].shape == (1, 3)
    assert torch.equal(result["action"], torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32))


def test_build_groot_sample_time_batch_preserves_observation_and_task():
    observation = {
        "observation.state": torch.zeros(4),
        "observation.images.front": torch.zeros(3, 8, 8),
    }

    batch = build_groot_sample_time_batch(
        observation,
        task="pick cube",
        extra_batch_fields={"task_index": torch.tensor(3)},
    )

    assert batch["task"] == "pick cube"
    assert torch.equal(batch["observation.state"], observation["observation.state"])
    assert torch.equal(batch["task_index"], torch.tensor(3))


def test_build_groot_sample_time_batch_rejects_chunk_transition_keys():
    observation = {
        "observation.state": torch.zeros(4),
        "action_chunk": torch.zeros(2, 4),
    }

    try:
        build_groot_sample_time_batch(observation)
    except ValueError as exc:
        assert "raw observation mapping" in str(exc)
    else:
        raise AssertionError("Expected sample-time batch builder to reject action_chunk inputs.")


def test_make_pre_post_processors_merges_groot_and_caller_overrides(tmp_path):
    config = _make_groot_config()
    stats = _make_stats()

    preprocessor, postprocessor = make_groot_pre_post_processors(config=config, dataset_stats=None)
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(tmp_path),
        dataset_stats=stats,
        preprocessor_overrides={
            "rename_observations_processor": {
                "rename_map": {"camera.front": "observation.images.front"},
            }
        },
        postprocessor_overrides={
            "groot_action_unpack_unnormalize_v1": {"output_mode": GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK},
            "device_processor": {"device": "cpu"},
        },
    )

    rename_step = loaded_preprocessor.steps[0]
    pack_step = loaded_preprocessor.steps[2]
    unpack_step = loaded_postprocessor.steps[0]

    assert rename_step.rename_map == {"camera.front": "observation.images.front"}
    assert isinstance(pack_step, GrootPackInputsStep)
    assert pack_step.normalize_min_max is True
    _assert_stats_match(pack_step.stats, stats)

    assert isinstance(unpack_step, GrootActionUnpackUnnormalizeStep)
    assert unpack_step.output_mode == GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK
    assert unpack_step.env_action_dim == 3
    assert unpack_step.normalize_min_max is True
    _assert_stats_match(unpack_step.stats, stats)


def test_make_pre_post_processors_accepts_legacy_groot_normalizer_override_aliases_on_resume(tmp_path):
    config = _make_groot_config()
    stats = _make_stats()

    preprocessor, postprocessor = make_groot_pre_post_processors(config=config, dataset_stats=None)
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(tmp_path),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {
                "rename_map": {"camera.front": "observation.images.front"},
            },
            "normalizer_processor": {
                "stats": stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            },
            "groot_action_unpack_unnormalize_v1": {
                "output_mode": GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK,
            },
        },
    )

    rename_step = loaded_preprocessor.steps[0]
    pack_step = loaded_preprocessor.steps[2]
    unpack_step = loaded_postprocessor.steps[0]

    assert rename_step.rename_map == {"camera.front": "observation.images.front"}
    assert isinstance(pack_step, GrootPackInputsStep)
    assert pack_step.normalize_min_max is True
    _assert_stats_match(pack_step.stats, stats)

    assert isinstance(unpack_step, GrootActionUnpackUnnormalizeStep)
    assert unpack_step.output_mode == GROOT_ACTION_OUTPUT_MODE_FULL_CHUNK
    assert unpack_step.env_action_dim == 3
    assert unpack_step.normalize_min_max is True
    _assert_stats_match(unpack_step.stats, stats)
