#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
from unittest.mock import Mock, patch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import load_json, write_json
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    PRETRAINED_MODEL_DIR,
    OBS_STATE,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.train_metadata import (
    KD_TEACHER_METADATA_FILENAME,
    create_kd_teacher_metadata,
    kd_teacher_metadata_to_dict,
    load_kd_teacher_metadata,
    resolve_kd_teacher_metadata_for_resume,
    save_kd_teacher_metadata,
)
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    load_training_step,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)


def make_train_cfg(
    output_dir: Path,
    *,
    resume: bool = False,
    **policy_kwargs,
) -> TrainPipelineConfig:
    policy = ACTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        push_to_hub=False,
        **policy_kwargs,
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test"),
        policy=policy,
        output_dir=output_dir,
        resume=resume,
    )


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint_writes_kd_teacher_metadata(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    cfg.kd_teacher_metadata = {
        "teacher_policy_path": "/teachers/original",
        "resolved_teacher_pretrained_path": "/teachers/pinned/pretrained_model",
        "teacher_source_kind": "output_dir_last",
        "teacher_checkpoint_step": 1200,
        "processor_compatibility_mode": "strict_preprocessor_match",
        "comparison_space": "normalized_action_space",
    }

    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)

    metadata_path = tmp_path / KD_TEACHER_METADATA_FILENAME
    assert metadata_path.is_file()
    assert load_json(metadata_path) == {
        "schema_version": 1,
        "comparison_space": "normalized_action_space",
        "processor_compatibility": "strict_preprocessor_match",
        "teacher": {
            "original_path": "/teachers/original",
            "pinned_pretrained_path": "/teachers/pinned/pretrained_model",
            "checkpoint_step": 1200,
            "source_kind": "output_dir_last",
        },
    }
    mock_save_training_state.assert_called_once()


def test_pin_kd_teacher_metadata_writes_run_level_metadata(tmp_path):
    cfg = make_train_cfg(tmp_path / "run")

    resolved_teacher_path = tmp_path / "teachers" / "resolved" / PRETRAINED_MODEL_DIR
    metadata_path = cfg.pin_kd_teacher_metadata(
        create_kd_teacher_metadata(
            teacher_source_path=tmp_path / "teachers" / "source",
            teacher_pretrained_path=resolved_teacher_path,
            teacher_source_kind="checkpoint_dir",
            teacher_checkpoint_step=2400,
            comparison_space="normalized_action_space",
            processor_compatibility="strict_preprocessor_match",
        )
    )

    assert metadata_path == cfg.output_dir / KD_TEACHER_METADATA_FILENAME
    assert load_json(metadata_path) == {
        "schema_version": 1,
        "comparison_space": "normalized_action_space",
        "processor_compatibility": "strict_preprocessor_match",
        "teacher": {
            "original_path": str(tmp_path / "teachers" / "source"),
            "pinned_pretrained_path": str(resolved_teacher_path),
            "checkpoint_step": 2400,
            "source_kind": "checkpoint_dir",
        },
    }
    assert cfg.get_pinned_kd_teacher_pretrained_path() == resolved_teacher_path
    assert cfg.get_runtime_teacher_source_path() == resolved_teacher_path


def test_pin_kd_teacher_metadata_canonicalizes_legacy_flat_schema(tmp_path):
    cfg = make_train_cfg(tmp_path / "run")
    metadata_path = cfg.pin_kd_teacher_metadata(
        {
            "teacher_policy_path": tmp_path / "teachers" / "legacy-source",
            "resolved_teacher_pretrained_path": tmp_path / "teachers" / "legacy-pinned" / PRETRAINED_MODEL_DIR,
            "teacher_source_kind": "pretrained_dir",
            "teacher_checkpoint_step": 512,
        }
    )

    assert metadata_path == cfg.output_dir / KD_TEACHER_METADATA_FILENAME
    assert load_json(metadata_path) == {
        "schema_version": 1,
        "teacher": {
            "original_path": str(tmp_path / "teachers" / "legacy-source"),
            "pinned_pretrained_path": str(tmp_path / "teachers" / "legacy-pinned" / PRETRAINED_MODEL_DIR),
            "checkpoint_step": 512,
            "source_kind": "pretrained_dir",
        },
    }


@patch("lerobot.utils.train_utils.save_training_state")
def test_run_and_checkpoint_kd_teacher_metadata_use_same_schema(
    mock_save_training_state,
    tmp_path,
    optimizer,
):
    run_dir = tmp_path / "run"
    cfg = make_train_cfg(run_dir)
    policy = Mock()
    canonical_metadata = create_kd_teacher_metadata(
        teacher_source_path=tmp_path / "teachers" / "source",
        teacher_pretrained_path=tmp_path / "teachers" / "pinned" / PRETRAINED_MODEL_DIR,
        teacher_source_kind="output_dir_last",
        teacher_checkpoint_step=3200,
        comparison_space="normalized_action_space",
        processor_compatibility="strict_preprocessor_match",
        metric_aggregation_mode="accelerate_mean",
    )

    cfg.pin_kd_teacher_metadata(canonical_metadata)
    save_checkpoint(tmp_path / "checkpoint", 10, cfg, policy, optimizer)

    assert load_json(run_dir / KD_TEACHER_METADATA_FILENAME) == load_json(
        tmp_path / "checkpoint" / KD_TEACHER_METADATA_FILENAME
    )
    mock_save_training_state.assert_called_once()


def test_load_kd_teacher_metadata_supports_legacy_flat_schema(tmp_path):
    write_json(
        {
            "teacher_policy_path": str(tmp_path / "teachers" / "legacy-source"),
            "resolved_teacher_pretrained_path": str(
                tmp_path / "teachers" / "legacy-pinned" / PRETRAINED_MODEL_DIR
            ),
            "teacher_source_kind": "checkpoint_dir",
            "teacher_checkpoint_step": 900,
            "processor_compatibility_mode": "strict_preprocessor_match",
        },
        tmp_path / KD_TEACHER_METADATA_FILENAME,
    )

    metadata = load_kd_teacher_metadata(tmp_path)

    assert metadata is not None
    assert metadata.teacher.original_path == tmp_path / "teachers" / "legacy-source"
    assert metadata.teacher.pinned_pretrained_path == tmp_path / "teachers" / "legacy-pinned" / PRETRAINED_MODEL_DIR
    assert metadata.teacher.source_kind == "checkpoint_dir"
    assert metadata.teacher.checkpoint_step == 900
    assert metadata.processor_compatibility == "strict_preprocessor_match"
    assert kd_teacher_metadata_to_dict(metadata) == {
        "schema_version": 1,
        "processor_compatibility": "strict_preprocessor_match",
        "teacher": {
            "original_path": str(tmp_path / "teachers" / "legacy-source"),
            "pinned_pretrained_path": str(tmp_path / "teachers" / "legacy-pinned" / PRETRAINED_MODEL_DIR),
            "checkpoint_step": 900,
            "source_kind": "checkpoint_dir",
        },
    }


@patch("lerobot.configs.train.parser.parse_arg")
@patch("lerobot.configs.train.parser.get_path_arg")
def test_resume_prefers_checkpoint_pinned_teacher_metadata(
    mock_get_path_arg,
    mock_parse_arg,
    tmp_path,
):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / CHECKPOINTS_DIR / "000010"
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / TRAIN_CONFIG_NAME).write_text("{}")

    stale_teacher_path = Path("/teachers/original-now-missing")
    cfg = make_train_cfg(
        run_dir,
        resume=True,
        kd=True,
        teacher_policy_path=stale_teacher_path,
    )
    cfg.kd_teacher_metadata = kd_teacher_metadata_to_dict(
        create_kd_teacher_metadata(
            teacher_pretrained_path=tmp_path / "teachers" / "from-train-config" / PRETRAINED_MODEL_DIR,
            teacher_source_kind="train_config",
        )
    )

    save_kd_teacher_metadata(
        {
            "schema_version": 1,
            "comparison_space": "normalized_action_space",
            "processor_compatibility": "strict_preprocessor_match",
            "teacher": {
                "pinned_pretrained_path": str(
                    tmp_path / "teachers" / "from-run-level" / PRETRAINED_MODEL_DIR
                ),
                "source_kind": "run_level",
            },
        },
        run_dir,
    )
    pinned_checkpoint_teacher_path = tmp_path / "teachers" / "from-checkpoint" / PRETRAINED_MODEL_DIR
    save_kd_teacher_metadata(
        {
            "schema_version": 1,
            "comparison_space": "normalized_action_space",
            "processor_compatibility": "strict_preprocessor_match",
            "teacher": {
                "original_path": str(stale_teacher_path),
                "pinned_pretrained_path": str(pinned_checkpoint_teacher_path),
                "source_kind": "output_dir_last",
                "checkpoint_step": 1200,
                "pinned_from_run_metadata": False,
            },
        },
        checkpoint_dir,
    )

    mock_get_path_arg.return_value = None
    mock_parse_arg.return_value = str(pretrained_dir / TRAIN_CONFIG_NAME)

    cfg.validate()

    assert cfg.checkpoint_path == checkpoint_dir
    assert cfg.policy.pretrained_path == pretrained_dir
    assert cfg.policy.teacher_policy_path == stale_teacher_path
    assert cfg.get_pinned_kd_teacher_pretrained_path() == pinned_checkpoint_teacher_path
    assert cfg.get_runtime_teacher_source_path() == pinned_checkpoint_teacher_path
    assert cfg.kd_teacher_metadata["teacher"]["source_kind"] == "output_dir_last"
    assert cfg.kd_teacher_metadata["teacher"]["checkpoint_step"] == 1200
    assert cfg.kd_teacher_metadata["comparison_space"] == "normalized_action_space"


@patch("lerobot.configs.train.parser.parse_arg")
@patch("lerobot.configs.train.parser.get_path_arg")
def test_load_pinned_kd_teacher_metadata_does_not_override_checkpoint_pin_with_run_level(
    mock_get_path_arg,
    mock_parse_arg,
    tmp_path,
):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / CHECKPOINTS_DIR / "000010"
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / TRAIN_CONFIG_NAME).write_text("{}")

    checkpoint_teacher_path = tmp_path / "teachers" / "checkpoint" / PRETRAINED_MODEL_DIR
    stale_run_level_teacher_path = tmp_path / "teachers" / "run-level-stale" / PRETRAINED_MODEL_DIR

    cfg = make_train_cfg(
        run_dir,
        resume=True,
        kd=True,
        teacher_policy_path=Path("/teachers/original-now-missing"),
    )

    save_kd_teacher_metadata(
        create_kd_teacher_metadata(
            teacher_pretrained_path=stale_run_level_teacher_path,
            teacher_source_kind="run_level",
            comparison_space="normalized_action_space",
        ),
        run_dir,
    )
    save_kd_teacher_metadata(
        create_kd_teacher_metadata(
            teacher_pretrained_path=checkpoint_teacher_path,
            teacher_source_kind="checkpoint_level",
            comparison_space="normalized_action_space",
        ),
        checkpoint_dir,
    )

    mock_get_path_arg.return_value = None
    mock_parse_arg.return_value = str(pretrained_dir / TRAIN_CONFIG_NAME)

    cfg.validate()
    assert cfg.get_pinned_kd_teacher_pretrained_path() == checkpoint_teacher_path

    reloaded_metadata = cfg.load_pinned_kd_teacher_metadata(run_dir)

    assert reloaded_metadata is not None
    assert reloaded_metadata.resolved_teacher_pretrained_path == checkpoint_teacher_path
    assert reloaded_metadata.teacher_source_kind == "checkpoint_level"
    assert cfg.get_pinned_kd_teacher_pretrained_path() == checkpoint_teacher_path


@patch("lerobot.configs.train.parser.parse_arg")
@patch("lerobot.configs.train.parser.get_path_arg")
def test_resume_falls_back_to_embedded_kd_teacher_metadata(
    mock_get_path_arg,
    mock_parse_arg,
    tmp_path,
):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / CHECKPOINTS_DIR / "000010"
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / TRAIN_CONFIG_NAME).write_text("{}")

    stale_teacher_path = Path("/teachers/source-no-longer-valid")
    cfg = make_train_cfg(
        run_dir,
        resume=True,
        kd=True,
        teacher_policy_path=stale_teacher_path,
    )
    embedded_teacher_path = tmp_path / "teachers" / "embedded" / PRETRAINED_MODEL_DIR
    cfg.kd_teacher_metadata = kd_teacher_metadata_to_dict(
        create_kd_teacher_metadata(
            teacher_source_path=stale_teacher_path,
            teacher_pretrained_path=embedded_teacher_path,
            teacher_source_kind="pretrained_dir",
        )
    )

    mock_get_path_arg.return_value = None
    mock_parse_arg.return_value = str(pretrained_dir / TRAIN_CONFIG_NAME)

    cfg.validate()

    assert cfg.get_pinned_kd_teacher_pretrained_path() == embedded_teacher_path
    assert cfg.get_runtime_teacher_source_path() == embedded_teacher_path
    assert cfg.policy.teacher_policy_path == stale_teacher_path


def test_resolve_kd_teacher_metadata_for_resume_prefers_checkpoint_then_run_then_embedded(tmp_path):
    checkpoint_dir = tmp_path / "run" / CHECKPOINTS_DIR / "000010"
    run_dir = tmp_path / "run"
    checkpoint_dir.mkdir(parents=True)

    save_kd_teacher_metadata(
        create_kd_teacher_metadata(
            teacher_pretrained_path=tmp_path / "teachers" / "run-level" / PRETRAINED_MODEL_DIR,
            teacher_source_kind="run_level",
            comparison_space="normalized_action_space",
        ),
        run_dir,
    )
    save_kd_teacher_metadata(
        create_kd_teacher_metadata(
            teacher_pretrained_path=tmp_path / "teachers" / "checkpoint" / PRETRAINED_MODEL_DIR,
            teacher_source_kind="checkpoint_level",
            comparison_space="normalized_action_space",
        ),
        checkpoint_dir,
    )

    resolved_metadata = resolve_kd_teacher_metadata_for_resume(
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        embedded_metadata=create_kd_teacher_metadata(
            teacher_pretrained_path=tmp_path / "teachers" / "embedded" / PRETRAINED_MODEL_DIR,
            teacher_source_kind="embedded",
        ),
    )

    assert resolved_metadata is not None
    assert resolved_metadata.resolved_teacher_pretrained_path == (
        tmp_path / "teachers" / "checkpoint" / PRETRAINED_MODEL_DIR
    )
    assert resolved_metadata.teacher_source_kind == "checkpoint_level"


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler
