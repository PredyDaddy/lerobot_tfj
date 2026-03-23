#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.distillation_utils import (
    KD_COMPARISON_SPACE_NORMALIZED_ACTION,
    KDProcessorCompatibilityReport,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


_EXPECTED_ACT_PREPROCESSOR_STEP_TYPES = (
    RenameObservationsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
)


def _raise_incompatible_processor(reason: str) -> None:
    raise ValueError(
        "Stage 1 ACT KD only supports processor-compatible normalized-action-space KD. "
        + reason
    )


def _assert_act_preprocessor_structure(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    *,
    role: str,
) -> None:
    step_types = tuple(type(step) for step in preprocessor.steps)
    if step_types != _EXPECTED_ACT_PREPROCESSOR_STEP_TYPES:
        _raise_incompatible_processor(
            f"{role} preprocessor must use the standard ACT step layout "
            f"{[step.__name__ for step in _EXPECTED_ACT_PREPROCESSOR_STEP_TYPES]}. "
            f"Got {[step.__name__ for step in step_types]}."
        )


def _get_act_rename_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> RenameObservationsProcessorStep:
    _assert_act_preprocessor_structure(preprocessor, role="ACT")
    return preprocessor.steps[0]


def _get_act_normalizer_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> NormalizerProcessorStep:
    _assert_act_preprocessor_structure(preprocessor, role="ACT")
    return preprocessor.steps[-1]


def _get_required_stat_names(
    feature_type,
    norm_map: dict[Any, Any],
) -> tuple[str, ...]:
    norm_mode = norm_map.get(feature_type, NormalizationMode.IDENTITY)
    if norm_mode == NormalizationMode.IDENTITY:
        return ()
    if norm_mode == NormalizationMode.MEAN_STD:
        return ("mean", "std")
    if norm_mode == NormalizationMode.MIN_MAX:
        return ("min", "max")
    if norm_mode == NormalizationMode.QUANTILES:
        return ("q01", "q99")
    if norm_mode == NormalizationMode.QUANTILE10:
        return ("q10", "q90")
    return ()


def _stat_tensors_match(student_tensor: torch.Tensor, teacher_tensor: torch.Tensor) -> bool:
    if student_tensor.shape != teacher_tensor.shape:
        return False

    if student_tensor.dtype == teacher_tensor.dtype and torch.equal(student_tensor, teacher_tensor):
        return True

    if student_tensor.dtype == torch.bool or teacher_tensor.dtype == torch.bool:
        return torch.equal(student_tensor.to(torch.bool), teacher_tensor.to(torch.bool))

    if torch.is_floating_point(student_tensor) or torch.is_floating_point(teacher_tensor):
        return torch.allclose(
            student_tensor.to(torch.float64),
            teacher_tensor.to(torch.float64),
            rtol=1e-5,
            atol=1e-6,
        )

    return torch.equal(student_tensor.to(torch.int64), teacher_tensor.to(torch.int64))


def _stats_match(
    student_stats: dict[str, dict[str, torch.Tensor]] | None,
    teacher_stats: dict[str, dict[str, torch.Tensor]] | None,
    *,
    features: dict[str, Any],
    norm_map: dict[Any, Any],
) -> bool:
    student_stats = student_stats or {}
    teacher_stats = teacher_stats or {}

    # Only the stats needed by the configured normalization mode affect ACT KD behavior.
    # Dataset metadata often carries additional bookkeeping keys or higher-precision copies
    # that do not change the normalized tensors but would fail a raw dict equality check.
    for key, feature in features.items():
        required_stat_names = _get_required_stat_names(feature.type, norm_map)
        if not required_stat_names:
            continue

        student_key_stats = student_stats.get(key) or {}
        teacher_key_stats = teacher_stats.get(key) or {}
        for stat_name in required_stat_names:
            if stat_name not in student_key_stats or stat_name not in teacher_key_stats:
                return False
            student_tensor = torch.as_tensor(student_key_stats[stat_name]).cpu()
            teacher_tensor = torch.as_tensor(teacher_key_stats[stat_name]).cpu()
            if not _stat_tensors_match(student_tensor, teacher_tensor):
                return False

    return True


def get_act_kd_processor_compatibility(
    *,
    student_config: ACTConfig,
    student_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    teacher_config: ACTConfig,
    teacher_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> KDProcessorCompatibilityReport:
    _assert_act_preprocessor_structure(student_preprocessor, role="Student")
    _assert_act_preprocessor_structure(teacher_preprocessor, role="Teacher")

    student_image_keys = list(student_config.image_features)
    teacher_image_keys = list(teacher_config.image_features)
    if student_image_keys != teacher_image_keys:
        _raise_incompatible_processor(
            "Student and teacher image key order must match for ACT KD. "
            f"Got student={student_image_keys} and teacher={teacher_image_keys}."
        )

    student_rename_step = _get_act_rename_step(student_preprocessor)
    teacher_rename_step = _get_act_rename_step(teacher_preprocessor)
    if student_rename_step.rename_map != teacher_rename_step.rename_map:
        _raise_incompatible_processor(
            "Student and teacher rename behavior must match. "
            f"Got student={student_rename_step.rename_map} and teacher={teacher_rename_step.rename_map}."
        )

    student_normalizer = _get_act_normalizer_step(student_preprocessor)
    teacher_normalizer = _get_act_normalizer_step(teacher_preprocessor)

    if student_normalizer.features != teacher_normalizer.features:
        _raise_incompatible_processor("Student and teacher normalization feature layouts must match exactly.")
    if student_normalizer.norm_map != teacher_normalizer.norm_map:
        _raise_incompatible_processor("Student and teacher normalization modes must match exactly.")
    if student_normalizer.normalize_observation_keys != teacher_normalizer.normalize_observation_keys:
        _raise_incompatible_processor(
            "Student and teacher normalization observation-key filters must match exactly."
        )
    if not _stats_match(
        student_normalizer.stats,
        teacher_normalizer.stats,
        features=student_normalizer.features,
        norm_map=student_normalizer.norm_map,
    ):
        _raise_incompatible_processor("Student and teacher normalization statistics must match exactly.")

    return KDProcessorCompatibilityReport(
        compatible=True,
        comparison_space=KD_COMPARISON_SPACE_NORMALIZED_ACTION,
    )


def assert_act_kd_processor_compatibility(
    *,
    student_config: ACTConfig,
    student_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    teacher_config: ACTConfig,
    teacher_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> KDProcessorCompatibilityReport:
    return get_act_kd_processor_compatibility(
        student_config=student_config,
        student_preprocessor=student_preprocessor,
        teacher_config=teacher_config,
        teacher_preprocessor=teacher_preprocessor,
    )


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
