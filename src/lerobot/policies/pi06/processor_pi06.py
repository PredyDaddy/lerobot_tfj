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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

if TYPE_CHECKING:
    from lerobot.policies.pi06.configuration_pi06 import PI06Config

PI06_IMAGES_KEY = "observation.pi06.images"
PI06_IMAGE_MASK_KEY = "observation.pi06.image_attention_mask"


def _pad_last_dim(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return functional.pad(vector, (0, new_dim - vector.shape[-1]))


@ProcessorStepRegistry.register(name="pi06_prepare_task_prompt")
@dataclass
class Pi06PrepareTaskPromptProcessorStep(ProcessorStep):
    task_key: str = "task"
    include_state_in_prompt: bool = True
    state_feature: str = OBS_STATE
    max_state_dim: int = 32
    state_discretization_bins: int = 256
    action_prompt_suffix: str = "Action: "

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "include_state_in_prompt": self.include_state_in_prompt,
            "state_feature": self.state_feature,
            "max_state_dim": self.max_state_dim,
            "state_discretization_bins": self.state_discretization_bins,
            "action_prompt_suffix": self.action_prompt_suffix,
        }

    @staticmethod
    def _clean_prompt(task: str) -> str:
        return str(task).strip().replace("_", " ").replace("\n", " ").strip()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.task_key not in complementary_data:
            raise KeyError(f"Missing task field '{self.task_key}' in complementary data.")

        tasks_raw = complementary_data[self.task_key]
        if isinstance(tasks_raw, str):
            tasks = [tasks_raw]
        elif isinstance(tasks_raw, Sequence) and all(isinstance(task, str) for task in tasks_raw):
            tasks = list(tasks_raw)
        else:
            raise TypeError(
                f"Expected task field '{self.task_key}' as sequence of strings, got {type(tasks_raw)}."
            )

        prompts: list[str] = []
        if self.include_state_in_prompt:
            if self.state_feature not in observation:
                raise KeyError(
                    f"Missing state feature '{self.state_feature}' while include_state_in_prompt=True."
                )

            state = observation[self.state_feature]
            if not isinstance(state, Tensor):
                state = torch.as_tensor(state)

            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.ndim != 2:
                raise ValueError(
                    f"Expected state tensor with shape [B, D], got {tuple(state.shape)} "
                    f"for feature '{self.state_feature}'."
                )

            state = state.detach().to(dtype=torch.float32, device="cpu")
            state = _pad_last_dim(state, self.max_state_dim)
            bins = np.linspace(-1.0, 1.0, self.state_discretization_bins + 1, dtype=np.float32)[:-1]
            discretized_state = np.digitize(state.numpy(), bins=bins) - 1

            if discretized_state.shape[0] != len(tasks):
                raise ValueError(
                    f"Task count ({len(tasks)}) does not match state batch size ({discretized_state.shape[0]})."
                )

            for i, task in enumerate(tasks):
                cleaned_task = self._clean_prompt(task)
                state_str = " ".join(map(str, discretized_state[i].tolist()))
                prompts.append(f"Task: {cleaned_task}, State: {state_str}\n{self.action_prompt_suffix}")
        else:
            prompts = [f"Task: {self._clean_prompt(task)}\n{self.action_prompt_suffix}" for task in tasks]

        complementary_data[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi06_prepare_images")
@dataclass
class Pi06PrepareImagesProcessorStep(ProcessorStep):
    camera_features: list[str]
    output_key: str = PI06_IMAGES_KEY
    output_mask_key: str = PI06_IMAGE_MASK_KEY

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_features": self.camera_features,
            "output_key": self.output_key,
            "output_mask_key": self.output_mask_key,
        }

    @staticmethod
    def _to_bchw(img_batch: Tensor) -> Tensor:
        if img_batch.ndim != 4:
            raise ValueError(f"Expected image batch rank 4, got shape {tuple(img_batch.shape)}.")

        if img_batch.shape[1] in {1, 3}:
            return img_batch
        if img_batch.shape[-1] in {1, 3}:
            return img_batch.permute(0, 3, 1, 2)
        raise ValueError(
            "Camera tensor must be channels-first or channels-last. "
            f"Got camera batch with shape={tuple(img_batch.shape)}."
        )

    def _process_camera_batch(self, img_batch: Tensor) -> Tensor:
        return self._to_bchw(img_batch).detach().to(dtype=torch.float32)

    def _prepare_images(self, observation: dict[str, Any]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.camera_features if key in observation]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(observation.keys())}"
            )

        reference_img = self._process_camera_batch(torch.as_tensor(observation[present_img_keys[0]]))
        bsize = reference_img.shape[0]
        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in observation:
                img = self._process_camera_batch(torch.as_tensor(observation[key]))
                if img.shape[0] != bsize:
                    raise ValueError(
                        f"Mismatched batch size across cameras. Camera '{key}' has {img.shape[0]}, expected {bsize}."
                    )
                if img.shape[1:] != reference_img.shape[1:]:
                    raise ValueError(
                        "Camera tensors must share the same [C,H,W] shape before model preprocessing. "
                        f"Camera '{key}' has {tuple(img.shape[1:])}, expected {tuple(reference_img.shape[1:])}."
                    )
                image_tensors.append(img)
                image_masks.append(torch.ones(bsize, dtype=torch.bool))
            else:
                image_tensors.append(torch.zeros_like(reference_img))
                image_masks.append(torch.zeros(bsize, dtype=torch.bool))

        images = torch.stack(image_tensors, dim=1)
        masks = torch.stack(image_masks, dim=1)
        return images, masks

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        images, image_attention_mask = self._prepare_images(observation)
        observation[self.output_key] = images.to(dtype=torch.float32)
        observation[self.output_mask_key] = image_attention_mask.to(dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi06_pre_post_processors(
    config: "PI06Config",
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [
            key
            for key in (config.input_features or {})
            if key.startswith(OBS_IMAGES) and not key.startswith(f"{OBS_IMAGES}.empty_camera_")
        ]
    if len(camera_features) > config.max_num_cameras:
        raise ValueError(
            f"Configured {len(camera_features)} pi06 camera features but max_num_cameras={config.max_num_cameras}."
        )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        Pi06PrepareTaskPromptProcessorStep(
            task_key=config.task_field,
            include_state_in_prompt=config.include_state_in_prompt,
            state_feature=config.state_feature,
            max_state_dim=config.max_state_dim,
            state_discretization_bins=config.state_discretization_bins,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        Pi06PrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
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
