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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("pi06")
@dataclass
class PI06Config(PI05Config):
    """Repo-local PI0.6-style policy config built on the PI0.5 flow stack."""

    task_field: str = "task"
    state_feature: str = OBS_STATE
    include_state_in_prompt: bool = True
    state_discretization_bins: int = 256

    camera_features: list[str] = field(default_factory=list)
    max_num_cameras: int = 4

    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    action_head_type: str = "flow"

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.task_field:
            raise ValueError("'task_field' must be non-empty.")
        if not self.state_feature:
            raise ValueError("'state_feature' must be non-empty.")
        if not self.state_feature.startswith("observation."):
            raise ValueError("'state_feature' must start with 'observation.'.")
        if self.max_num_cameras <= 0:
            raise ValueError("'max_num_cameras' must be greater than 0.")
        if self.state_discretization_bins < 2:
            raise ValueError("'state_discretization_bins' must be at least 2.")
        if not self.text_tokenizer_name:
            raise ValueError("'text_tokenizer_name' must be non-empty.")
        if self.paligemma_variant != "gemma_2b":
            raise ValueError(
                "The current repo-local pi06 implementation only supports "
                "'paligemma_variant=gemma_2b'. The gemma_300m setting hits a vision/text width mismatch."
            )
        if self.action_head_type != "flow":
            raise ValueError("Only flow action heads are supported for the current pi06 implementation.")
        if len(self.camera_features) > self.max_num_cameras:
            raise ValueError(
                f"Configured {len(self.camera_features)} camera features but max_num_cameras="
                f"{self.max_num_cameras}."
            )
        if len(set(self.camera_features)) != len(self.camera_features):
            raise ValueError("'camera_features' must not contain duplicates.")
        invalid_camera_features = [key for key in self.camera_features if not key.startswith(OBS_IMAGES)]
        if invalid_camera_features:
            raise ValueError(
                "All 'camera_features' entries must start with "
                f"'{OBS_IMAGES}'. Invalid values: {invalid_camera_features}"
            )

    def validate_features(self) -> None:
        super().validate_features()

        if self.input_features is None:
            return

        if not self.camera_features:
            self.camera_features = [
                key
                for key in self.input_features
                if key.startswith(OBS_IMAGES) and not key.startswith(f"{OBS_IMAGES}.empty_camera_")
            ]

        if len(self.camera_features) > self.max_num_cameras:
            raise ValueError(
                f"Found {len(self.camera_features)} camera features but max_num_cameras={self.max_num_cameras}."
            )

        missing_camera_features = [key for key in self.camera_features if key not in self.input_features]
        if missing_camera_features:
            raise ValueError(
                "All configured 'camera_features' must be present in 'input_features'. "
                f"Missing: {missing_camera_features}"
            )
