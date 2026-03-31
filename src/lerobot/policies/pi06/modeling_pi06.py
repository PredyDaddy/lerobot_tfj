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

from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch
from lerobot.policies.pi06.configuration_pi06 import PI06Config
from lerobot.policies.pi06.processor_pi06 import PI06_IMAGE_MASK_KEY, PI06_IMAGES_KEY

T = TypeVar("T", bound="PI06Policy")


class PI06Policy(PI05Policy):
    """Repo-local PI0.6-style policy built on the PI0.5 flow-action runtime."""

    config_class = PI06Config
    name = "pi06"

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T:
        print(
            "PI06Policy here is a repo-local B-light implementation. "
            "It is not an official upstream pi0.6 release."
        )
        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def __init__(self, config: PI06Config, **kwargs):
        super().__init__(config=config, **kwargs)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        packed_images = batch.get(PI06_IMAGES_KEY)
        if packed_images is None:
            return super()._preprocess_images(batch)

        if packed_images.ndim != 5:
            raise ValueError(
                f"Expected packed pi06 images with rank 5 [B,N,C,H,W] or [B,N,H,W,C], got {packed_images.shape}."
            )
        max_num_cameras = getattr(self.config, "max_num_cameras", packed_images.shape[1])
        if packed_images.shape[1] > max_num_cameras:
            raise ValueError(
                f"Packed pi06 images include {packed_images.shape[1]} cameras but "
                f"max_num_cameras={max_num_cameras}."
            )

        packed_masks = batch.get(PI06_IMAGE_MASK_KEY)
        if packed_masks is None:
            packed_masks = torch.ones(
                packed_images.shape[:2],
                dtype=torch.bool,
                device=packed_images.device,
            )
        elif packed_masks.ndim != 2:
            raise ValueError(
                f"Expected pi06 image attention mask with rank 2 [B,N], got {packed_masks.shape}."
            )
        elif packed_masks.shape != packed_images.shape[:2]:
            raise ValueError(
                "Packed pi06 images and image attention mask must agree on [B,N]. "
                f"Got images={packed_images.shape[:2]} mask={packed_masks.shape}."
            )

        device = next(self.parameters()).device
        packed_images = packed_images.to(device=device, dtype=torch.float32)
        packed_masks = packed_masks.to(device=device, dtype=torch.bool)
        if not torch.all(packed_masks.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera for pi06.")

        images: list[Tensor] = []
        img_masks: list[Tensor] = []

        for camera_idx in range(packed_images.shape[1]):
            img = packed_images[:, camera_idx]
            if img.shape[1] in {1, 3}:
                is_channels_first = True
            elif img.shape[-1] in {1, 3}:
                is_channels_first = False
            else:
                raise ValueError(
                    "Packed pi06 camera tensors must be channels-first or channels-last. "
                    f"Got per-camera shape {tuple(img.shape)}."
                )

            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            img = img * 2.0 - 1.0
            img = img.permute(0, 3, 1, 2)
            camera_mask = packed_masks[:, camera_idx].view(-1, 1, 1, 1)
            img = torch.where(camera_mask, img, torch.full_like(img, -1.0))

            images.append(img)
            img_masks.append(packed_masks[:, camera_idx])

        return images, img_masks
