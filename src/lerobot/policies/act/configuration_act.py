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
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.distillation_utils import get_kd_segment_layout


@dataclass
class ACTDecoderProjectionConfig:
    enabled: bool = False
    # Keep these as plain strings so repo-native draccus reload paths remain compatible.
    kind: str = "linear"
    placement: str = "student_only"
    output_dim: int | None = None
    bias: bool = False

    def __post_init__(self) -> None:
        if self.kind != "linear":
            raise ValueError(f"`decoder_kd.projection.kind` must be `linear`. Got {self.kind}.")
        if self.placement != "student_only":
            raise ValueError(
                "`decoder_kd.projection.placement` must be `student_only`. "
                f"Got {self.placement}."
            )
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError(
                "`decoder_kd.projection.output_dim` must be strictly positive when provided. "
                f"Got {self.output_dim}."
            )


@dataclass
class ACTDecoderKDConfig:
    enabled: bool = False
    peak_weight: float = 0.0
    # Keep Stage-2 enum-like fields as plain strings so `save_pretrained` / `from_pretrained`
    # and `TrainPipelineConfig.from_pretrained(...)` can round-trip through draccus.
    loss_type: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    latent_mode: str = "zero"
    overlap_steps: int | None = None
    temporal_decay: float | None = None
    prefix_weight: float | None = None
    tail_weight: float | None = None
    require_action_kd: bool = True
    start_step: int = 0
    ramp_steps: int = 0
    anneal_start_step: int | None = None
    end_step: int | None = None
    enable_noise_gate: bool = True
    enable_grad_gate: bool = True
    log_grad_ratio: bool = False
    projection: ACTDecoderProjectionConfig | None = None

    def __post_init__(self) -> None:
        if self.peak_weight < 0:
            raise ValueError(
                f"`decoder_kd.peak_weight` must be non-negative. Got {self.peak_weight}."
            )
        if self.loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(
                "`decoder_kd.loss_type` must be one of `smooth_l1` or `mse`. "
                f"Got {self.loss_type}."
            )
        if self.smooth_l1_beta <= 0:
            raise ValueError(
                f"`decoder_kd.smooth_l1_beta` must be strictly positive. Got {self.smooth_l1_beta}."
            )
        if self.latent_mode != "zero":
            raise ValueError(
                f"`decoder_kd.latent_mode` is frozen to `zero` for Stage-2 v1. Got {self.latent_mode}."
            )
        if self.overlap_steps is not None and self.overlap_steps <= 0:
            raise ValueError(
                "`decoder_kd.overlap_steps` must be strictly positive when provided. "
                f"Got {self.overlap_steps}."
            )
        if self.temporal_decay is not None and self.temporal_decay < 0:
            raise ValueError(
                "`decoder_kd.temporal_decay` must be non-negative when provided. "
                f"Got {self.temporal_decay}."
            )
        if self.prefix_weight is not None and self.prefix_weight < 0:
            raise ValueError(
                "`decoder_kd.prefix_weight` must be non-negative when provided. "
                f"Got {self.prefix_weight}."
            )
        if self.tail_weight is not None and self.tail_weight < 0:
            raise ValueError(
                "`decoder_kd.tail_weight` must be non-negative when provided. "
                f"Got {self.tail_weight}."
            )
        if self.prefix_weight == 0 and self.tail_weight == 0:
            raise ValueError(
                "At least one of `decoder_kd.prefix_weight` or `decoder_kd.tail_weight` must be positive."
            )
        if self.start_step < 0:
            raise ValueError(f"`decoder_kd.start_step` must be non-negative. Got {self.start_step}.")
        if self.ramp_steps < 0:
            raise ValueError(f"`decoder_kd.ramp_steps` must be non-negative. Got {self.ramp_steps}.")
        if self.anneal_start_step is not None and self.anneal_start_step < 0:
            raise ValueError(
                "`decoder_kd.anneal_start_step` must be non-negative when provided. "
                f"Got {self.anneal_start_step}."
            )
        if self.end_step is not None and self.end_step < 0:
            raise ValueError(
                f"`decoder_kd.end_step` must be non-negative when provided. Got {self.end_step}."
            )
        if self.anneal_start_step is not None and self.anneal_start_step < self.start_step + self.ramp_steps:
            raise ValueError(
                "`decoder_kd.anneal_start_step` must be greater than or equal to "
                "`start_step + ramp_steps`."
            )
        if self.end_step is not None:
            min_end_step = self.anneal_start_step if self.anneal_start_step is not None else self.start_step
            if self.end_step < min_end_step:
                raise ValueError(
                    "`decoder_kd.end_step` must be greater than or equal to the last active schedule boundary. "
                    f"Got end_step={self.end_step}, minimum={min_end_step}."
                )


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    kd: bool = False
    teacher_policy_path: Path | None = None
    teacher_train_config: Path | None = None
    kd_weight: float = 1.0
    kd_overlap_steps: int | None = None
    kd_temporal_decay: float = 0.0
    kd_strict_processor_compatibility: bool = True
    kd_prefix_weight: float = 1.0
    kd_tail_weight: float = 1.0
    # Legacy field for Phase-1 / existing worktree compatibility only.
    # Stage-2 canonical decoder feature dimension remains `dim_model`.
    decoder_out_dim: int = 1024
    decoder_kd: ACTDecoderKDConfig = field(default_factory=ACTDecoderKDConfig)

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.decoder_kd, dict):
            self.decoder_kd = ACTDecoderKDConfig(**self.decoder_kd)
        elif not isinstance(self.decoder_kd, ACTDecoderKDConfig):
            raise TypeError(
                "`decoder_kd` must be an `ACTDecoderKDConfig` or a dict compatible with it. "
                f"Got {type(self.decoder_kd)}."
            )
        if isinstance(self.decoder_kd.projection, dict):
            self.decoder_kd.projection = ACTDecoderProjectionConfig(**self.decoder_kd.projection)
        elif self.decoder_kd.projection is not None and not isinstance(
            self.decoder_kd.projection, ACTDecoderProjectionConfig
        ):
            raise TypeError(
                "`decoder_kd.projection` must be an `ACTDecoderProjectionConfig` or a dict compatible with it. "
                f"Got {type(self.decoder_kd.projection)}."
            )

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.kd:
            if self.teacher_policy_path is None and self.teacher_train_config is None:
                raise ValueError(
                    "When `kd=True`, please set `teacher_policy_path` (preferred) or `teacher_train_config`."
                )
            if not self.kd_strict_processor_compatibility:
                raise NotImplementedError(
                    "Stage 1 ACT KD only supports strict processor-compatible normalized-action-space KD."
                )
            if self.kd_weight <= 0:
                raise ValueError(f"`kd_weight` must be strictly positive. Got {self.kd_weight}.")
            if self.kd_overlap_steps is not None and self.kd_overlap_steps <= 0:
                raise ValueError(
                    f"`kd_overlap_steps` must be strictly positive when provided. Got {self.kd_overlap_steps}."
                )
            if self.kd_temporal_decay < 0:
                raise ValueError(
                    f"`kd_temporal_decay` must be non-negative. Got {self.kd_temporal_decay}."
                )
            if self.kd_prefix_weight < 0:
                raise ValueError(
                    f"`kd_prefix_weight` must be non-negative. Got {self.kd_prefix_weight}."
                )
            if self.kd_tail_weight < 0:
                raise ValueError(
                    f"`kd_tail_weight` must be non-negative. Got {self.kd_tail_weight}."
                )
            if self.kd_prefix_weight == 0 and self.kd_tail_weight == 0:
                raise ValueError("At least one of `kd_prefix_weight` or `kd_tail_weight` must be positive.")
            max_overlap_steps = self.chunk_size
            if self.kd_overlap_steps is not None:
                max_overlap_steps = min(max_overlap_steps, self.kd_overlap_steps)
            get_kd_segment_layout(
                overlap_steps=max_overlap_steps,
                n_action_steps=self.n_action_steps,
                kd_prefix_weight=self.kd_prefix_weight,
                kd_tail_weight=self.kd_tail_weight,
            )
        if self.decoder_kd.enabled:
            if self.decoder_kd.require_action_kd and not self.kd:
                raise ValueError(
                    "Stage-2 `decoder_kd` requires Phase-1 action KD to be enabled when "
                    "`decoder_kd.require_action_kd=True`."
                )
            if self.decoder_kd.peak_weight <= 0:
                raise ValueError(
                    "When `decoder_kd.enabled=True`, `decoder_kd.peak_weight` must be strictly positive. "
                    f"Got {self.decoder_kd.peak_weight}."
                )

            effective_overlap_steps = self.chunk_size
            if self.decoder_kd.overlap_steps is not None:
                effective_overlap_steps = min(effective_overlap_steps, self.decoder_kd.overlap_steps)

            effective_prefix_weight = (
                self.decoder_kd.prefix_weight if self.decoder_kd.prefix_weight is not None else self.kd_prefix_weight
            )
            effective_tail_weight = (
                self.decoder_kd.tail_weight if self.decoder_kd.tail_weight is not None else self.kd_tail_weight
            )
            get_kd_segment_layout(
                overlap_steps=effective_overlap_steps,
                n_action_steps=self.n_action_steps,
                kd_prefix_weight=effective_prefix_weight,
                kd_tail_weight=effective_tail_weight,
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
