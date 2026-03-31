from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from common import install_siglip_check_shim
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from trt_runtime import TensorRTRunner


install_siglip_check_shim()

from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch


def _resolve_visual_keys(config: PI05Config) -> tuple[str, str]:
    visual_keys = list(config.image_features)
    if len(visual_keys) != 2:
        raise ValueError(f"PI05 TRT runner expects exactly 2 visual inputs, got {visual_keys}")

    top_key = next((key for key in visual_keys if key.endswith(".top")), None)
    wrist_key = next((key for key in visual_keys if key.endswith(".wrist")), None)
    if top_key is not None and wrist_key is not None:
        return top_key, wrist_key

    return visual_keys[0], visual_keys[1]


@dataclass(frozen=True)
class PiTrtArtifacts:
    engine_dir: Path
    vision_engine: Path
    prefix_engine: Path
    denoise_engine: Path
    metadata_path: Path | None = None
    metadata: dict[str, Any] | None = None


class TrtPi05PolicyAdapter(PreTrainedPolicy):
    config_class = PI05Config
    name = "pi05_trt"

    def __init__(
        self,
        config: PI05Config,
        *,
        artifacts: PiTrtArtifacts,
        trt_device: str = "cuda:0",
        num_inference_steps: int | None = None,
        noise_seed: int | None = None,
        fixed_noise: bool = False,
    ) -> None:
        super().__init__(config)
        if not str(trt_device).startswith("cuda"):
            raise ValueError("TrtPi05PolicyAdapter requires a CUDA device.")

        self.config = config
        self.device = torch.device(trt_device)
        self.artifacts = artifacts
        self.top_image_key, self.wrist_image_key = _resolve_visual_keys(config)
        self.original_action_dim = int(self.config.output_features["action"].shape[0])
        self.chunk_size = int(self.config.chunk_size)
        self.max_action_dim = int(self.config.max_action_dim)
        self.num_inference_steps = int(num_inference_steps or self.config.num_inference_steps)
        self.noise_seed = noise_seed
        self.fixed_noise = fixed_noise
        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")

        image_resolution = tuple(int(value) for value in self.config.image_resolution)
        if len(image_resolution) != 2:
            raise ValueError(f"Unexpected image_resolution: {self.config.image_resolution}")
        self.image_resolution = image_resolution

        self.vision_runner = TensorRTRunner(self.artifacts.vision_engine, device=str(self.device))
        self.prefix_runner = TensorRTRunner(self.artifacts.prefix_engine, device=str(self.device))
        self.denoise_runner = TensorRTRunner(self.artifacts.denoise_engine, device=str(self.device))
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )
        self._noise_generator = torch.Generator(device="cpu") if noise_seed is not None else None
        if self._noise_generator is not None:
            self._noise_generator.manual_seed(int(noise_seed))
        self._fixed_noise_cache: Tensor | None = None

        self._validate_engine_contract()
        self.reset()

    def _tensor_shape(self, runner: TensorRTRunner, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in runner.tensor_meta(name).shape)

    def _static_dim(self, runner: TensorRTRunner, name: str, axis: int, default: int) -> int:
        shape = self._tensor_shape(runner, name)
        if axis >= len(shape):
            return int(default)
        value = int(shape[axis])
        if value > 0:
            return value
        profile_shape = runner.tensor_meta(name).profile_shape
        if profile_shape is None:
            return int(default)
        profile_dims = profile_shape.get("opt", [])
        if axis >= len(profile_dims):
            return int(default)
        profile_value = int(profile_dims[axis])
        return profile_value if profile_value > 0 else int(default)

    def _require_exact_shape(
        self,
        runner: TensorRTRunner,
        name: str,
        expected_shape: tuple[int, ...],
        *,
        owner: str,
    ) -> None:
        actual_shape = self._tensor_shape(runner, name)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{owner} tensor `{name}` shape mismatch: expected {expected_shape}, got {actual_shape}"
            )

    def _validate_engine_contract(self) -> None:
        if "image" not in self.vision_runner.input_names:
            raise ValueError(f"Vision engine missing `image` input: {self.vision_runner.input_names}")
        if "image_embs" not in self.vision_runner.output_names:
            raise ValueError(f"Vision engine missing `image_embs` output: {self.vision_runner.output_names}")

        expected_image_shape = (1, 3, *self.image_resolution)
        self._require_exact_shape(self.vision_runner, "image", expected_image_shape, owner="Vision engine")
        vision_output_shape = self._tensor_shape(self.vision_runner, "image_embs")

        expected_prefix_inputs = {
            "image_embs_top",
            "image_embs_wrist",
            "image_mask_top",
            "image_mask_wrist",
            "tokens",
            "token_attention_mask",
        }
        missing_prefix_inputs = expected_prefix_inputs.difference(self.prefix_runner.input_names)
        if missing_prefix_inputs:
            raise ValueError(
                f"Prefix engine missing inputs {sorted(missing_prefix_inputs)}: {self.prefix_runner.input_names}"
            )
        self._require_exact_shape(self.prefix_runner, "image_embs_top", vision_output_shape, owner="Prefix engine")
        self._require_exact_shape(self.prefix_runner, "image_embs_wrist", vision_output_shape, owner="Prefix engine")
        self._require_exact_shape(self.prefix_runner, "image_mask_top", (1,), owner="Prefix engine")
        self._require_exact_shape(self.prefix_runner, "image_mask_wrist", (1,), owner="Prefix engine")
        self._require_exact_shape(
            self.prefix_runner,
            "tokens",
            (1, int(self.config.tokenizer_max_length)),
            owner="Prefix engine",
        )
        self._require_exact_shape(
            self.prefix_runner,
            "token_attention_mask",
            (1, int(self.config.tokenizer_max_length)),
            owner="Prefix engine",
        )

        if "prefix_pad_masks" not in self.prefix_runner.output_names:
            raise ValueError(
                "Prefix engine missing `prefix_pad_masks` output: "
                f"{self.prefix_runner.output_names}"
            )
        self.cache_output_names = [
            name for name in self.prefix_runner.output_names if name != "prefix_pad_masks"
        ]
        if not self.cache_output_names:
            raise ValueError("Prefix engine did not expose any cache outputs.")

        expected_denoise_inputs = {"x_t", "timestep", "prefix_pad_masks", *self.cache_output_names}
        missing_denoise_inputs = expected_denoise_inputs.difference(self.denoise_runner.input_names)
        if missing_denoise_inputs:
            raise ValueError(
                f"Denoise engine missing inputs {sorted(missing_denoise_inputs)}: {self.denoise_runner.input_names}"
            )
        if "v_t" not in self.denoise_runner.output_names:
            raise ValueError(f"Denoise engine missing `v_t` output: {self.denoise_runner.output_names}")
        self._require_exact_shape(
            self.denoise_runner,
            "x_t",
            (1, self.chunk_size, self.max_action_dim),
            owner="Denoise engine",
        )
        self._require_exact_shape(self.denoise_runner, "timestep", (1,), owner="Denoise engine")

        prefix_pad_masks_shape = self._tensor_shape(self.prefix_runner, "prefix_pad_masks")
        self._require_exact_shape(
            self.denoise_runner,
            "prefix_pad_masks",
            prefix_pad_masks_shape,
            owner="Denoise engine",
        )
        for cache_name in self.cache_output_names:
            expected_cache_shape = self._tensor_shape(self.prefix_runner, cache_name)
            self._require_exact_shape(
                self.denoise_runner,
                cache_name,
                expected_cache_shape,
                owner="Denoise engine",
            )

        denoise_output_shape = self._tensor_shape(self.denoise_runner, "v_t")
        expected_output_shape = (1, self.chunk_size, self.max_action_dim)
        if denoise_output_shape != expected_output_shape:
            raise ValueError(
                "Denoise engine `v_t` output shape mismatch: "
                f"expected {expected_output_shape}, got {denoise_output_shape}"
            )

    def close(self) -> None:
        self.vision_runner.close()
        self.prefix_runner.close()
        self.denoise_runner.close()

    def __enter__(self) -> "TrtPi05PolicyAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("TrtPi05PolicyAdapter does not support training forward().")

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.config.rtc_config is not None and self.config.rtc_config.enabled

    def describe_engines(self) -> dict[str, dict[str, Any]]:
        return {
            "vision_encoder": self.vision_runner.engine_summary(),
            "prefix_cache": self.prefix_runner.engine_summary(),
            "denoise_step": self.denoise_runner.engine_summary(),
        }

    def runtime_summary(self) -> dict[str, Any]:
        metadata = self.artifacts.metadata if isinstance(self.artifacts.metadata, dict) else {}
        requested_precision = metadata.get("verified_trt_requested_precision")
        if requested_precision is None:
            requested_precision = metadata.get("requested_trt_precision")
        if requested_precision is None:
            engine_build_settings = metadata.get("engine_build_settings", {})
            if isinstance(engine_build_settings, dict):
                requested_precision = engine_build_settings.get("precision")
        return {
            "trt_device": str(self.device),
            "variant": metadata.get("variant"),
            "requested_precision": requested_precision,
            "rtc_enabled": self._rtc_enabled(),
            "rtc_debug_enabled": (
                self.rtc_processor.is_debug_enabled() if self.rtc_processor is not None else False
            ),
            "image_keys": [self.top_image_key, self.wrist_image_key],
            "image_resolution": list(self.image_resolution),
            "chunk_size": self.chunk_size,
            "max_action_dim": self.max_action_dim,
            "original_action_dim": self.original_action_dim,
            "n_action_steps": int(self.config.n_action_steps),
            "num_inference_steps": self.num_inference_steps,
            "denoise_accepts_timestep": True,
            "fixed_noise": self.fixed_noise,
            "noise_seed": self.noise_seed,
            "metadata_path": (
                self.artifacts.metadata_path.as_posix() if self.artifacts.metadata_path is not None else None
            ),
        }

    def build_preflight_batch(self) -> dict[str, Tensor]:
        batch_size = self._static_dim(self.vision_runner, "image", 0, default=1)
        token_length = self._static_dim(
            self.prefix_runner,
            "tokens",
            1,
            default=int(self.config.tokenizer_max_length),
        )
        height, width = self.image_resolution

        num_image_values = batch_size * 3 * height * width
        base_image = torch.linspace(
            0.0,
            1.0,
            steps=num_image_values,
            dtype=torch.float32,
            device=self.device,
        ).view(batch_size, 3, height, width)
        wrist_image = torch.remainder(base_image * 0.75 + 0.2, 1.0)
        token_values = (torch.arange(token_length, dtype=torch.int64, device=self.device) % 255) + 1
        tokens = token_values.unsqueeze(0).expand(batch_size, -1).contiguous()
        token_attention_mask = torch.ones(batch_size, token_length, dtype=torch.int32, device=self.device)

        return {
            self.top_image_key: base_image.contiguous(),
            self.wrist_image_key: wrist_image.contiguous(),
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: token_attention_mask,
        }

    def _make_preflight_noise(self, batch_size: int) -> Tensor:
        total_values = batch_size * self.chunk_size * self.max_action_dim
        return torch.linspace(
            -0.5,
            0.5,
            steps=total_values,
            dtype=torch.float32,
            device=self.device,
        ).view(batch_size, self.chunk_size, self.max_action_dim)

    @torch.no_grad()
    def run_preflight(self, *, warmup_num_inference_steps: int | None = None) -> dict[str, Any]:
        batch = self.build_preflight_batch()
        runtime_inputs = self._extract_runtime_inputs(batch)
        batch_size = int(runtime_inputs["tokens"].shape[0])

        top_image_embs = self.vision_runner.infer({"image": runtime_inputs["top_image"]})["image_embs"]
        wrist_image_embs = self.vision_runner.infer({"image": runtime_inputs["wrist_image"]})["image_embs"]
        prefix_outputs = self.prefix_runner.infer(
            {
                "image_embs_top": top_image_embs,
                "image_embs_wrist": wrist_image_embs,
                "image_mask_top": runtime_inputs["image_mask_top"],
                "image_mask_wrist": runtime_inputs["image_mask_wrist"],
                "tokens": runtime_inputs["tokens"],
                "token_attention_mask": runtime_inputs["token_attention_mask"],
            }
        )

        shared_denoise_inputs = {
            "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
            **{name: prefix_outputs[name] for name in self.cache_output_names},
        }
        noise = self._make_preflight_noise(batch_size)
        timestep_hi = torch.full((batch_size,), 1.0, dtype=torch.float32, device=self.device)
        timestep_lo = torch.full((batch_size,), 0.5, dtype=torch.float32, device=self.device)
        denoise_hi = self.denoise_runner.infer({"x_t": noise, "timestep": timestep_hi, **shared_denoise_inputs})["v_t"]
        denoise_lo = self.denoise_runner.infer({"x_t": noise, "timestep": timestep_lo, **shared_denoise_inputs})["v_t"]
        if not torch.isfinite(denoise_hi).all() or not torch.isfinite(denoise_lo).all():
            raise RuntimeError("TRT denoise preflight produced non-finite outputs.")

        timestep_delta_max_abs = float((denoise_hi - denoise_lo).abs().max().item())
        if timestep_delta_max_abs <= 1e-6:
            raise RuntimeError(
                "TRT denoise preflight detected no response to changing the live `timestep` input."
            )

        warmup_steps = int(warmup_num_inference_steps or min(self.num_inference_steps, 3))
        if warmup_steps <= 0:
            raise ValueError(f"warmup_num_inference_steps must be positive, got {warmup_steps}")
        action_chunk = self.predict_action_chunk(
            batch,
            noise=noise.clone(),
            num_inference_steps=warmup_steps,
        )
        expected_shape = (batch_size, self.chunk_size, self.original_action_dim)
        actual_shape = tuple(int(value) for value in action_chunk.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"TRT warmup action chunk shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
        if not torch.isfinite(action_chunk).all():
            raise RuntimeError("TRT warmup action chunk contains non-finite values.")

        self.reset()
        return {
            "status": "pass",
            "batch_size": batch_size,
            "token_length": int(runtime_inputs["tokens"].shape[1]),
            "vision_embedding_shape": list(top_image_embs.shape),
            "prefix_pad_masks_shape": list(prefix_outputs["prefix_pad_masks"].shape),
            "cache_tensor_count": len(self.cache_output_names),
            "cache_tensor_shape": list(prefix_outputs[self.cache_output_names[0]].shape),
            "warmup_num_inference_steps": warmup_steps,
            "action_chunk_shape": list(action_chunk.shape),
            "action_chunk_max_abs": float(action_chunk.abs().max().item()),
            "action_chunk_mean_abs": float(action_chunk.abs().mean().item()),
            "denoise_timestep_delta_max_abs": timestep_delta_max_abs,
        }

    def _require_tensor(self, batch: dict[str, Tensor], key: str) -> Tensor:
        if key not in batch:
            raise KeyError(f"Missing runtime batch key: {key}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Runtime batch key `{key}` must be a torch.Tensor, got {type(value)}")
        return value

    def _prepare_image(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got shape={tuple(image.shape)}")

        image = image.to(device=self.device, dtype=torch.float32).contiguous()
        if image.shape[1] == 3:
            is_channels_first = True
            image = image.permute(0, 2, 3, 1)
        elif image.shape[-1] == 3:
            is_channels_first = False
        else:
            raise ValueError(
                "Expected image tensor in BCHW or BHWC layout with 3 channels, "
                f"got shape={tuple(image.shape)}"
            )
        if tuple(int(value) for value in image.shape[1:3]) != self.image_resolution:
            image = resize_with_pad_torch(image, *self.image_resolution)
        image = image * 2.0 - 1.0
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        else:
            image = image.permute(0, 3, 1, 2)
        return image.contiguous()

    def _extract_runtime_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        top_image = self._prepare_image(self._require_tensor(batch, self.top_image_key))
        wrist_image = self._prepare_image(self._require_tensor(batch, self.wrist_image_key))
        tokens = self._require_tensor(batch, OBS_LANGUAGE_TOKENS).to(
            device=self.device,
            dtype=torch.int64,
        )
        token_attention_mask = self._require_tensor(batch, OBS_LANGUAGE_ATTENTION_MASK).to(
            device=self.device,
            dtype=torch.int32,
        )

        batch_size = int(top_image.shape[0])
        return {
            "top_image": top_image,
            "wrist_image": wrist_image,
            "image_mask_top": torch.ones(batch_size, dtype=torch.int32, device=self.device),
            "image_mask_wrist": torch.ones(batch_size, dtype=torch.int32, device=self.device),
            "tokens": tokens.contiguous(),
            "token_attention_mask": token_attention_mask.contiguous(),
        }

    def _init_noise(self, batch_size: int, noise: Tensor | None) -> Tensor:
        expected_shape = (batch_size, self.chunk_size, self.max_action_dim)
        if noise is not None:
            if not isinstance(noise, torch.Tensor):
                raise TypeError(f"`noise` must be a torch.Tensor, got {type(noise)}")
            if tuple(int(value) for value in noise.shape) != expected_shape:
                raise ValueError(f"`noise` shape must be {expected_shape}, got {tuple(noise.shape)}")
            return noise.to(device=self.device, dtype=torch.float32).contiguous()

        if self.fixed_noise and self._fixed_noise_cache is not None:
            if tuple(int(value) for value in self._fixed_noise_cache.shape) == expected_shape:
                return self._fixed_noise_cache

        if self._noise_generator is not None:
            sampled_noise = torch.randn(
                expected_shape,
                generator=self._noise_generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=self.device)
        else:
            sampled_noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=expected_shape,
                dtype=torch.float32,
                device=self.device,
            )

        sampled_noise = sampled_noise.contiguous()
        if self.fixed_noise:
            self._fixed_noise_cache = sampled_noise
        return sampled_noise

    def _prepare_prev_chunk_left_over(self, prev_chunk_left_over: Tensor | None, batch_size: int) -> Tensor | None:
        if prev_chunk_left_over is None:
            return None
        if not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError(
                "`prev_chunk_left_over` must be a torch.Tensor or None, "
                f"got {type(prev_chunk_left_over)}"
            )
        if prev_chunk_left_over.ndim not in (2, 3):
            raise ValueError(
                "`prev_chunk_left_over` must have shape (T, A) or (B, T, A), "
                f"got {tuple(prev_chunk_left_over.shape)}"
            )

        prev_chunk_left_over = prev_chunk_left_over.to(
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        if prev_chunk_left_over.shape[0] == 1 and batch_size > 1:
            prev_chunk_left_over = prev_chunk_left_over.expand(batch_size, -1, -1).contiguous()
        elif prev_chunk_left_over.shape[0] != batch_size:
            raise ValueError(
                "`prev_chunk_left_over` batch dimension must be 1 or match the runtime batch size, "
                f"got {prev_chunk_left_over.shape[0]} vs {batch_size}"
            )
        return prev_chunk_left_over

    def _resolve_rtc_kwargs(
        self,
        kwargs: dict[str, Any],
        batch_size: int,
    ) -> tuple[Tensor | None, int | None, int | None]:
        prev_chunk_left_over = self._prepare_prev_chunk_left_over(
            kwargs.get("prev_chunk_left_over"),
            batch_size,
        )
        inference_delay = kwargs.get("inference_delay")
        if inference_delay is not None:
            inference_delay = int(inference_delay)
        execution_horizon = kwargs.get("execution_horizon")
        if execution_horizon is not None:
            execution_horizon = int(execution_horizon)
        if prev_chunk_left_over is not None and inference_delay is None:
            raise ValueError("`inference_delay` is required when `prev_chunk_left_over` is provided.")
        return prev_chunk_left_over, inference_delay, execution_horizon

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        runtime_inputs = self._extract_runtime_inputs(batch)

        top_image_embs = self.vision_runner.infer({"image": runtime_inputs["top_image"]})["image_embs"]
        wrist_image_embs = self.vision_runner.infer({"image": runtime_inputs["wrist_image"]})["image_embs"]

        prefix_outputs = self.prefix_runner.infer(
            {
                "image_embs_top": top_image_embs,
                "image_embs_wrist": wrist_image_embs,
                "image_mask_top": runtime_inputs["image_mask_top"],
                "image_mask_wrist": runtime_inputs["image_mask_wrist"],
                "tokens": runtime_inputs["tokens"],
                "token_attention_mask": runtime_inputs["token_attention_mask"],
            }
        )

        batch_size = int(runtime_inputs["tokens"].shape[0])
        x_t = self._init_noise(batch_size, kwargs.get("noise"))
        num_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        if num_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_steps}")
        prev_chunk_left_over, inference_delay, execution_horizon = self._resolve_rtc_kwargs(
            kwargs,
            batch_size,
        )

        dt = torch.tensor(-1.0 / float(num_steps), dtype=torch.float32, device=self.device)
        timestep_values = 1.0 - (
            torch.arange(num_steps, dtype=torch.float32, device=self.device) / float(num_steps)
        )

        shared_denoise_inputs = {
            "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
            **{name: prefix_outputs[name] for name in self.cache_output_names},
        }

        for timestep_value in timestep_values:
            expanded_timestep = timestep_value.expand(batch_size)

            def denoise_step_partial_call(input_x_t: Tensor, current_timestep: Tensor = expanded_timestep) -> Tensor:
                denoise_feed = {
                    "x_t": input_x_t,
                    "timestep": current_timestep,
                    **shared_denoise_inputs,
                }
                return self.denoise_runner.infer(denoise_feed)["v_t"].to(dtype=torch.float32)

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=timestep_value,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            x_t = (x_t + dt * v_t).contiguous()
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=timestep_value, x_t=x_t, v_t=v_t)

        return x_t[:, :, : self.original_action_dim].to(dtype=torch.float32).contiguous()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()
