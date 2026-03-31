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
from onnx_runtime import OnnxRuntimeRunner


install_siglip_check_shim()

from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch


def _resolve_visual_keys(config: PI05Config) -> tuple[str, str]:
    visual_keys = list(config.image_features)
    if len(visual_keys) != 2:
        raise ValueError(f"PI05 ONNX runner expects exactly 2 visual inputs, got {visual_keys}")

    top_key = next((key for key in visual_keys if key.endswith(".top")), None)
    wrist_key = next((key for key in visual_keys if key.endswith(".wrist")), None)
    if top_key is not None and wrist_key is not None:
        return top_key, wrist_key

    return visual_keys[0], visual_keys[1]


@dataclass(frozen=True)
class OnnxPiArtifacts:
    onnx_dir: Path
    vision_onnx: Path
    prefix_onnx: Path
    denoise_onnx: Path
    stage2_report_path: Path | None = None
    stage2_payload: dict[str, Any] | None = None


class OnnxPi05PolicyAdapter(PreTrainedPolicy):
    config_class = PI05Config
    name = "pi05_onnx"

    def __init__(
        self,
        config: PI05Config,
        *,
        artifacts: OnnxPiArtifacts,
        onnx_provider: str = "auto",
        num_inference_steps: int | None = None,
        noise_seed: int | None = None,
        fixed_noise: bool = False,
    ) -> None:
        super().__init__(config)

        self.config = config
        self.device = torch.device("cpu")
        self.artifacts = artifacts
        self.onnx_provider = onnx_provider
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

        self.vision_runner = OnnxRuntimeRunner(
            self.artifacts.vision_onnx,
            provider=onnx_provider,
            optimization_level="all",
        )
        self.prefix_runner = OnnxRuntimeRunner(
            self.artifacts.prefix_onnx,
            provider=onnx_provider,
            optimization_level="disable",
        )
        self.denoise_runner = OnnxRuntimeRunner(
            self.artifacts.denoise_onnx,
            provider=onnx_provider,
            optimization_level="disable",
        )
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )
        self._noise_generator = torch.Generator(device="cpu") if noise_seed is not None else None
        if self._noise_generator is not None:
            self._noise_generator.manual_seed(int(noise_seed))
        self._fixed_noise_cache: Tensor | None = None

        self._validate_session_contract()
        self.reset()

    def _validate_session_contract(self) -> None:
        if "image" not in self.vision_runner.input_names:
            raise ValueError(f"Vision ONNX missing `image` input: {self.vision_runner.input_names}")
        if "image_embs" not in self.vision_runner.output_names:
            raise ValueError(f"Vision ONNX missing `image_embs` output: {self.vision_runner.output_names}")

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
                f"Prefix ONNX missing inputs {sorted(missing_prefix_inputs)}: {self.prefix_runner.input_names}"
            )

        if "prefix_pad_masks" not in self.prefix_runner.output_names:
            raise ValueError(
                "Prefix ONNX missing `prefix_pad_masks` output: "
                f"{self.prefix_runner.output_names}"
            )
        self.cache_output_names = [
            name for name in self.prefix_runner.output_names if name != "prefix_pad_masks"
        ]
        if not self.cache_output_names:
            raise ValueError("Prefix ONNX did not expose any cache outputs.")

        expected_denoise_inputs = {"x_t", "prefix_pad_masks", *self.cache_output_names}
        missing_denoise_inputs = expected_denoise_inputs.difference(self.denoise_runner.input_names)
        if missing_denoise_inputs:
            raise ValueError(
                f"Denoise ONNX missing inputs {sorted(missing_denoise_inputs)}: {self.denoise_runner.input_names}"
            )
        if "v_t" not in self.denoise_runner.output_names:
            raise ValueError(f"Denoise ONNX missing `v_t` output: {self.denoise_runner.output_names}")

        self.denoise_accepts_timestep = "timestep" in self.denoise_runner.input_names

    def close(self) -> None:
        self.vision_runner.close()
        self.prefix_runner.close()
        self.denoise_runner.close()

    def __enter__(self) -> "OnnxPi05PolicyAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("OnnxPi05PolicyAdapter does not support training forward().")

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
        return {
            "onnx_provider": self.onnx_provider,
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
            "denoise_accepts_timestep": self.denoise_accepts_timestep,
            "fixed_noise": self.fixed_noise,
            "noise_seed": self.noise_seed,
            "stage2_report_path": (
                self.artifacts.stage2_report_path.as_posix()
                if self.artifacts.stage2_report_path is not None
                else None
            ),
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
            image = image.permute(0, 2, 3, 1)
        elif image.shape[-1] != 3:
            raise ValueError(
                "Expected image tensor in BCHW or BHWC layout with 3 channels, "
                f"got shape={tuple(image.shape)}"
            )

        if tuple(int(value) for value in image.shape[1:3]) != self.image_resolution:
            image = resize_with_pad_torch(image, *self.image_resolution)
        image = image * 2.0 - 1.0
        return image.permute(0, 3, 1, 2).contiguous()

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
                device=self.device,
            )
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
                    **shared_denoise_inputs,
                }
                if self.denoise_accepts_timestep:
                    denoise_feed["timestep"] = current_timestep
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
