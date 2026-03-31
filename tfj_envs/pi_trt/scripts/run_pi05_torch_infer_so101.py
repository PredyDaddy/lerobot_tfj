#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PI_TRT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common import (
    DEFAULT_POSTPROCESSOR_CONFIG_FILENAME,
    discover_local_tokenizer_path,
    load_policy_preprocessor_from_checkpoint,
    read_json,
    resolve_checkpoint_dir,
)
from pi05_chunk_runtime import (
    AsyncChunkPrefetcher,
    ChunkPredictionResult,
    build_chunk_predict_kwargs,
    estimate_prefetch_threshold,
    merge_chunk_prediction_result,
)

from lerobot import policies  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.robots import make_robot_from_config, so101_follower  # noqa: F401
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging


DEFAULT_POLICY_PATH = REPO_ROOT / "pi_model" / "pretrained_model"
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _discover_default_safe_trt_path() -> Path | None:
    results_root = PI_TRT_ROOT / "docs" / "results"
    if not results_root.is_dir():
        return None

    candidates: list[tuple[str, str, Path]] = []
    for metadata_path in sorted(results_root.glob("**/pi_trt_metadata.json")):
        try:
            metadata = read_json(metadata_path)
        except Exception:
            continue

        if metadata.get("variant") != "pi05":
            continue
        if not metadata.get("checkpoint_dir"):
            continue

        stage_status = metadata.get("stage_status", {})
        if _normalize_status(stage_status.get("stage5_verify_trt")) != "pass":
            continue

        created_at = str(metadata.get("created_at") or "")
        candidates.append((created_at, metadata_path.as_posix(), metadata_path.resolve(strict=False)))

    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][2]


DEFAULT_TRT_PATH = _discover_default_safe_trt_path()
DEFAULT_TRT_METADATA_PATH = DEFAULT_TRT_PATH


@dataclass(frozen=True)
class ResolvedRTCRuntimeConfig:
    checkpoint_enabled: bool | None
    config: RTCConfig
    enabled_by_cli: bool
    override_applied: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "prefix_attention_schedule": self.config.prefix_attention_schedule.value,
            "max_guidance_weight": float(self.config.max_guidance_weight),
            "execution_horizon": int(self.config.execution_horizon),
            "debug": bool(self.config.debug),
            "debug_maxlen": int(self.config.debug_maxlen),
            "checkpoint_enabled": self.checkpoint_enabled,
            "enabled_by_cli": self.enabled_by_cli,
            "override_applied": self.override_applied,
        }


class TorchChunkPolicyRuntime:
    """Keep the shared chunk helper path while preserving optional CUDA AMP."""

    def __init__(self, policy: PI05Policy, *, device: torch.device, use_amp: bool) -> None:
        self.policy = policy
        self.device = device
        self.use_amp = bool(use_amp)
        self.config = policy.config

    def _rtc_enabled(self) -> bool:
        rtc_config = getattr(self.config, "rtc_config", None)
        return bool(rtc_config is not None and getattr(rtc_config, "enabled", False))

    def _predict_grad_context(self) -> Any:
        if not self._rtc_enabled():
            return torch.inference_mode()

        is_inference_mode_enabled = getattr(torch, "is_inference_mode_enabled", None)
        if callable(is_inference_mode_enabled) and bool(is_inference_mode_enabled()):
            raise RuntimeError(
                "RTC-enabled PI05 torch runtime requires autograd during chunk denoising, "
                "but `predict_action_chunk(...)` entered with `torch.inference_mode()` already active. "
                "Remove the outer inference_mode wrapper for RTC-enabled runs."
            )
        return torch.enable_grad()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # RTC guidance calls torch.autograd.grad during chunk denoising, so the
        # launcher keeps inference_mode only for the non-RTC path.
        with self._predict_grad_context(), (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        ):
            return self.policy.predict_action_chunk(batch, **kwargs)


def stage(message: str) -> None:
    print(f"[STAGE] {message}", flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return float(value)


def parse_rtc_attention_schedule(value: str | None) -> RTCAttentionSchedule | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    try:
        return RTCAttentionSchedule(normalized)
    except ValueError as exc:
        valid = ", ".join(item.value for item in RTCAttentionSchedule)
        raise ValueError(f"Unsupported RTC attention schedule {value!r}. Expected one of: {valid}") from exc


def add_rtc_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    rtc_choices = [item.value.lower() for item in RTCAttentionSchedule]
    parser.add_argument(
        "--rtc-enable",
        "--rtc-enabled",
        dest="rtc_enable",
        action="store_true",
        help=(
            "Enable RTC-aware chunk runtime. `--rtc-enabled` is kept as a compatibility alias. "
            "Default remains off unless this or another --rtc-* override is set."
        ),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=None,
        help="RTC execution horizon in action steps. If set, RTC is enabled for the launcher runtime.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=None,
        help="RTC max guidance weight. If set, RTC is enabled for the launcher runtime.",
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        type=str.lower,
        choices=rtc_choices,
        default=None,
        help="RTC prefix attention schedule. If set, RTC is enabled for the launcher runtime.",
    )
    parser.add_argument(
        "--rtc-debug",
        action="store_true",
        help="Enable RTC debug tracking. Also enables RTC for the launcher runtime.",
    )
    parser.add_argument(
        "--rtc-debug-maxlen",
        type=int,
        default=None,
        help="RTC debug ring-buffer size. If set, RTC is enabled for the launcher runtime.",
    )


def _rtc_override_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            bool(getattr(args, "rtc_enable", False)),
            getattr(args, "rtc_execution_horizon", None) is not None,
            getattr(args, "rtc_max_guidance_weight", None) is not None,
            getattr(args, "rtc_prefix_attention_schedule", None) is not None,
            bool(getattr(args, "rtc_debug", False)),
            getattr(args, "rtc_debug_maxlen", None) is not None,
        )
    )


def resolve_rtc_runtime_config(args: argparse.Namespace, policy_cfg: object) -> ResolvedRTCRuntimeConfig:
    if getattr(policy_cfg, "type", None) != "pi05":
        raise ValueError(f"Expected PI05 policy, got {getattr(policy_cfg, 'type', None)!r}")

    checkpoint_cfg = getattr(policy_cfg, "rtc_config", None)
    if checkpoint_cfg is not None and not isinstance(checkpoint_cfg, RTCConfig):
        raise TypeError(f"Expected rtc_config to be RTCConfig or None, got {type(checkpoint_cfg)}")

    base_cfg = checkpoint_cfg or RTCConfig()
    override_applied = _rtc_override_requested(args)
    enabled = bool(getattr(args, "rtc_enable", False) or override_applied)
    schedule = (
        parse_rtc_attention_schedule(getattr(args, "rtc_prefix_attention_schedule", None))
        or base_cfg.prefix_attention_schedule
    )
    max_guidance_weight = (
        float(args.rtc_max_guidance_weight)
        if getattr(args, "rtc_max_guidance_weight", None) is not None
        else float(base_cfg.max_guidance_weight)
    )
    execution_horizon = (
        int(args.rtc_execution_horizon)
        if getattr(args, "rtc_execution_horizon", None) is not None
        else int(base_cfg.execution_horizon)
    )
    debug = bool(getattr(args, "rtc_debug", False) or (enabled and bool(base_cfg.debug) and not override_applied))
    debug_maxlen = (
        int(args.rtc_debug_maxlen)
        if getattr(args, "rtc_debug_maxlen", None) is not None
        else int(base_cfg.debug_maxlen)
    )

    if execution_horizon <= 0:
        raise ValueError(f"--rtc-execution-horizon must be positive, got {execution_horizon}")
    if max_guidance_weight <= 0.0:
        raise ValueError(f"--rtc-max-guidance-weight must be positive, got {max_guidance_weight}")
    if debug_maxlen <= 0:
        raise ValueError(f"--rtc-debug-maxlen must be positive, got {debug_maxlen}")

    resolved_cfg = RTCConfig(
        enabled=enabled,
        prefix_attention_schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
        debug=debug,
        debug_maxlen=debug_maxlen,
    )
    policy_cfg.rtc_config = resolved_cfg
    return ResolvedRTCRuntimeConfig(
        checkpoint_enabled=(checkpoint_cfg.enabled if checkpoint_cfg is not None else None),
        config=resolved_cfg,
        enabled_by_cli=bool(getattr(args, "rtc_enable", False)),
        override_applied=override_applied,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PI0.5 PyTorch/model.safetensors inference on a real SO101 follower robot.")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument(
        "--robot-max-relative-target",
        type=parse_optional_float,
        default=None,
        help="Optional SO101 per-step relative target clamp. Smaller is safer and less jumpy.",
    )

    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-use-amp", action="store_true", help="Enable AMP during PyTorch inference.")
    parser.add_argument("--policy-n-action-steps", type=parse_optional_int, default=None)
    parser.add_argument("--policy-num-inference-steps", type=parse_optional_int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=parse_optional_float, default=None)

    parser.add_argument(
        "--trt-path",
        default=str(DEFAULT_TRT_PATH) if DEFAULT_TRT_PATH is not None else None,
        help=(
            "Optional TensorRT artifact path kept for CLI compatibility. "
            "The torch launcher does not execute TRT inference, but can preflight these artifacts."
        ),
    )
    parser.add_argument(
        "--trt-metadata-path",
        default=str(DEFAULT_TRT_METADATA_PATH) if DEFAULT_TRT_METADATA_PATH is not None else None,
        help=(
            "Optional pi_trt_metadata.json path kept for CLI compatibility. "
            "Used only for TensorRT artifact preflight in the torch launcher."
        ),
    )
    parser.add_argument(
        "--trt-device",
        default="cuda:0",
        help="Accepted for CLI compatibility. Torch inference still runs on --policy-device.",
    )
    parser.add_argument("--local-tokenizer-path", default=None)

    parser.add_argument("--task", default="grasp block in bin")
    parser.add_argument("--run-time-s", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument(
        "--prefetch-threshold",
        type=int,
        default=None,
        help="Queue size threshold for starting async chunk prefetch. Default is latency-aware and computed at runtime.",
    )
    parser.add_argument(
        "--sync-refill-timeout-s",
        type=float,
        default=1.0,
        help="Grace period for collecting a just-finished async chunk before a synchronous refill.",
    )
    parser.add_argument(
        "--joint-delta-limit",
        type=parse_optional_float,
        default=None,
        help="Optional per-step clamp in robot command space for arm joints. Uses current observation as reference.",
    )
    parser.add_argument(
        "--gripper-delta-limit",
        type=parse_optional_float,
        default=None,
        help="Optional per-step clamp in robot command space for the gripper. Defaults to --joint-delta-limit.",
    )
    parser.add_argument(
        "--joint-action-alpha",
        type=parse_optional_float,
        default=None,
        help="Optional smoothing factor in (0, 1]. Smaller values make arm commands less jittery.",
    )
    parser.add_argument(
        "--gripper-action-alpha",
        type=parse_optional_float,
        default=None,
        help="Optional smoothing factor in (0, 1] for gripper commands. Defaults to --joint-action-alpha.",
    )
    add_rtc_runtime_arguments(parser)

    parser.add_argument("--skip-camera-preflight", action="store_true")
    parser.add_argument("--skip-policy-preflight", action="store_true")
    parser.add_argument("--skip-trt-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    policy_dir = resolve_checkpoint_dir(args.policy_path)
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")
    tokenizer_dir = discover_local_tokenizer_path(args.local_tokenizer_path, require=False)
    return policy_dir, calib_dir, tokenizer_dir


def preflight_cameras(args: argparse.Namespace) -> None:
    import cv2

    for camera_index in [args.top_cam_index, args.wrist_cam_index]:
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Camera {camera_index} failed to open")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera {camera_index} opened but failed to read a frame")
            info(f"Camera {camera_index} OK: frame_shape={tuple(frame.shape)}")
        finally:
            cap.release()


def preflight_trt_artifacts(args: argparse.Namespace) -> None:
    trt_path_value = args.trt_path or args.trt_metadata_path
    if trt_path_value is None:
        info("No TRT artifact path provided. Skipping TRT preflight for torch runtime.")
        return

    trt_path = Path(trt_path_value).expanduser().resolve(strict=False)
    if not trt_path.exists():
        raise FileNotFoundError(f"TRT artifact path not found: {trt_path}")

    metadata_path: Path | None = None
    if args.trt_metadata_path:
        metadata_path = Path(args.trt_metadata_path).expanduser().resolve(strict=False)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"TRT metadata path not found: {metadata_path}")
    elif trt_path.is_file() and trt_path.suffix.lower() == ".json":
        metadata_path = trt_path
    elif trt_path.is_dir():
        candidate = trt_path / "pi_trt_metadata.json"
        if candidate.is_file():
            metadata_path = candidate.resolve(strict=False)

    info(
        "TRT preflight (torch runtime only, artifacts are not used for live inference): "
        f"path={trt_path}"
    )
    if metadata_path is None:
        warn("No TRT metadata json was resolved. Continuing because this launcher runs the PyTorch checkpoint.")
        return

    try:
        metadata = read_json(metadata_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TRT metadata json: {metadata_path}") from exc

    stage_status = metadata.get("stage_status", {})
    info(f"TRT metadata path: {metadata_path}")
    info(f"TRT metadata stage_status: {stage_status or '<missing>'}")
    if _normalize_status(stage_status.get("stage5_verify_trt")) != "pass":
        warn(
            "TRT metadata does not show stage5_verify_trt=pass. "
            "Torch checkpoint inference can still continue, but these TRT artifacts are not confirmed safe."
        )


def build_robot_config(args: argparse.Namespace, calib_dir: Path) -> SO101FollowerConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
    }
    return SO101FollowerConfig(
        id=args.robot_id,
        calibration_dir=calib_dir,
        port=args.robot_port,
        max_relative_target=args.robot_max_relative_target,
        cameras=cameras,
    )


def apply_pi_runtime_overrides(args: argparse.Namespace, policy_cfg: object) -> ResolvedRTCRuntimeConfig:
    if getattr(policy_cfg, "type", None) != "pi05":
        raise ValueError(f"Expected PI05 policy, got {getattr(policy_cfg, 'type', None)!r}")

    chunk_size = int(policy_cfg.chunk_size)
    if args.policy_n_action_steps is not None:
        if not 1 <= args.policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {args.policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = int(args.policy_n_action_steps)

    if args.policy_num_inference_steps is not None:
        if args.policy_num_inference_steps <= 0:
            raise ValueError("--policy-num-inference-steps must be positive")
        policy_cfg.num_inference_steps = int(args.policy_num_inference_steps)

    if args.policy_temporal_ensemble_coeff is not None:
        raise ValueError("PI05 PyTorch runtime does not support temporal ensembling.")

    policy_cfg.use_amp = bool(args.policy_use_amp)
    return resolve_rtc_runtime_config(args, policy_cfg)


def load_policy_config(policy_dir: Path, policy_device: str) -> PreTrainedConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = str(policy_dir)
    policy_cfg.device = policy_device
    if policy_cfg.type != "pi05":
        raise ValueError(f"Expected PI05 policy, got {policy_cfg.type}")
    return policy_cfg


def load_postprocessor(policy_dir: Path) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename=DEFAULT_POSTPROCESSOR_CONFIG_FILENAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        local_files_only=True,
    )


def preflight_policy(policy_dir: Path, policy_cfg: object):
    policy = PI05Policy.from_pretrained(
        str(policy_dir),
        config=policy_cfg,
        local_files_only=True,
        strict=False,
    )
    policy.eval()
    info(
        "PI05 PyTorch policy OK: "
        f"device={policy_cfg.device}, use_amp={policy_cfg.use_amp}, "
        f"rtc_enabled={getattr(getattr(policy_cfg, 'rtc_config', None), 'enabled', False)}"
    )
    info(
        "PI05 runtime: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"num_inference_steps={policy_cfg.num_inference_steps}"
    )
    return policy


def print_summary(
    args: argparse.Namespace,
    policy_dir: Path,
    calib_dir: Path,
    tokenizer_dir: Path | None,
    policy_cfg: object,
    preprocessor_details: dict,
    rtc_runtime: ResolvedRTCRuntimeConfig,
) -> None:
    info(f"Python: {sys.executable}")
    info(f"Policy path: {policy_dir}")
    info(f"Policy type: {getattr(policy_cfg, 'type', '<unknown>')}")
    info(
        "PI05 runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"num_inference_steps={policy_cfg.num_inference_steps}, "
        f"use_amp={policy_cfg.use_amp}"
    )
    info(f"Resolved RTC config: {rtc_runtime.as_dict()}")
    if rtc_runtime.checkpoint_enabled and not rtc_runtime.config.enabled:
        warn(
            "Checkpoint RTC config is enabled, but launcher runtime keeps RTC off by default "
            "unless --rtc-enable/--rtc-enabled or another --rtc-* override is provided."
        )
    info(f"Policy device: {args.policy_device}")
    info(f"Calibration dir: {calib_dir}")
    info(f"Robot port: {args.robot_port}")
    info(f"Robot max_relative_target: {args.robot_max_relative_target}")
    info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    info(f"Task: {args.task}")
    info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    info(f"sync_refill_timeout_s: {args.sync_refill_timeout_s}")
    info(
        "prefetch_threshold: "
        f"{args.prefetch_threshold if args.prefetch_threshold is not None else '<latency-aware>'}"
    )
    info(
        "Tokenizer path: "
        f"{preprocessor_details.get('local_tokenizer_path') or tokenizer_dir or '<unresolved>'}"
    )
    info(
        "TRT compatibility args: "
        f"path={Path(args.trt_path).expanduser().resolve(strict=False) if args.trt_path else '<unset>'}, "
        f"metadata={Path(args.trt_metadata_path).expanduser().resolve(strict=False) if args.trt_metadata_path else '<unset>'}, "
        f"trt_device={args.trt_device}"
    )
    info("Torch runtime note: --trt-* flags are accepted for preflight/compatibility only; live inference uses pi_model weights.")
    info(f"Script joint_delta_limit: {args.joint_delta_limit}")
    info(
        "Script gripper_delta_limit: "
        f"{args.gripper_delta_limit if args.gripper_delta_limit is not None else args.joint_delta_limit}"
    )
    info(f"Script joint_action_alpha: {args.joint_action_alpha}")
    info(
        "Script gripper_action_alpha: "
        f"{args.gripper_action_alpha if args.gripper_action_alpha is not None else args.joint_action_alpha}"
    )


def build_dataset_features(robot) -> dict:
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return dataset_features


def make_hold_robot_action(observation: dict, action_keys: list[str]) -> dict[str, float]:
    missing = [key for key in action_keys if key not in observation]
    if missing:
        raise KeyError(f"Observation missing action keys required for hold action: {missing}")
    return {key: float(observation[key]) for key in action_keys}


def smooth_robot_action(
    robot_action: dict[str, float],
    reference_action: dict[str, float],
    *,
    joint_action_alpha: float | None,
    gripper_action_alpha: float | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if joint_action_alpha is None and gripper_action_alpha is None:
        return robot_action, {}

    effective_gripper_alpha = gripper_action_alpha if gripper_action_alpha is not None else joint_action_alpha
    smoothed_action = dict(robot_action)
    smoothed_joints: dict[str, dict[str, float]] = {}

    for key, target in robot_action.items():
        if not key.endswith(".pos") or key not in reference_action:
            continue

        alpha = effective_gripper_alpha if key == "gripper.pos" else joint_action_alpha
        if alpha is None:
            continue

        reference_value = float(reference_action[key])
        safe_target = reference_value + float(alpha) * (float(target) - reference_value)
        smoothed_action[key] = safe_target
        if abs(safe_target - float(target)) > 1e-6:
            smoothed_joints[key] = {
                "original_target": float(target),
                "smoothed_target": safe_target,
                "reference_action": reference_value,
            }

    return smoothed_action, smoothed_joints


def clamp_robot_action_delta(
    robot_action: dict[str, float],
    observation: dict,
    *,
    joint_delta_limit: float | None,
    gripper_delta_limit: float | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if joint_delta_limit is None and gripper_delta_limit is None:
        return robot_action, {}

    effective_gripper_limit = gripper_delta_limit if gripper_delta_limit is not None else joint_delta_limit
    clipped_action = dict(robot_action)
    clipped_joints: dict[str, dict[str, float]] = {}

    for key, target in robot_action.items():
        if not key.endswith(".pos") or key not in observation:
            continue

        limit = effective_gripper_limit if key == "gripper.pos" else joint_delta_limit
        if limit is None or limit <= 0:
            continue

        current = float(observation[key])
        min_target = current - limit
        max_target = current + limit
        safe_target = min(max(float(target), min_target), max_target)
        clipped_action[key] = safe_target
        if abs(safe_target - float(target)) > 1e-6:
            clipped_joints[key] = {
                "original_target": float(target),
                "clipped_target": safe_target,
                "reference_observation": current,
            }

    return clipped_action, clipped_joints


def assert_finite_robot_action(robot_action: dict[str, float]) -> None:
    non_finite = {
        key: float(value)
        for key, value in robot_action.items()
        if not math.isfinite(float(value))
    }
    if non_finite:
        raise RuntimeError(f"Refusing to send non-finite robot action values: {non_finite}")


def estimate_inference_delay_steps(
    *,
    rtc_enabled: bool,
    n_action_steps: int,
    chunk_latency_s: float | None,
    step_time_s: float | None,
    fallback_fps: float,
) -> int:
    if not rtc_enabled:
        return 0

    effective_step_time_s = step_time_s
    if effective_step_time_s is None:
        effective_step_time_s = 1.0 / max(fallback_fps, 1e-6)
    if chunk_latency_s is None or effective_step_time_s <= 0:
        return 0

    return max(0, min(int(n_action_steps), math.ceil(float(chunk_latency_s) / float(effective_step_time_s))))


def main() -> int:
    args = build_parser().parse_args()

    register_third_party_devices()
    init_logging()

    stage("Validate environment")
    policy_dir, calib_dir, tokenizer_dir = validate_paths(args)
    policy_cfg = load_policy_config(policy_dir, args.policy_device)
    rtc_runtime = apply_pi_runtime_overrides(args, policy_cfg)
    for flag_name, value in [
        ("--joint-action-alpha", args.joint_action_alpha),
        ("--gripper-action-alpha", args.gripper_action_alpha),
    ]:
        if value is not None and not (0.0 < value <= 1.0):
            raise ValueError(f"{flag_name} must be within (0, 1], got {value}")
    for flag_name, value in [
        ("--joint-delta-limit", args.joint_delta_limit),
        ("--gripper-delta-limit", args.gripper_delta_limit),
    ]:
        if value is not None and value <= 0.0:
            raise ValueError(f"{flag_name} must be > 0 when provided, got {value}")
    if args.robot_max_relative_target is not None and args.robot_max_relative_target <= 0.0:
        raise ValueError(
            "--robot-max-relative-target must be > 0 when provided; "
            f"got {args.robot_max_relative_target}"
        )

    preprocessor, preprocessor_details = load_policy_preprocessor_from_checkpoint(
        policy_dir,
        device=args.policy_device,
        local_tokenizer_path=args.local_tokenizer_path,
        require_local_tokenizer=True,
    )
    postprocessor = load_postprocessor(policy_dir)
    print_summary(args, policy_dir, calib_dir, tokenizer_dir, policy_cfg, preprocessor_details, rtc_runtime)

    if args.dry_run:
        info("Dry run only. Exiting before any preflight or hardware access.")
        return 0

    stage("Preflight checks")
    if not args.skip_camera_preflight:
        preflight_cameras(args)
    if not args.skip_trt_preflight:
        preflight_trt_artifacts(args)

    policy = None
    if not args.skip_policy_preflight:
        policy = preflight_policy(policy_dir, policy_cfg)

    if args.preflight_only:
        info("Preflight completed. Exiting before robot connect.")
        return 0

    stage("Build robot and processors")
    robot_cfg = build_robot_config(args, calib_dir)
    robot = make_robot_from_config(robot_cfg)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    if policy is None:
        stage("Load PI05 PyTorch policy")
        policy = preflight_policy(policy_dir, policy_cfg)

    policy_device = get_safe_torch_device(policy_cfg.device)
    chunk_runtime_policy = TorchChunkPolicyRuntime(
        policy,
        device=policy_device,
        use_amp=bool(policy_cfg.use_amp),
    )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        stage("Connect robot")
        robot.connect()
        policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()
        dataset_features = build_dataset_features(robot)
        info("Robot connected. Preparing async PI05 PyTorch inference loop.")

        if args.prefetch_threshold is not None and args.prefetch_threshold < 0:
            raise ValueError("--prefetch-threshold must be >= 0")
        if args.sync_refill_timeout_s < 0.0:
            raise ValueError(f"--sync-refill-timeout-s must be >= 0, got {args.sync_refill_timeout_s}")

        prefetcher = AsyncChunkPrefetcher(
            policy=chunk_runtime_policy,
            device=policy_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
            robot_type=robot.robot_type,
            n_action_steps=int(policy_cfg.n_action_steps),
            thread_name_prefix="pi05_torch_prefetch",
        )
        action_queue = ActionQueue(rtc_runtime.config)
        last_chunk_latency_s: float | None = None
        chunk_latency_ema_s: float | None = None
        step_time_ema_s: float | None = None
        chunk_count = 0
        queue_underrun_count = 0
        hold_step_count = 0
        sync_refill_count = 0
        last_real_delay = 0
        last_refill_mode = "startup"
        smoothing_event_count = 0
        delta_clip_event_count = 0
        action_keys = list(robot.action_features.keys())
        last_sent_action: dict[str, float] | None = None

        stage("Warm up initial chunk")
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
        initial_chunk: ChunkPredictionResult = prefetcher.predict_sync(
            observation_frame,
            action_index_before_inference=0,
            predict_kwargs=build_chunk_predict_kwargs(
                rtc_runtime=rtc_runtime,
                action_queue=action_queue,
                predicted_delay_steps=0,
            ),
        )
        last_real_delay = merge_chunk_prediction_result(
            action_queue,
            initial_chunk,
            action_index_before_inference=0,
            real_delay=0,
        )
        chunk_count += 1
        last_refill_mode = "initial_sync"
        last_chunk_latency_s = initial_chunk.total_time_s
        chunk_latency_ema_s = initial_chunk.total_time_s
        recommended_min_steps = max(1, math.ceil(initial_chunk.total_time_s * args.camera_fps))
        info(
            "Initial chunk ready: "
            f"steps={initial_chunk.num_actions} total={initial_chunk.total_time_s:.3f}s "
            f"(pre={initial_chunk.preprocess_time_s:.3f}s infer={initial_chunk.inference_time_s:.3f}s "
            f"post={initial_chunk.postprocess_time_s:.3f}s)"
        )
        if int(policy_cfg.n_action_steps) < recommended_min_steps:
            warn(
                "Configured n_action_steps is shorter than one chunk's measured compute time at current fps. "
                f"n_action_steps={policy_cfg.n_action_steps}, recommended>={recommended_min_steps}. "
                "Async prefetch is enabled, but occasional stalls are still possible."
            )
        current_threshold = estimate_prefetch_threshold(
            configured_threshold=args.prefetch_threshold,
            n_action_steps=int(policy_cfg.n_action_steps),
            chunk_latency_s=chunk_latency_ema_s,
            step_time_s=step_time_ema_s,
            fallback_fps=float(args.camera_fps),
        )
        info(
            "Async chunk settings: "
            f"prefetch_threshold={current_threshold}, sync_refill_timeout_s={args.sync_refill_timeout_s}, "
            f"rtc_enabled={rtc_runtime.config.enabled}"
        )

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                info("Reached requested run_time_s. Exiting inference loop.")
                break

            loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            generated_new_chunk = False

            completed_chunk = prefetcher.maybe_collect()
            if completed_chunk is not None:
                completed_chunk = completed_chunk.with_real_delay(
                    merge_chunk_prediction_result(action_queue, completed_chunk)
                )
                last_real_delay = int(completed_chunk.real_delay or 0)
                last_refill_mode = "async_collect"
                last_chunk_latency_s = completed_chunk.total_time_s
                chunk_latency_ema_s = (
                    completed_chunk.total_time_s
                    if chunk_latency_ema_s is None
                    else 0.7 * chunk_latency_ema_s + 0.3 * completed_chunk.total_time_s
                )
                chunk_count += 1
                generated_new_chunk = True

            current_threshold = estimate_prefetch_threshold(
                configured_threshold=args.prefetch_threshold,
                n_action_steps=int(policy_cfg.n_action_steps),
                chunk_latency_s=chunk_latency_ema_s,
                step_time_s=step_time_ema_s,
                fallback_fps=float(args.camera_fps),
            )
            predicted_delay_steps = estimate_inference_delay_steps(
                rtc_enabled=rtc_runtime.config.enabled,
                n_action_steps=int(policy_cfg.n_action_steps),
                chunk_latency_s=chunk_latency_ema_s,
                step_time_s=step_time_ema_s,
                fallback_fps=float(args.camera_fps),
            )

            if action_queue.qsize() <= current_threshold:
                _ = prefetcher.maybe_submit(
                    observation_frame,
                    action_index_before_inference=action_queue.get_action_index(),
                    predict_kwargs=build_chunk_predict_kwargs(
                        rtc_runtime=rtc_runtime,
                        action_queue=action_queue,
                        predicted_delay_steps=predicted_delay_steps,
                    ),
                )

            waited_for_prefetch_result = False
            if action_queue.empty() and prefetcher.has_future():
                waited_for_prefetch_result = True
                waited_chunk = prefetcher.wait_for_result(args.sync_refill_timeout_s)
                if waited_chunk is not None:
                    waited_chunk = waited_chunk.with_real_delay(
                        merge_chunk_prediction_result(action_queue, waited_chunk)
                    )
                    last_real_delay = int(waited_chunk.real_delay or 0)
                    last_refill_mode = "async_wait"
                    last_chunk_latency_s = waited_chunk.total_time_s
                    chunk_latency_ema_s = (
                        waited_chunk.total_time_s
                        if chunk_latency_ema_s is None
                        else 0.7 * chunk_latency_ema_s + 0.3 * waited_chunk.total_time_s
                    )
                    chunk_count += 1
                    generated_new_chunk = True

            used_hold_action = False
            if action_queue.empty() and prefetcher.has_future():
                queue_underrun_count += 1
                hold_step_count += 1
                used_hold_action = True
                last_refill_mode = "hold_pending_async"
                robot_action_to_send = make_hold_robot_action(obs, action_keys)
                if queue_underrun_count == 1 or queue_underrun_count % max(args.log_interval, 1) == 0:
                    waited_prefix = ""
                    if waited_for_prefetch_result and args.sync_refill_timeout_s > 0.0:
                        waited_prefix = (
                            f" after waiting up to {args.sync_refill_timeout_s:.3f}s "
                            "for the in-flight async chunk"
                        )
                    warn(
                        f"Action queue drained{waited_prefix} while async chunk inference was still running. "
                        "Holding current pose instead of launching a concurrent sync inference."
                    )
            elif action_queue.empty():
                sync_refill_count += 1
                sync_refill_reason = "queue_empty_no_async_future"
                last_refill_mode = "sync_refill"
                warn(
                    f"refill_mode=sync_refill reason={sync_refill_reason} "
                    f"sync_refill_count={sync_refill_count} rtc_enabled={rtc_runtime.config.enabled} "
                    "Running synchronous chunk generation; "
                    "`real_delay=0` on this refill path only reflects blocking refill semantics, not healthy async overlap."
                )
                action_index_before_sync = action_queue.get_action_index()
                sync_chunk: ChunkPredictionResult = prefetcher.predict_sync(
                    observation_frame,
                    action_index_before_inference=action_index_before_sync,
                    predict_kwargs=build_chunk_predict_kwargs(
                        rtc_runtime=rtc_runtime,
                        action_queue=action_queue,
                        predicted_delay_steps=predicted_delay_steps,
                    ),
                )
                sync_chunk = sync_chunk.with_real_delay(
                    merge_chunk_prediction_result(
                        action_queue,
                        sync_chunk,
                        action_index_before_inference=action_index_before_sync,
                        real_delay=0,
                    )
                )
                last_real_delay = int(sync_chunk.real_delay or 0)
                last_chunk_latency_s = sync_chunk.total_time_s
                chunk_latency_ema_s = (
                    sync_chunk.total_time_s
                    if chunk_latency_ema_s is None
                    else 0.7 * chunk_latency_ema_s + 0.3 * sync_chunk.total_time_s
                )
                chunk_count += 1
                generated_new_chunk = True

            if not used_hold_action:
                action_values = action_queue.get()
                if action_values is None:
                    raise RuntimeError("Action queue unexpectedly returned no action after refill.")
                action_dict = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, obs))

            smoothing_reference = last_sent_action or make_hold_robot_action(obs, action_keys)
            robot_action_to_send, smoothed_joints = smooth_robot_action(
                robot_action_to_send,
                smoothing_reference,
                joint_action_alpha=args.joint_action_alpha,
                gripper_action_alpha=args.gripper_action_alpha,
            )
            if smoothed_joints:
                smoothing_event_count += 1

            robot_action_to_send, clipped_joints = clamp_robot_action_delta(
                robot_action_to_send,
                obs,
                joint_delta_limit=args.joint_delta_limit,
                gripper_delta_limit=args.gripper_delta_limit,
            )
            if clipped_joints:
                delta_clip_event_count += 1

            assert_finite_robot_action(robot_action_to_send)
            sent_action = robot.send_action(robot_action_to_send)
            effective_sent_action = sent_action if isinstance(sent_action, dict) else robot_action_to_send
            last_sent_action = {key: float(value) for key, value in effective_sent_action.items() if key in action_keys}

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                info(
                    f"Step {step} | elapsed={elapsed:.2f}s | "
                    f"generated_new_chunk={generated_new_chunk} | "
                    f"queue_size={action_queue.qsize()} | "
                    f"prefetch_pending={prefetcher.has_pending()} | "
                    f"chunk_count={chunk_count} | "
                    f"queue_underrun_count={queue_underrun_count} | "
                    f"hold_step_count={hold_step_count} | "
                    f"sync_refill_count={sync_refill_count} | "
                    f"refill_mode={last_refill_mode} | "
                    f"rtc_enabled={rtc_runtime.config.enabled} | "
                    f"real_delay={last_real_delay} | "
                    f"chunk_latency_s={last_chunk_latency_s if last_chunk_latency_s is not None else -1.0:.3f} | "
                    f"prefetch_threshold={current_threshold} | "
                    f"smoothing_events={smoothing_event_count} | "
                    f"delta_clip_events={delta_clip_event_count}"
                )
                if smoothed_joints:
                    info(f"Last smoothing details: {smoothed_joints}")
                if clipped_joints:
                    info(f"Last delta clip details: {clipped_joints}")

            dt_s = time.perf_counter() - loop_t
            step_time_ema_s = dt_s if step_time_ema_s is None else 0.8 * step_time_ema_s + 0.2 * dt_s
            precise_sleep(max(1 / args.camera_fps - dt_s, 0.0))
    except KeyboardInterrupt:
        info("KeyboardInterrupt received. Stopping inference.")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if "robot" in locals() and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        if "prefetcher" in locals():
            prefetcher.close(wait=True)
        info("Inference finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
