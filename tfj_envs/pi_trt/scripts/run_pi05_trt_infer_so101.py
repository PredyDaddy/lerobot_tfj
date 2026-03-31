#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from trt_pi_adapter import PiTrtArtifacts, TrtPi05PolicyAdapter


DEFAULT_POLICY_PATH = REPO_ROOT / "pi_model" / "pretrained_model"
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")

ENGINE_FILENAMES = {
    "vision_encoder": "pi_shared_vision_encoder.engine",
    "prefix_cache": "pi_shared_prefix_cache.engine",
    "denoise_step": "pi05_denoise_step.engine",
}
BUILD_REPORT_FILENAMES = {
    "vision_encoder": "vision_encoder_build_report.json",
    "prefix_cache": "prefix_cache_build_report.json",
    "denoise_step": "denoise_step_build_report.json",
}


@dataclass(frozen=True)
class ResolvedPaths:
    policy_dir: Path
    calib_dir: Path
    artifacts: PiTrtArtifacts
    metadata_checkpoint_dir: Path | None
    local_tokenizer_path: Path | None
    artifact_safety: "TrtArtifactSafetyReport"


@dataclass(frozen=True)
class TrtArtifactSafetyReport:
    allow_unsafe_override: bool
    metadata_path: Path | None
    metadata_checkpoint_dir: Path | None
    resolved_variant: str | None
    resolved_requested_precision: str | None
    metadata_stage_status: dict[str, str]
    last_completed_stage: str | None
    stage4_report_path: Path | None
    stage4_report_status: str | None
    stage5_report_path: Path | None
    stage5_report_status: str | None
    build_report_paths: dict[str, str]
    build_report_status: dict[str, str]
    effective_precision_evidence: dict[str, Any]
    resolved_engine_paths: dict[str, str]
    blocking_reasons: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def is_safe(self) -> bool:
        return len(self.blocking_reasons) == 0


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
    if lowered in {"", "none", "null"}:
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
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


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _single_value_or_none(values: set[str]) -> str | None:
    if len(values) != 1:
        return None
    return next(iter(values))


def _engine_io_dtype_summary(engine_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(engine_summary, dict):
        return {"inputs": {}, "outputs": {}}

    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for tensor in engine_summary.get("tensors", []):
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        mode = tensor.get("mode")
        if not isinstance(name, str) or dtype is None:
            continue
        if mode == "input":
            inputs[name] = str(dtype)
        elif mode == "output":
            outputs[name] = str(dtype)
    return {
        "inputs": inputs,
        "outputs": outputs,
    }


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PI0.5 TensorRT inference on a real SO101 follower robot.")
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
    parser.add_argument("--top-cam-fourcc", default=None)
    parser.add_argument("--wrist-cam-fourcc", default=None)

    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-n-action-steps", type=parse_optional_int, default=None)
    parser.add_argument("--policy-num-inference-steps", type=parse_optional_int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=parse_optional_float, default=None)
    parser.add_argument("--policy-noise-seed", type=int, default=None)
    parser.add_argument(
        "--policy-fixed-noise",
        action="store_true",
        help="Reuse the same denoise initialization each chunk to reduce stochastic jitter.",
    )

    parser.add_argument(
        "--trt-path",
        default=str(DEFAULT_TRT_PATH) if DEFAULT_TRT_PATH is not None else None,
        help=(
            "Path to the TensorRT artifact set: a verified PI TRT metadata json, run directory, engine directory, "
            "or one engine file. Required when no verified-pass default artifact is available."
        ),
    )
    parser.add_argument("--trt-metadata-path", default=None)
    parser.add_argument("--trt-device", default="cuda:0")
    parser.add_argument("--local-tokenizer-path", default=None)
    parser.add_argument(
        "--allow-unsafe-trt-artifacts",
        action="store_true",
        help=(
            "Override fail-closed metadata/provenance checks for warning, failed, or mismatched TRT artifacts. "
            "Use only when you intentionally accept the safety risk."
        ),
    )

    parser.add_argument("--task", default="grasp block in bin")
    parser.add_argument("--run-time-s", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument(
        "--prefetch-threshold",
        type=parse_optional_int,
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
    parser.add_argument("--skip-trt-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_file_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend(
            [
                path / "pi_trt_metadata.json",
                path.parent / "pi_trt_metadata.json",
                path.parent.parent / "pi_trt_metadata.json",
            ]
        )
    else:
        candidates.extend([path, path.parent / "pi_trt_metadata.json", path.parent.parent / "pi_trt_metadata.json"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_dir_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend([path, path / "artifacts" / "engines"])
    else:
        candidates.extend([path.parent, path.parent / "artifacts" / "engines"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_metadata_path(trt_path: Path, explicit_metadata_path: str | None) -> Path | None:
    if explicit_metadata_path:
        candidate = Path(explicit_metadata_path).expanduser().resolve(strict=False)
        if not candidate.is_file():
            raise FileNotFoundError(f"TensorRT metadata path not found: {candidate}")
        return candidate

    if trt_path.suffix == ".json" and trt_path.is_file():
        return trt_path

    for candidate in _resolve_file_candidates(trt_path):
        if candidate.is_file():
            return candidate
    return None


def _resolve_path_from_metadata(raw_path: str | Path, metadata_path: Path, metadata: dict) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    candidate_roots = [
        metadata_path.parent,
        metadata_path.parent / "artifacts" / "engines",
    ]
    run_dir_value = metadata.get("run_dir")
    if run_dir_value:
        run_dir = Path(run_dir_value).expanduser().resolve(strict=False)
        candidate_roots.extend([run_dir, run_dir / "artifacts" / "engines"])

    for root in candidate_roots:
        resolved = (root / candidate).resolve(strict=False)
        if resolved.is_file():
            return resolved
    return (metadata_path.parent / candidate).resolve(strict=False)


def _resolve_complete_engine_set(candidate_dirs: list[Path]) -> tuple[dict[str, Path], list[Path]]:
    searched_dirs: list[Path] = []
    for candidate_dir in candidate_dirs:
        resolved_dir = candidate_dir.expanduser().resolve(strict=False)
        if resolved_dir in searched_dirs:
            continue
        searched_dirs.append(resolved_dir)
        candidate_paths = {
            subgraph: (resolved_dir / filename).resolve(strict=False)
            for subgraph, filename in ENGINE_FILENAMES.items()
        }
        if all(path.is_file() for path in candidate_paths.values()):
            return candidate_paths, searched_dirs
    return {}, searched_dirs


def _load_json_if_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload)}")
    return payload


def _resolve_metadata_artifact_path(metadata: dict[str, Any], metadata_path: Path, key: str) -> Path | None:
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    raw_path = artifacts.get(key)
    if not raw_path:
        return None
    return _resolve_path_from_metadata(raw_path, metadata_path, metadata)


def _report_search_roots(trt_path: Path, metadata_path: Path | None) -> list[Path]:
    roots: list[Path] = []
    if trt_path.is_dir():
        roots.extend([trt_path, trt_path.parent])
    else:
        roots.extend([trt_path.parent, trt_path.parent.parent])
    if metadata_path is not None:
        roots.extend([metadata_path.parent, metadata_path.parent.parent])

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=False)
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        unique.append(resolved_root)
    return unique


def _extract_engine_paths_from_report(report: dict[str, Any]) -> dict[str, Path]:
    engine_paths: dict[str, Path] = {}
    artifact_paths = report.get("artifact_paths", {})
    for subgraph in ENGINE_FILENAMES:
        raw_engine_path = None
        artifact_entry = artifact_paths.get(subgraph) if isinstance(artifact_paths, dict) else None
        if isinstance(artifact_entry, dict):
            raw_engine_path = artifact_entry.get("engine")
        if raw_engine_path is None:
            result_entry = report.get("results", {}).get(subgraph, {})
            if isinstance(result_entry, dict):
                raw_engine_path = result_entry.get("engine_path")
        if raw_engine_path:
            engine_paths[subgraph] = Path(raw_engine_path).expanduser().resolve(strict=False)
    return engine_paths


def _find_coherent_trt_report(
    *,
    trt_path: Path,
    metadata_path: Path | None,
    metadata: dict[str, Any] | None,
    patterns: tuple[str, ...],
) -> tuple[Path | None, dict[str, Any] | None]:
    expected_run_dir = None
    if metadata is not None and metadata.get("run_dir"):
        expected_run_dir = Path(metadata["run_dir"]).expanduser().resolve(strict=False)

    candidate_paths: list[Path] = []
    if trt_path.suffix == ".json" and trt_path.is_file():
        candidate_paths.append(trt_path)
    if metadata_path is not None and metadata_path not in candidate_paths:
        candidate_paths.append(metadata_path)

    for root in _report_search_roots(trt_path, metadata_path):
        if not root.exists():
            continue
        for pattern in patterns:
            candidate_paths.extend(sorted(root.glob(pattern)))

    seen: set[Path] = set()
    for candidate_path in candidate_paths:
        resolved_candidate = candidate_path.expanduser().resolve(strict=False)
        if resolved_candidate in seen or not resolved_candidate.is_file():
            continue
        seen.add(resolved_candidate)

        report = _load_json_if_file(resolved_candidate)
        if report is None:
            continue

        report_run_dir = report.get("run_dir")
        if expected_run_dir is not None and report_run_dir:
            resolved_run_dir = Path(report_run_dir).expanduser().resolve(strict=False)
            if resolved_run_dir != expected_run_dir:
                continue

        engine_paths = _extract_engine_paths_from_report(report)
        if len(engine_paths) != len(ENGINE_FILENAMES):
            continue
        if not all(path.is_file() for path in engine_paths.values()):
            continue
        engine_dirs = {path.parent for path in engine_paths.values()}
        if len(engine_dirs) != 1:
            continue
        report_engine_dir = report.get("engine_dir")
        if report_engine_dir:
            resolved_report_engine_dir = Path(report_engine_dir).expanduser().resolve(strict=False)
            if resolved_report_engine_dir != next(iter(engine_dirs)):
                continue
        return resolved_candidate, report

    return None, None


def resolve_trt_artifacts(trt_path_arg: str, metadata_path_arg: str | None) -> tuple[PiTrtArtifacts, Path | None]:
    from trt_pi_adapter import PiTrtArtifacts

    trt_path = Path(trt_path_arg).expanduser().resolve(strict=False)
    metadata_path = _resolve_metadata_path(trt_path, metadata_path_arg)
    metadata = read_json(metadata_path) if metadata_path is not None else None

    resolved_engines: dict[str, Path] = {}
    candidate_dirs: list[Path] = []
    if metadata is not None:
        metadata_engine_paths = metadata.get("engine_paths", {})
        for subgraph in ENGINE_FILENAMES:
            raw_engine_path = metadata_engine_paths.get(subgraph)
            if raw_engine_path:
                resolved_path = _resolve_path_from_metadata(raw_engine_path, metadata_path, metadata)
                resolved_engines[subgraph] = resolved_path
                candidate_dirs.append(resolved_path.parent)
        run_dir_value = metadata.get("run_dir")
        if run_dir_value:
            candidate_dirs.extend(_resolve_dir_candidates(Path(run_dir_value).expanduser().resolve(strict=False)))

    candidate_dirs.extend(_resolve_dir_candidates(trt_path))
    complete_engine_set, searched_dirs = _resolve_complete_engine_set(candidate_dirs)
    report_engine_set: dict[str, Path] = {}
    report_path, report_payload = _find_coherent_trt_report(
        trt_path=trt_path,
        metadata_path=metadata_path,
        metadata=metadata,
        patterns=("stage5_verify_trt*.json", "stage4_build_engines*.json"),
    )
    if report_payload is not None:
        report_engine_set = _extract_engine_paths_from_report(report_payload)

    if any(subgraph not in resolved_engines or not resolved_engines[subgraph].is_file() for subgraph in ENGINE_FILENAMES):
        if complete_engine_set:
            resolved_engines = complete_engine_set
        elif report_engine_set:
            if resolved_engines:
                mismatched_subgraphs = [
                    subgraph
                    for subgraph in report_engine_set
                    if subgraph in resolved_engines
                    and resolved_engines[subgraph].is_file()
                    and resolved_engines[subgraph] != report_engine_set[subgraph]
                ]
                if mismatched_subgraphs:
                    mismatch_details = ", ".join(
                        f"{subgraph}: metadata={resolved_engines[subgraph]} report={report_engine_set[subgraph]}"
                        for subgraph in mismatched_subgraphs
                    )
                    raise RuntimeError(
                        "TensorRT report provenance mismatch while resolving artifacts. "
                        f"The engine set is not coherent: {mismatch_details}"
                    )
            resolved_engines = report_engine_set
    elif complete_engine_set:
        mismatched_subgraphs = [
            subgraph
            for subgraph in ENGINE_FILENAMES
            if resolved_engines[subgraph].is_file() and resolved_engines[subgraph] != complete_engine_set[subgraph]
        ]
        if mismatched_subgraphs:
            mismatch_details = ", ".join(
                f"{subgraph}: metadata={resolved_engines[subgraph]} discovered={complete_engine_set[subgraph]}"
                for subgraph in mismatched_subgraphs
            )
            raise RuntimeError(
                "TensorRT engine provenance mismatch while resolving artifacts. "
                f"The engine set is not coherent: {mismatch_details}"
            )

    missing = [
        subgraph
        for subgraph in ENGINE_FILENAMES
        if subgraph not in resolved_engines or not resolved_engines[subgraph].is_file()
    ]
    if missing:
        searched = ", ".join(path.as_posix() for path in searched_dirs) or "<none>"
        report_hint = report_path.as_posix() if report_path is not None else "<none>"
        raise FileNotFoundError(
            "Could not resolve all PI TRT engines. "
            f"Missing={missing}. Searched directories: {searched}. "
            f"coherent_report={report_hint}. "
            f"metadata_path={metadata_path.as_posix() if metadata_path else '<none>'}"
        )

    engine_dirs = {path.parent for path in resolved_engines.values()}
    if len(engine_dirs) != 1:
        engine_dirs_text = ", ".join(path.as_posix() for path in sorted(engine_dirs, key=lambda item: item.as_posix()))
        raise RuntimeError(
            "TensorRT engines must come from one engine directory. "
            f"Resolved directories: {engine_dirs_text}"
        )
    engine_dir = next(iter(engine_dirs))
    metadata_checkpoint_dir = None
    if metadata is not None and metadata.get("checkpoint_dir"):
        metadata_checkpoint_dir = Path(metadata["checkpoint_dir"]).expanduser().resolve(strict=False)

    return (
        PiTrtArtifacts(
            engine_dir=engine_dir,
            vision_engine=resolved_engines["vision_encoder"],
            prefix_engine=resolved_engines["prefix_cache"],
            denoise_engine=resolved_engines["denoise_step"],
            metadata_path=metadata_path,
            metadata=metadata,
        ),
        metadata_checkpoint_dir,
    )


def assess_trt_artifact_safety(
    *,
    policy_dir: Path,
    artifacts: PiTrtArtifacts,
    metadata_checkpoint_dir: Path | None,
    allow_unsafe_override: bool,
) -> TrtArtifactSafetyReport:
    metadata = artifacts.metadata if isinstance(artifacts.metadata, dict) else {}
    metadata_path = artifacts.metadata_path
    stage_status_raw = metadata.get("stage_status", {}) if isinstance(metadata, dict) else {}
    metadata_stage_status = {
        str(stage_name): (_normalize_status(stage_value) or "<missing>")
        for stage_name, stage_value in stage_status_raw.items()
    }
    last_completed_stage = metadata.get("last_completed_stage") if isinstance(metadata, dict) else None

    blocking_reasons: list[str] = []
    notes: list[str] = []
    resolved_engine_paths = {
        "vision_encoder": artifacts.vision_engine.as_posix(),
        "prefix_cache": artifacts.prefix_engine.as_posix(),
        "denoise_step": artifacts.denoise_engine.as_posix(),
    }

    if metadata_path is None:
        blocking_reasons.append(
            "No TensorRT metadata file was resolved. Live robot use requires pi_trt_metadata.json "
            "for checkpoint and verification provenance."
        )
    else:
        if metadata.get("variant") not in {None, "pi05"}:
            blocking_reasons.append(
                f"TensorRT metadata variant mismatch: expected 'pi05', got {metadata.get('variant')!r}."
            )
        if metadata_checkpoint_dir is None:
            blocking_reasons.append("TensorRT metadata is missing checkpoint_dir.")
        elif metadata_checkpoint_dir != policy_dir:
            blocking_reasons.append(
                "Policy checkpoint does not match TensorRT metadata checkpoint_dir: "
                f"policy={policy_dir}, metadata={metadata_checkpoint_dir}"
            )

        stage5_status = _normalize_status(stage_status_raw.get("stage5_verify_trt"))
        if stage5_status != "pass":
            blocking_reasons.append(
                "TensorRT metadata stage5_verify_trt must be 'pass' for real-robot use, "
                f"got {stage5_status or '<missing>'}."
            )
        for stage_name in ("stage2_export_onnx", "stage3_verify_onnx", "stage4_build_engines"):
            stage_value = _normalize_status(stage_status_raw.get(stage_name))
            if stage_value in {"warn", "error", "fail", "failed"}:
                blocking_reasons.append(
                    f"TensorRT metadata {stage_name} status is {stage_value!r}; refusing live use."
                )

        metadata_engine_paths = metadata.get("engine_paths", {}) if isinstance(metadata, dict) else {}
        if not isinstance(metadata_engine_paths, dict) or not metadata_engine_paths:
            notes.append("TensorRT metadata is missing engine_paths; relying on build/stage reports for engine provenance.")
        elif metadata_path is not None:
            for subgraph in ENGINE_FILENAMES:
                raw_engine_path = metadata_engine_paths.get(subgraph)
                if not raw_engine_path:
                    notes.append(f"TensorRT metadata is missing engine_paths[{subgraph!r}].")
                    continue
                expected_engine_path = _resolve_path_from_metadata(raw_engine_path, metadata_path, metadata)
                actual_engine_path = Path(resolved_engine_paths[subgraph]).expanduser().resolve(strict=False)
                if expected_engine_path != actual_engine_path:
                    blocking_reasons.append(
                        f"Resolved engine for {subgraph} does not match metadata engine_paths: "
                        f"resolved={actual_engine_path}, metadata={expected_engine_path}"
                    )

    build_report_paths: dict[str, str] = {}
    build_report_status: dict[str, str] = {}
    build_precision_values: set[str] = set()
    build_device_values: set[str] = set()
    build_onnx_dirs: set[Path] = set()
    force_fp32_layer_types: dict[str, list[str]] = {}
    precision_constraint_matches: dict[str, int] = {}
    engine_io_dtypes: dict[str, dict[str, Any]] = {}
    for subgraph, filename in BUILD_REPORT_FILENAMES.items():
        report_path = (artifacts.engine_dir / filename).resolve(strict=False)
        build_report_paths[subgraph] = report_path.as_posix()
        report_payload = _load_json_if_file(report_path)
        if report_payload is None:
            blocking_reasons.append(f"Missing TensorRT build report for {subgraph}: {report_path}")
            build_report_status[subgraph] = "<missing>"
            continue

        report_status = _normalize_status(report_payload.get("status")) or "<missing>"
        build_report_status[subgraph] = report_status
        if report_status != "pass":
            blocking_reasons.append(
                f"TensorRT build report {report_path.name} status is {report_status!r}; refusing live use."
            )

        report_engine = report_payload.get("engine")
        if not report_engine:
            blocking_reasons.append(f"TensorRT build report {report_path.name} is missing the built engine path.")
        else:
            report_engine_path = Path(report_engine).expanduser().resolve(strict=False)
            actual_engine_path = Path(resolved_engine_paths[subgraph]).expanduser().resolve(strict=False)
            if report_engine_path != actual_engine_path:
                blocking_reasons.append(
                    f"TensorRT build report {report_path.name} does not match resolved engine for {subgraph}: "
                    f"report={report_engine_path}, resolved={actual_engine_path}"
                )

        report_onnx = report_payload.get("onnx")
        if not report_onnx:
            blocking_reasons.append(f"TensorRT build report {report_path.name} is missing the ONNX source path.")
        else:
            build_onnx_dirs.add(Path(report_onnx).expanduser().resolve(strict=False).parent)

        if report_payload.get("precision") is not None:
            build_precision_values.add(str(report_payload["precision"]))
        if report_payload.get("device") is not None:
            build_device_values.add(str(report_payload["device"]))
        precision_constraints = report_payload.get("precision_constraints", {})
        if isinstance(precision_constraints, dict):
            force_fp32_layer_types[subgraph] = [
                str(name) for name in precision_constraints.get("forced_layer_types", [])
            ]
            precision_constraint_matches[subgraph] = int(precision_constraints.get("matched_count", 0))
        else:
            force_fp32_layer_types[subgraph] = []
            precision_constraint_matches[subgraph] = 0
        engine_io_dtypes[subgraph] = _engine_io_dtype_summary(report_payload.get("engine_summary"))

    if len(build_onnx_dirs) > 1:
        blocking_reasons.append(
            "TensorRT build reports disagree about their ONNX source directory: "
            + ", ".join(path.as_posix() for path in sorted(build_onnx_dirs, key=lambda item: item.as_posix()))
        )
    if len(build_precision_values) > 1:
        blocking_reasons.append(
            "TensorRT build reports disagree about precision: " + ", ".join(sorted(build_precision_values))
        )
    if len(build_device_values) > 1:
        blocking_reasons.append(
            "TensorRT build reports disagree about build device: " + ", ".join(sorted(build_device_values))
        )

    stage4_report_path: Path | None = None
    stage4_report_status: str | None = None
    stage4_report: dict[str, Any] | None = None
    referenced_stage4_path = None
    if metadata_path is not None:
        referenced_stage4_path = _resolve_metadata_artifact_path(metadata, metadata_path, "stage4_build_engines")
        if referenced_stage4_path is not None:
            stage4_report_path = referenced_stage4_path
            stage4_report = _load_json_if_file(stage4_report_path)
    if stage4_report is None:
        fallback_stage4_path, fallback_stage4_report = _find_coherent_trt_report(
            trt_path=artifacts.engine_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            patterns=("stage4_build_engines*.json",),
        )
        if fallback_stage4_report is not None:
            if referenced_stage4_path is not None and not referenced_stage4_path.is_file():
                notes.append(
                    "TensorRT metadata referenced a missing stage4_build_engines report; using a nearby coherent report "
                    f"instead: {fallback_stage4_path}"
                )
            stage4_report_path = fallback_stage4_path
            stage4_report = fallback_stage4_report

    if stage4_report is None:
        blocking_reasons.append("Could not resolve a coherent stage4_build_engines report for the selected TensorRT artifact set.")
    else:
        stage4_report_status = _normalize_status(stage4_report.get("overall_status"))
        if stage4_report_status != "pass":
            blocking_reasons.append(
                "stage4_build_engines overall_status must be 'pass' for coherent TensorRT provenance, "
                f"got {stage4_report_status or '<missing>'}."
            )

        report_engine_dir = stage4_report.get("engine_dir")
        if report_engine_dir:
            resolved_report_engine_dir = Path(report_engine_dir).expanduser().resolve(strict=False)
            if resolved_report_engine_dir != artifacts.engine_dir:
                blocking_reasons.append(
                    "stage4_build_engines engine_dir does not match the resolved TensorRT engine dir: "
                    f"resolved={artifacts.engine_dir}, report={resolved_report_engine_dir}"
                )

        stage4_engine_paths = _extract_engine_paths_from_report(stage4_report)
        if stage4_engine_paths:
            for subgraph in ENGINE_FILENAMES:
                resolved_stage4_engine = stage4_engine_paths.get(subgraph)
                if resolved_stage4_engine is None:
                    continue
                actual_engine_path = Path(resolved_engine_paths[subgraph]).expanduser().resolve(strict=False)
                if resolved_stage4_engine != actual_engine_path:
                    blocking_reasons.append(
                        f"stage4_build_engines artifact path for {subgraph} does not match the resolved engine: "
                        f"resolved={actual_engine_path}, report={resolved_stage4_engine}"
                    )

    stage5_report_path: Path | None = None
    stage5_report_status: str | None = None
    stage5_report: dict[str, Any] | None = None
    referenced_stage5_path = None
    if metadata_path is not None:
        referenced_stage5_path = _resolve_metadata_artifact_path(metadata, metadata_path, "stage5_verify_trt")
        if referenced_stage5_path is not None:
            stage5_report_path = referenced_stage5_path
            stage5_report = _load_json_if_file(stage5_report_path)
    if stage5_report is None:
        fallback_stage5_path, fallback_stage5_report = _find_coherent_trt_report(
            trt_path=artifacts.engine_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            patterns=("stage5_verify_trt*.json",),
        )
        if fallback_stage5_report is not None:
            if referenced_stage5_path is not None and not referenced_stage5_path.is_file():
                notes.append(
                    "TensorRT metadata referenced a missing stage5_verify_trt report; using a nearby coherent report "
                    f"instead: {fallback_stage5_path}"
                )
            stage5_report_path = fallback_stage5_path
            stage5_report = fallback_stage5_report

    if stage5_report is None:
        blocking_reasons.append("Could not resolve a coherent stage5_verify_trt report for the selected TensorRT artifact set.")
    else:
        stage5_report_status = _normalize_status(stage5_report.get("overall_status"))
        if stage5_report_status != "pass":
            blocking_reasons.append(
                "stage5_verify_trt overall_status must be 'pass' for live robot use, "
                f"got {stage5_report_status or '<missing>'}."
            )

        report_policy_dir = stage5_report.get("policy_dir")
        if report_policy_dir:
            resolved_report_policy = resolve_checkpoint_dir(report_policy_dir)
            if resolved_report_policy != policy_dir:
                blocking_reasons.append(
                    "stage5_verify_trt policy_dir does not match the requested checkpoint: "
                    f"policy={policy_dir}, report={resolved_report_policy}"
                )

        report_engine_dir = stage5_report.get("engine_dir")
        if report_engine_dir:
            resolved_report_engine_dir = Path(report_engine_dir).expanduser().resolve(strict=False)
            if resolved_report_engine_dir != artifacts.engine_dir:
                blocking_reasons.append(
                    "stage5_verify_trt engine_dir does not match the resolved TensorRT engine dir: "
                    f"resolved={artifacts.engine_dir}, report={resolved_report_engine_dir}"
                )

        artifact_paths = stage5_report.get("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            for subgraph in ENGINE_FILENAMES:
                raw_stage5_engine = None
                artifact_entry = artifact_paths.get(subgraph)
                if isinstance(artifact_entry, dict):
                    raw_stage5_engine = artifact_entry.get("engine")
                if raw_stage5_engine:
                    resolved_stage5_engine = Path(raw_stage5_engine).expanduser().resolve(strict=False)
                    actual_engine_path = Path(resolved_engine_paths[subgraph]).expanduser().resolve(strict=False)
                    if resolved_stage5_engine != actual_engine_path:
                        blocking_reasons.append(
                            f"stage5_verify_trt artifact path for {subgraph} does not match the resolved engine: "
                            f"resolved={actual_engine_path}, report={resolved_stage5_engine}"
                        )

    resolved_variant = None
    if isinstance(metadata, dict):
        raw_variant = metadata.get("variant")
        if raw_variant is not None:
            resolved_variant = str(raw_variant)

    resolved_requested_precision = _single_value_or_none(build_precision_values)
    if resolved_requested_precision is None and stage4_report is not None:
        build_settings = stage4_report.get("build_settings", {})
        if isinstance(build_settings, dict) and build_settings.get("precision") is not None:
            resolved_requested_precision = str(build_settings.get("precision"))
    if resolved_requested_precision is None and isinstance(metadata, dict):
        engine_build_settings = metadata.get("engine_build_settings", {})
        if isinstance(engine_build_settings, dict) and engine_build_settings.get("precision") is not None:
            resolved_requested_precision = str(engine_build_settings.get("precision"))

    effective_precision_evidence = {
        "requested_precision_values": sorted(build_precision_values),
        "build_device_values": sorted(build_device_values),
        "build_report_paths": build_report_paths,
        "force_fp32_layer_types": force_fp32_layer_types,
        "precision_constraint_matched_count": precision_constraint_matches,
        "engine_io_dtypes": engine_io_dtypes,
        "note": (
            "These fields prove requested precision and visible engine I/O dtypes/provenance. "
            "They do not guarantee per-layer effective execution precision."
        ),
    }

    if allow_unsafe_override and blocking_reasons:
        notes.append(
            "Unsafe TRT artifact override is enabled. The launcher will continue despite blocking artifact checks."
        )
    elif not blocking_reasons:
        notes.append("TensorRT artifact provenance and verification checks passed.")

    return TrtArtifactSafetyReport(
        allow_unsafe_override=allow_unsafe_override,
        metadata_path=metadata_path,
        metadata_checkpoint_dir=metadata_checkpoint_dir,
        resolved_variant=resolved_variant,
        resolved_requested_precision=resolved_requested_precision,
        metadata_stage_status=metadata_stage_status,
        last_completed_stage=last_completed_stage,
        stage4_report_path=stage4_report_path,
        stage4_report_status=stage4_report_status,
        stage5_report_path=stage5_report_path,
        stage5_report_status=stage5_report_status,
        build_report_paths=build_report_paths,
        build_report_status=build_report_status,
        effective_precision_evidence=effective_precision_evidence,
        resolved_engine_paths=resolved_engine_paths,
        blocking_reasons=tuple(blocking_reasons),
        notes=tuple(notes),
    )


def validate_paths(args: argparse.Namespace) -> ResolvedPaths:
    policy_dir = resolve_checkpoint_dir(args.policy_path)
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    artifact_path = args.trt_path or args.trt_metadata_path
    if artifact_path is None:
        raise FileNotFoundError(
            "No verified-pass default TensorRT artifact is available. "
            "Provide --trt-path (or --trt-metadata-path) explicitly."
        )

    artifacts, metadata_checkpoint_dir = resolve_trt_artifacts(artifact_path, args.trt_metadata_path)
    artifact_safety = assess_trt_artifact_safety(
        policy_dir=policy_dir,
        artifacts=artifacts,
        metadata_checkpoint_dir=metadata_checkpoint_dir,
        allow_unsafe_override=args.allow_unsafe_trt_artifacts,
    )
    if not artifact_safety.is_safe and not args.allow_unsafe_trt_artifacts:
        reasons = "\n".join(f"  - {reason}" for reason in artifact_safety.blocking_reasons)
        raise RuntimeError(
            "Refusing to use TensorRT artifacts for real-robot inference because the provenance/verification checks "
            "did not pass.\n"
            f"{reasons}\n"
            "Use --allow-unsafe-trt-artifacts only if you intentionally accept that risk."
        )
    local_tokenizer_path = discover_local_tokenizer_path(args.local_tokenizer_path, require=False)

    return ResolvedPaths(
        policy_dir=policy_dir,
        calib_dir=calib_dir,
        artifacts=artifacts,
        metadata_checkpoint_dir=metadata_checkpoint_dir,
        local_tokenizer_path=local_tokenizer_path,
        artifact_safety=artifact_safety,
    )


def preflight_cameras(args: argparse.Namespace) -> None:
    import cv2

    camera_settings = [
        ("top", args.top_cam_index, args.top_cam_fourcc),
        ("wrist", args.wrist_cam_index, args.wrist_cam_fourcc),
    ]
    for camera_name, camera_index, camera_fourcc in camera_settings:
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Camera {camera_name} ({camera_index}) failed to open")
            if args.camera_width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
            if args.camera_height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
            if args.camera_fps is not None:
                cap.set(cv2.CAP_PROP_FPS, int(args.camera_fps))
            if camera_fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_fourcc))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera {camera_name} ({camera_index}) opened but failed to read a frame")
            info(
                f"Camera {camera_name} ({camera_index}) OK: "
                f"frame_shape={tuple(frame.shape)}, fourcc={camera_fourcc or '<default>'}"
            )
        finally:
            cap.release()


def build_robot_config(args: argparse.Namespace, calib_dir: Path) -> SO101FollowerConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.wrist_cam_fourcc,
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
        raise ValueError("PI05 TRT runtime does not support temporal ensembling.")

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


def preflight_trt_adapter(
    policy_cfg: object,
    artifacts: PiTrtArtifacts,
    trt_device: str,
    noise_seed: int | None,
    fixed_noise: bool,
    *,
    run_warmup: bool,
):
    from trt_pi_adapter import TrtPi05PolicyAdapter

    trt_policy = TrtPi05PolicyAdapter(
        policy_cfg,
        artifacts=artifacts,
        trt_device=trt_device,
        noise_seed=noise_seed,
        fixed_noise=fixed_noise,
    )
    trt_policy.eval()
    info(f"PI05 TRT policy OK: trt_device={trt_device}")
    info(f"PI05 TRT runtime: {trt_policy.runtime_summary()}")
    for engine_name, summary in trt_policy.describe_engines().items():
        info(
            f"Engine `{engine_name}`: "
            f"inputs={summary['input_names']} outputs={summary['output_names']}"
        )
    if run_warmup:
        preflight_summary = trt_policy.run_preflight()
        info(f"PI05 TRT warmup preflight: {preflight_summary}")
    return trt_policy


def print_summary(
    args: argparse.Namespace,
    resolved: ResolvedPaths,
    policy_cfg: object,
    preprocessor_details: dict,
    rtc_runtime: ResolvedRTCRuntimeConfig,
) -> None:
    trt_path_value = args.trt_path or args.trt_metadata_path or "<unset>"
    info(f"Python: {sys.executable}")
    info(f"Policy path: {resolved.policy_dir}")
    info(f"Policy type: {getattr(policy_cfg, 'type', '<unknown>')}")
    info(
        "PI05 runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"num_inference_steps={policy_cfg.num_inference_steps}"
    )
    info(f"Resolved RTC config: {rtc_runtime.as_dict()}")
    if rtc_runtime.checkpoint_enabled and not rtc_runtime.config.enabled:
        warn(
            "Checkpoint RTC config is enabled, but launcher runtime keeps RTC off by default "
            "without --rtc-enable/--rtc-enabled."
        )
    info(f"Policy device: {args.policy_device}")
    info(f"TRT device: {args.trt_device}")
    info(f"TRT path: {Path(trt_path_value).expanduser().resolve(strict=False)}")
    info(
        "TRT metadata: "
        f"{resolved.artifacts.metadata_path if resolved.artifacts.metadata_path is not None else '<none>'}"
    )
    info(f"TRT engine dir: {resolved.artifacts.engine_dir}")
    info(f"TRT model variant: {resolved.artifact_safety.resolved_variant or '<missing>'}")
    info(
        "TRT requested precision: "
        f"{resolved.artifact_safety.resolved_requested_precision or '<missing>'}"
    )
    info(f"TRT allow_unsafe override: {args.allow_unsafe_trt_artifacts}")
    info(f"TRT metadata stage_status: {resolved.artifact_safety.metadata_stage_status or '<missing>'}")
    info(
        "TRT build report status: "
        f"{resolved.artifact_safety.build_report_status or '<missing>'}"
    )
    info(
        "TRT stage4 report: "
        f"{resolved.artifact_safety.stage4_report_path if resolved.artifact_safety.stage4_report_path is not None else '<none>'}"
    )
    info(f"TRT stage4 overall_status: {resolved.artifact_safety.stage4_report_status or '<missing>'}")
    info(
        "TRT stage5 report: "
        f"{resolved.artifact_safety.stage5_report_path if resolved.artifact_safety.stage5_report_path is not None else '<none>'}"
    )
    info(f"TRT stage5 overall_status: {resolved.artifact_safety.stage5_report_status or '<missing>'}")
    info(f"Calibration dir: {resolved.calib_dir}")
    info(f"Robot port: {args.robot_port}")
    info(f"Robot max_relative_target: {args.robot_max_relative_target}")
    info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    info(f"Camera fourcc: top={args.top_cam_fourcc or '<default>'}, wrist={args.wrist_cam_fourcc or '<default>'}")
    info(f"Task: {args.task}")
    info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    info(f"sync_refill_timeout_s: {args.sync_refill_timeout_s}")
    info(
        "prefetch_threshold: "
        f"{args.prefetch_threshold if args.prefetch_threshold is not None else '<latency-aware>'}"
    )
    info(
        "Tokenizer path: "
        f"{preprocessor_details.get('local_tokenizer_path') or resolved.local_tokenizer_path or '<unresolved>'}"
    )
    info(f"Policy fixed_noise: {args.policy_fixed_noise}")
    info(f"Policy noise_seed: {args.policy_noise_seed}")
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
    for note in resolved.artifact_safety.notes:
        info(f"TRT artifact note: {note}")
    for reason in resolved.artifact_safety.blocking_reasons:
        warn(f"TRT artifact block: {reason}")


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


def main() -> int:
    args = build_parser().parse_args()

    register_third_party_devices()
    init_logging()

    stage("Validate environment")
    resolved = validate_paths(args)
    policy_cfg = load_policy_config(resolved.policy_dir, args.policy_device)
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
        resolved.policy_dir,
        device=args.policy_device,
        local_tokenizer_path=args.local_tokenizer_path,
        require_local_tokenizer=True,
    )
    postprocessor = load_postprocessor(resolved.policy_dir)
    print_summary(args, resolved, policy_cfg, preprocessor_details, rtc_runtime)

    if args.dry_run:
        info("Dry run only. Exiting before any preflight or hardware access.")
        return 0

    stage("Preflight checks")
    if not args.skip_camera_preflight:
        preflight_cameras(args)

    trt_policy = None
    if not args.skip_trt_preflight:
        trt_policy = preflight_trt_adapter(
            policy_cfg,
            resolved.artifacts,
            args.trt_device,
            args.policy_noise_seed,
            args.policy_fixed_noise,
            run_warmup=True,
        )

    if args.preflight_only:
        if trt_policy is not None:
            trt_policy.close()
        info("Preflight completed. Exiting before robot connect.")
        return 0

    stage("Build robot and processors")
    robot_cfg = build_robot_config(args, resolved.calib_dir)
    robot = make_robot_from_config(robot_cfg)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    if trt_policy is None:
        stage("Load PI05 TRT policy")
        trt_policy = preflight_trt_adapter(
            policy_cfg,
            resolved.artifacts,
            args.trt_device,
            args.policy_noise_seed,
            args.policy_fixed_noise,
            run_warmup=False,
        )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        stage("Connect robot")
        robot.connect()
        trt_policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()
        if args.prefetch_threshold is not None and args.prefetch_threshold < 0:
            raise ValueError("--prefetch-threshold must be >= 0")
        if args.sync_refill_timeout_s < 0.0:
            raise ValueError(f"--sync-refill-timeout-s must be >= 0, got {args.sync_refill_timeout_s}")
        dataset_features = build_dataset_features(robot)
        action_keys = list(robot.action_features.keys())
        last_sent_action: dict[str, float] | None = None
        smoothing_event_count = 0
        delta_clip_event_count = 0
        policy_device = get_safe_torch_device(policy_cfg.device)
        prefetcher = AsyncChunkPrefetcher(
            policy=trt_policy,
            device=policy_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
            robot_type=robot.robot_type,
            n_action_steps=int(policy_cfg.n_action_steps),
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
        info("Robot connected. Starting PI05 FP32 TRT chunk runtime.")

        stage("Warm up initial chunk")
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
        initial_chunk = prefetcher.predict_sync(
            observation_frame,
            action_index_before_inference=0,
            predict_kwargs=build_chunk_predict_kwargs(
                rtc_runtime=rtc_runtime,
                action_queue=action_queue,
                predicted_delay_steps=0,
            ),
        )
        initial_chunk = initial_chunk.with_real_delay(
            merge_chunk_prediction_result(action_queue, initial_chunk)
        )
        chunk_count += 1
        last_real_delay = int(initial_chunk.real_delay or 0)
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
                    predict_kwargs=build_chunk_predict_kwargs(
                        rtc_runtime=rtc_runtime,
                        action_queue=action_queue,
                        predicted_delay_steps=predicted_delay_steps,
                    ),
                    action_index_before_inference=action_queue.get_action_index(),
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
                sync_refill_reason = "async_wait_timeout" if waited_for_prefetch_result else "no_inflight_async_chunk"
                warn(
                    "Action queue drained and requires blocking refill. "
                    f"refill_mode=sync_refill reason={sync_refill_reason} "
                    f"sync_refill_count={sync_refill_count} rtc_enabled={rtc_runtime.config.enabled} "
                    f"prefetch_pending={prefetcher.has_pending()} predicted_delay_steps={predicted_delay_steps}. "
                    "`real_delay=0` on this refill path only reflects blocking refill semantics, not healthy async overlap."
                )
                action_index_before_sync = action_queue.get_action_index()
                sync_chunk = prefetcher.predict_sync(
                    observation_frame,
                    action_index_before_inference=action_index_before_sync,
                    predict_kwargs=build_chunk_predict_kwargs(
                        rtc_runtime=rtc_runtime,
                        action_queue=action_queue,
                        predicted_delay_steps=predicted_delay_steps,
                    ),
                )
                sync_chunk = sync_chunk.with_real_delay(
                    merge_chunk_prediction_result(action_queue, sync_chunk)
                )
                last_real_delay = int(sync_chunk.real_delay or 0)
                last_refill_mode = "sync_refill"
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
                    f"rtc_enabled={rtc_runtime.config.enabled} | "
                    f"generated_new_chunk={generated_new_chunk} | "
                    f"queue_size={action_queue.qsize()} | "
                    f"prefetch_pending={prefetcher.has_pending()} | "
                    f"chunk_count={chunk_count} | "
                    f"queue_underrun_count={queue_underrun_count} | "
                    f"hold_step_count={hold_step_count} | "
                    f"sync_refill_count={sync_refill_count} | "
                    f"refill_mode={last_refill_mode} | "
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
        if trt_policy is not None:
            trt_policy.close()
        info("Inference finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
