#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
from onnx_pi_adapter import OnnxPi05PolicyAdapter, OnnxPiArtifacts
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


DEFAULT_POLICY_PATH = REPO_ROOT / "pi_model" / "pretrained_model"
DEFAULT_ONNX_PATH = PI_TRT_ROOT / "docs" / "results" / "pi05_onnx_fix_20260311_230500" / "onnx"
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")

ONNX_FILENAMES = {
    "vision_encoder": "pi_shared_vision_encoder.onnx",
    "prefix_cache": "pi_shared_prefix_cache.onnx",
    "denoise_step": "pi05_denoise_step.onnx",
}


@dataclass(frozen=True)
class ResolvedPaths:
    policy_dir: Path
    calib_dir: Path
    artifacts: OnnxPiArtifacts
    artifact_safety: "OnnxArtifactSafetyReport"
    local_tokenizer_path: Path | None


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


@dataclass(frozen=True)
class OnnxArtifactSafetyReport:
    stage2_report_path: Path
    stage2_gate_status: str | None
    stage3_report_path: Path
    stage3_gate_status: str | None
    stage3_overall_status: str | None
    stage2_policy_dir: Path | None
    stage3_policy_dir: Path | None
    stage2_run_dir: Path | None
    stage3_run_dir: Path | None
    stage2_onnx_dir: Path | None
    stage3_onnx_dir: Path | None
    resolved_onnx_paths: dict[str, str]
    blocking_reasons: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def is_safe(self) -> bool:
        return len(self.blocking_reasons) == 0


def stage(message: str) -> None:
    print(f"[STAGE] {message}", flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _resolve_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser().resolve(strict=False)

    normalized = str(value).strip()
    if not normalized:
        return None
    return Path(normalized).expanduser().resolve(strict=False)


def _single_resolved_path_or_none(values: set[Path]) -> Path | None:
    if len(values) != 1:
        return None
    return next(iter(values))


def _extract_gate_status(payload: dict[str, Any], *, stage_name: str) -> tuple[str | None, str]:
    gate_field = "overall_status"
    gate_status = _normalize_status(payload.get("overall_status"))
    acceptance_field = "stage2_acceptance" if stage_name == "stage2_export_onnx" else "stage3_acceptance"
    acceptance_payload = payload.get(acceptance_field)
    if isinstance(acceptance_payload, dict):
        acceptance_status = _normalize_status(acceptance_payload.get("status"))
        if acceptance_status is not None:
            gate_field = f"{acceptance_field}.status"
            gate_status = acceptance_status
    return gate_status, gate_field


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PI0.5 ONNX Runtime inference on a real SO101 follower robot.")
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
    parser.add_argument(
        "--policy-device",
        default="cpu",
        help="Preprocessor device. CPU is the safer default for ONNX Runtime live inference.",
    )
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
        "--onnx-path",
        default=str(DEFAULT_ONNX_PATH),
        help="Path to the ONNX directory, a stage2_export_onnx json, or one ONNX file.",
    )
    parser.add_argument("--onnx-stage2-report-path", default=None)
    parser.add_argument(
        "--onnx-provider",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="onnxruntime provider selection.",
    )
    parser.add_argument("--local-tokenizer-path", default=None)

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
    parser.add_argument("--skip-onnx-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_report_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend(
            [
                path / "stage2_export_onnx.json",
                path.parent / "stage2_export_onnx.json",
                path.parent.parent / "stage2_export_onnx.json",
            ]
        )
    else:
        candidates.extend(
            [
                path,
                path.parent / "stage2_export_onnx.json",
                path.parent.parent / "stage2_export_onnx.json",
            ]
        )

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
        candidates.extend([path, path / "onnx", path / "artifacts" / "onnx"])
    else:
        candidates.extend(
            [
                path.parent,
                path.parent / "onnx",
                path.parent / "artifacts" / "onnx",
                path.parent.parent / "onnx",
                path.parent.parent / "artifacts" / "onnx",
            ]
        )

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_stage2_report_path(onnx_path: Path, explicit_report_path: str | None) -> Path | None:
    if explicit_report_path:
        candidate = Path(explicit_report_path).expanduser().resolve(strict=False)
        if not candidate.is_file():
            raise FileNotFoundError(f"stage2_export_onnx report not found: {candidate}")
        return candidate

    if onnx_path.name == "stage2_export_onnx.json" and onnx_path.is_file():
        return onnx_path

    for candidate in _resolve_report_candidates(onnx_path):
        if candidate.is_file():
            return candidate
    return None


def _resolve_stage3_report_path(stage2_report_path: Path, stage2_payload: dict[str, Any]) -> Path | None:
    candidates = [stage2_report_path.parent / "stage3_verify_onnx.json"]
    run_dir = _resolve_optional_path(stage2_payload.get("run_dir"))
    if run_dir is not None:
        candidates.insert(0, run_dir / "stage3_verify_onnx.json")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _resolve_path_from_stage2(raw_path: str | Path, stage2_path: Path, stage2_payload: dict) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    candidate_roots = [
        stage2_path.parent,
        stage2_path.parent / "onnx",
        stage2_path.parent / "artifacts" / "onnx",
    ]
    run_dir_value = stage2_payload.get("run_dir")
    if run_dir_value:
        run_dir = Path(run_dir_value).expanduser().resolve(strict=False)
        candidate_roots.extend([run_dir, run_dir / "onnx", run_dir / "artifacts" / "onnx"])

    for root in candidate_roots:
        resolved = (root / candidate).resolve(strict=False)
        if resolved.is_file():
            return resolved
    return (stage2_path.parent / candidate).resolve(strict=False)


def assess_onnx_artifact_safety(
    *,
    onnx_path: Path,
    stage2_report_path: Path,
    stage2_payload: dict[str, Any],
    stage3_report_path: Path,
    stage3_payload: dict[str, Any],
) -> OnnxArtifactSafetyReport:
    blocking_reasons: list[str] = []
    notes: list[str] = []

    if _normalize_status(stage2_payload.get("stage")) != "stage2_export_onnx":
        blocking_reasons.append(
            "Selected stage2 report does not declare stage='stage2_export_onnx': "
            f"path={stage2_report_path} stage={stage2_payload.get('stage')!r}"
        )
    stage2_gate_status, stage2_gate_field = _extract_gate_status(stage2_payload, stage_name="stage2_export_onnx")
    if stage2_gate_status != "pass":
        blocking_reasons.append(
            "stage2_export_onnx gate must be 'pass' before live ONNX launch: "
            f"{stage2_gate_field}={stage2_gate_status or '<missing>'} report={stage2_report_path}"
        )

    if _normalize_status(stage3_payload.get("stage")) != "stage3_verify_onnx":
        blocking_reasons.append(
            "Selected stage3 report does not declare stage='stage3_verify_onnx': "
            f"path={stage3_report_path} stage={stage3_payload.get('stage')!r}"
        )
    stage3_gate_status, stage3_gate_field = _extract_gate_status(stage3_payload, stage_name="stage3_verify_onnx")
    if stage3_gate_status != "pass":
        blocking_reasons.append(
            "stage3_verify_onnx gate must be 'pass' before live ONNX launch: "
            f"{stage3_gate_field}={stage3_gate_status or '<missing>'} report={stage3_report_path}"
        )

    stage3_overall_status = _normalize_status(stage3_payload.get("overall_status"))
    if stage3_overall_status not in {None, "pass"} and stage3_gate_status == "pass":
        notes.append(
            "stage3_verify_onnx overall_status is not 'pass' even though the gate passed: "
            f"overall_status={stage3_overall_status} report={stage3_report_path}"
        )

    stage2_policy_dir = _resolve_optional_path(stage2_payload.get("policy_dir"))
    if stage2_policy_dir is None:
        blocking_reasons.append(f"stage2_export_onnx report is missing policy_dir: {stage2_report_path}")
    stage3_policy_dir = _resolve_optional_path(stage3_payload.get("policy_dir"))
    if stage3_policy_dir is None:
        blocking_reasons.append(f"stage3_verify_onnx report is missing policy_dir: {stage3_report_path}")
    if (
        stage2_policy_dir is not None
        and stage3_policy_dir is not None
        and stage2_policy_dir != stage3_policy_dir
    ):
        blocking_reasons.append(
            "stage2/stage3 policy_dir mismatch for the selected ONNX artifact set: "
            f"stage2={stage2_policy_dir}, stage3={stage3_policy_dir}"
        )

    stage2_run_dir = _resolve_optional_path(stage2_payload.get("run_dir"))
    if stage2_run_dir is None:
        blocking_reasons.append(f"stage2_export_onnx report is missing run_dir: {stage2_report_path}")
    stage3_run_dir = _resolve_optional_path(stage3_payload.get("run_dir"))
    if stage3_run_dir is None:
        blocking_reasons.append(f"stage3_verify_onnx report is missing run_dir: {stage3_report_path}")
    if stage2_run_dir is not None and stage3_run_dir is not None and stage2_run_dir != stage3_run_dir:
        blocking_reasons.append(
            "stage2/stage3 run_dir mismatch for the selected ONNX artifact set: "
            f"stage2={stage2_run_dir}, stage3={stage3_run_dir}"
        )

    stage2_onnx_dir = _resolve_optional_path(stage2_payload.get("onnx_dir"))
    if stage2_onnx_dir is None:
        blocking_reasons.append(f"stage2_export_onnx report is missing onnx_dir: {stage2_report_path}")
    stage3_onnx_dir = _resolve_optional_path(stage3_payload.get("onnx_dir"))
    if stage3_onnx_dir is None:
        blocking_reasons.append(f"stage3_verify_onnx report is missing onnx_dir: {stage3_report_path}")
    if stage2_onnx_dir is not None and stage3_onnx_dir is not None and stage2_onnx_dir != stage3_onnx_dir:
        blocking_reasons.append(
            "stage2/stage3 onnx_dir mismatch for the selected ONNX artifact set: "
            f"stage2={stage2_onnx_dir}, stage3={stage3_onnx_dir}"
        )

    report_paths = stage2_payload.get("onnx_paths", {})
    if not isinstance(report_paths, dict):
        blocking_reasons.append(
            "stage2_export_onnx report is missing the onnx_paths mapping required for provenance."
        )
        report_paths = {}

    resolved_onnx: dict[str, Path] = {}
    for subgraph in ONNX_FILENAMES:
        raw_onnx_path = report_paths.get(subgraph)
        if not raw_onnx_path:
            blocking_reasons.append(
                f"stage2_export_onnx report is missing onnx_paths[{subgraph!r}]: {stage2_report_path}"
            )
            continue
        resolved_path = _resolve_path_from_stage2(raw_onnx_path, stage2_report_path, stage2_payload)
        resolved_onnx[subgraph] = resolved_path
        if not resolved_path.is_file():
            blocking_reasons.append(
                f"stage2_export_onnx artifact for {subgraph} does not exist: {resolved_path}"
            )

    coherent_onnx_dir = None
    if resolved_onnx:
        coherent_onnx_dir = _single_resolved_path_or_none({path.parent for path in resolved_onnx.values()})
        if coherent_onnx_dir is None:
            blocking_reasons.append(
                "stage2_export_onnx artifact paths do not form a single coherent ONNX directory: "
                + ", ".join(f"{name}={path}" for name, path in sorted(resolved_onnx.items()))
            )
    if coherent_onnx_dir is not None and stage2_onnx_dir is not None and coherent_onnx_dir != stage2_onnx_dir:
        blocking_reasons.append(
            "stage2_export_onnx onnx_dir does not match the resolved artifact directory: "
            f"report={stage2_onnx_dir}, resolved={coherent_onnx_dir}"
        )
    if coherent_onnx_dir is not None and stage3_onnx_dir is not None and coherent_onnx_dir != stage3_onnx_dir:
        blocking_reasons.append(
            "stage3_verify_onnx onnx_dir does not match the resolved artifact directory: "
            f"report={stage3_onnx_dir}, resolved={coherent_onnx_dir}"
        )

    stage3_artifact_paths = stage3_payload.get("artifact_paths", {})
    if not isinstance(stage3_artifact_paths, dict):
        blocking_reasons.append(
            "stage3_verify_onnx report is missing the artifact_paths mapping required for provenance."
        )
        stage3_artifact_paths = {}
    for subgraph in ONNX_FILENAMES:
        raw_stage3_path = stage3_artifact_paths.get(subgraph)
        if not raw_stage3_path:
            blocking_reasons.append(
                f"stage3_verify_onnx report is missing artifact_paths[{subgraph!r}]: {stage3_report_path}"
            )
            continue
        resolved_stage3_path = Path(raw_stage3_path).expanduser().resolve(strict=False)
        stage2_path = resolved_onnx.get(subgraph)
        if stage2_path is not None and resolved_stage3_path != stage2_path:
            blocking_reasons.append(
                f"stage3_verify_onnx artifact path for {subgraph} does not match stage2_export_onnx: "
                f"stage2={stage2_path}, stage3={resolved_stage3_path}"
            )

    stage2_context = stage3_payload.get("stage2_context")
    if isinstance(stage2_context, dict):
        stage2_context_report = _resolve_optional_path(stage2_context.get("stage2_report_path"))
        if stage2_context_report is not None and stage2_context_report != stage2_report_path:
            blocking_reasons.append(
                "stage3_verify_onnx references a different stage2_export_onnx report: "
                f"resolved={stage2_report_path}, stage3_context={stage2_context_report}"
            )
        stage2_context_paths = stage2_context.get("stage2_onnx_paths", {})
        if isinstance(stage2_context_paths, dict):
            for subgraph in ONNX_FILENAMES:
                raw_context_path = stage2_context_paths.get(subgraph)
                if not raw_context_path:
                    continue
                resolved_context_path = Path(raw_context_path).expanduser().resolve(strict=False)
                stage2_path = resolved_onnx.get(subgraph)
                if stage2_path is not None and resolved_context_path != stage2_path:
                    blocking_reasons.append(
                        f"stage3_verify_onnx stage2_context path for {subgraph} does not match the resolved stage2 artifact: "
                        f"resolved={stage2_path}, stage3_context={resolved_context_path}"
                    )

    if onnx_path.is_file() and onnx_path.suffix.lower() == ".onnx" and onnx_path not in resolved_onnx.values():
        blocking_reasons.append(
            "Explicit --onnx-path file does not belong to the coherent stage2/stage3 artifact set: "
            f"requested={onnx_path}, resolved={list(sorted(path.as_posix() for path in resolved_onnx.values()))}"
        )

    return OnnxArtifactSafetyReport(
        stage2_report_path=stage2_report_path,
        stage2_gate_status=stage2_gate_status,
        stage3_report_path=stage3_report_path,
        stage3_gate_status=stage3_gate_status,
        stage3_overall_status=stage3_overall_status,
        stage2_policy_dir=stage2_policy_dir,
        stage3_policy_dir=stage3_policy_dir,
        stage2_run_dir=stage2_run_dir,
        stage3_run_dir=stage3_run_dir,
        stage2_onnx_dir=stage2_onnx_dir,
        stage3_onnx_dir=stage3_onnx_dir,
        resolved_onnx_paths={name: path.as_posix() for name, path in resolved_onnx.items()},
        blocking_reasons=tuple(blocking_reasons),
        notes=tuple(notes),
    )


def resolve_onnx_artifacts(
    onnx_path_arg: str,
    stage2_report_arg: str | None,
) -> tuple[OnnxPiArtifacts, OnnxArtifactSafetyReport]:
    onnx_path = Path(onnx_path_arg).expanduser().resolve(strict=False)
    stage2_report_path = _resolve_stage2_report_path(onnx_path, stage2_report_arg)
    if stage2_report_path is None:
        raise FileNotFoundError(
            "Could not resolve stage2_export_onnx report for the selected ONNX path. "
            "Refusing to guess a mixed artifact set without stage2 provenance."
        )

    stage2_payload = read_json(stage2_report_path)
    if not isinstance(stage2_payload, dict):
        raise TypeError(f"stage2_export_onnx report must be a JSON object: {stage2_report_path}")

    stage3_report_path = _resolve_stage3_report_path(stage2_report_path, stage2_payload)
    if stage3_report_path is None:
        raise FileNotFoundError(
            "Could not resolve stage3_verify_onnx report for the selected ONNX artifacts. "
            "Refusing to continue without stage3 provenance."
        )
    stage3_payload = read_json(stage3_report_path)
    if not isinstance(stage3_payload, dict):
        raise TypeError(f"stage3_verify_onnx report must be a JSON object: {stage3_report_path}")

    artifact_safety = assess_onnx_artifact_safety(
        onnx_path=onnx_path,
        stage2_report_path=stage2_report_path,
        stage2_payload=stage2_payload,
        stage3_report_path=stage3_report_path,
        stage3_payload=stage3_payload,
    )
    if not artifact_safety.is_safe:
        reasons = "\n".join(f"  - {reason}" for reason in artifact_safety.blocking_reasons)
        raise ValueError(
            "Refusing to launch PI05 ONNX runtime without coherent stage2/stage3 provenance:\n"
            f"{reasons}"
        )

    resolved_onnx = {
        name: Path(path).expanduser().resolve(strict=False)
        for name, path in artifact_safety.resolved_onnx_paths.items()
    }
    onnx_dir = (
        artifact_safety.stage2_onnx_dir
        or artifact_safety.stage3_onnx_dir
        or resolved_onnx["vision_encoder"].parent
    )
    return (
        OnnxPiArtifacts(
            onnx_dir=onnx_dir,
            vision_onnx=resolved_onnx["vision_encoder"],
            prefix_onnx=resolved_onnx["prefix_cache"],
            denoise_onnx=resolved_onnx["denoise_step"],
            stage2_report_path=stage2_report_path,
            stage2_payload=stage2_payload,
        ),
        artifact_safety,
    )


def validate_paths(args: argparse.Namespace) -> ResolvedPaths:
    policy_dir = resolve_checkpoint_dir(args.policy_path)
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    artifacts, artifact_safety = resolve_onnx_artifacts(args.onnx_path, args.onnx_stage2_report_path)
    if artifact_safety.stage2_policy_dir is not None and artifact_safety.stage2_policy_dir != policy_dir:
        raise ValueError(
            "Policy path does not match stage2_export_onnx policy_dir: "
            f"policy={policy_dir}, stage2={artifact_safety.stage2_policy_dir}"
        )
    if artifact_safety.stage3_policy_dir is not None and artifact_safety.stage3_policy_dir != policy_dir:
        raise ValueError(
            "Policy path does not match stage3_verify_onnx policy_dir: "
            f"policy={policy_dir}, stage3={artifact_safety.stage3_policy_dir}"
        )
    local_tokenizer_path = discover_local_tokenizer_path(args.local_tokenizer_path, require=False)

    return ResolvedPaths(
        policy_dir=policy_dir,
        calib_dir=calib_dir,
        artifacts=artifacts,
        artifact_safety=artifact_safety,
        local_tokenizer_path=local_tokenizer_path,
    )


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
        raise ValueError("PI05 ONNX runtime does not support temporal ensembling.")

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


def preflight_onnx_adapter(
    policy_cfg: object,
    artifacts: OnnxPiArtifacts,
    onnx_provider: str,
    noise_seed: int | None,
    fixed_noise: bool,
):
    onnx_policy = OnnxPi05PolicyAdapter(
        policy_cfg,
        artifacts=artifacts,
        onnx_provider=onnx_provider,
        noise_seed=noise_seed,
        fixed_noise=fixed_noise,
    )
    onnx_policy.eval()
    info(f"PI05 ONNX policy OK: onnx_provider={onnx_provider}")
    info(f"PI05 ONNX runtime: {onnx_policy.runtime_summary()}")
    for subgraph_name, summary in onnx_policy.describe_engines().items():
        info(
            f"ONNX `{subgraph_name}`: "
            f"inputs={summary['input_names']} outputs={summary['output_names']} "
            f"providers={summary['active_providers']}"
        )
    return onnx_policy


def print_summary(
    args: argparse.Namespace,
    resolved: ResolvedPaths,
    policy_cfg: object,
    preprocessor_details: dict,
    rtc_runtime: ResolvedRTCRuntimeConfig,
) -> None:
    stage2_report_path = resolved.artifact_safety.stage2_report_path
    stage3_report_path = resolved.artifact_safety.stage3_report_path
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
            "unless --rtc-enable/--rtc-enabled or another --rtc-* override is provided."
        )
    info(f"Policy device: {args.policy_device}")
    info(f"ONNX provider: {args.onnx_provider}")
    info(f"ONNX path: {Path(args.onnx_path).expanduser().resolve(strict=False)}")
    info(
        "stage2_export_onnx report: "
        f"{stage2_report_path if stage2_report_path is not None else '<none>'}"
    )
    info(f"stage2_export_onnx gate: {resolved.artifact_safety.stage2_gate_status or '<missing>'}")
    info(
        "stage3_verify_onnx report: "
        f"{stage3_report_path if stage3_report_path is not None else '<none>'}"
    )
    info(f"stage3_verify_onnx gate: {resolved.artifact_safety.stage3_gate_status or '<missing>'}")
    info(f"stage3_verify_onnx overall_status: {resolved.artifact_safety.stage3_overall_status or '<missing>'}")
    info(f"ONNX provenance policy_dir: {resolved.artifact_safety.stage2_policy_dir or '<missing>'}")
    info(f"ONNX provenance run_dir: {resolved.artifact_safety.stage2_run_dir or '<missing>'}")
    info(f"ONNX provenance onnx_dir: {resolved.artifact_safety.stage2_onnx_dir or resolved.artifacts.onnx_dir}")
    for subgraph, artifact_path in sorted(resolved.artifact_safety.resolved_onnx_paths.items()):
        info(f"ONNX artifact `{subgraph}`: {artifact_path}")
    info(f"Calibration dir: {resolved.calib_dir}")
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
        info(f"ONNX artifact note: {note}")


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

        if key == "gripper.pos":
            limit = effective_gripper_limit
        else:
            limit = joint_delta_limit
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

        if key == "gripper.pos":
            alpha = effective_gripper_alpha
        else:
            alpha = joint_action_alpha
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

    onnx_policy = None
    if not args.skip_onnx_preflight:
        onnx_policy = preflight_onnx_adapter(
            policy_cfg,
            resolved.artifacts,
            args.onnx_provider,
            args.policy_noise_seed,
            args.policy_fixed_noise,
        )

    if args.preflight_only:
        if onnx_policy is not None:
            onnx_policy.close()
        info("Preflight completed. Exiting before robot connect.")
        return 0

    stage("Build robot and processors")
    robot_cfg = build_robot_config(args, resolved.calib_dir)
    robot = make_robot_from_config(robot_cfg)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    if onnx_policy is None:
        stage("Load PI05 ONNX policy")
        onnx_policy = preflight_onnx_adapter(
            policy_cfg,
            resolved.artifacts,
            args.onnx_provider,
            args.policy_noise_seed,
            args.policy_fixed_noise,
        )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        stage("Connect robot")
        robot.connect()
        onnx_policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()
        dataset_features = build_dataset_features(robot)
        info("Robot connected. Preparing async PI05 ONNX inference loop.")

        if args.prefetch_threshold is not None and args.prefetch_threshold < 0:
            raise ValueError("--prefetch-threshold must be >= 0")
        if args.sync_refill_timeout_s < 0.0:
            raise ValueError(f"--sync-refill-timeout-s must be >= 0, got {args.sync_refill_timeout_s}")

        prefetcher = AsyncChunkPrefetcher(
            policy=onnx_policy,
            device=get_safe_torch_device(policy_cfg.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
            robot_type=robot.robot_type,
            n_action_steps=int(policy_cfg.n_action_steps),
            thread_name_prefix="pi05_onnx_prefetch",
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
        delta_clip_event_count = 0
        smoothing_event_count = 0
        action_keys = list(robot.action_features.keys())
        last_sent_action: dict[str, float] | None = None

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
                last_real_delay = merge_chunk_prediction_result(action_queue, completed_chunk)
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
                    last_real_delay = merge_chunk_prediction_result(action_queue, waited_chunk)
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
                warn(
                    "Action queue drained with no async chunk available after the grace wait. "
                    "Running synchronous chunk generation; `real_delay=0` on this refill path only reflects "
                    "blocking refill semantics."
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
                last_real_delay = merge_chunk_prediction_result(
                    action_queue,
                    sync_chunk,
                    action_index_before_inference=action_index_before_sync,
                    real_delay=0,
                )
                sync_refill_count += 1
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
        if onnx_policy is not None:
            onnx_policy.close()
        info("Inference finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
