#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import (
    DEFAULT_CONDA_ENV,
    ensure_dir,
    load_policy,
    resolve_policy_dir,
    resolve_tensorrt_py_dir,
    resolve_tmpdir,
    validate_engine_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot import policies  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots import make_robot_from_config
import lerobot.robots.so101_follower  # noqa: F401
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging


DEFAULT_POLICY_PATH = REPO_ROOT / "tmp" / "train" / "groot_grasp" / "checkpoints" / "010000"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "trt" / "groot_self_run_20260311_161210"
DEFAULT_ENGINE_DIR = DEFAULT_RUN_DIR / "gr00t_engine_api_trt1013"
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")
CONTRACT_SCHEMA_VERSION = "so101_pickplace/v1"
DEFAULT_SAFETY_PROFILE = "default"
SAFE_SAFETY_PROFILES: dict[str, float] = {
    "default": 8.0,
    "strict": 4.0,
}
UNSAFE_SAFETY_PROFILES = {"off", "unsafe", "disabled", "none"}


@dataclass(frozen=True)
class RuntimeContract:
    task_text: str
    task_text_input: str
    intent_payload: Any | None
    resolved_intent_source: str
    safety_profile: str
    max_relative_target: float | None
    events_jsonl_path: Path | None
    run_id: str
    mode: str


@dataclass(frozen=True)
class GuardResult:
    status: str
    action: dict[str, float]
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class JsonlEventLogger:
    def __init__(self, path: Path | None, *, run_id: str):
        self.path = path
        self.run_id = run_id
        self._event_id = 0

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def log(self, event_type: str, **fields: Any) -> None:
        if self.path is None:
            return

        self._event_id += 1
        record = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "event_id": self._event_id,
            "event_type": event_type,
            "run_id": self.run_id,
            "timestamp_unix_s": time.time(),
            **fields,
        }

        try:
            ensure_dir(self.path.parent)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            raise RuntimeError(f"Failed to write events JSONL to {self.path}: {exc}") from exc


class JointSafetyGuard:
    def __init__(self, *, max_relative_target: float):
        self.max_relative_target = float(max_relative_target)

    def validate(self, action: Mapping[str, Any], obs: Mapping[str, Any]) -> GuardResult:
        joint_state = extract_joint_positions(obs)
        if not joint_state:
            return GuardResult(
                status="REJECT",
                action={},
                error_code="missing_joint_state",
                details={"reason": "No `.pos` observation fields available for safety validation."},
            )

        final_action: dict[str, float] = {}
        clamped: dict[str, dict[str, float]] = {}

        for key, value in action.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                return GuardResult(
                    status="REJECT",
                    action={},
                    error_code="non_numeric_action",
                    details={"action_key": key, "raw_value": repr(value), "error": str(exc)},
                )
            if not math.isfinite(numeric_value):
                return GuardResult(
                    status="REJECT",
                    action={},
                    error_code="nan_or_inf_action",
                    details={"action_key": key, "raw_value": numeric_value},
                )

            if not key.endswith(".pos"):
                final_action[key] = numeric_value
                continue

            if key not in joint_state:
                return GuardResult(
                    status="REJECT",
                    action={},
                    error_code="missing_joint_state",
                    details={"action_key": key, "available_joint_keys": sorted(joint_state)},
                )

            current_value = joint_state[key]
            lower = current_value - self.max_relative_target
            upper = current_value + self.max_relative_target
            bounded_value = min(max(numeric_value, lower), upper)
            final_action[key] = bounded_value

            if bounded_value != numeric_value:
                clamped[key] = {
                    "requested": numeric_value,
                    "current": current_value,
                    "sent": bounded_value,
                }

        if clamped:
            return GuardResult(
                status="CLAMP_AND_ACCEPT",
                action=final_action,
                details={
                    "max_relative_target": self.max_relative_target,
                    "clamped_joints": clamped,
                },
            )
        return GuardResult(
            status="ACCEPT",
            action=final_action,
            details={"max_relative_target": self.max_relative_target},
        )


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


def maybe_add_tensorrt_python_path(path: str | None) -> str | None:
    resolved = resolve_tensorrt_py_dir(path)
    if resolved:
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        os.environ["TENSORRT_PY_DIR"] = resolved
    return resolved


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing Python module `{module_name}` in current env.\n"
            f"Install hint: {install_hint}\n"
            f"Current python: {sys.executable}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GROOT TensorRT inference on a real SO101 follower robot."
    )
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration-dir", default=str(DEFAULT_CALIB_DIR))

    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-n-action-steps", type=parse_optional_int, default=None)

    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--engine-dir", default=str(DEFAULT_ENGINE_DIR))
    parser.add_argument("--trt-device", default="cuda:0")
    parser.add_argument("--tensorrt-py-dir", default=None)
    parser.add_argument("--tmpdir", default=None)

    parser.add_argument("--task", default=os.getenv("TASK_TEXT", "grasp block in bin"))
    parser.add_argument("--intent-json", default=os.getenv("INTENT_JSON", os.getenv("TASK_INTENT_JSON", "")))
    parser.add_argument("--task-intent-json", dest="intent_json", help=argparse.SUPPRESS)
    parser.add_argument("--safety-profile", default=os.getenv("SAFETY_PROFILE", DEFAULT_SAFETY_PROFILE))
    parser.add_argument(
        "--events-jsonl",
        default=os.getenv("EVENTS_JSONL_PATH", os.getenv("EVENTS_PATH", "")),
    )
    parser.add_argument("--events-jsonl-path", dest="events_jsonl", help=argparse.SUPPRESS)
    parser.add_argument("--run-time-s", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--allow-unsafe", action="store_true")

    parser.add_argument("--skip-camera-preflight", action="store_true")
    parser.add_argument("--skip-trt-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_engine_dir(run_dir: Path, engine_dir_arg: str | None) -> Path:
    if engine_dir_arg:
        return Path(engine_dir_arg).expanduser().resolve()
    return (run_dir / "gr00t_engine_api_trt1013").resolve()


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    policy_dir = resolve_policy_dir(args.policy_path)
    run_dir = Path(args.run_dir).expanduser().resolve()
    engine_dir = resolve_engine_dir(run_dir, args.engine_dir)
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()

    validate_engine_dir(engine_dir)
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    return policy_dir, run_dir, engine_dir, calib_dir


def normalize_optional_json(value: str | None, *, field_name: str) -> Any | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"`{field_name}` must be valid JSON: {exc}") from exc


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def infer_task_text(task_text: str, intent_payload: Any | None) -> tuple[str, str]:
    normalized_task = task_text.strip()
    if intent_payload is not None:
        if isinstance(intent_payload, dict):
            for key in ("task_text", "task", "raw_task_text", "text", "prompt"):
                raw_value = intent_payload.get(key)
                if isinstance(raw_value, str) and raw_value.strip():
                    return raw_value.strip(), "intent_json"

            verb = str(intent_payload.get("verb", "")).strip()
            target_object = str(intent_payload.get("target_object", "")).strip()
            target_container = str(intent_payload.get("target_container", "")).strip()
            synthesized_parts = [part for part in [verb, target_object, target_container] if part]
            if synthesized_parts:
                return " ".join(synthesized_parts), "intent_json"
        return compact_json(intent_payload), "intent_json"

    if normalized_task:
        return normalized_task, "task"

    raise ValueError("Provide either `--task` or `--intent-json`.")


def resolve_runtime_contract(args: argparse.Namespace) -> RuntimeContract:
    if args.dry_run and args.preflight_only:
        raise ValueError("`--dry-run` and `--preflight-only` are mutually exclusive.")

    intent_payload = normalize_optional_json(args.intent_json, field_name="intent_json")
    task_text, resolved_intent_source = infer_task_text(args.task, intent_payload)

    safety_profile = (args.safety_profile or DEFAULT_SAFETY_PROFILE).strip().lower() or DEFAULT_SAFETY_PROFILE
    max_relative_target: float | None = None

    if safety_profile in UNSAFE_SAFETY_PROFILES:
        if not args.dry_run:
            raise ValueError(
                "Unsafe safety profiles are rejected for real TRT robot runs. "
                "Use `default` or `strict`, or limit unsafe experiments to `--dry-run --allow-unsafe`."
            )
        if not args.allow_unsafe:
            raise ValueError("Unsafe safety profiles require `--allow-unsafe`, and only in `--dry-run` mode.")
        warn(f"Unsafe safety profile `{safety_profile}` is allowed only because `--dry-run --allow-unsafe` was set.")
    else:
        if safety_profile not in SAFE_SAFETY_PROFILES:
            allowed = ", ".join(sorted(SAFE_SAFETY_PROFILES))
            raise ValueError(f"Unsupported safety profile `{safety_profile}`. Allowed values: {allowed}.")
        max_relative_target = SAFE_SAFETY_PROFILES[safety_profile]

    events_jsonl_path = None
    if args.events_jsonl and args.events_jsonl.strip():
        events_jsonl_path = Path(args.events_jsonl).expanduser().resolve()

    mode = "dry_run_contract_only" if args.dry_run else "preflight_only" if args.preflight_only else "robot_run"
    return RuntimeContract(
        task_text=task_text,
        task_text_input=args.task,
        intent_payload=intent_payload,
        resolved_intent_source=resolved_intent_source,
        safety_profile=safety_profile,
        max_relative_target=max_relative_target,
        events_jsonl_path=events_jsonl_path,
        run_id=uuid.uuid4().hex[:12],
        mode=mode,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def extract_joint_positions(obs: Mapping[str, Any]) -> dict[str, float]:
    joints: dict[str, float] = {}
    for key, value in obs.items():
        if not key.endswith(".pos"):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            joints[key] = numeric_value
    return joints


def build_hold_action(obs: Mapping[str, Any]) -> dict[str, float]:
    return extract_joint_positions(obs)


def summarize_joint_action(action: Mapping[str, Any]) -> dict[str, Any]:
    numeric_items = []
    for key, value in action.items():
        try:
            numeric_items.append((key, float(value)))
        except (TypeError, ValueError):
            continue
    if not numeric_items:
        return {"joint_count": 0}

    values = [value for _, value in numeric_items]
    return {
        "joint_count": len(numeric_items),
        "joint_keys": [key for key, _ in numeric_items],
        "min_value": min(values),
        "max_value": max(values),
    }


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


def build_robot_config(args: argparse.Namespace, calib_dir: Path, *, max_relative_target: float) -> SO101FollowerConfig:
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
        max_relative_target=max_relative_target,
        cameras=cameras,
    )


def apply_groot_runtime_overrides(args: argparse.Namespace, policy_cfg: object) -> None:
    if getattr(policy_cfg, "type", None) != "groot":
        raise ValueError(f"Expected GROOT policy, got {getattr(policy_cfg, 'type', None)!r}")

    chunk_size = int(policy_cfg.chunk_size)
    if args.policy_n_action_steps is not None:
        if not 1 <= args.policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {args.policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = int(args.policy_n_action_steps)


def load_pre_post_processors(
    policy_dir: Path,
    policy_device: str,
) -> tuple[
    PolicyProcessorPipeline[dict[str, object], dict[str, object]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": policy_device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def preflight_trt_adapter(
    policy_cfg: object,
    base_policy: object,
    engine_dir: Path,
    trt_device: str,
):
    from groot_trt_adapter_local import TrtGrootPolicyAdapter

    trt_policy = TrtGrootPolicyAdapter(
        policy_cfg,
        base_policy=base_policy,
        engine_dir=engine_dir,
        trt_device=trt_device,
    )
    info(f"GROOT TRT policy OK: trt_device={trt_device}")
    for engine_name, tensors in trt_policy.describe_engines().items():
        info(f"Engine `{engine_name}`:")
        for tensor in tensors:
            info(
                f"  name={tensor['name']} mode={tensor['mode']} "
                f"dtype={tensor['dtype']} shape={tensor['shape']}"
            )
    return trt_policy


def print_summary(
    args: argparse.Namespace,
    contract: RuntimeContract,
    policy_dir: Path,
    run_dir: Path,
    engine_dir: Path,
    calib_dir: Path,
    tmpdir: str,
    tensorrt_py_dir: str | None,
    policy_cfg: object,
) -> None:
    info(f"Python: {sys.executable}")
    info(f"Conda env(default): {DEFAULT_CONDA_ENV}")
    info(f"Policy path: {policy_dir}")
    info(f"Run dir: {run_dir}")
    info(f"Engine dir: {engine_dir}")
    info(f"Calibration dir: {calib_dir}")
    info(f"Policy type: {getattr(policy_cfg, 'type', '<unknown>')}")
    info(f"Policy device: {getattr(policy_cfg, 'device', '<unknown>')}")
    info(f"GROOT runtime config: chunk_size={policy_cfg.chunk_size}, n_action_steps={policy_cfg.n_action_steps}")
    info(f"Run mode: {contract.mode}")
    info(f"TRT device: {args.trt_device}")
    info(f"TENSORRT_PY_DIR: {tensorrt_py_dir or '<unset>'}")
    info(f"TMPDIR: {tmpdir}")
    info(f"Robot port: {args.robot_port}")
    info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    info(f"Task input: {contract.task_text_input}")
    info(f"Resolved task: {contract.task_text}")
    info(f"Resolved intent source: {contract.resolved_intent_source}")
    info(f"Safety profile: {contract.safety_profile}")
    info(f"max_relative_target: {contract.max_relative_target}")
    info(f"Events JSONL: {contract.events_jsonl_path or '<disabled>'}")
    info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")


def print_contract_summary(args: argparse.Namespace, contract: RuntimeContract, tmpdir: str, tensorrt_py_dir: str | None) -> None:
    info(f"Python: {sys.executable}")
    info(f"Conda env(default): {DEFAULT_CONDA_ENV}")
    info(f"Run mode: {contract.mode}")
    info(f"Task input: {contract.task_text_input}")
    info(f"Resolved task: {contract.task_text}")
    info(f"Resolved intent source: {contract.resolved_intent_source}")
    info(f"Safety profile: {contract.safety_profile}")
    info(f"max_relative_target: {contract.max_relative_target}")
    info(f"Events JSONL: {contract.events_jsonl_path or '<disabled>'}")
    info(f"Run dir: {Path(args.run_dir).expanduser().resolve()}")
    info(f"Engine dir: {resolve_engine_dir(Path(args.run_dir).expanduser().resolve(), args.engine_dir)}")
    info(f"TENSORRT_PY_DIR: {tensorrt_py_dir or '<unset>'}")
    info(f"TMPDIR: {tmpdir}")
    info("`--dry-run` only validates the config contract. It does not prove TensorRT artifacts, cameras, or robot access.")


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


def main() -> int:
    args = build_parser().parse_args()
    tensorrt_py_dir = maybe_add_tensorrt_python_path(args.tensorrt_py_dir)
    tmpdir = resolve_tmpdir(args.tmpdir, Path(args.run_dir).expanduser().resolve())
    os.environ["TMPDIR"] = tmpdir

    register_third_party_devices()
    init_logging()
    contract = resolve_runtime_contract(args)
    event_logger = JsonlEventLogger(contract.events_jsonl_path, run_id=contract.run_id)
    event_logger.log(
        "run_start",
        mode=contract.mode,
        task_text=contract.task_text,
        task_text_input=contract.task_text_input,
        resolved_intent_source=contract.resolved_intent_source,
        intent_payload=contract.intent_payload,
        safety_profile=contract.safety_profile,
    )

    if args.dry_run:
        stage("Validate runtime contract")
        print_contract_summary(args, contract, tmpdir, tensorrt_py_dir)
        event_logger.log("run_end", mode=contract.mode, status="ok")
        return 0

    stage("Validate environment")
    require_module("tensorrt", "set --tensorrt-py-dir or export TENSORRT_PY_DIR to your TensorRT Python package path")
    policy_dir, run_dir, engine_dir, calib_dir = validate_paths(args)
    ensure_dir(run_dir / "logs")

    policy_cfg, _, base_policy = load_policy(policy_dir, device=args.policy_device, strict=False)
    apply_groot_runtime_overrides(args, policy_cfg)
    print_summary(args, contract, policy_dir, run_dir, engine_dir, calib_dir, tmpdir, tensorrt_py_dir, policy_cfg)

    stage("Preflight checks")
    if not args.skip_camera_preflight:
        preflight_cameras(args)

    trt_policy = None
    if not args.skip_trt_preflight:
        trt_policy = preflight_trt_adapter(policy_cfg, base_policy, engine_dir, args.trt_device)

    if args.preflight_only:
        event_logger.log("run_end", mode=contract.mode, status="ok")
        info("Preflight completed. Exiting before robot connect.")
        return 0

    stage("Build robot and processors")
    if contract.max_relative_target is None:
        raise RuntimeError("Real robot runs require a resolved max_relative_target safety limit.")

    robot_cfg = build_robot_config(args, calib_dir, max_relative_target=contract.max_relative_target)
    robot = make_robot_from_config(robot_cfg)
    preprocessor, postprocessor = load_pre_post_processors(policy_dir, args.policy_device)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    safety_guard = JointSafetyGuard(max_relative_target=contract.max_relative_target)

    if trt_policy is None:
        stage("Load GROOT TRT policy")
        trt_policy = preflight_trt_adapter(policy_cfg, base_policy, engine_dir, args.trt_device)

    trt_policy.eval()

    step = 0
    run_end_status = "ok"
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        stage("Connect robot")
        robot.connect()
        trt_policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        dataset_features = build_dataset_features(robot)
        info("Robot connected. Starting GROOT TRT inference loop.")

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                info("Reached requested run_time_s. Exiting inference loop.")
                break

            loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            queue_was_empty = len(trt_policy._action_queue) == 0
            action_values = predict_action(
                observation=observation_frame,
                policy=trt_policy,
                device=get_safe_torch_device(policy_cfg.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(getattr(policy_cfg, "use_amp", False)),
                task=contract.task_text,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_values, dataset_features)
            requested_robot_action = robot_action_processor((action_dict, obs))
            guard_result = safety_guard.validate(requested_robot_action, obs)
            if guard_result.status == "REJECT":
                event_logger.log(
                    "guard_reject",
                    step_id=step + 1,
                    error_code=guard_result.error_code,
                    guard_result=guard_result.details,
                    requested_action_summary=summarize_joint_action(requested_robot_action),
                )
                hold_action = build_hold_action(obs)
                if hold_action:
                    robot.send_action(hold_action)
                    event_logger.log(
                        "fail_safe_hold",
                        step_id=step + 1,
                        action_summary=summarize_joint_action(hold_action),
                    )
                raise RuntimeError(
                    f"Safety guard rejected TRT action at step {step + 1}: {guard_result.error_code}"
                )

            robot_action_to_send = guard_result.action
            if guard_result.status == "CLAMP_AND_ACCEPT":
                event_logger.log(
                    "guard_clamp",
                    step_id=step + 1,
                    guard_result=guard_result.details,
                    requested_action_summary=summarize_joint_action(requested_robot_action),
                    final_action_summary=summarize_joint_action(robot_action_to_send),
                )
            sent_action = robot.send_action(robot_action_to_send)

            step += 1
            event_logger.log(
                "step",
                step_id=step,
                generated_new_chunk=queue_was_empty,
                queue_size=len(trt_policy._action_queue),
                action_summary=summarize_joint_action(sent_action),
                guard_status=guard_result.status,
            )
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                info(
                    f"Step {step} | elapsed={elapsed:.2f}s | "
                    f"generated_new_chunk={queue_was_empty} | queue_size={len(trt_policy._action_queue)}"
                )

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.camera_fps - dt_s, 0.0))
    except KeyboardInterrupt:
        run_end_status = "interrupted"
        info("KeyboardInterrupt received. Stopping inference.")
    except Exception as exc:
        event_logger.log("run_end", mode=contract.mode, status="error", error=f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if "robot" in locals() and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        info("Inference finished.")

    event_logger.log("run_end", mode=contract.mode, status=run_end_status, steps=step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
