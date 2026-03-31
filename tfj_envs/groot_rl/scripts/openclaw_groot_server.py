#!/usr/bin/env python

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_so101_pickplace_infer.sh"
LOG_DIR = REPO_ROOT / "outputs" / "openclaw_jobs"
HOST = os.environ.get("OPENCLAW_GROOT_HOST", "127.0.0.1")
PORT = int(os.environ.get("OPENCLAW_GROOT_PORT", "8765"))
DEFAULT_BACKEND = "groot"
DEFAULT_ROBOT_ID = "so101_follower"
DEFAULT_TASK = "Put the block in the bin"
DEFAULT_SAFETY_PROFILE = "default"
SHORTCUT_POLICY_DEVICE = "cuda"
SHORTCUT_ROBOT_PORT = "/dev/ttyACM0"
SHORTCUT_ROBOT_ID = "my_so101"
SHORTCUT_ROBOT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
SHORTCUT_TOP_CAMERA_INDEX = 4
SHORTCUT_WRIST_CAMERA_INDEX = 6
SHORTCUT_CAMERA_WIDTH = 640
SHORTCUT_CAMERA_HEIGHT = 480
SHORTCUT_CAMERA_FPS = 30
SHORTCUT_DATASET_ROOT = "/data/tfj/lerobot_tfj/outputs/feishu_so101_run"
SHORTCUT_EVENTS_JSONL_PATH = "/data/tfj/lerobot_tfj/outputs/feishu_so101_run.events.jsonl"
SHORTCUT_DISPLAY_DATA = False
SHORTCUT_CANONICAL_TASK = "pick block and place to box"
SNAPSHOT_DIR = Path(SHORTCUT_DATASET_ROOT) / "snapshots"
ALLOWED_SNAPSHOT_MEDIA_DIR = Path.home() / ".openclaw" / "workspace" / "feishu_so101_snapshots"
DEFAULT_SNAPSHOT_CAMERA = "top"
DEFAULT_SNAPSHOT_WARMUP_FRAMES = 8
DEFAULT_SNAPSHOT_WARMUP_DELAY_S = 0.05
DEFAULT_SNAPSHOT_OPEN_RETRIES = 4
DEFAULT_SNAPSHOT_OPEN_RETRY_DELAY_S = 0.25
DEFAULT_VISION_BACKEND = os.environ.get("OPENCLAW_VISION_BACKEND", "codex").strip().lower() or "codex"
DEFAULT_VISION_CODEX_BIN = os.environ.get("OPENCLAW_VISION_CODEX_BIN", "codex").strip() or "codex"
DEFAULT_VISION_TIMEOUT_S = int(os.environ.get("OPENCLAW_VISION_TIMEOUT_S", "180"))
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_VISION_MODEL = os.environ.get("OPENCLAW_VISION_MODEL", "")
DEFAULT_VISION_DETAIL = os.environ.get("OPENCLAW_VISION_DETAIL", "high").strip().lower() or "high"
DEFAULT_VISION_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENCLAW_VISION_MAX_OUTPUT_TOKENS", "240"))
DEFAULT_VISION_PROMPT = (
    "请基于这张机器人相机图，简洁回答用户问题。"
    "如果用户没有给出具体问题，就描述画面里最重要的物体、方块、盒子、机械臂和抓取相关信息。"
    "不要编造看不到的内容。"
)
SUPPORTED_BACKENDS = {"groot", "smolvla", "pi05", "act", "policy_record"}
BACKEND_ALIASES = {
    "groot": "groot",
    "smolvla": "smolvla",
    "pi": "pi05",
    "pi05": "pi05",
    "pi0.5": "pi05",
    "act": "act",
    "act_distill": "act",
    "policy": "policy_record",
    "policy_record": "policy_record",
}
LEGACY_PAYLOAD_KEYS = {
    "task_text",
    "instruction",
    "intent",
    "task_intent_json",
    "events_path",
    "clear_dataset",
    "model_backend",
    "policy_backend",
}
CAMERA_ALIASES = {
    "top": "top",
    "顶部": "top",
    "顶视": "top",
    "顶视角": "top",
    "俯视": "top",
    "俯视角": "top",
    "topcamera": "top",
    "overhead": "top",
    "wrist": "wrist",
    "wristcamera": "wrist",
    "hand": "wrist",
    "handcamera": "wrist",
    "gripper": "wrist",
    "grippercamera": "wrist",
    "腕": "wrist",
    "腕部": "wrist",
    "手腕": "wrist",
    "手腕相机": "wrist",
    "夹爪": "wrist",
    "夹爪相机": "wrist",
}
ARM_SDK_SCRIPT = WORKSPACE_ROOT / "tfj_envs" / "so101_control_pause_20260319" / "scripts" / "so101_sdk.py"
ARM_URDF_PATH = Path(
    os.environ.get("OPENCLAW_ARM_URDF_PATH", os.fspath(WORKSPACE_ROOT / "so101_new_calib.urdf"))
).expanduser()
ARM_DEFAULT_IK_SOLVER = str(os.environ.get("OPENCLAW_ARM_IK_SOLVER", "placo")).strip().lower() or "placo"
ARM_DEFAULT_MOVE_CM = float(os.environ.get("OPENCLAW_ARM_DEFAULT_MOVE_CM", "0.5"))
ARM_DEFAULT_VERTICAL_MOVE_CM = float(os.environ.get("OPENCLAW_ARM_DEFAULT_VERTICAL_MOVE_CM", "0.3"))
ARM_MAX_XY_STEP_MM = float(os.environ.get("OPENCLAW_ARM_MAX_XY_STEP_MM", "20.0"))
ARM_MAX_Z_STEP_MM = float(os.environ.get("OPENCLAW_ARM_MAX_Z_STEP_MM", "10.0"))
ARM_MAX_COMMAND_DELTA_DEG = float(os.environ.get("OPENCLAW_ARM_MAX_COMMAND_DELTA_DEG", "8.0"))
ARM_MAX_RELATIVE_TARGET_DEG = float(os.environ.get("OPENCLAW_ARM_MAX_RELATIVE_TARGET_DEG", "8.0"))
ARM_GRIPPER_MAX_RELATIVE_TARGET = float(os.environ.get("OPENCLAW_ARM_GRIPPER_MAX_RELATIVE_TARGET", "40.0"))
ARM_SETTLE_S = float(os.environ.get("OPENCLAW_ARM_SETTLE_S", "0.15"))
ARM_DIRECTION_ALIASES = {
    "up": "up",
    "上": "up",
    "上移": "up",
    "向上": "up",
    "向上移动": "up",
    "down": "down",
    "下": "down",
    "下移": "down",
    "向下": "down",
    "向下移动": "down",
    "left": "left",
    "左": "left",
    "左移": "left",
    "向左": "left",
    "向左移动": "left",
    "right": "right",
    "右": "right",
    "右移": "right",
    "向右": "right",
    "向右移动": "right",
    "forward": "forward",
    "front": "forward",
    "前": "forward",
    "前进": "forward",
    "向前": "forward",
    "向前移动": "forward",
    "back": "back",
    "backward": "back",
    "后": "back",
    "后退": "back",
    "向后": "back",
    "向后移动": "back",
    "open": "open",
    "打开": "open",
    "开爪": "open",
    "张开": "open",
    "close": "close",
    "关闭": "close",
    "闭合": "close",
    "合上": "close",
    "抓取": "close",
    "夹紧": "close",
    "home": "home",
    "归位": "home",
    "回零": "home",
    "复位": "home",
}
ARM_CARTESIAN_DIRECTIONS = {"up", "down", "left", "right", "forward", "back"}
ARM_GRIPPER_ACTIONS = {"open", "close"}
arm_lock = Lock()
_arm_sdk_runtime: tuple[Any, Any] | None = None


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def _placo_available() -> bool:
    return importlib.util.find_spec("placo") is not None


def _normalize_arm_direction(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _normalize_free_text(value)
    if not normalized:
        return None
    return ARM_DIRECTION_ALIASES.get(normalized)


def _normalize_gripper_command(payload: dict[str, Any]) -> str | None:
    if "gripper_closed" in payload:
        return "close" if _coerce_bool(payload.get("gripper_closed"), default=False) else "open"

    _, raw_value = _first_present(payload, "gripper", "gripper_action", "gripper_command", "action")
    if raw_value is None:
        return None

    if isinstance(raw_value, bool):
        return "close" if raw_value else "open"

    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
        if numeric > 0:
            return "open"
        if numeric < 0:
            return "close"
        return None

    command = _normalize_arm_direction(raw_value)
    if command in ARM_GRIPPER_ACTIONS:
        return command
    if _normalize_free_text(raw_value) in {"keep", "hold", "stay", "保持", "不动", "none"}:
        return None
    raise ValueError(f"unsupported gripper command: {raw_value}")


def _axis_limit_mm(direction: str) -> float:
    if direction in {"up", "down"}:
        return ARM_MAX_Z_STEP_MM
    return ARM_MAX_XY_STEP_MM


def _default_step_cm(direction: str) -> float:
    if direction in {"up", "down"}:
        return ARM_DEFAULT_VERTICAL_MOVE_CM
    return ARM_DEFAULT_MOVE_CM


def _validate_move_cm(direction: str, cm: float) -> float:
    if cm <= 0:
        raise ValueError("cm must be > 0")
    max_mm = _axis_limit_mm(direction)
    if cm * 10.0 > max_mm + 1e-9:
        raise ValueError(f"{direction} step exceeds limit: {cm * 10.0:.1f}mm > {max_mm:.1f}mm")
    return cm


def _load_arm_sdk_runtime() -> tuple[Any, Any]:
    global _arm_sdk_runtime

    if _arm_sdk_runtime is not None:
        return _arm_sdk_runtime

    if not ARM_SDK_SCRIPT.is_file():
        raise FileNotFoundError(f"SO101 SDK not found: {ARM_SDK_SCRIPT}")

    module_name = "_openclaw_so101_sdk"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, ARM_SDK_SCRIPT)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load SDK module from {ARM_SDK_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    try:
        sdk_cls = getattr(module, "SO101SDK")
        cfg_cls = getattr(module, "SO101SDKConfig")
    except AttributeError as exc:
        raise ImportError(f"incomplete SDK module: {ARM_SDK_SCRIPT}") from exc

    _arm_sdk_runtime = (sdk_cls, cfg_cls)
    return _arm_sdk_runtime


def _build_arm_sdk(*, robot_id: str, control_mode: str, require_ik: bool) -> Any:
    sdk_cls, cfg_cls = _load_arm_sdk_runtime()

    if control_mode not in {"joint", "ik"}:
        raise ValueError(f"unsupported control_mode: {control_mode}")

    if require_ik:
        if not _placo_available():
            raise RuntimeError(
                "placo is not installed; Cartesian arm control requires placo. "
                "Install `placo>=0.9.6,<0.10.0` or `lerobot[kinematics]`."
            )
        if not ARM_URDF_PATH.is_file():
            raise FileNotFoundError(f"URDF not found: {ARM_URDF_PATH}")

    cfg = cfg_cls(
        robot_port=SHORTCUT_ROBOT_PORT,
        robot_id=robot_id,
        calibration_dir=SHORTCUT_ROBOT_CALIB_DIR,
        calibrate_on_connect=False,
        dry_run=False,
        control_mode=control_mode,
        ik_solver=ARM_DEFAULT_IK_SOLVER,
        urdf_path=str(ARM_URDF_PATH),
        max_command_delta_deg=ARM_MAX_COMMAND_DELTA_DEG,
        max_relative_target_deg=ARM_MAX_RELATIVE_TARGET_DEG,
        gripper_max_relative_target=ARM_GRIPPER_MAX_RELATIVE_TARGET,
        settle_s=ARM_SETTLE_S,
    )
    return sdk_cls(cfg)


def _run_arm_sdk(robot_id: str, *, control_mode: str, require_ik: bool, operation: Any) -> dict[str, Any]:
    session_id = f"arm:{uuid.uuid4().hex[:8]}"
    with jobs_lock:
        blocking_job_id = _acquire_robot_session_locked(robot_id, session_id)
        if blocking_job_id is not None:
            blocking_job = jobs.get(blocking_job_id)
            raise RuntimeError(
                json.dumps(
                    {
                        "type": "robot_busy",
                        "robot_id": robot_id,
                        "blocking_job": _job_to_dict(blocking_job)
                        if blocking_job is not None
                        else {"job_id": blocking_job_id},
                    },
                    ensure_ascii=False,
                )
            )

    try:
        with arm_lock:
            sdk = _build_arm_sdk(robot_id=robot_id, control_mode=control_mode, require_ik=require_ik)
            sdk.connect()
            try:
                return operation(sdk)
            finally:
                sdk.disconnect()
    finally:
        with jobs_lock:
            _release_robot_session_locked(robot_id, session_id)


def _read_arm_state(robot_id: str) -> dict[str, Any]:
    def operation(sdk: Any) -> dict[str, Any]:
        return {"state": sdk.state()}

    return _run_arm_sdk(robot_id, control_mode="joint", require_ik=False, operation=operation)


def _run_arm_move(payload: dict[str, Any]) -> dict[str, Any]:
    robot_id = str(payload.get("robot_id") or SHORTCUT_ROBOT_ID).strip() or SHORTCUT_ROBOT_ID
    direction = _normalize_arm_direction(payload.get("direction") or payload.get("action"))
    if direction is None:
        raise ValueError("direction is required")

    if direction in ARM_GRIPPER_ACTIONS:
        def operation(sdk: Any) -> dict[str, Any]:
            if direction == "open":
                state = sdk.open_gripper()
            else:
                state = sdk.close_gripper()
            return {
                "direction": direction,
                "state": state,
                "control_mode": "joint",
            }

        return _run_arm_sdk(robot_id, control_mode="joint", require_ik=False, operation=operation)

    if direction == "home":
        def operation(sdk: Any) -> dict[str, Any]:
            state = sdk.home()
            return {
                "direction": direction,
                "state": state,
                "control_mode": "joint",
            }

        return _run_arm_sdk(robot_id, control_mode="joint", require_ik=False, operation=operation)

    if direction not in ARM_CARTESIAN_DIRECTIONS:
        raise ValueError(f"unsupported arm direction: {direction}")

    cm = _coerce_float(payload.get("cm"), default=_default_step_cm(direction))
    if cm is None:
        cm = _default_step_cm(direction)
    cm = _validate_move_cm(direction, float(cm))

    def operation(sdk: Any) -> dict[str, Any]:
        state = sdk.move(direction, cm, exact=True)
        return {
            "direction": direction,
            "cm": cm,
            "control_mode": "ik",
            "ik_solver": ARM_DEFAULT_IK_SOLVER,
            "state": state,
        }

    return _run_arm_sdk(robot_id, control_mode="ik", require_ik=True, operation=operation)


def _run_arm_jog(payload: dict[str, Any]) -> dict[str, Any]:
    robot_id = str(payload.get("robot_id") or SHORTCUT_ROBOT_ID).strip() or SHORTCUT_ROBOT_ID
    dx_mm = _coerce_float(payload.get("dx_mm"), default=0.0) or 0.0
    dy_mm = _coerce_float(payload.get("dy_mm"), default=0.0) or 0.0
    dz_mm = _coerce_float(payload.get("dz_mm"), default=0.0) or 0.0
    gripper_command = _normalize_gripper_command(payload)

    if abs(dx_mm) > ARM_MAX_XY_STEP_MM + 1e-9:
        raise ValueError(f"dx_mm exceeds limit: {dx_mm:.1f}mm")
    if abs(dy_mm) > ARM_MAX_XY_STEP_MM + 1e-9:
        raise ValueError(f"dy_mm exceeds limit: {dy_mm:.1f}mm")
    if abs(dz_mm) > ARM_MAX_Z_STEP_MM + 1e-9:
        raise ValueError(f"dz_mm exceeds limit: {dz_mm:.1f}mm")
    if abs(dx_mm) < 1e-9 and abs(dy_mm) < 1e-9 and abs(dz_mm) < 1e-9 and gripper_command is None:
        raise ValueError("arm jog requires non-zero dx_mm/dy_mm/dz_mm or gripper command")

    planned_moves: list[tuple[str, float]] = []
    if abs(dx_mm) >= 1e-9:
        planned_moves.append(("forward" if dx_mm > 0 else "back", abs(dx_mm) / 10.0))
    if abs(dy_mm) >= 1e-9:
        planned_moves.append(("left" if dy_mm > 0 else "right", abs(dy_mm) / 10.0))
    if abs(dz_mm) >= 1e-9:
        planned_moves.append(("up" if dz_mm > 0 else "down", abs(dz_mm) / 10.0))

    require_ik = bool(planned_moves)
    control_mode = "ik" if require_ik else "joint"

    def operation(sdk: Any) -> dict[str, Any]:
        applied_moves: list[dict[str, Any]] = []
        state: dict[str, float] | None = None
        for direction, cm in planned_moves:
            state = sdk.move(direction, cm, exact=True)
            applied_moves.append({"direction": direction, "cm": cm})

        if gripper_command == "open":
            state = sdk.open_gripper()
            applied_moves.append({"gripper": "open"})
        elif gripper_command == "close":
            state = sdk.close_gripper()
            applied_moves.append({"gripper": "close"})

        if state is None:
            state = sdk.state()

        return {
            "dx_mm": dx_mm,
            "dy_mm": dy_mm,
            "dz_mm": dz_mm,
            "gripper": gripper_command,
            "control_mode": control_mode,
            "ik_solver": ARM_DEFAULT_IK_SOLVER if require_ik else None,
            "applied_moves": applied_moves,
            "state": state,
        }

    return _run_arm_sdk(robot_id, control_mode=control_mode, require_ik=require_ik, operation=operation)


def _arm_exception_response(exc: Exception) -> tuple[HTTPStatus, dict[str, Any]]:
    message = str(exc).strip() or exc.__class__.__name__

    if isinstance(exc, ValueError):
        return HTTPStatus.BAD_REQUEST, {"ok": False, "error": message}

    if isinstance(exc, FileNotFoundError):
        return (
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "ok": False,
                "error": message,
                "dependency": "file",
                "sdk_path": str(ARM_SDK_SCRIPT),
                "urdf_path": str(ARM_URDF_PATH),
            },
        )

    if isinstance(exc, ImportError):
        return (
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "ok": False,
                "error": message,
                "dependency": "sdk_import",
                "sdk_path": str(ARM_SDK_SCRIPT),
            },
        )

    if isinstance(exc, RuntimeError):
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("type") == "robot_busy":
            return (
                HTTPStatus.CONFLICT,
                {
                    "ok": False,
                    "error": f"robot_id `{parsed.get('robot_id')}` is already busy",
                    "robot_id": parsed.get("robot_id"),
                    "blocking_job": parsed.get("blocking_job"),
                },
            )

        dependency = None
        if "placo" in message.lower():
            dependency = "placo"
        elif "urdf" in message.lower():
            dependency = "urdf"
        elif "sdk" in message.lower():
            dependency = "sdk"

        payload: dict[str, Any] = {"ok": False, "error": message}
        if dependency is not None:
            payload["dependency"] = dependency
            payload["sdk_path"] = str(ARM_SDK_SCRIPT)
            payload["urdf_path"] = str(ARM_URDF_PATH)
            return HTTPStatus.SERVICE_UNAVAILABLE, payload
        return HTTPStatus.INTERNAL_SERVER_ERROR, payload

    return HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": message}


def _is_pretrained_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def _resolve_default_groot_policy_path() -> str:
    prefer_stage2_rl = os.environ.get("OPENCLAW_GROOT_PREFER_STAGE2_RL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    train_root = REPO_ROOT / "outputs" / "train"
    if prefer_stage2_rl and train_root.is_dir():
        stage2_runs = sorted(
            train_root.glob("groot_offline_rl_stage2_run_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in stage2_runs:
            pretrained_dir = run_dir / "checkpoints" / "last" / "pretrained_model"
            if _is_pretrained_model_dir(pretrained_dir):
                return str(pretrained_dir)

    return "/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model"


SHORTCUT_POLICY_PATH = _resolve_default_groot_policy_path()


@dataclass
class Job:
    job_id: str
    process: subprocess.Popen
    log_path: Path
    payload: dict[str, Any]
    robot_id: str
    compat_aliases: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def status(self) -> str:
        return "running" if self.process.poll() is None else "finished"

    def return_code(self) -> int | None:
        return self.process.poll()


jobs: dict[str, Job] = {}
robot_sessions: dict[str, str] = {}
reserved_job_ids: set[str] = set()
jobs_lock = Lock()


def _normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return default


def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return default
    return int(text)


def _first_present(payload: dict[str, Any], *keys: str) -> tuple[str | None, Any]:
    for key in keys:
        if key in payload:
            value = payload[key]
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            return key, value
    return None, None


def _has_explicit_value(payload: dict[str, Any], *keys: str) -> bool:
    source, _ = _first_present(payload, *keys)
    return source is not None


def _json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_backend(value: Any) -> str:
    backend = str(value).strip().lower()
    return BACKEND_ALIASES.get(backend, backend)


def _normalize_free_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    for old, new in {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
    }.items():
        text = text.replace(old, new)
    text = "".join(text.split())
    return text


def _normalize_camera_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _normalize_free_text(value)
    if not normalized:
        return None
    return CAMERA_ALIASES.get(normalized)


def _resolve_snapshot_camera(payload: dict[str, Any]) -> tuple[str, int]:
    if _has_explicit_value(payload, "camera_index"):
        camera_index = _coerce_int(payload.get("camera_index"))
        if camera_index is None:
            raise ValueError("camera_index is empty")
        return "custom", camera_index

    if _has_explicit_value(payload, "camera", "camera_name"):
        _, raw_camera = _first_present(payload, "camera", "camera_name")
        if isinstance(raw_camera, (int, float)) or (
            isinstance(raw_camera, str) and raw_camera.strip().lstrip("-").isdigit()
        ):
            camera_index = _coerce_int(raw_camera)
            if camera_index is None:
                raise ValueError("camera is empty")
            return "custom", camera_index

        camera_name = _normalize_camera_name(raw_camera)
        if camera_name is None:
            raise ValueError(f"unsupported camera: {raw_camera}")
    else:
        camera_name = DEFAULT_SNAPSHOT_CAMERA

    if camera_name == "top":
        return camera_name, _coerce_int(payload.get("top_camera_index"), default=SHORTCUT_TOP_CAMERA_INDEX) or 0
    if camera_name == "wrist":
        return camera_name, _coerce_int(payload.get("wrist_camera_index"), default=SHORTCUT_WRIST_CAMERA_INDEX) or 0

    raise ValueError(f"unsupported camera: {camera_name}")


def _capture_snapshot_frame(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int = DEFAULT_SNAPSHOT_WARMUP_FRAMES,
    warmup_delay_s: float = DEFAULT_SNAPSHOT_WARMUP_DELAY_S,
    open_retries: int = DEFAULT_SNAPSHOT_OPEN_RETRIES,
    open_retry_delay_s: float = DEFAULT_SNAPSHOT_OPEN_RETRY_DELAY_S,
) -> Any:
    last_error = f"failed to open camera index {camera_index}"

    for attempt in range(max(open_retries, 1)):
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            last_error = f"failed to open camera index {camera_index}"
            if attempt + 1 < max(open_retries, 1):
                time.sleep(max(open_retry_delay_s, 0.0))
            continue

        try:
            if width > 0:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps > 0:
                camera.set(cv2.CAP_PROP_FPS, fps)

            frame = None
            for _ in range(max(warmup_frames, 1)):
                ok, candidate = camera.read()
                if ok and candidate is not None:
                    frame = candidate
                time.sleep(max(warmup_delay_s, 0.0))

            if frame is not None:
                return frame
            last_error = f"failed to read frame from camera index {camera_index}"
        finally:
            camera.release()

        if attempt + 1 < max(open_retries, 1):
            time.sleep(max(open_retry_delay_s, 0.0))

    raise RuntimeError(last_error)


def _build_snapshot_path(camera_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return SNAPSHOT_DIR / f"{camera_name}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"


def _mirror_snapshot_media_to_allowed_path(original_path: Path) -> Path | None:
    if not original_path.is_file():
        return None
    try:
        ALLOWED_SNAPSHOT_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        mirror_name = f"{original_path.stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}{original_path.suffix}"
        destination = ALLOWED_SNAPSHOT_MEDIA_DIR / mirror_name
        shutil.copy2(original_path, destination)
        return destination
    except Exception:
        return None


def _encode_frame_as_jpeg(frame: Any, *, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode frame as jpeg")
    return encoded.tobytes()


def _load_openai_api_key() -> str:
    env_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if env_key:
        return env_key

    if CODEX_AUTH_PATH.is_file():
        try:
            auth = json.loads(CODEX_AUTH_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to parse {CODEX_AUTH_PATH}: {exc}") from exc
        file_key = str(auth.get("OPENAI_API_KEY", "")).strip()
        if file_key:
            return file_key

    raise RuntimeError("OPENAI_API_KEY is not configured")


def _load_codex_model_config() -> dict[str, str]:
    if not CODEX_CONFIG_PATH.is_file():
        return {}

    try:
        config = tomllib.loads(CODEX_CONFIG_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"failed to parse {CODEX_CONFIG_PATH}: {exc}") from exc

    provider_name = str(config.get("model_provider") or "").strip()
    providers = config.get("model_providers") or {}
    provider = providers.get(provider_name) or {}
    base_url = str(provider.get("base_url") or "").strip()
    model = str(config.get("model") or "").strip()
    resolved: dict[str, str] = {}
    if base_url:
        resolved["base_url"] = base_url
    if model:
        resolved["model"] = model
    return resolved


def _extract_response_text(response: Any) -> str:
    direct_text = str(getattr(response, "output_text", "") or "").strip()
    if direct_text:
        return direct_text

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(str(text).strip())
    merged = "\n".join(fragment for fragment in fragments if fragment)
    if merged:
        return merged

    return str(response)


def _extract_codex_exec_text(stdout: str) -> str:
    last_text = ""
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") != "agent_message":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            last_text = text
    return last_text


def _describe_snapshot_with_openai(
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
    max_output_tokens: int,
    detail: str,
) -> str:
    from openai import OpenAI

    api_key = _load_openai_api_key()
    codex_config = _load_codex_model_config()
    data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if codex_config.get("base_url"):
        client_kwargs["base_url"] = codex_config["base_url"]
    client = OpenAI(**client_kwargs)
    response = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
    )
    text = _extract_response_text(response).strip()
    if not text:
        raise RuntimeError("vision model returned empty text")
    return text


def _describe_snapshot_with_codex_cli(
    *,
    image_path: Path,
    prompt: str,
    model: str,
    timeout_s: int,
) -> str:
    inner_cmd = [
        DEFAULT_VISION_CODEX_BIN,
        "exec",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-c",
        "suppress_unstable_features_warning=true",
    ]
    if model:
        inner_cmd.extend(["--model", model])
    inner_cmd.extend(["--image", os.fspath(image_path), "--", prompt])
    script_cmd = [
        "script",
        "-qfc",
        " ".join(shlex.quote(part) for part in inner_cmd),
        "/dev/null",
    ]

    completed = subprocess.run(
        script_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=max(timeout_s, 1),
        check=False,
    )
    answer = _extract_codex_exec_text(completed.stdout)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if not stderr:
            stderr = answer or completed.stdout.strip() or f"codex exited with code {completed.returncode}"
        raise RuntimeError(stderr)
    if not answer:
        raise RuntimeError("codex vision returned empty text")
    return answer


def _position_label(center_x: float, width: int) -> str:
    ratio = center_x / max(width, 1)
    if ratio < 0.33:
        return "左侧"
    if ratio < 0.45:
        return "中间偏左"
    if ratio < 0.55:
        return "中间"
    if ratio < 0.67:
        return "中间偏右"
    return "右侧"


def _detect_scene_regions_from_frame(frame: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    _, th_dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arm = None
    box = None
    panel = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 800:
            continue
        aspect = w / max(h, 1)

        if (
            arm is None
            and x < int(width * 0.45)
            and y < int(height * 0.7)
            and 0.8 < aspect < 3.5
            and area > 4000
        ):
            arm = {"x": x, "y": y, "w": w, "h": h, "area": area}
            continue

        if (
            box is None
            and int(width * 0.45) <= x <= int(width * 0.8)
            and int(height * 0.25) <= y <= int(height * 0.8)
            and 0.75 <= aspect <= 1.35
            and area > 2500
        ):
            box = {"x": x, "y": y, "w": w, "h": h, "area": area}
            continue

        if (
            panel is None
            and x >= int(width * 0.75)
            and h > int(height * 0.2)
            and w < int(width * 0.12)
        ):
            panel = {"x": x, "y": y, "w": w, "h": h, "area": area}

    block = None
    block_source = None
    if box is not None:
        pad = 10
        x1 = max(box["x"] + pad, 0)
        y1 = max(box["y"] + pad, 0)
        x2 = min(box["x"] + box["w"] - pad, width)
        y2 = min(box["y"] + box["h"] - pad, height)
        inner = gray[y1:y2, x1:x2]
        if inner.size > 0:
            mask = (inner > 90).astype("uint8") * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: list[dict[str, Any]] = []
            for contour in contours:
                bx, by, bw, bh = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if area < 40:
                    continue
                if bw < 6 or bh < 6:
                    continue
                aspect = bw / max(bh, 1)
                box_area = max((x2 - x1) * (y2 - y1), 1)
                # Reject thin labels/reflections inside the box. The actual block in this setup
                # occupies a larger, near-square region.
                if aspect < 0.6 or aspect > 1.8:
                    continue
                if area < box_area * 0.06:
                    continue
                candidates.append(
                    {
                        "x": x1 + bx,
                        "y": y1 + by,
                        "w": bw,
                        "h": bh,
                        "area": area,
                    }
                )
            if candidates:
                block = max(candidates, key=lambda item: item["area"])
                block_source = "box"

    if block is None:
        mask = (gray < 160).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 500 or area > 5000:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.7 or aspect > 1.4:
                continue
            # Search across the tabletop workspace, not only the lower half.
            if y < int(height * 0.2) or y > int(height * 0.9):
                continue
            if x < int(width * 0.2) or x > int(width * 0.8):
                continue
            if box is not None:
                if not (x + w < box["x"] or x > box["x"] + box["w"] or y + h < box["y"] or y > box["y"] + box["h"]):
                    continue
            if panel is not None:
                if not (x + w < panel["x"] or x > panel["x"] + panel["w"] or y + h < panel["y"] or y > panel["y"] + panel["h"]):
                    continue
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                }
            )
        if candidates:
            block = max(candidates, key=lambda item: item["area"])
            block_source = "table"

    return {
        "image_width": width,
        "image_height": height,
        "arm": arm,
        "box": box,
        "panel": panel,
        "block": block,
        "block_source": block_source,
    }


def _heuristic_scene_answer(prompt: str, detections: dict[str, Any]) -> str:
    width = int(detections["image_width"])
    arm = detections.get("arm")
    box = detections.get("box")
    panel = detections.get("panel")
    block = detections.get("block")
    block_source = detections.get("block_source")
    normalized_prompt = _normalize_free_text(prompt)

    block_phrase = ""
    if block is not None:
        center_x = block["x"] + block["w"] / 2
        block_phrase = f"浅色方块大致在画面{_position_label(center_x, width)}"
        if block_source == "box" and box is not None:
            block_phrase += "，并且看起来在黑色盒子里面。"
        elif block_source == "table":
            block_phrase += "，看起来还在桌面上，不在盒子里。"
        else:
            block_phrase += "。"
    elif box is not None:
        center_x = box["x"] + box["w"] / 2
        block_phrase = f"当前没有稳定看到方块，只看到画面{_position_label(center_x, width)}有一个黑色盒子。"
    else:
        block_phrase = "当前没有稳定识别到方块位置。"

    if any(token in normalized_prompt for token in ("方块在哪里", "物块在哪里", "block在哪里", "方块位置", "物块位置")):
        return block_phrase

    if any(token in normalized_prompt for token in ("盒子在哪里", "box在哪里", "bin在哪里")) and box is not None:
        center_x = box["x"] + box["w"] / 2
        return f"黑色盒子大致在画面{_position_label(center_x, width)}。"

    parts: list[str] = []
    if arm is not None:
        center_x = arm["x"] + arm["w"] / 2
        parts.append(f"左边有一只机械臂，主体在画面{_position_label(center_x, width)}")
    if box is not None:
        center_x = box["x"] + box["w"] / 2
        parts.append(f"中右区域有一个黑色方形盒子，位置大致在画面{_position_label(center_x, width)}")
    if block is not None:
        center_x = block["x"] + block["w"] / 2
        if block_source == "box":
            parts.append(f"盒子里还能看到一个浅色方块，位置大致在画面{_position_label(center_x, width)}")
        elif block_source == "table":
            parts.append(f"桌面上还能看到一个浅色方块，位置大致在画面{_position_label(center_x, width)}")
        else:
            parts.append(f"还能看到一个浅色方块，位置大致在画面{_position_label(center_x, width)}")
    if panel is not None:
        center_x = panel["x"] + panel["w"] / 2
        parts.append(f"最右侧还有一个竖着的长条面板，位置在画面{_position_label(center_x, width)}")

    if parts:
        return "。".join(parts) + "。"
    return "当前没能从这张图里稳定识别出方块、盒子和机械臂的相对位置。"


def _describe_snapshot_with_heuristic(*, image_bytes: bytes, prompt: str) -> str:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode image bytes for heuristic vision")
    detections = _detect_scene_regions_from_frame(image)
    if detections.get("arm") is None and detections.get("box") is None and detections.get("block") is None:
        raise RuntimeError("heuristic vision could not find stable scene landmarks")
    return _heuristic_scene_answer(prompt, detections)


def _format_detection_hint_from_image_bytes(image_bytes: bytes) -> str:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return "- 辅助检测不可用：图片解码失败"

    detections = _detect_scene_regions_from_frame(image)
    width = int(detections.get("image_width") or image.shape[1] or 1)

    lines: list[str] = []

    arm = detections.get("arm")
    if arm is None:
        lines.append("- 机械臂：未稳定检测到")
    else:
        arm_center_x = arm["x"] + arm["w"] / 2
        lines.append(f"- 机械臂：可见，位置大致在画面{_position_label(arm_center_x, width)}")

    box = detections.get("box")
    if box is None:
        lines.append("- 盒子：未稳定检测到")
    else:
        box_center_x = box["x"] + box["w"] / 2
        lines.append(f"- 盒子：可见，位置大致在画面{_position_label(box_center_x, width)}")

    block = detections.get("block")
    block_source = detections.get("block_source")
    if block is None:
        lines.append("- 方块：未稳定检测到")
    else:
        block_center_x = block["x"] + block["w"] / 2
        location = _position_label(block_center_x, width)
        if block_source == "box":
            lines.append(f"- 方块：可见，位置大致在画面{location}，看起来在盒子里")
        elif block_source == "table":
            lines.append(f"- 方块：可见，位置大致在画面{location}，看起来在桌面上")
        else:
            lines.append(f"- 方块：可见，位置大致在画面{location}")

    panel = detections.get("panel")
    if panel is None:
        lines.append("- 右侧长条面板：未稳定检测到")
    else:
        panel_center_x = panel["x"] + panel["w"] / 2
        lines.append(f"- 右侧长条面板：可见，位置大致在画面{_position_label(panel_center_x, width)}")

    return "\n".join(lines)


def _build_multimodal_prompt(*, user_prompt: str, image_bytes: bytes) -> str:
    normalized_prompt = str(user_prompt or "").strip() or DEFAULT_VISION_PROMPT
    detection_hint = _format_detection_hint_from_image_bytes(image_bytes)
    return (
        "你正在分析一张当前时刻的机器人相机图片。\n"
        "只根据这一张当前图片回答，绝对不要沿用之前任何一次回答，也不要假设场景延续上一帧。\n"
        "如果当前图片里没有清楚看到目标，请直接说当前没看到，不要猜，不要补全。\n"
        "回答要求：中文，1到2句，直接回答用户问题。\n"
        f"用户问题：{normalized_prompt}\n"
        "当前帧辅助检测（仅供参考；如果与图片明显矛盾，以图片为准）：\n"
        f"{detection_hint}"
    )


def _describe_snapshot(
    *,
    image_bytes: bytes,
    image_path: Path,
    prompt: str,
    model: str,
    max_output_tokens: int,
    detail: str,
    backend: str,
    timeout_s: int,
) -> tuple[str, str]:
    multimodal_prompt = _build_multimodal_prompt(user_prompt=prompt, image_bytes=image_bytes)

    if backend == "auto":
        try:
            return (
                _describe_snapshot_with_codex_cli(
                    image_path=image_path,
                    prompt=multimodal_prompt,
                    model=model,
                    timeout_s=timeout_s,
                ),
                "codex",
            )
        except Exception:
            try:
                return (
                    _describe_snapshot_with_openai(
                        image_bytes=image_bytes,
                        prompt=multimodal_prompt,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        detail=detail,
                    ),
                    "openai",
                )
            except Exception:
                return _describe_snapshot_with_heuristic(image_bytes=image_bytes, prompt=prompt), "heuristic"
    if backend == "codex":
        return (
            _describe_snapshot_with_codex_cli(
                image_path=image_path,
                prompt=multimodal_prompt,
                model=model,
                timeout_s=timeout_s,
            ),
            "codex",
        )
    if backend == "openai":
        return (
            _describe_snapshot_with_openai(
                image_bytes=image_bytes,
                prompt=multimodal_prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                detail=detail,
            ),
            "openai",
        )
    if backend == "heuristic":
        return _describe_snapshot_with_heuristic(image_bytes=image_bytes, prompt=prompt), "heuristic"
    raise ValueError(f"unsupported vision backend: {backend}")


def _is_pick_place_shortcut(task_text: str) -> bool:
    text = _normalize_free_text(task_text)
    if not text:
        return False

    object_tokens = ("物块", "方块", "积木", "block")
    container_tokens = ("盒子", "盒里", "盒中", "箱子", "box", "bin")
    place_tokens = ("放到", "放进", "放入", "放", "place", "put")
    pick_tokens = ("抓", "拿", "pick", "pickup", "pickandplace")
    grasp_only_tokens = ("抓起来", "抓起", "拿起来", "拿起", "夹起来", "grasp", "pickblock", "pickupblock")

    has_object = any(token in text for token in object_tokens)
    has_container = any(token in text for token in container_tokens)
    has_place = any(token in text for token in place_tokens)
    has_pick = any(token in text for token in pick_tokens)
    has_grasp_only = any(token in text for token in grasp_only_tokens)

    if "pickblockandplacetobox" in text:
        return True
    if "puttheblockinthebin" in text:
        return True
    return has_object and ((has_container and (has_place or has_pick)) or has_grasp_only)


def _apply_pick_place_shortcut_defaults(payload: dict[str, Any], raw_payload: dict[str, Any]) -> None:
    if not _has_explicit_value(raw_payload, "backend", "model_backend", "policy_backend"):
        payload["backend"] = _normalize_backend(DEFAULT_BACKEND)
    payload["task"] = SHORTCUT_CANONICAL_TASK

    shortcut_defaults = [
        (("policy_path",), "policy_path", SHORTCUT_POLICY_PATH),
        (("policy_device",), "policy_device", SHORTCUT_POLICY_DEVICE),
        (("robot_port",), "robot_port", SHORTCUT_ROBOT_PORT),
        (("robot_id",), "robot_id", SHORTCUT_ROBOT_ID),
        (("robot_calib_dir",), "robot_calib_dir", SHORTCUT_ROBOT_CALIB_DIR),
        (("top_camera_index",), "top_camera_index", SHORTCUT_TOP_CAMERA_INDEX),
        (("wrist_camera_index",), "wrist_camera_index", SHORTCUT_WRIST_CAMERA_INDEX),
        (("camera_width",), "camera_width", SHORTCUT_CAMERA_WIDTH),
        (("camera_height",), "camera_height", SHORTCUT_CAMERA_HEIGHT),
        (("camera_fps",), "camera_fps", SHORTCUT_CAMERA_FPS),
        (("dataset_root",), "dataset_root", SHORTCUT_DATASET_ROOT),
        (("events_jsonl_path", "events_path"), "events_jsonl_path", SHORTCUT_EVENTS_JSONL_PATH),
        (("display_data",), "display_data", SHORTCUT_DISPLAY_DATA),
        (("safety_profile", "safety", "guard_profile"), "safety_profile", DEFAULT_SAFETY_PROFILE),
    ]
    for raw_keys, payload_key, value in shortcut_defaults:
        if not _has_explicit_value(raw_payload, *raw_keys):
            payload[payload_key] = value


def _normalize_dataset_repo_id(value: Any, *, backend: str, job_id: str) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        raw_value = f"local/eval_openclaw_{backend}_{job_id}"

    owner, dataset_name = ("local", raw_value)
    if "/" in raw_value:
        owner, dataset_name = raw_value.split("/", 1)

    dataset_name = dataset_name.strip()
    if not dataset_name.startswith("eval_"):
        dataset_name = f"eval_{dataset_name}"

    return f"{owner}/{dataset_name}"


def _normalize_dataset_root(
    value: Any,
    *,
    job_dir: Path,
    dataset_repo_id: str,
    clear_dataset_root: bool,
) -> str:
    if value is None or str(value).strip() == "":
        return os.fspath(job_dir / "dataset")

    root = Path(os.fspath(value)).expanduser()
    if clear_dataset_root:
        return os.fspath(root)

    dataset_dir_name = dataset_repo_id.split("/", 1)[-1]
    if root.name == dataset_dir_name:
        return os.fspath(root)

    if root.exists():
        return os.fspath(root / dataset_dir_name)

    return os.fspath(root)


def _normalize_payload(raw_payload: dict[str, Any], job_id: str, job_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    compat_aliases: dict[str, str] = {}
    payload = {key: value for key, value in raw_payload.items() if key not in LEGACY_PAYLOAD_KEYS}

    backend_source, backend_value = _first_present(raw_payload, "backend", "model_backend", "policy_backend")
    backend = _normalize_backend(backend_value or DEFAULT_BACKEND)
    if backend_source not in {None, "backend"}:
        compat_aliases["backend"] = backend_source
    payload["backend"] = backend

    task_source, task_value = _first_present(raw_payload, "task", "task_text", "instruction")
    task = str(task_value or DEFAULT_TASK)
    if task_source not in {None, "task"}:
        compat_aliases["task"] = task_source
    payload["task"] = task

    robot_id_source, robot_id_value = _first_present(raw_payload, "robot_id")
    robot_id = str(robot_id_value or DEFAULT_ROBOT_ID)
    if robot_id_source not in {None, "robot_id"}:
        compat_aliases["robot_id"] = robot_id_source
    payload["robot_id"] = robot_id

    intent_source, intent_value = _first_present(raw_payload, "intent_json", "task_intent_json", "intent")
    if intent_value is not None:
        payload["intent_json"] = _json_dumps(intent_value)
        if intent_source not in {None, "intent_json"}:
            compat_aliases["intent_json"] = intent_source

    safety_source, safety_value = _first_present(raw_payload, "safety_profile", "safety", "guard_profile")
    payload["safety_profile"] = str(safety_value or DEFAULT_SAFETY_PROFILE)
    if safety_source not in {None, "safety_profile"}:
        compat_aliases["safety_profile"] = safety_source

    events_source, events_value = _first_present(raw_payload, "events_jsonl_path", "events_path")
    if events_value is None:
        events_value = job_dir / "events.jsonl"
    payload["events_jsonl_path"] = os.fspath(events_value)
    if events_source not in {None, "events_jsonl_path"}:
        compat_aliases["events_jsonl_path"] = events_source

    clear_source, clear_value = _first_present(raw_payload, "clear_dataset_root", "clear_dataset")
    payload["clear_dataset_root"] = _coerce_bool(clear_value, default=False)
    if clear_source not in {None, "clear_dataset_root"}:
        compat_aliases["clear_dataset_root"] = clear_source

    if payload["backend"] == "groot" and _is_pick_place_shortcut(payload["task"]):
        _apply_pick_place_shortcut_defaults(payload, raw_payload)
        compat_aliases["task_shortcut"] = "so101_pick_place_cn"

    payload["job_id"] = job_id
    return payload, compat_aliases


def _persist_request(job_dir: Path, raw_payload: dict[str, Any], payload: dict[str, Any], compat_aliases: dict[str, str]) -> None:
    request_path = job_dir / "request.json"
    request_payload = {
        "raw_payload": raw_payload,
        "normalized_payload": payload,
        "compat_aliases_used": compat_aliases,
        "runner_script": str(RUNNER_SCRIPT),
        "written_at": time.time(),
    }
    request_path.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_env(payload: dict[str, Any], job_id: str, job_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env_map = {
        "backend": "BACKEND",
        "task": "TASK_TEXT",
        "leader_port": "LEADER_PORT",
        "policy_path": "POLICY_PATH",
        "policy_device": "POLICY_DEVICE",
        "robot_port": "ROBOT_PORT",
        "robot_id": "ROBOT_ID",
        "robot_calib_dir": "ROBOT_CALIB_DIR",
        "top_camera_index": "TOP_CAMERA_INDEX",
        "wrist_camera_index": "WRIST_CAMERA_INDEX",
        "camera_width": "CAMERA_WIDTH",
        "camera_height": "CAMERA_HEIGHT",
        "camera_fps": "CAMERA_FPS",
        "leader_id": "LEADER_ID",
        "leader_calib_dir": "LEADER_CALIB_DIR",
        "num_episodes": "NUM_EPISODES",
        "episode_time_s": "EPISODE_TIME_S",
        "reset_time_s": "RESET_TIME_S",
        "display_data": "DISPLAY_DATA",
        "python_bin": "PYTHON_BIN",
        "intent_json": "INTENT_JSON",
        "safety_profile": "SAFETY_PROFILE",
        "events_jsonl_path": "EVENTS_JSONL_PATH",
    }

    dataset_repo_id = _normalize_dataset_repo_id(payload.get("dataset_repo_id"), backend=payload["backend"], job_id=job_id)
    dataset_root = _normalize_dataset_root(
        payload.get("dataset_root"),
        job_dir=job_dir,
        dataset_repo_id=dataset_repo_id,
        clear_dataset_root=bool(payload.get("clear_dataset_root")),
    )
    env["DATASET_REPO_ID"] = _normalize_value(dataset_repo_id)
    env["DATASET_ROOT"] = _normalize_value(dataset_root)
    env["BACKEND"] = payload["backend"]
    env["TASK_TEXT"] = payload["task"]
    env["ROBOT_ID"] = payload["robot_id"]
    env["INTENT_JSON"] = payload.get("intent_json", "")
    env["TASK_INTENT_JSON"] = env["INTENT_JSON"]
    env["SAFETY_PROFILE"] = payload.get("safety_profile", "")
    env["EVENTS_JSONL_PATH"] = payload["events_jsonl_path"]
    env["EVENTS_PATH"] = env["EVENTS_JSONL_PATH"]
    env["CLEAR_DATASET_ROOT"] = "1" if payload.get("clear_dataset_root") else "0"
    env["OPENCLAW_JOB_ID"] = job_id
    env["OPENCLAW_JOB_DIR"] = str(job_dir)
    env["OPENCLAW_REQUEST_JSON"] = str(job_dir / "request.json")

    for request_key, env_key in env_map.items():
        if request_key in payload and payload[request_key] is not None:
            env[env_key] = _normalize_value(payload[request_key])

    return env


def _release_robot_session_locked(robot_id: str, job_id: str) -> None:
    if robot_sessions.get(robot_id) == job_id:
        robot_sessions.pop(robot_id, None)


def _release_job_reservation_locked(robot_id: str, job_id: str) -> None:
    reserved_job_ids.discard(job_id)
    _release_robot_session_locked(robot_id, job_id)


def _reap_finished_jobs_locked() -> None:
    for job in jobs.values():
        if job.process.poll() is not None:
            _release_robot_session_locked(job.robot_id, job.job_id)


def _acquire_robot_session_locked(robot_id: str, job_id: str) -> str | None:
    _reap_finished_jobs_locked()
    active_job_id = robot_sessions.get(robot_id)
    if active_job_id is not None:
        return active_job_id
    robot_sessions[robot_id] = job_id
    return None


def _reserve_job_start_locked(job_id: str, robot_id: str) -> tuple[str, str] | None:
    _reap_finished_jobs_locked()
    if job_id in reserved_job_ids or job_id in jobs:
        return ("job_id", job_id)

    active_job_id = robot_sessions.get(robot_id)
    if active_job_id is not None:
        return ("robot_id", active_job_id)

    reserved_job_ids.add(job_id)
    robot_sessions[robot_id] = job_id
    return None


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length == 0:
        return {}
    raw_body = handler.rfile.read(content_length)
    return json.loads(raw_body.decode("utf-8"))


def _job_to_dict(job: Job) -> dict[str, Any]:
    log_tail = ""
    if job.log_path.exists():
        with job.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            log_tail = handle.read()[-4000:]

    return {
        "job_id": job.job_id,
        "pid": job.process.pid,
        "status": job.status(),
        "return_code": job.return_code(),
        "started_at": job.started_at,
        "backend": job.payload.get("backend", DEFAULT_BACKEND),
        "task": job.payload.get("task", ""),
        "robot_id": job.robot_id,
        "log_path": str(job.log_path),
        "log_tail": log_tail,
        "payload": job.payload,
        "compat_aliases_used": job.compat_aliases,
    }


class OpenClawGrootHandler(BaseHTTPRequestHandler):
    server_version = "OpenClawGrootHTTP/0.2"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "runner_script": str(RUNNER_SCRIPT),
                    "log_dir": str(LOG_DIR),
                    "supported_backends": sorted(SUPPORTED_BACKENDS),
                    "arm": {
                        "sdk_path": str(ARM_SDK_SCRIPT),
                        "urdf_path": str(ARM_URDF_PATH),
                        "placo_available": _placo_available(),
                    },
                },
            )
            return

        if self.path == "/arm/state":
            self._handle_arm_state()
            return

        if self.path == "/jobs":
            with jobs_lock:
                _reap_finished_jobs_locked()
                payload = {"jobs": [_job_to_dict(job) for job in jobs.values()]}
            self._send_json(HTTPStatus.OK, payload)
            return

        if self.path.startswith("/jobs/"):
            job_id = self.path.removeprefix("/jobs/").strip("/")
            with jobs_lock:
                _reap_finished_jobs_locked()
                job = jobs.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "job not found"})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "job": _job_to_dict(job)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run":
            self._handle_run()
            return

        if self.path == "/arm/move":
            self._handle_arm_move()
            return

        if self.path == "/arm/jog":
            self._handle_arm_jog()
            return

        if self.path == "/snapshot":
            self._handle_snapshot()
            return

        if self.path == "/describe":
            self._handle_describe()
            return

        if self.path.startswith("/jobs/") and self.path.endswith("/stop"):
            job_id = self.path.removeprefix("/jobs/").removesuffix("/stop").strip("/")
            self._handle_stop(job_id)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})

    def _handle_arm_state(self) -> None:
        try:
            result = _read_arm_state(SHORTCUT_ROBOT_ID)
        except Exception as exc:  # noqa: BLE001
            status, payload = _arm_exception_response(exc)
            self._send_json(status, payload)
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "arm": {
                    "robot_id": SHORTCUT_ROBOT_ID,
                    "control_mode": "joint",
                    "state": result["state"],
                },
                "message": "arm state ready",
            },
        )

    def _handle_arm_move(self) -> None:
        try:
            raw_payload = _read_json(self)
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid json: {exc}"})
            return

        try:
            result = _run_arm_move(raw_payload)
        except Exception as exc:  # noqa: BLE001
            status, payload = _arm_exception_response(exc)
            self._send_json(status, payload)
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "arm": {
                    "robot_id": str(raw_payload.get("robot_id") or SHORTCUT_ROBOT_ID).strip() or SHORTCUT_ROBOT_ID,
                    **result,
                },
                "message": "arm move executed",
            },
        )

    def _handle_arm_jog(self) -> None:
        try:
            raw_payload = _read_json(self)
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid json: {exc}"})
            return

        try:
            result = _run_arm_jog(raw_payload)
        except Exception as exc:  # noqa: BLE001
            status, payload = _arm_exception_response(exc)
            self._send_json(status, payload)
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "arm": {
                    "robot_id": str(raw_payload.get("robot_id") or SHORTCUT_ROBOT_ID).strip() or SHORTCUT_ROBOT_ID,
                    **result,
                },
                "message": "arm jog executed",
            },
        )

    def _handle_run(self) -> None:
        if not RUNNER_SCRIPT.is_file():
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "runner script missing"})
            return

        try:
            raw_payload = _read_json(self)
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid json: {exc}"})
            return

        job_id = raw_payload.get("job_id") or uuid.uuid4().hex[:10]
        job_dir = LOG_DIR / job_id

        payload, compat_aliases = _normalize_payload(raw_payload, job_id=job_id, job_dir=job_dir)
        if payload["backend"] not in SUPPORTED_BACKENDS:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": f"unsupported backend: {payload['backend']}",
                    "supported_backends": sorted(SUPPORTED_BACKENDS),
                },
            )
            return

        with jobs_lock:
            conflict = _reserve_job_start_locked(job_id, payload["robot_id"])
            if conflict is not None:
                conflict_type, blocking_job_id = conflict
                blocking_job = jobs.get(blocking_job_id)
                if conflict_type == "job_id":
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {
                            "ok": False,
                            "error": f"job_id `{job_id}` is already in use",
                            "job_id": job_id,
                            "blocking_job": _job_to_dict(blocking_job)
                            if blocking_job is not None
                            else {"job_id": blocking_job_id},
                        },
                    )
                    return
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "ok": False,
                        "error": f"robot_id `{payload['robot_id']}` is already controlled by job `{blocking_job_id}`",
                        "robot_id": payload["robot_id"],
                        "blocking_job": _job_to_dict(blocking_job) if blocking_job is not None else {"job_id": blocking_job_id},
                    },
                )
                return

        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            log_path = job_dir / "run.log"
            _persist_request(job_dir, raw_payload, payload, compat_aliases)
            env = _build_env(payload, job_id, job_dir)
            log_handle = log_path.open("a", encoding="utf-8")
        except Exception as exc:
            with jobs_lock:
                _release_job_reservation_locked(payload["robot_id"], job_id)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"failed to prepare job: {exc}"},
            )
            return

        try:
            process = subprocess.Popen(
                ["bash", str(RUNNER_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            with jobs_lock:
                _release_job_reservation_locked(payload["robot_id"], job_id)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"failed to start job: {exc}"},
            )
            return
        finally:
            log_handle.close()

        job = Job(
            job_id=job_id,
            process=process,
            log_path=log_path,
            payload=payload,
            robot_id=payload["robot_id"],
            compat_aliases=compat_aliases,
        )
        with jobs_lock:
            reserved_job_ids.discard(job_id)
            jobs[job_id] = job

        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "ok": True,
                "job": _job_to_dict(job),
                "message": "job started",
            },
        )

    def _handle_stop(self, job_id: str) -> None:
        with jobs_lock:
            _reap_finished_jobs_locked()
            job = jobs.get(job_id)

        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "job not found"})
            return

        if job.process.poll() is not None:
            with jobs_lock:
                _release_robot_session_locked(job.robot_id, job.job_id)
            self._send_json(HTTPStatus.OK, {"ok": True, "job": _job_to_dict(job), "message": "already stopped"})
            return

        os.killpg(job.process.pid, signal.SIGINT)
        self._send_json(HTTPStatus.OK, {"ok": True, "job": _job_to_dict(job), "message": "stop signal sent"})

    def _handle_snapshot(self) -> None:
        try:
            raw_payload = _read_json(self)
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid json: {exc}"})
            return

        try:
            camera_name, camera_index = _resolve_snapshot_camera(raw_payload)
            width = _coerce_int(raw_payload.get("camera_width"), default=SHORTCUT_CAMERA_WIDTH) or SHORTCUT_CAMERA_WIDTH
            height = _coerce_int(raw_payload.get("camera_height"), default=SHORTCUT_CAMERA_HEIGHT) or SHORTCUT_CAMERA_HEIGHT
            fps = _coerce_int(raw_payload.get("camera_fps"), default=SHORTCUT_CAMERA_FPS) or SHORTCUT_CAMERA_FPS
            warmup_frames = _coerce_int(raw_payload.get("warmup_frames"), default=DEFAULT_SNAPSHOT_WARMUP_FRAMES)
            if warmup_frames is None:
                warmup_frames = DEFAULT_SNAPSHOT_WARMUP_FRAMES
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        output_dir = Path(os.fspath(raw_payload.get("output_dir") or SNAPSHOT_DIR)).expanduser()
        output_path = Path(os.fspath(raw_payload.get("output_path") or _build_snapshot_path(camera_name))).expanduser()
        if not output_path.is_absolute():
            output_path = output_dir / output_path

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame = _capture_snapshot_frame(
                camera_index=camera_index,
                width=width,
                height=height,
                fps=fps,
                warmup_frames=warmup_frames,
            )
            if not cv2.imwrite(os.fspath(output_path), frame):
                raise RuntimeError(f"failed to write snapshot to {output_path}")
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"failed to capture snapshot: {exc}"},
            )
            return

        actual_height, actual_width = frame.shape[:2]
        mirrored = _mirror_snapshot_media_to_allowed_path(output_path)
        media_url = mirrored.as_uri() if mirrored is not None else output_path.as_uri()
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "snapshot": {
                    "camera": camera_name,
                    "camera_index": camera_index,
                    "path": str(output_path),
                    "media_url": media_url,
                    "width": actual_width,
                    "height": actual_height,
                    "captured_at": time.time(),
                },
                "message": "snapshot captured",
            },
        )

    def _handle_describe(self) -> None:
        try:
            raw_payload = _read_json(self)
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid json: {exc}"})
            return

        try:
            camera_name, camera_index = _resolve_snapshot_camera(raw_payload)
            width = _coerce_int(raw_payload.get("camera_width"), default=SHORTCUT_CAMERA_WIDTH) or SHORTCUT_CAMERA_WIDTH
            height = _coerce_int(raw_payload.get("camera_height"), default=SHORTCUT_CAMERA_HEIGHT) or SHORTCUT_CAMERA_HEIGHT
            fps = _coerce_int(raw_payload.get("camera_fps"), default=SHORTCUT_CAMERA_FPS) or SHORTCUT_CAMERA_FPS
            warmup_frames = _coerce_int(raw_payload.get("warmup_frames"), default=DEFAULT_SNAPSHOT_WARMUP_FRAMES)
            if warmup_frames is None:
                warmup_frames = DEFAULT_SNAPSHOT_WARMUP_FRAMES
            max_output_tokens = _coerce_int(
                raw_payload.get("max_output_tokens"),
                default=DEFAULT_VISION_MAX_OUTPUT_TOKENS,
            ) or DEFAULT_VISION_MAX_OUTPUT_TOKENS
            prompt = str(
                raw_payload.get("prompt")
                or raw_payload.get("question")
                or raw_payload.get("task")
                or DEFAULT_VISION_PROMPT
            ).strip()
            if not prompt:
                prompt = DEFAULT_VISION_PROMPT
            vision_backend = str(raw_payload.get("vision_backend") or DEFAULT_VISION_BACKEND).strip().lower()
            configured_model = _load_codex_model_config().get("model", "")
            model = str(raw_payload.get("model") or DEFAULT_VISION_MODEL or configured_model or "gpt-5.4").strip()
            detail = str(raw_payload.get("detail") or DEFAULT_VISION_DETAIL).strip().lower()
            if detail not in {"low", "high", "auto"}:
                raise ValueError(f"unsupported detail: {detail}")
            timeout_s = _coerce_int(
                raw_payload.get("timeout_s"),
                default=DEFAULT_VISION_TIMEOUT_S,
            ) or DEFAULT_VISION_TIMEOUT_S
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        output_dir = Path(os.fspath(raw_payload.get("output_dir") or SNAPSHOT_DIR)).expanduser()
        output_path = Path(os.fspath(raw_payload.get("output_path") or _build_snapshot_path(camera_name))).expanduser()
        if not output_path.is_absolute():
            output_path = output_dir / output_path

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame = _capture_snapshot_frame(
                camera_index=camera_index,
                width=width,
                height=height,
                fps=fps,
                warmup_frames=warmup_frames,
            )
            image_bytes = _encode_frame_as_jpeg(frame)
            output_path.write_bytes(image_bytes)
            answer, resolved_backend = _describe_snapshot(
                image_bytes=image_bytes,
                image_path=output_path,
                prompt=prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                detail=detail,
                backend=vision_backend,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"failed to describe snapshot: {exc}"},
            )
            return

        actual_height, actual_width = frame.shape[:2]
        mirrored = _mirror_snapshot_media_to_allowed_path(output_path)
        media_url = mirrored.as_uri() if mirrored is not None else output_path.as_uri()
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "snapshot": {
                    "camera": camera_name,
                    "camera_index": camera_index,
                    "path": str(output_path),
                    "media_url": media_url,
                    "width": actual_width,
                    "height": actual_height,
                    "captured_at": time.time(),
                },
                "vision": {
                    "prompt": prompt,
                    "model": model,
                    "detail": detail,
                    "backend": resolved_backend,
                    "requested_backend": vision_backend,
                    "answer": answer,
                },
                "message": "snapshot described",
            },
        )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((HOST, PORT), OpenClawGrootHandler)
    print(f"OpenClaw GROOT server listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
