#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("so101_geom_grasp")
DEFAULT_CALIB_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
DEFAULT_URDF_PATH = REPO_ROOT / "so101_new_calib.urdf"
DEFAULT_CAMERA_TO_BASE_XYZ = [0.0, 0.0, 0.0]
DEFAULT_CAMERA_TO_BASE_RPY_DEG = [0.0, 0.0, 0.0]
DEFAULT_GRASP_RPY_DEG = [180.0, 0.0, 0.0]


class ArgumentParserError(Exception):
    """Raised instead of argparse exiting so the script can still emit JSON."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentParserError(message)


class FloatVectorAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str] | str,
        option_string: str | None = None,
    ) -> None:
        raw_values = values if isinstance(values, list) else [values]
        flattened: list[str] = []
        for raw_value in raw_values:
            normalized_value = str(raw_value).strip()
            if normalized_value[:1] in {"[", "("} and normalized_value[-1:] in {"]", ")"}:
                normalized_value = normalized_value[1:-1].strip()
            pieces = [piece for piece in re.split(r"[\s,;]+", normalized_value) if piece]
            flattened.extend(pieces)

        if not flattened:
            raise argparse.ArgumentError(self, f"{option_string} requires at least one numeric value")

        try:
            parsed = [float(piece) for piece in flattened]
        except ValueError as exc:
            raise argparse.ArgumentError(self, f"{option_string} expects numeric values, got {raw_values!r}") from exc

        expected_len = getattr(self, "expected_len", None)
        if expected_len is not None and len(parsed) != expected_len:
            raise argparse.ArgumentError(
                self,
                f"{option_string} expects {expected_len} values, got {len(parsed)} from {raw_values!r}",
            )

        setattr(namespace, self.dest, parsed)


class Vector3Action(FloatVectorAction):
    expected_len = 3


def safe_error_text(exc: BaseException | str | None) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, str):
        return exc
    return f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def optional_import(module_name: str) -> tuple[Any | None, BaseException | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return None, exc


@lru_cache(maxsize=1)
def load_so101_runtime_symbols() -> tuple[dict[str, Any] | None, BaseException | None]:
    try:
        kinematics_module = importlib.import_module("lerobot.model.kinematics")
        config_module = importlib.import_module("lerobot.robots.so101_follower.config_so101_follower")
        robot_module = importlib.import_module("lerobot.robots.so101_follower.so101_follower")
        return {
            "RobotKinematics": kinematics_module.RobotKinematics,
            "SO101FollowerConfig": config_module.SO101FollowerConfig,
            "SO101Follower": robot_module.SO101Follower,
        }, None
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return None, exc


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_alias_argument(parser: argparse.ArgumentParser, *option_strings: str, **kwargs: Any) -> argparse.Action:
    return parser.add_argument(*option_strings, **kwargs)


def add_bool_argument(parser: argparse.ArgumentParser, *option_strings: str, default: bool, help: str) -> argparse.Action:
    return parser.add_argument(
        *option_strings,
        type=parse_bool,
        nargs="?",
        const=True,
        default=default,
        help=help,
    )


def build_arg_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(
        description="Standalone SO101 geometric grasp runtime using RealSense RGB-D and RobotKinematics.",
    )

    add_alias_argument(parser, "--task", default="Grasp the block from the table")
    add_alias_argument(parser, "--robot_port", "--robot-port", default="/dev/ttyACM0")
    add_alias_argument(parser, "--robot_id", "--robot-id", default="so101_follower")
    add_alias_argument(
        parser,
        "--robot_calib_dir",
        "--robot-calib-dir",
        "--robot-calibration-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
    )
    add_bool_argument(
        parser,
        "--robot_use_degrees",
        "--robot-use-degrees",
        default=True,
        help="Use degree-normalized joint commands for the arm motors.",
    )

    add_alias_argument(parser, "--realsense_serial", "--camera_serial", "--camera-serial", default=None)
    add_alias_argument(parser, "--camera_width", "--camera-width", type=int, default=640)
    add_alias_argument(parser, "--camera_height", "--camera-height", type=int, default=480)
    add_alias_argument(parser, "--camera_fps", "--camera-fps", type=int, default=30)
    add_alias_argument(parser, "--camera_warmup_frames", "--camera-warmup-frames", type=int, default=20)
    add_alias_argument(parser, "--camera_timeout_ms", "--camera-timeout-ms", type=int, default=5000)
    add_alias_argument(parser, "--perception_consistency_frames", "--perception-consistency-frames", type=int, default=3)
    add_alias_argument(
        parser,
        "--perception_min_consistent_detections",
        "--perception-min-consistent-detections",
        type=int,
        default=2,
    )
    add_alias_argument(
        parser,
        "--perception_position_std_max_m",
        "--perception-position-std-max",
        type=float,
        default=0.015,
    )
    add_alias_argument(
        parser,
        "--perception_mask_area_rel_std_max",
        "--perception-mask-area-rel-std-max",
        type=float,
        default=0.20,
    )

    add_alias_argument(
        parser,
        "--camera_to_base_xyz",
        "--camera-to-base-xyz",
        action=Vector3Action,
        nargs="+",
        default=None,
    )
    add_alias_argument(
        parser,
        "--camera_to_base_rpy_deg",
        "--camera-to-base-rpy-deg",
        action=Vector3Action,
        nargs="+",
        default=None,
        help="Fixed-axis roll/pitch/yaw in degrees for camera-to-base rotation.",
    )
    add_alias_argument(
        parser,
        "--workspace_min_xyz",
        "--workspace-min",
        "--workspace-min-xyz",
        action=Vector3Action,
        nargs="+",
        default=None,
    )
    add_alias_argument(
        parser,
        "--workspace_max_xyz",
        "--workspace-max",
        "--workspace-max-xyz",
        action=Vector3Action,
        nargs="+",
        default=None,
    )
    add_alias_argument(parser, "--depth_min_m", "--depth-min", type=float, default=0.10)
    add_alias_argument(parser, "--depth_max_m", "--depth-max", type=float, default=1.20)
    add_alias_argument(
        parser,
        "--foreground_min_height_m",
        "--foreground-min-height",
        type=float,
        default=0.015,
    )
    add_alias_argument(parser, "--min_component_area_px", "--min-component-area-px", type=int, default=250)
    add_alias_argument(parser, "--morph_kernel_size", "--morph-kernel-size", type=int, default=5)
    add_alias_argument(parser, "--table_depth_bin_size_m", "--table-depth-bin-size", type=float, default=0.005)
    add_alias_argument(parser, "--top_surface_percentile", "--top-surface-percentile", type=float, default=0.20)
    add_alias_argument(parser, "--top_surface_band_m", "--top-surface-band", type=float, default=0.008)
    add_alias_argument(parser, "--graspable_min_height_m", "--graspable-min-height", type=float, default=0.01)
    add_alias_argument(parser, "--graspable_max_height_m", "--graspable-max-height", type=float, default=0.20)
    add_alias_argument(parser, "--graspable_min_mask_area_px", "--graspable-min-mask-area-px", type=int, default=400)
    add_alias_argument(parser, "--graspable_max_mask_area_px", "--graspable-max-mask-area-px", type=int, default=25000)

    add_alias_argument(parser, "--pregrasp_offset_m", "--pregrasp-offset", type=float, default=0.08)
    add_alias_argument(parser, "--grasp_z_offset_m", "--grasp-z-offset", type=float, default=0.0)
    add_alias_argument(parser, "--lift_offset_m", "--lift-offset", type=float, default=0.10)
    add_alias_argument(parser, "--gripper_open_pos", "--gripper-open", type=float, default=80.0)
    add_alias_argument(
        parser,
        "--gripper_close_pos",
        "--gripper-close",
        "--gripper-closed",
        type=float,
        default=20.0,
    )
    add_alias_argument(
        parser,
        "--robot_max_relative_target",
        "--robot-max-relative-target",
        type=float,
        default=None,
    )
    add_alias_argument(parser, "--safety_profile", "--safety-profile", default="default")
    add_alias_argument(parser, "--position_weight", "--position-weight", type=float, default=1.0)
    add_alias_argument(parser, "--orientation_weight", "--orientation-weight", type=float, default=0.05)
    add_alias_argument(
        parser,
        "--grasp_rpy_deg",
        "--grasp-rpy-deg",
        action=Vector3Action,
        nargs="+",
        default=None,
        help="Explicit roll/pitch/yaw in degrees for grasp orientation. Defaults to a top-down pose.",
    )
    add_alias_argument(parser, "--urdf_path", "--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    add_alias_argument(parser, "--target_frame_name", "--target-frame-name", default="gripper_frame_link")
    add_alias_argument(parser, "--move_sleep_s", "--move-sleep-s", type=float, default=1.2)
    add_alias_argument(parser, "--settle_s", "--settle-s", type=float, default=0.5)
    add_alias_argument(parser, "--verification_warmup_frames", "--verification-warmup-frames", type=int, default=5)
    add_alias_argument(parser, "--verification_position_tol_m", "--verification-position-tol", type=float, default=0.04)
    add_alias_argument(parser, "--verification_height_tol_m", "--verification-height-tol", type=float, default=0.03)
    add_bool_argument(parser, "--dry_run", "--dry-run", default=True, help="Run perception and planning without actuation.")
    add_bool_argument(parser, "--display_data", "--display-data", default=False, help="Show the detection overlay window.")
    add_alias_argument(parser, "--log_level", "--log-level", default="INFO")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def collect_provided_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    provided_dests: set[str] = set()
    option_map = parser._option_string_actions  # type: ignore[attr-defined]
    for token in argv:
        if not token.startswith("-"):
            continue
        option_string = token.split("=", 1)[0]
        action = option_map.get(option_string)
        if action is not None:
            provided_dests.add(action.dest)
    return provided_dests


def normalize_vector(value: list[float] | tuple[float, ...] | None, fallback: list[float]) -> list[float]:
    if value is None:
        return [float(item) for item in fallback]
    return [float(item) for item in value]


def normalize_args(args: argparse.Namespace, provided_dests: set[str]) -> dict[str, bool]:
    if args.realsense_serial in {"", "none", "null", "None"}:
        args.realsense_serial = None

    args.camera_to_base_xyz = normalize_vector(args.camera_to_base_xyz, DEFAULT_CAMERA_TO_BASE_XYZ)
    args.camera_to_base_rpy_deg = normalize_vector(args.camera_to_base_rpy_deg, DEFAULT_CAMERA_TO_BASE_RPY_DEG)
    args.grasp_rpy_deg = normalize_vector(args.grasp_rpy_deg, DEFAULT_GRASP_RPY_DEG)
    args.workspace_min_xyz = None if args.workspace_min_xyz is None else [float(item) for item in args.workspace_min_xyz]
    args.workspace_max_xyz = None if args.workspace_max_xyz is None else [float(item) for item in args.workspace_max_xyz]

    args.safety_profile = str(args.safety_profile).strip().lower() or "default"
    if args.safety_profile not in {"off", "default", "strict"}:
        raise ValueError(
            f"safety_profile must be one of off/default/strict, got {args.safety_profile!r}"
        )

    if (args.workspace_min_xyz is None) != (args.workspace_max_xyz is None):
        raise ValueError("workspace_min_xyz and workspace_max_xyz must be provided together")
    if args.workspace_min_xyz is not None and args.workspace_max_xyz is not None:
        if any(low > high for low, high in zip(args.workspace_min_xyz, args.workspace_max_xyz, strict=True)):
            raise ValueError(
                f"workspace_min_xyz={args.workspace_min_xyz} must be <= workspace_max_xyz={args.workspace_max_xyz}"
            )

    if args.depth_min_m < 0.0:
        raise ValueError(f"depth_min_m must be >= 0, got {args.depth_min_m}")
    if args.depth_max_m <= args.depth_min_m:
        raise ValueError(f"depth_max_m must be greater than depth_min_m, got {args.depth_max_m} <= {args.depth_min_m}")
    if args.foreground_min_height_m <= 0.0:
        raise ValueError(
            f"foreground_min_height_m must be positive, got {args.foreground_min_height_m}"
        )
    if not args.dry_run and args.safety_profile == "off":
        raise ValueError("Live geom_grasp runs require safety_profile=default or strict")
    if args.robot_max_relative_target is None and args.safety_profile == "strict":
        args.robot_max_relative_target = 4.0
    elif args.robot_max_relative_target is None and args.safety_profile == "default":
        args.robot_max_relative_target = 8.0
    if args.robot_max_relative_target is not None and args.robot_max_relative_target <= 0.0:
        raise ValueError(
            "robot_max_relative_target must be positive when provided, "
            f"got {args.robot_max_relative_target}"
        )
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        raise ValueError("camera_width, camera_height, and camera_fps must all be positive")
    if args.camera_timeout_ms <= 0:
        raise ValueError(f"camera_timeout_ms must be positive, got {args.camera_timeout_ms}")
    if args.perception_consistency_frames <= 0:
        raise ValueError("perception_consistency_frames must be positive")
    if args.perception_min_consistent_detections <= 0:
        raise ValueError("perception_min_consistent_detections must be positive")
    if args.perception_min_consistent_detections > args.perception_consistency_frames:
        raise ValueError(
            "perception_min_consistent_detections cannot exceed perception_consistency_frames"
        )
    if args.perception_position_std_max_m <= 0.0:
        raise ValueError("perception_position_std_max_m must be positive")
    if args.perception_mask_area_rel_std_max <= 0.0:
        raise ValueError("perception_mask_area_rel_std_max must be positive")
    if args.min_component_area_px <= 0:
        raise ValueError(f"min_component_area_px must be positive, got {args.min_component_area_px}")
    if args.morph_kernel_size <= 0:
        raise ValueError(f"morph_kernel_size must be positive, got {args.morph_kernel_size}")
    if not 0.0 < args.top_surface_percentile <= 0.5:
        raise ValueError("top_surface_percentile must be in (0, 0.5]")
    if args.top_surface_band_m <= 0.0:
        raise ValueError("top_surface_band_m must be positive")
    if args.graspable_min_height_m <= 0.0:
        raise ValueError("graspable_min_height_m must be positive")
    if args.graspable_max_height_m <= args.graspable_min_height_m:
        raise ValueError("graspable_max_height_m must exceed graspable_min_height_m")
    if args.graspable_min_mask_area_px <= 0:
        raise ValueError("graspable_min_mask_area_px must be positive")
    if args.graspable_max_mask_area_px < args.graspable_min_mask_area_px:
        raise ValueError("graspable_max_mask_area_px must be >= graspable_min_mask_area_px")
    if args.orientation_weight < 0.0:
        raise ValueError("orientation_weight must be >= 0")
    if args.verification_warmup_frames <= 0:
        raise ValueError("verification_warmup_frames must be positive")
    if args.verification_position_tol_m <= 0.0 or args.verification_height_tol_m <= 0.0:
        raise ValueError("verification tolerances must be positive")

    tracked_dests = [
        "task",
        "robot_port",
        "robot_id",
        "robot_calib_dir",
        "robot_use_degrees",
        "realsense_serial",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_warmup_frames",
        "camera_timeout_ms",
        "perception_consistency_frames",
        "perception_min_consistent_detections",
        "perception_position_std_max_m",
        "perception_mask_area_rel_std_max",
        "camera_to_base_xyz",
        "camera_to_base_rpy_deg",
        "grasp_rpy_deg",
        "workspace_min_xyz",
        "workspace_max_xyz",
        "depth_min_m",
        "depth_max_m",
        "foreground_min_height_m",
        "top_surface_percentile",
        "top_surface_band_m",
        "graspable_min_height_m",
        "graspable_max_height_m",
        "graspable_min_mask_area_px",
        "graspable_max_mask_area_px",
        "pregrasp_offset_m",
        "grasp_z_offset_m",
        "lift_offset_m",
        "gripper_open_pos",
        "gripper_close_pos",
        "robot_max_relative_target",
        "safety_profile",
        "dry_run",
        "display_data",
        "orientation_weight",
        "position_weight",
        "move_sleep_s",
        "settle_s",
        "verification_warmup_frames",
        "verification_position_tol_m",
        "verification_height_tol_m",
    ]
    used_defaults = {dest: dest not in provided_dests for dest in tracked_dests}
    used_defaults["camera_to_base_extrinsics"] = used_defaults["camera_to_base_xyz"] and used_defaults["camera_to_base_rpy_deg"]
    return used_defaults


def to_jsonable(value: Any) -> Any:
    numpy_module, _ = optional_import("numpy")
    if numpy_module is not None and isinstance(value, numpy_module.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: to_jsonable(value) for key, value in vars(args).items() if not key.startswith("_")}


def dependency_entry(*, available: bool, error: BaseException | str | None = None) -> dict[str, Any]:
    return {
        "available": bool(available),
        "error": None if available else safe_error_text(error),
    }


def build_dependency_status(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    numpy_module, numpy_error = optional_import("numpy")
    cv2_module, cv2_error = optional_import("cv2")
    rs_module, rs_error = optional_import("pyrealsense2")
    placo_module, placo_error = optional_import("placo")
    runtime_symbols, runtime_error = load_so101_runtime_symbols()

    urdf_exists = Path(args.urdf_path).is_file()
    status = {
        "numpy": dependency_entry(available=numpy_module is not None, error=numpy_error),
        "cv2": dependency_entry(available=cv2_module is not None, error=cv2_error),
        "pyrealsense2": dependency_entry(available=rs_module is not None, error=rs_error),
        "placo": dependency_entry(available=placo_module is not None, error=placo_error),
        "so101_runtime": dependency_entry(available=runtime_symbols is not None, error=runtime_error),
        "urdf_exists": dependency_entry(
            available=urdf_exists,
            error=None if urdf_exists else f"URDF not found: {Path(args.urdf_path)}",
        ),
    }
    status["can_perceive"] = all(status[name]["available"] for name in ("numpy", "cv2", "pyrealsense2"))
    status["can_plan"] = status["can_perceive"] and all(
        status[name]["available"] for name in ("placo", "so101_runtime", "urdf_exists")
    )
    status["can_execute"] = bool(status["can_plan"])
    status["missing_dependencies"] = [
        name
        for name in ("numpy", "cv2", "pyrealsense2", "placo", "so101_runtime", "urdf_exists")
        if not status[name]["available"]
    ]
    return status, runtime_symbols


def make_stage_state() -> dict[str, Any]:
    return {
        "attempted": False,
        "ok": False,
        "skipped": False,
        "error": None,
    }


def build_base_summary(argv: list[str]) -> dict[str, Any]:
    return {
        "ok": False,
        "dry_run": False,
        "executed": False,
        "dependency_status": {},
        "preflight": {},
        "parsed_args": None,
        "raw_argv": argv,
        "detection": None,
        "perception_consistency": None,
        "graspability": None,
        "object_center_camera_xyz": None,
        "object_center_base_xyz": None,
        "verification": None,
        "used_defaults": {},
        "error": None,
        "error_stage": None,
        "stage_status": {
            "perception": make_stage_state(),
            "robot_connection": make_stage_state(),
            "planning": make_stage_state(),
            "execution": make_stage_state(),
            "verification": make_stage_state(),
        },
    }


def emit_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, sort_keys=True))


def to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    numpy_module, _ = optional_import("numpy")
    if numpy_module is not None and isinstance(value, numpy_module.ndarray):
        flattened = value.astype(float).reshape(-1)
        return [float(item) for item in flattened.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def rotation_matrix_from_rpy_deg(rpy_deg: list[float] | tuple[float, float, float]) -> Any:
    numpy_module, _ = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError("numpy is required to build camera transforms")

    roll, pitch, yaw = [math.radians(float(value)) for value in rpy_deg]
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = numpy_module.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = numpy_module.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz = numpy_module.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def find_realsense_serial(requested_serial: str | None) -> str:
    rs_module, rs_error = optional_import("pyrealsense2")
    if rs_module is None:
        raise RuntimeError(f"pyrealsense2 import failed: {safe_error_text(rs_error)}")

    context = rs_module.context()
    devices = list(context.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices detected")

    if requested_serial:
        for device in devices:
            serial = device.get_info(rs_module.camera_info.serial_number)
            if serial == requested_serial:
                return str(serial)
        available = [device.get_info(rs_module.camera_info.serial_number) for device in devices]
        raise RuntimeError(f"Requested RealSense serial {requested_serial!r} not found. Available: {available}")

    return str(devices[0].get_info(rs_module.camera_info.serial_number))


def start_realsense_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any, str, float]:
    rs_module, rs_error = optional_import("pyrealsense2")
    if rs_module is None:
        raise RuntimeError(f"pyrealsense2 import failed: {safe_error_text(rs_error)}")

    serial = find_realsense_serial(args.realsense_serial)
    pipeline = rs_module.pipeline()
    config = rs_module.config()
    config.enable_device(serial)
    config.enable_stream(rs_module.stream.color, args.camera_width, args.camera_height, rs_module.format.bgr8, args.camera_fps)
    config.enable_stream(rs_module.stream.depth, args.camera_width, args.camera_height, rs_module.format.z16, args.camera_fps)
    profile = pipeline.start(config)
    align = rs_module.align(rs_module.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    LOGGER.info(
        "Started RealSense serial=%s width=%s height=%s fps=%s depth_scale=%s",
        serial,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        depth_scale,
    )
    return pipeline, align, profile, serial, depth_scale


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
    }


def get_aligned_rgbd_frame(
    pipeline: Any,
    align: Any,
    *,
    timeout_ms: int,
    warmup_frames: int,
) -> tuple[Any, Any, Any]:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    color_image = None
    depth_image = None
    intrinsics = None

    for _ in range(max(int(warmup_frames), 1)):
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color_image = numpy_module.asanyarray(color_frame.get_data())
        depth_image = numpy_module.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    if color_image is None or depth_image is None or intrinsics is None:
        raise RuntimeError("Failed to acquire aligned RGB-D frame from RealSense")

    return color_image, depth_image, intrinsics


def estimate_table_depth(valid_depths_m: Any, *, bin_size: float) -> float:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    if valid_depths_m.size == 0:
        raise RuntimeError("No valid depth samples available to estimate table depth")

    lower = float(valid_depths_m.min())
    upper = float(valid_depths_m.max())
    if upper <= lower:
        return float(numpy_module.median(valid_depths_m))

    step = max(float(bin_size), 1e-4)
    bins = numpy_module.arange(lower, upper + step, step, dtype=float)
    if bins.size < 2:
        return float(numpy_module.median(valid_depths_m))

    hist, edges = numpy_module.histogram(valid_depths_m, bins=bins)
    best_idx = int(numpy_module.argmax(hist))
    band_min = float(edges[best_idx])
    band_max = float(edges[best_idx + 1])
    in_band = valid_depths_m[(valid_depths_m >= band_min) & (valid_depths_m < band_max)]
    if in_band.size == 0:
        in_band = valid_depths_m
    return float(numpy_module.median(in_band))


def detect_foreground_object(
    color_bgr: Any,
    depth_raw: Any,
    *,
    intrinsics: Any,
    depth_scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    numpy_module, numpy_error = optional_import("numpy")
    cv2_module, cv2_error = optional_import("cv2")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")
    if cv2_module is None:
        raise RuntimeError(f"cv2 import failed: {safe_error_text(cv2_error)}")

    depth_m = depth_raw.astype(numpy_module.float32) * float(depth_scale)
    valid_mask = numpy_module.isfinite(depth_m) & (depth_m >= float(args.depth_min_m)) & (depth_m <= float(args.depth_max_m))
    valid_depths = depth_m[valid_mask]
    if valid_depths.size == 0:
        raise RuntimeError("No valid depth pixels found in the configured depth range")

    table_depth_m = estimate_table_depth(valid_depths, bin_size=float(args.table_depth_bin_size_m))
    height_above_table_m = table_depth_m - depth_m
    foreground_mask = valid_mask & (height_above_table_m >= float(args.foreground_min_height_m))

    kernel_size = max(int(args.morph_kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = numpy_module.ones((kernel_size, kernel_size), dtype=numpy_module.uint8)
    mask_u8 = foreground_mask.astype(numpy_module.uint8) * 255
    mask_u8 = cv2_module.morphologyEx(mask_u8, cv2_module.MORPH_OPEN, kernel)
    mask_u8 = cv2_module.morphologyEx(mask_u8, cv2_module.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2_module.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No foreground connected component detected above the table")

    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2_module.CC_STAT_AREA])
        if area < int(args.min_component_area_px):
            continue
        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        raise RuntimeError(
            "Foreground components were found, but none passed the minimum area threshold "
            f"of {args.min_component_area_px} px"
        )

    component_mask = labels == best_label
    centroid_x = float(centroids[best_label][0])
    centroid_y = float(centroids[best_label][1])
    component_depths = depth_m[component_mask]
    component_depth_m = float(numpy_module.median(component_depths))

    x0 = int(stats[best_label, cv2_module.CC_STAT_LEFT])
    y0 = int(stats[best_label, cv2_module.CC_STAT_TOP])
    width = int(stats[best_label, cv2_module.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2_module.CC_STAT_HEIGHT])
    bbox = [x0, y0, x0 + width, y0 + height]

    grasp_point = compute_grasp_point_from_component(
        component_mask=component_mask,
        depth_m=depth_m,
        intrinsics=intrinsics,
        args=args,
    )
    camera_xyz = grasp_point["camera_xyz"]
    object_depth_m = float(grasp_point["depth_m"])
    object_height_m = float(table_depth_m - object_depth_m)

    display = color_bgr.copy()
    overlay = display.copy()
    overlay[component_mask] = (0, 255, 0)
    cv2_module.addWeighted(overlay, 0.25, display, 0.75, 0.0, dst=display)
    cv2_module.circle(display, (int(round(centroid_x)), int(round(centroid_y))), 6, (0, 0, 255), -1)
    cv2_module.circle(
        display,
        (int(round(grasp_point["pixel_xy"][0])), int(round(grasp_point["pixel_xy"][1]))),
        7,
        (255, 0, 255),
        -1,
    )
    cv2_module.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2_module.putText(
        display,
        f"table={table_depth_m:.3f}m",
        (10, 24),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    cv2_module.putText(
        display,
        f"obj={object_depth_m:.3f}m h={object_height_m:.3f}m",
        (10, 48),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    cv2_module.putText(
        display,
        f"grasp={grasp_point['method']}",
        (10, 72),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    detection = {
        "table_depth_m": float(table_depth_m),
        "component_depth_m": float(component_depth_m),
        "object_depth_m": float(object_depth_m),
        "object_height_m": float(object_height_m),
        "centroid_px": [float(centroid_x), float(centroid_y)],
        "grasp_point_px": [float(grasp_point["pixel_xy"][0]), float(grasp_point["pixel_xy"][1])],
        "bbox_xyxy": [int(value) for value in bbox],
        "mask_area_px": int(best_area),
        "depth_scale_m": float(depth_scale),
        "camera_intrinsics": intrinsics_to_dict(intrinsics),
        "grasp_point_camera_xyz": to_float_list(grasp_point["camera_xyz"]),
        "grasp_point_depth_m": float(grasp_point["depth_m"]),
        "grasp_point_method": grasp_point["method"],
        "top_surface_band_pixels": int(grasp_point["band_pixels"]),
        "top_surface_band_ratio": float(grasp_point["band_ratio"]),
        "top_surface_depth_m": float(grasp_point["top_surface_depth_m"]),
    }
    return detection, camera_xyz, display


def compute_grasp_point_from_component(
    *,
    component_mask: Any,
    depth_m: Any,
    intrinsics: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    valid_component_mask = component_mask & numpy_module.isfinite(depth_m)
    component_depths = depth_m[valid_component_mask]
    if component_depths.size == 0:
        raise RuntimeError("Selected foreground component has no valid depth values")

    top_surface_depth_m = float(
        numpy_module.quantile(component_depths, float(args.top_surface_percentile))
    )
    band_mask = valid_component_mask & (depth_m <= (top_surface_depth_m + float(args.top_surface_band_m)))
    band_pixels = int(numpy_module.count_nonzero(band_mask))
    min_band_pixels = max(25, int(args.min_component_area_px * 0.05))
    method = "top_surface_band"
    if band_pixels < min_band_pixels:
        band_mask = valid_component_mask
        band_pixels = int(numpy_module.count_nonzero(band_mask))
        method = "component_median_fallback"

    ys, xs = numpy_module.nonzero(band_mask)
    band_depths = depth_m[band_mask]
    pixel_xy = [float(numpy_module.median(xs)), float(numpy_module.median(ys))]
    depth_value = float(numpy_module.median(band_depths))

    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    xs_f = xs.astype(float)
    ys_f = ys.astype(float)
    points_x = (xs_f - ppx) / fx * band_depths
    points_y = (ys_f - ppy) / fy * band_depths
    camera_xyz = numpy_module.asarray(
        [
            float(numpy_module.median(points_x)),
            float(numpy_module.median(points_y)),
            float(numpy_module.median(band_depths)),
        ],
        dtype=float,
    )

    return {
        "pixel_xy": pixel_xy,
        "depth_m": depth_value,
        "camera_xyz": camera_xyz,
        "method": method,
        "band_pixels": band_pixels,
        "band_ratio": float(band_pixels / max(int(numpy_module.count_nonzero(component_mask)), 1)),
        "top_surface_depth_m": top_surface_depth_m,
    }


def aggregate_perception_samples(
    samples: list[dict[str, Any]],
    *,
    min_consistent_detections: int,
) -> dict[str, Any]:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    if len(samples) < min_consistent_detections:
        raise RuntimeError(
            f"Only {len(samples)} valid perception samples collected, "
            f"need at least {min_consistent_detections}"
        )

    base_points = numpy_module.asarray([sample["object_center_base_xyz"] for sample in samples], dtype=float)
    mask_areas = numpy_module.asarray([sample["detection"]["mask_area_px"] for sample in samples], dtype=float)
    heights = numpy_module.asarray([sample["detection"]["object_height_m"] for sample in samples], dtype=float)
    point_mean = base_points.mean(axis=0)
    point_std_per_axis = base_points.std(axis=0)
    point_std_m = float(numpy_module.linalg.norm(point_std_per_axis))
    area_rel_std = float(mask_areas.std() / max(mask_areas.mean(), 1.0))
    height_std_m = float(heights.std())
    representative_index = int(numpy_module.argmin(numpy_module.linalg.norm(base_points - point_mean, axis=1)))

    representative = dict(samples[representative_index])
    representative["consistency"] = {
        "requested_frames": len(samples),
        "successful_frames": len(samples),
        "selected_sample_index": representative_index,
        "position_mean_base_xyz": to_float_list(point_mean),
        "position_std_per_axis_m": to_float_list(point_std_per_axis),
        "position_std_m": point_std_m,
        "mask_area_mean_px": float(mask_areas.mean()),
        "mask_area_rel_std": area_rel_std,
        "object_height_std_m": height_std_m,
    }
    return representative


def assess_graspability(perception: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    detection = perception["detection"]
    consistency = perception["consistency"]
    reasons: list[str] = []

    mask_area_px = int(detection["mask_area_px"])
    object_height_m = float(detection["object_height_m"])
    position_std_m = float(consistency["position_std_m"])
    area_rel_std = float(consistency["mask_area_rel_std"])

    if mask_area_px < int(args.graspable_min_mask_area_px):
        reasons.append(
            f"mask area {mask_area_px} px is below graspable_min_mask_area_px={args.graspable_min_mask_area_px}"
        )
    if mask_area_px > int(args.graspable_max_mask_area_px):
        reasons.append(
            f"mask area {mask_area_px} px exceeds graspable_max_mask_area_px={args.graspable_max_mask_area_px}"
        )
    if object_height_m < float(args.graspable_min_height_m):
        reasons.append(
            f"object height {object_height_m:.4f} m is below graspable_min_height_m={args.graspable_min_height_m:.4f}"
        )
    if object_height_m > float(args.graspable_max_height_m):
        reasons.append(
            f"object height {object_height_m:.4f} m exceeds graspable_max_height_m={args.graspable_max_height_m:.4f}"
        )
    if position_std_m > float(args.perception_position_std_max_m):
        reasons.append(
            f"perception position std {position_std_m:.4f} m exceeds threshold {args.perception_position_std_max_m:.4f}"
        )
    if area_rel_std > float(args.perception_mask_area_rel_std_max):
        reasons.append(
            f"mask area rel std {area_rel_std:.4f} exceeds threshold {args.perception_mask_area_rel_std_max:.4f}"
        )

    return {
        "graspable": not reasons,
        "reasons": reasons or ["graspability checks passed"],
        "metrics": {
            "mask_area_px": mask_area_px,
            "object_height_m": object_height_m,
            "perception_position_std_m": position_std_m,
            "perception_mask_area_rel_std": area_rel_std,
        },
    }


def transform_camera_point_to_base(
    camera_xyz: Any,
    *,
    camera_to_base_xyz: list[float] | tuple[float, float, float],
    camera_to_base_rpy_deg: list[float] | tuple[float, float, float],
) -> Any:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    rotation = rotation_matrix_from_rpy_deg(camera_to_base_rpy_deg)
    translation = numpy_module.asarray(camera_to_base_xyz, dtype=float)
    return rotation @ numpy_module.asarray(camera_xyz, dtype=float) + translation


def assert_within_workspace(point_xyz: Any, args: argparse.Namespace) -> None:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    if args.workspace_min_xyz is None or args.workspace_max_xyz is None:
        return
    workspace_min = numpy_module.asarray(args.workspace_min_xyz, dtype=float)
    workspace_max = numpy_module.asarray(args.workspace_max_xyz, dtype=float)
    point = numpy_module.asarray(point_xyz, dtype=float)
    if numpy_module.any(point < workspace_min) or numpy_module.any(point > workspace_max):
        raise RuntimeError(
            f"Point {point.tolist()} is outside workspace bounds min={workspace_min.tolist()} max={workspace_max.tolist()}"
        )


def build_target_pose(position_xyz: Any, rotation: Any) -> Any:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    pose = numpy_module.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = numpy_module.asarray(position_xyz, dtype=float)
    return pose


def extract_joint_vector(observation: dict[str, Any], motor_names: list[str]) -> Any:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")
    return numpy_module.asarray([float(observation[f"{name}.pos"]) for name in motor_names], dtype=float)


def make_joint_action(motor_names: list[str], joint_positions: Any, gripper_pos: float) -> dict[str, float]:
    action: dict[str, float] = {}
    for index, name in enumerate(motor_names):
        if name == "gripper":
            action[f"{name}.pos"] = float(gripper_pos)
        else:
            action[f"{name}.pos"] = float(joint_positions[index])
    return action


def compute_stage_actions(
    *,
    robot: Any,
    kinematics: Any,
    object_center_base_xyz: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    numpy_module, numpy_error = optional_import("numpy")
    if numpy_module is None:
        raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

    observation = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())
    current_joints = extract_joint_vector(observation, motor_names)
    desired_rotation = rotation_matrix_from_rpy_deg(args.grasp_rpy_deg)

    object_center_base_xyz = numpy_module.asarray(object_center_base_xyz, dtype=float)
    pregrasp_xyz = object_center_base_xyz + numpy_module.array([0.0, 0.0, float(args.pregrasp_offset_m)], dtype=float)
    grasp_xyz = object_center_base_xyz + numpy_module.array([0.0, 0.0, float(args.grasp_z_offset_m)], dtype=float)
    lift_xyz = grasp_xyz + numpy_module.array([0.0, 0.0, float(args.lift_offset_m)], dtype=float)

    for stage_name, point in {
        "pregrasp": pregrasp_xyz,
        "grasp": grasp_xyz,
        "lift": lift_xyz,
    }.items():
        assert_within_workspace(point, args)
        LOGGER.info("%s target base xyz=%s", stage_name, numpy_module.round(point, 4).tolist())

    pregrasp_pose = build_target_pose(pregrasp_xyz, desired_rotation)
    pregrasp_joints = kinematics.inverse_kinematics(
        current_joint_pos=current_joints,
        desired_ee_pose=pregrasp_pose,
        position_weight=float(args.position_weight),
        orientation_weight=float(args.orientation_weight),
    )

    grasp_pose = build_target_pose(grasp_xyz, desired_rotation)
    grasp_joints = kinematics.inverse_kinematics(
        current_joint_pos=pregrasp_joints,
        desired_ee_pose=grasp_pose,
        position_weight=float(args.position_weight),
        orientation_weight=float(args.orientation_weight),
    )

    lift_pose = build_target_pose(lift_xyz, desired_rotation)
    lift_joints = kinematics.inverse_kinematics(
        current_joint_pos=grasp_joints,
        desired_ee_pose=lift_pose,
        position_weight=float(args.position_weight),
        orientation_weight=float(args.orientation_weight),
    )

    return {
        "motor_names": motor_names,
        "current_joints": current_joints,
        "desired_rotation": desired_rotation,
        "desired_grasp_rpy_deg": [float(value) for value in args.grasp_rpy_deg],
        "orientation_weight_used": float(args.orientation_weight),
        "pregrasp_xyz": pregrasp_xyz,
        "grasp_xyz": grasp_xyz,
        "lift_xyz": lift_xyz,
        "pregrasp_action": make_joint_action(motor_names, pregrasp_joints, float(args.gripper_open_pos)),
        "grasp_action": make_joint_action(motor_names, grasp_joints, float(args.gripper_open_pos)),
        "close_action": {"gripper.pos": float(args.gripper_close_pos)},
        "lift_action": make_joint_action(motor_names, lift_joints, float(args.gripper_close_pos)),
    }


def validate_action_candidate(action: dict[str, Any]) -> None:
    for key, value in action.items():
        try:
            numeric = float(value)
        except Exception as exc:
            raise RuntimeError(f"Action field {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(numeric):
            raise RuntimeError(f"Action field {key!r} is not finite: {value!r}")
        if key == "gripper.pos" and not (0.0 <= numeric <= 100.0):
            raise RuntimeError(f"gripper.pos must be in [0, 100], got {numeric}")


def validate_plan(plan: dict[str, Any]) -> None:
    for key in ("pregrasp_action", "grasp_action", "close_action", "lift_action"):
        validate_action_candidate(plan[key])


def connect_robot(args: argparse.Namespace, runtime_symbols: dict[str, Any]) -> Any:
    robot_config = runtime_symbols["SO101FollowerConfig"](
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=Path(args.robot_calib_dir),
        disable_torque_on_disconnect=True,
        max_relative_target=args.robot_max_relative_target,
        cameras={},
        use_degrees=bool(args.robot_use_degrees),
    )
    robot = runtime_symbols["SO101Follower"](robot_config)
    robot.connect(calibrate=False)
    return robot


def create_kinematics(args: argparse.Namespace, runtime_symbols: dict[str, Any], motor_names: list[str]) -> Any:
    return runtime_symbols["RobotKinematics"](
        urdf_path=str(args.urdf_path),
        target_frame_name=args.target_frame_name,
        joint_names=motor_names,
    )


def maybe_show_detection(display_bgr: Any, *, enabled: bool) -> None:
    if not enabled:
        return
    cv2_module, cv2_error = optional_import("cv2")
    if cv2_module is None:
        LOGGER.warning("display_data requested but cv2 is unavailable: %s", safe_error_text(cv2_error))
        return
    try:
        cv2_module.imshow("so101_geom_grasp", display_bgr)
        cv2_module.waitKey(1)
    except Exception as exc:  # pragma: no cover - depends on display environment
        LOGGER.warning("Failed to display detection overlay: %s", exc)


def execute_actions(robot: Any, plan: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    sent_actions: list[dict[str, Any]] = []
    stages = [
        ("pregrasp", plan["pregrasp_action"]),
        ("grasp", plan["grasp_action"]),
        ("close_gripper", plan["close_action"]),
        ("lift", plan["lift_action"]),
    ]
    for stage_name, action in stages:
        LOGGER.info("Sending %s action=%s", stage_name, {key: round(float(val), 3) for key, val in action.items()})
        sent = robot.send_action(action)
        sent_actions.append({stage_name: {key: float(val) for key, val in sent.items()}})
        time.sleep(float(args.move_sleep_s))
        if stage_name in {"grasp", "close_gripper"}:
            time.sleep(float(args.settle_s))
    return sent_actions


def plan_to_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "planned_grasp_rpy_deg": [float(value) for value in plan["desired_grasp_rpy_deg"]],
        "orientation_weight_used": float(plan["orientation_weight_used"]),
        "planned_pregrasp_xyz": to_float_list(plan["pregrasp_xyz"]),
        "planned_grasp_xyz": to_float_list(plan["grasp_xyz"]),
        "planned_lift_xyz": to_float_list(plan["lift_xyz"]),
        "planned_actions": {
            "pregrasp": {key: float(value) for key, value in plan["pregrasp_action"].items()},
            "grasp": {key: float(value) for key, value in plan["grasp_action"].items()},
            "close_gripper": {key: float(value) for key, value in plan["close_action"].items()},
            "lift": {key: float(value) for key, value in plan["lift_action"].items()},
        },
    }


def run_perception(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = None
    frame_errors: list[str] = []
    try:
        pipeline, align, profile, serial, depth_scale = start_realsense_pipeline(args)
        samples: list[dict[str, Any]] = []
        requested_frames = max(int(args.perception_consistency_frames), 1)
        for frame_index in range(requested_frames):
            warmup_frames = int(args.camera_warmup_frames) if frame_index == 0 else 1
            try:
                color_bgr, depth_raw, intrinsics = get_aligned_rgbd_frame(
                    pipeline,
                    align,
                    timeout_ms=int(args.camera_timeout_ms),
                    warmup_frames=warmup_frames,
                )
                detection, camera_xyz, display_bgr = detect_foreground_object(
                    color_bgr,
                    depth_raw,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                    args=args,
                )
                object_center_base_xyz = transform_camera_point_to_base(
                    camera_xyz,
                    camera_to_base_xyz=args.camera_to_base_xyz,
                    camera_to_base_rpy_deg=args.camera_to_base_rpy_deg,
                )
                assert_within_workspace(object_center_base_xyz, args)
                samples.append(
                    {
                        "frame_index": frame_index,
                        "detection": detection,
                        "object_center_camera_xyz": camera_xyz,
                        "object_center_base_xyz": object_center_base_xyz,
                        "display_bgr": display_bgr,
                    }
                )
            except Exception as exc:
                frame_errors.append(f"frame[{frame_index}] {exc}")

        perception = aggregate_perception_samples(
            samples,
            min_consistent_detections=int(args.perception_min_consistent_detections),
        )
        perception["realsense_serial"] = serial
        perception["frame_errors"] = frame_errors
        perception["consistency"]["requested_frames"] = requested_frames
        return perception
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:  # pragma: no cover - cleanup path
                LOGGER.warning("Failed to stop RealSense pipeline cleanly: %s", exc)


def verify_post_grasp_result(
    *,
    args: argparse.Namespace,
    pre_perception: dict[str, Any],
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "attempted": False,
        "ok": None,
        "reason": None,
        "post_detection": None,
        "post_object_center_base_xyz": None,
    }
    try:
        verification["attempted"] = True
        verify_args = argparse.Namespace(**vars(args))
        verify_args.perception_consistency_frames = 1
        verify_args.perception_min_consistent_detections = 1
        verify_args.camera_warmup_frames = int(args.verification_warmup_frames)
        post_perception = run_perception(verify_args)
        verification["post_detection"] = post_perception["detection"]
        verification["post_object_center_base_xyz"] = to_float_list(post_perception["object_center_base_xyz"])

        numpy_module, numpy_error = optional_import("numpy")
        if numpy_module is None:
            raise RuntimeError(f"numpy import failed: {safe_error_text(numpy_error)}")

        post_xyz = numpy_module.asarray(post_perception["object_center_base_xyz"], dtype=float)
        pre_xyz = numpy_module.asarray(pre_perception["object_center_base_xyz"], dtype=float)
        position_delta_m = float(numpy_module.linalg.norm(post_xyz - pre_xyz))
        height_delta_m = abs(
            float(post_perception["detection"]["object_height_m"]) - float(pre_perception["detection"]["object_height_m"])
        )
        verification["position_delta_from_pre_m"] = position_delta_m
        verification["height_delta_from_pre_m"] = height_delta_m

        if (
            position_delta_m <= float(args.verification_position_tol_m)
            and height_delta_m <= float(args.verification_height_tol_m)
        ):
            verification["ok"] = False
            verification["reason"] = "Post-grasp tabletop target remains near the original pre-grasp location"
        else:
            verification["ok"] = True
            verification["reason"] = "Post-grasp foreground changed from the original tabletop location"
        return verification
    except Exception as exc:
        verification["ok"] = None
        verification["reason"] = f"Verification inconclusive: {exc}"
        return verification


def populate_preflight(summary: dict[str, Any], dependency_status: dict[str, Any]) -> None:
    summary["preflight"] = {
        "can_perceive": bool(dependency_status.get("can_perceive", False)),
        "can_plan": bool(dependency_status.get("can_plan", False)),
        "can_execute": bool(dependency_status.get("can_execute", False)),
        "missing_dependencies": list(dependency_status.get("missing_dependencies", [])),
    }


def build_parse_failure_summary(parser: argparse.ArgumentParser, argv: list[str], exc: BaseException) -> dict[str, Any]:
    defaults = parser.parse_args([])
    used_defaults = normalize_args(defaults, set())
    dependency_status, _ = build_dependency_status(defaults)
    summary = build_base_summary(argv)
    summary["parsed_args"] = serialize_args(defaults)
    summary["used_defaults"] = used_defaults
    summary["dependency_status"] = dependency_status
    populate_preflight(summary, dependency_status)
    summary["error"] = str(exc)
    summary["error_stage"] = "argument_parsing"
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    provided_dests = collect_provided_dests(parser, argv)
    summary = build_base_summary(argv)

    try:
        args = parser.parse_args(argv)
        used_defaults = normalize_args(args, provided_dests)
    except (ArgumentParserError, ValueError) as exc:
        configure_logging("INFO")
        summary = build_parse_failure_summary(parser, argv, exc)
        emit_summary(summary)
        return 2

    configure_logging(args.log_level)
    summary["dry_run"] = bool(args.dry_run)
    summary["parsed_args"] = serialize_args(args)
    summary["used_defaults"] = used_defaults

    dependency_status, runtime_symbols = build_dependency_status(args)
    summary["dependency_status"] = dependency_status
    populate_preflight(summary, dependency_status)

    robot = None
    exit_code = 1

    try:
        if not dependency_status["can_perceive"]:
            missing = ", ".join(dependency_status["missing_dependencies"])
            summary["error"] = f"Perception prerequisites missing: {missing}"
            summary["error_stage"] = "perception"
            summary["stage_status"]["perception"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": summary["error"],
            }
            return exit_code

        summary["stage_status"]["perception"]["attempted"] = True
        perception = run_perception(args)
        maybe_show_detection(perception["display_bgr"], enabled=bool(args.display_data))
        summary["realsense_serial"] = perception["realsense_serial"]
        summary["detection"] = perception["detection"]
        summary["perception_consistency"] = perception["consistency"]
        if perception.get("frame_errors"):
            summary["perception_frame_errors"] = list(perception["frame_errors"])
        summary["object_center_camera_xyz"] = to_float_list(perception["object_center_camera_xyz"])
        summary["object_center_base_xyz"] = to_float_list(perception["object_center_base_xyz"])
        summary["graspability"] = assess_graspability(perception, args)
        summary["stage_status"]["perception"] = {
            "attempted": True,
            "ok": True,
            "skipped": False,
            "error": None,
        }
        LOGGER.info(
            "Detected object center camera xyz=%s base xyz=%s",
            summary["object_center_camera_xyz"],
            summary["object_center_base_xyz"],
        )
        LOGGER.info("Perception consistency=%s", summary["perception_consistency"])

        if not bool(summary["graspability"]["graspable"]):
            summary["error"] = "Perception target is not graspable: " + "; ".join(summary["graspability"]["reasons"])
            summary["error_stage"] = "planning"
            summary["stage_status"]["robot_connection"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": "Robot connection skipped because graspability gate did not pass",
            }
            summary["stage_status"]["planning"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": summary["error"],
            }
            return exit_code

        if not dependency_status["can_plan"]:
            missing = ", ".join(dependency_status["missing_dependencies"])
            summary["error"] = f"Planning prerequisites missing: {missing}"
            summary["error_stage"] = "planning"
            summary["stage_status"]["robot_connection"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": "Robot connection skipped because planning prerequisites are missing",
            }
            summary["stage_status"]["planning"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": summary["error"],
            }
            return exit_code

        if runtime_symbols is None:
            raise RuntimeError("SO101 runtime symbols unexpectedly unavailable after dependency probe")

        summary["stage_status"]["robot_connection"]["attempted"] = True
        robot = connect_robot(args, runtime_symbols)
        summary["stage_status"]["robot_connection"] = {
            "attempted": True,
            "ok": True,
            "skipped": False,
            "error": None,
        }
        LOGGER.info(
            "Connected robot id=%s port=%s max_relative_target=%s",
            args.robot_id,
            args.robot_port,
            args.robot_max_relative_target,
        )

        summary["stage_status"]["planning"]["attempted"] = True
        kinematics = create_kinematics(args, runtime_symbols, list(robot.bus.motors.keys()))
        plan = compute_stage_actions(
            robot=robot,
            kinematics=kinematics,
            object_center_base_xyz=perception["object_center_base_xyz"],
            args=args,
        )
        validate_plan(plan)
        summary.update(plan_to_summary(plan))
        summary["stage_status"]["planning"] = {
            "attempted": True,
            "ok": True,
            "skipped": False,
            "error": None,
        }

        if args.dry_run:
            summary["ok"] = True
            summary["stage_status"]["execution"] = {
                "attempted": False,
                "ok": False,
                "skipped": True,
                "error": "Execution skipped because dry_run=true",
            }
            exit_code = 0
            return exit_code

        summary["stage_status"]["execution"]["attempted"] = True
        sent_actions = execute_actions(robot, plan, args)
        summary["sent_actions"] = sent_actions
        summary["executed"] = True
        summary["stage_status"]["execution"] = {
            "attempted": True,
            "ok": True,
            "skipped": False,
            "error": None,
        }
        summary["stage_status"]["verification"]["attempted"] = True
        verification = verify_post_grasp_result(args=args, pre_perception=perception)
        summary["verification"] = verification
        if verification["ok"] is False:
            summary["error"] = str(verification["reason"])
            summary["error_stage"] = "verification"
            summary["stage_status"]["verification"] = {
                "attempted": True,
                "ok": False,
                "skipped": False,
                "error": str(verification["reason"]),
            }
            return exit_code

        summary["ok"] = True
        summary["stage_status"]["verification"] = {
            "attempted": True,
            "ok": True if verification["ok"] is True else False,
            "skipped": True if verification["ok"] is None else False,
            "error": None if verification["ok"] is True else verification["reason"],
        }
        exit_code = 0
        return exit_code
    except Exception as exc:
        LOGGER.error("Geometric grasp runtime failed: %s", exc)
        LOGGER.debug("Failure traceback:\n%s", traceback.format_exc())
        if summary["error"] is None:
            summary["error"] = str(exc)
        if summary["error_stage"] is None:
            if summary["stage_status"]["verification"].get("attempted"):
                summary["error_stage"] = "verification"
                summary["stage_status"]["verification"] = {
                    "attempted": True,
                    "ok": False,
                    "skipped": False,
                    "error": str(exc),
                }
            elif summary["stage_status"]["execution"].get("attempted"):
                summary["error_stage"] = "execution"
                summary["stage_status"]["execution"] = {
                    "attempted": True,
                    "ok": False,
                    "skipped": False,
                    "error": str(exc),
                }
            elif summary["stage_status"]["planning"].get("attempted"):
                summary["error_stage"] = "planning"
                summary["stage_status"]["planning"] = {
                    "attempted": True,
                    "ok": False,
                    "skipped": False,
                    "error": str(exc),
                }
            elif summary["stage_status"]["robot_connection"].get("attempted"):
                summary["error_stage"] = "robot_connection"
                summary["stage_status"]["robot_connection"] = {
                    "attempted": True,
                    "ok": False,
                    "skipped": False,
                    "error": str(exc),
                }
            else:
                summary["error_stage"] = "perception"
                summary["stage_status"]["perception"] = {
                    "attempted": True,
                    "ok": False,
                    "skipped": False,
                    "error": str(exc),
                }
        return exit_code
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:  # pragma: no cover - cleanup path
                LOGGER.warning("Failed to disconnect robot cleanly: %s", exc)
        if bool(getattr(args, "display_data", False)):
            cv2_module, _ = optional_import("cv2")
            if cv2_module is not None:
                try:
                    cv2_module.destroyAllWindows()
                except Exception:
                    pass
        emit_summary(summary)


if __name__ == "__main__":
    raise SystemExit(main())
