#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger("so101_camera_to_base_calib")


def load_geom_runtime_module():
    runtime_path = Path(__file__).with_name("lerobot_run_so101_geom_grasp.py")
    spec = importlib.util.spec_from_file_location("so101_geom_grasp_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load runtime module from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNTIME = load_geom_runtime_module()
DEFAULT_CALIB_DIR = RUNTIME.DEFAULT_CALIB_DIR
DEFAULT_URDF_PATH = RUNTIME.DEFAULT_URDF_PATH


def parse_vector3(value: str) -> list[float]:
    pieces = [piece for piece in value.replace("[", "").replace("]", "").replace(",", " ").split() if piece]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 values, got {value!r}")
    try:
        return [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected numeric values, got {value!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive SO101 camera-to-base calibration using point correspondences."
    )
    parser.add_argument("--robot_port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="my_so101")
    parser.add_argument("--robot_calib_dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--robot_use_degrees", type=RUNTIME.parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--robot_max_relative_target", type=float, default=8.0)
    parser.add_argument("--realsense_serial", default=None)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--camera_warmup_frames", type=int, default=20)
    parser.add_argument("--camera_timeout_ms", type=int, default=5000)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--target_frame_name", default="gripper_frame_link")
    parser.add_argument(
        "--tcp_offset_xyz",
        type=parse_vector3,
        default=[0.0, 0.0, 0.0],
        help="Optional TCP offset in the end-effector frame, in meters.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=6,
        help="Minimum number of correspondences required before solving.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "camera_to_base_calibration",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> list[float]:
    sy = -float(rotation[2, 0])
    pitch = math.asin(max(min(sy, 1.0), -1.0))
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def solve_rigid_transform(camera_points: np.ndarray, base_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_centroid = camera_points.mean(axis=0)
    base_centroid = base_points.mean(axis=0)
    centered_camera = camera_points - camera_centroid
    centered_base = base_points - base_centroid

    covariance = centered_camera.T @ centered_base
    u_matrix, _, v_t = np.linalg.svd(covariance)
    rotation = v_t.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        v_t[-1, :] *= -1.0
        rotation = v_t.T @ u_matrix.T

    translation = base_centroid - rotation @ camera_centroid
    predicted = (rotation @ camera_points.T).T + translation
    residuals = predicted - base_points
    return rotation, translation, residuals


def compute_ee_base_xyz(robot: Any, kinematics: Any, tcp_offset_xyz: list[float]) -> tuple[np.ndarray, np.ndarray]:
    observation = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())
    joint_vector = RUNTIME.extract_joint_vector(observation, motor_names)
    ee_pose = kinematics.forward_kinematics(joint_vector)
    ee_base_xyz = np.asarray(ee_pose[:3, 3], dtype=float)
    tcp_offset = np.asarray(tcp_offset_xyz, dtype=float)
    if np.linalg.norm(tcp_offset) > 0.0:
        ee_base_xyz = ee_base_xyz + np.asarray(ee_pose[:3, :3], dtype=float) @ tcp_offset
    return ee_base_xyz, joint_vector


def capture_frame(args: argparse.Namespace, pipeline: Any, align: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    color_bgr, depth_raw, intrinsics = RUNTIME.get_aligned_rgbd_frame(
        pipeline,
        align,
        timeout_ms=int(args.camera_timeout_ms),
        warmup_frames=1,
    )
    return np.asarray(color_bgr), np.asarray(depth_raw), intrinsics


def pixel_to_camera_xyz(depth_raw: np.ndarray, intrinsics: Any, depth_scale: float, pixel_xy: tuple[int, int]) -> tuple[np.ndarray, float]:
    rs_module, rs_error = RUNTIME.optional_import("pyrealsense2")
    if rs_module is None:
        raise RuntimeError(f"pyrealsense2 import failed: {RUNTIME.safe_error_text(rs_error)}")

    x_coord, y_coord = pixel_xy
    if y_coord < 0 or y_coord >= depth_raw.shape[0] or x_coord < 0 or x_coord >= depth_raw.shape[1]:
        raise RuntimeError(f"Selected pixel {pixel_xy} is outside image bounds")

    depth_m = float(depth_raw[y_coord, x_coord]) * float(depth_scale)
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise RuntimeError(f"Selected pixel {pixel_xy} has invalid depth {depth_m}")

    camera_xyz = np.asarray(
        rs_module.rs2_deproject_pixel_to_point(intrinsics, [float(x_coord), float(y_coord)], float(depth_m)),
        dtype=float,
    )
    return camera_xyz, depth_m


def draw_overlay(
    frame_bgr: np.ndarray,
    *,
    frozen: bool,
    selected_pixel: tuple[int, int] | None,
    num_records: int,
    min_points: int,
    status_lines: list[str],
) -> np.ndarray:
    cv2_module, cv2_error = RUNTIME.optional_import("cv2")
    if cv2_module is None:
        raise RuntimeError(f"cv2 import failed: {RUNTIME.safe_error_text(cv2_error)}")

    display = frame_bgr.copy()
    header = "FROZEN: click point, s=save, r=resume, u=undo, q=solve" if frozen else "LIVE: move tip to point, f=freeze, q=quit"
    cv2_module.putText(display, header, (10, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2_module.putText(
        display,
        f"saved_points={num_records} min_points={min_points}",
        (10, 48),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    if selected_pixel is not None:
        x_coord, y_coord = selected_pixel
        cv2_module.drawMarker(
            display,
            (int(x_coord), int(y_coord)),
            (0, 0, 255),
            markerType=cv2_module.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    for index, line in enumerate(status_lines[:8]):
        cv2_module.putText(
            display,
            line,
            (10, 80 + index * 22),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return display


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    cv2_module, cv2_error = RUNTIME.optional_import("cv2")
    if cv2_module is None:
        raise RuntimeError(f"cv2 import failed: {RUNTIME.safe_error_text(cv2_error)}")

    runtime_symbols, runtime_error = RUNTIME.load_so101_runtime_symbols()
    if runtime_symbols is None:
        raise RuntimeError(f"Failed to load SO101 runtime: {RUNTIME.safe_error_text(runtime_error)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"so101_camera_to_base_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    robot = None
    records: list[dict[str, Any]] = []
    selected_pixel: tuple[int, int] | None = None
    frozen_sample: dict[str, Any] | None = None
    window_name = "so101_camera_to_base_calibration"

    def on_mouse(event: int, x_coord: int, y_coord: int, _flags: int, _param: Any) -> None:
        nonlocal selected_pixel
        if event == cv2_module.EVENT_LBUTTONDOWN and frozen_sample is not None:
            selected_pixel = (int(x_coord), int(y_coord))

    cv2_module.namedWindow(window_name, cv2_module.WINDOW_NORMAL)
    cv2_module.setMouseCallback(window_name, on_mouse)

    try:
        pipeline, align, _profile, serial, depth_scale = RUNTIME.start_realsense_pipeline(args)
        robot = RUNTIME.connect_robot(args, runtime_symbols)
        kinematics = RUNTIME.create_kinematics(args, runtime_symbols, list(robot.bus.motors.keys()))
        LOGGER.info("Calibration session started with RealSense serial=%s robot_id=%s", serial, args.robot_id)

        live_frame = None
        live_status = "Move the TCP tip onto a visible physical point, then press f to freeze."
        while True:
            if frozen_sample is None:
                try:
                    color_bgr, depth_raw, intrinsics = capture_frame(args, pipeline, align)
                    live_frame = {
                        "color_bgr": color_bgr,
                        "depth_raw": depth_raw,
                        "intrinsics": intrinsics,
                    }
                except Exception as exc:
                    live_status = f"live capture error: {exc}"

            active_frame = frozen_sample["color_bgr"] if frozen_sample is not None else live_frame["color_bgr"]
            status_lines = [live_status]
            if frozen_sample is not None:
                status_lines.append(f"frozen_tcp_base_xyz={np.round(frozen_sample['tcp_base_xyz'], 4).tolist()}")
            if selected_pixel is not None and frozen_sample is not None:
                try:
                    camera_xyz, depth_m = pixel_to_camera_xyz(
                        frozen_sample["depth_raw"],
                        frozen_sample["intrinsics"],
                        depth_scale,
                        selected_pixel,
                    )
                    status_lines.append(f"pixel={selected_pixel} depth_m={depth_m:.4f}")
                    status_lines.append(f"camera_xyz={np.round(camera_xyz, 4).tolist()}")
                except Exception as exc:
                    status_lines.append(f"pixel error: {exc}")

            display = draw_overlay(
                active_frame,
                frozen=frozen_sample is not None,
                selected_pixel=selected_pixel,
                num_records=len(records),
                min_points=int(args.min_points),
                status_lines=status_lines,
            )
            cv2_module.imshow(window_name, display)
            key = cv2_module.waitKey(30) & 0xFF

            if key == 27:
                LOGGER.info("Calibration aborted by user")
                return 1

            if key == ord("f"):
                if live_frame is None:
                    LOGGER.warning("Cannot freeze because no live frame is available yet")
                    continue
                tcp_base_xyz, joint_vector = compute_ee_base_xyz(robot, kinematics, args.tcp_offset_xyz)
                frozen_sample = {
                    **live_frame,
                    "tcp_base_xyz": tcp_base_xyz,
                    "joint_vector_deg": np.asarray(joint_vector, dtype=float),
                    "captured_at": time.time(),
                }
                selected_pixel = None
                live_status = "Frame frozen. Click the same physical point in the image, then press s to save."
                continue

            if key == ord("r"):
                frozen_sample = None
                selected_pixel = None
                live_status = "Resumed live stream."
                continue

            if key == ord("u"):
                if records:
                    removed = records.pop()
                    LOGGER.info("Removed point #%s", removed["index"])
                else:
                    LOGGER.info("No saved points to undo")
                continue

            if key == ord("s"):
                if frozen_sample is None:
                    LOGGER.warning("Freeze a frame first with 'f'")
                    continue
                if selected_pixel is None:
                    LOGGER.warning("Click a point in the frozen image before saving")
                    continue
                try:
                    camera_xyz, depth_m = pixel_to_camera_xyz(
                        frozen_sample["depth_raw"],
                        frozen_sample["intrinsics"],
                        depth_scale,
                        selected_pixel,
                    )
                except Exception as exc:
                    LOGGER.error("Failed to convert selected pixel to camera point: %s", exc)
                    continue

                index = len(records) + 1
                frame_path = output_dir / f"point_{index:02d}.jpg"
                if not cv2_module.imwrite(str(frame_path), frozen_sample["color_bgr"]):
                    raise RuntimeError(f"Failed to write calibration image to {frame_path}")

                record = {
                    "index": index,
                    "captured_at": float(frozen_sample["captured_at"]),
                    "pixel_xy": [int(selected_pixel[0]), int(selected_pixel[1])],
                    "depth_m": float(depth_m),
                    "camera_xyz": [float(value) for value in camera_xyz.tolist()],
                    "base_xyz": [float(value) for value in frozen_sample["tcp_base_xyz"].tolist()],
                    "joint_vector_deg": [float(value) for value in frozen_sample["joint_vector_deg"].tolist()],
                    "frame_path": str(frame_path),
                }
                records.append(record)
                LOGGER.info(
                    "Saved point #%s camera_xyz=%s base_xyz=%s",
                    index,
                    np.round(camera_xyz, 4).tolist(),
                    np.round(frozen_sample["tcp_base_xyz"], 4).tolist(),
                )
                frozen_sample = None
                selected_pixel = None
                live_status = f"Saved point #{index}. Move to the next point and press f again."
                continue

            if key == ord("q"):
                if len(records) < int(args.min_points):
                    LOGGER.warning("Need at least %s points before solving, currently have %s", args.min_points, len(records))
                    continue
                break

        camera_points = np.asarray([record["camera_xyz"] for record in records], dtype=float)
        base_points = np.asarray([record["base_xyz"] for record in records], dtype=float)
        rotation, translation, residuals = solve_rigid_transform(camera_points, base_points)
        residual_norms = np.linalg.norm(residuals, axis=1)
        rpy_deg = rotation_matrix_to_rpy_deg(rotation)

        result = {
            "robot_id": args.robot_id,
            "realsense_serial": serial,
            "num_points": len(records),
            "camera_to_base_xyz": [float(value) for value in translation.tolist()],
            "camera_to_base_rpy_deg": [float(value) for value in rpy_deg],
            "camera_to_base_rotation_matrix": rotation.tolist(),
            "residuals_m": {
                "per_point": [float(value) for value in residual_norms.tolist()],
                "mean": float(residual_norms.mean()),
                "max": float(residual_norms.max()),
            },
            "tcp_offset_xyz": [float(value) for value in args.tcp_offset_xyz],
            "records": records,
            "recommended_flags": {
                "camera_to_base_xyz": [float(value) for value in translation.tolist()],
                "camera_to_base_rpy_deg": [float(value) for value in rpy_deg],
            },
            "recommended_command_snippet": (
                f"--camera_to_base_xyz='[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]' "
                f"--camera_to_base_rpy_deg='[{rpy_deg[0]:.3f}, {rpy_deg[1]:.3f}, {rpy_deg[2]:.3f}]'"
            ),
        }

        result_path = output_dir / "camera_to_base_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(result, ensure_ascii=False, indent=2))
        LOGGER.info("Calibration result written to %s", result_path)
        return 0
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        try:
            cv2_module.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
