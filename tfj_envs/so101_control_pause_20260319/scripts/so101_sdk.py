#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from so101_manual_console import (  # noqa: E402
    CARTESIAN_DIRECTIONS,
    JOINT_NAMES,
    ManualConsole,
    build_parser,
)


Direction = Literal["up", "down", "left", "right", "forward", "back"]
JointName = Literal["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class SO101SDKConfig:
    # Connection
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "my_so101"
    calibration_dir: str = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
    calibrate_on_connect: bool = False
    dry_run: bool = False

    # Control mode
    control_mode: Literal["joint", "ik"] = "ik"
    ik_solver: Literal["placo", "dls"] = "placo"
    urdf_path: str = str(Path(__file__).resolve().parents[3] / "so101_new_calib.urdf")
    target_frame_name: str = "gripper_frame_link"
    tcp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # IK / safety defaults
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.0
    ik_exact_step_m: float = 0.004
    ik_target_tol_mm: float = 1.0
    ik_max_iters: int = 60
    ik_jacobian_eps_deg: float = 0.5
    ik_damping: float = 0.002
    ik_min_step_progress_mm: float = 0.05
    ik_stuck_steps: int = 4
    ik_max_auto_flips: int = 1
    ik_auto_flip_axis: bool = False
    ik_verbose_steps: bool = False

    max_command_delta_deg: float = 8.0
    max_relative_target_deg: float = 8.0
    gripper_max_relative_target: float | None = 40.0
    settle_s: float = 0.15

    # Direction mapping
    direction_map_file: str = str(
        Path(__file__).resolve().parents[1] / "runtime" / "so101_direction_map.json"
    )
    sign_up: float = 1.0
    sign_left: float = 1.0
    sign_forward: float = 1.0
    sign_cartesian_x: float = 1.0
    sign_cartesian_y: float = 1.0
    sign_cartesian_z: float = 1.0

    # Gripper
    gripper_open_value: float = 100.0
    gripper_close_value: float = 0.0

    # Other parser fields
    cm_to_deg: float = 3.0
    default_cm: float = 1.0
    calibration_probe_cm: float = 0.6
    calibration_min_progress_mm: float = 0.5
    calibration_improve_margin_mm: float = 0.3
    calibration_auto_save: bool = False


class SO101SDK:
    """Simple programmatic SDK wrapper around `so101_manual_console.py`."""

    def __init__(self, config: SO101SDKConfig | None = None):
        self.config = config or SO101SDKConfig()
        self._console: ManualConsole | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._console is not None

    def _build_args(self) -> argparse.Namespace:
        args = build_parser().parse_args([])
        for key, value in vars(self.config).items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.cmd = ""
        if args.gripper_max_relative_target is not None and args.gripper_max_relative_target <= 0:
            args.gripper_max_relative_target = None
        return args

    def connect(self) -> "SO101SDK":
        with self._lock:
            if self._console is not None:
                return self
            self._console = ManualConsole(self._build_args())
            self._console.connect()
            return self

    def disconnect(self) -> None:
        with self._lock:
            if self._console is None:
                return
            self._console.disconnect()
            self._console = None

    def __enter__(self) -> "SO101SDK":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def _require_console(self) -> ManualConsole:
        if self._console is None:
            raise RuntimeError("SO101SDK is not connected. Call connect() first.")
        return self._console

    def execute(self, command: str) -> None:
        with self._lock:
            console = self._require_console()
            keep_running = console.execute(command)
            if not keep_running:
                raise RuntimeError("Command requested stop/exit.")

    def state(self) -> dict[str, float]:
        with self._lock:
            console = self._require_console()
            return console._read_positions()

    def move(self, direction: Direction, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        if direction not in CARTESIAN_DIRECTIONS:
            raise ValueError(f"Unsupported direction: {direction}")
        with self._lock:
            console = self._require_console()
            use_exact = (console.args.control_mode == "ik") if exact is None else bool(exact)
            cmd = f"{direction}_exact {float(cm):g}" if use_exact else f"{direction} {float(cm):g}"
            self.execute(cmd)
            return console._read_positions()

    def up(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("up", cm, exact=exact)

    def down(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("down", cm, exact=exact)

    def left(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("left", cm, exact=exact)

    def right(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("right", cm, exact=exact)

    def forward(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("forward", cm, exact=exact)

    def back(self, cm: float, *, exact: bool | None = None) -> dict[str, float]:
        return self.move("back", cm, exact=exact)

    def open_gripper(self) -> dict[str, float]:
        self.execute("open")
        return self.state()

    def close_gripper(self) -> dict[str, float]:
        self.execute("close")
        return self.state()

    def set_joint(self, joint: JointName | str, value: float) -> dict[str, float]:
        self.execute(f"set {joint} {float(value):g}")
        return self.state()

    def delta_joint(self, joint: JointName | str, delta_deg: float) -> dict[str, float]:
        self.execute(f"j {joint} {float(delta_deg):g}")
        return self.state()

    def home(self) -> dict[str, float]:
        self.execute("home")
        return self.state()

    def calibrate_directions(self, probe_cm: float | None = None) -> None:
        if probe_cm is None:
            self.execute("calibrate")
            return
        self.execute(f"calibrate {float(probe_cm):g}")

    def frame_set(self, direction: Direction, axis_token: str) -> None:
        self.execute(f"frame set {direction} {axis_token}")

    def frame_save(self) -> None:
        self.execute("frame save")

    def frame_load(self) -> None:
        self.execute("frame load")

    def frame_reset(self) -> None:
        self.execute("frame reset")

    def frame_show(self) -> str:
        with self._lock:
            console = self._require_console()
            return console._format_direction_map()

    def tcp_xyz(self) -> tuple[float, float, float] | None:
        with self._lock:
            console = self._require_console()
            if console.kinematics is None:
                return None
            positions = console._read_positions()
            xyz = console._ee_xyz_from_positions(positions)
            return tuple(float(value) for value in xyz.tolist())


def _demo() -> int:
    # Keep demo dependency-light: joint mode does not require placo.
    cfg = SO101SDKConfig(dry_run=True, control_mode="joint")
    with SO101SDK(cfg) as arm:
        arm.up(1.0)
        arm.left(1.0)
        arm.open_gripper()
        print("state:", arm.state())
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
