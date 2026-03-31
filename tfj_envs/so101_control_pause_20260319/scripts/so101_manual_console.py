#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from lerobot.model.kinematics import RobotKinematics  # noqa: E402
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig  # noqa: E402


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CARTESIAN_DIRECTIONS = ("up", "down", "left", "right", "forward", "back")
DIRECTION_OPPOSITES = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "forward": "back",
    "back": "forward",
}

AXIS_TOKEN_VECTORS = {
    "x+": np.asarray([1.0, 0.0, 0.0], dtype=float),
    "x-": np.asarray([-1.0, 0.0, 0.0], dtype=float),
    "y+": np.asarray([0.0, 1.0, 0.0], dtype=float),
    "y-": np.asarray([0.0, -1.0, 0.0], dtype=float),
    "z+": np.asarray([0.0, 0.0, 1.0], dtype=float),
    "z-": np.asarray([0.0, 0.0, -1.0], dtype=float),
}

AXIS_TOKEN_ALIASES = {
    "x+": "x+",
    "+x": "x+",
    "x-": "x-",
    "-x": "x-",
    "y+": "y+",
    "+y": "y+",
    "y-": "y-",
    "-y": "y-",
    "z+": "z+",
    "+z": "z+",
    "z-": "z-",
    "-z": "z-",
}


@dataclass(frozen=True)
class AxisBinding:
    joint: str
    sign: float


@dataclass(frozen=True)
class IKMoveResult:
    direction: str
    target_m: float
    achieved_m: float
    iterations: int
    flipped_count: int
    stalled: bool
    converged: bool
    start_xyz: np.ndarray
    end_xyz: np.ndarray


class ManualConsole:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot: SO101Follower | None = None
        self.home_positions: dict[str, float] | None = None
        self.last_positions: dict[str, float] = {}
        self.kinematics: RobotKinematics | None = None
        self.ik_joint_names: list[str] = []

        self.axis_bindings = {
            "up": AxisBinding("shoulder_lift", args.sign_up),
            "down": AxisBinding("shoulder_lift", -args.sign_up),
            "left": AxisBinding("shoulder_pan", args.sign_left),
            "right": AxisBinding("shoulder_pan", -args.sign_left),
            "forward": AxisBinding("elbow_flex", args.sign_forward),
            "back": AxisBinding("elbow_flex", -args.sign_forward),
        }
        self.direction_vectors = self._default_direction_vectors()
        self.direction_map_path = Path(self.args.direction_map_file).expanduser()
        self.tcp_offset_xyz = np.asarray(self.args.tcp_offset_xyz, dtype=float)
        self._load_direction_map_if_exists()

    def _default_direction_vectors(self) -> dict[str, np.ndarray]:
        return {
            "up": np.asarray([0.0, 0.0, float(self.args.sign_cartesian_z)], dtype=float),
            "down": np.asarray([0.0, 0.0, -float(self.args.sign_cartesian_z)], dtype=float),
            "left": np.asarray([0.0, float(self.args.sign_cartesian_y), 0.0], dtype=float),
            "right": np.asarray([0.0, -float(self.args.sign_cartesian_y), 0.0], dtype=float),
            "forward": np.asarray([float(self.args.sign_cartesian_x), 0.0, 0.0], dtype=float),
            "back": np.asarray([-float(self.args.sign_cartesian_x), 0.0, 0.0], dtype=float),
        }

    def _normalize_unit_vector(self, raw_vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(raw_vec, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            raise ValueError("Direction vector norm is zero.")
        return vec / norm

    def _direction_axis_unit(self, direction: str) -> np.ndarray:
        if direction not in CARTESIAN_DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        if direction not in self.direction_vectors:
            raise ValueError(f"Direction mapping missing for: {direction}")
        return self._normalize_unit_vector(self.direction_vectors[direction])

    def _direction_delta_m(self, direction: str, cm: float) -> tuple[float, float, float]:
        step_m = float(cm) * 0.01
        axis = self._direction_axis_unit(direction)
        delta = axis * step_m
        return float(delta[0]), float(delta[1]), float(delta[2])

    def _frame_pose_from_q(self, q_deg: np.ndarray) -> np.ndarray:
        if self.kinematics is None:
            raise RuntimeError("Kinematics is not initialized.")
        return np.asarray(self.kinematics.forward_kinematics(np.asarray(q_deg, dtype=float)), dtype=float)

    def _tcp_xyz_from_frame_pose(self, pose: np.ndarray) -> np.ndarray:
        tcp_xyz = np.asarray(pose[:3, 3], dtype=float)
        if float(np.linalg.norm(self.tcp_offset_xyz)) > 0.0:
            tcp_xyz = tcp_xyz + np.asarray(pose[:3, :3], dtype=float) @ self.tcp_offset_xyz
        return tcp_xyz

    def _ee_xyz_from_positions(self, positions: dict[str, float]) -> np.ndarray:
        q = np.asarray([positions[joint] for joint in self.ik_joint_names], dtype=float)
        return self._fk_xyz_from_q(q)

    def _fk_xyz_from_q(self, q_deg: np.ndarray) -> np.ndarray:
        return self._tcp_xyz_from_frame_pose(self._frame_pose_from_q(q_deg))

    def _q_from_positions(self, positions: dict[str, float]) -> np.ndarray:
        return np.asarray([positions[joint] for joint in self.ik_joint_names], dtype=float)

    def _numerical_position_jacobian(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_xyz = self._fk_xyz_from_q(q_deg)
        dof = len(self.ik_joint_names)
        jac = np.zeros((3, dof), dtype=float)
        eps_deg = float(self.args.ik_jacobian_eps_deg)
        for idx in range(dof):
            q_pert = np.array(q_deg, copy=True)
            q_pert[idx] += eps_deg
            xyz_pert = self._fk_xyz_from_q(q_pert)
            jac[:, idx] = (xyz_pert - base_xyz) / eps_deg
        return base_xyz, jac

    def _ik_dls_step(self, q_deg: np.ndarray, delta_xyz_m: np.ndarray) -> np.ndarray:
        _, jac = self._numerical_position_jacobian(q_deg)
        damping = float(self.args.ik_damping)
        jj_t = jac @ jac.T
        lhs = jj_t + (damping**2) * np.eye(3, dtype=float)
        dq = jac.T @ np.linalg.solve(lhs, delta_xyz_m)

        max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
        limit = abs(float(self.args.max_command_delta_deg))
        if max_abs > limit and max_abs > 1e-9:
            dq = dq * (limit / max_abs)
        return q_deg + dq

    def _ik_pose_step(self, q_deg: np.ndarray, delta_xyz_m: np.ndarray) -> np.ndarray:
        if self.kinematics is None:
            raise RuntimeError("Kinematics is not initialized.")
        current_pose = self._frame_pose_from_q(q_deg)
        current_tcp_xyz = self._tcp_xyz_from_frame_pose(current_pose)
        desired_tcp_xyz = current_tcp_xyz + np.asarray(delta_xyz_m, dtype=float)
        desired_pose = np.array(current_pose, copy=True)
        desired_pose[:3, 3] = desired_tcp_xyz - np.asarray(current_pose[:3, :3], dtype=float) @ self.tcp_offset_xyz
        q_next = self.kinematics.inverse_kinematics(
            current_joint_pos=np.asarray(q_deg, dtype=float),
            desired_ee_pose=desired_pose,
            position_weight=float(self.args.ik_position_weight),
            orientation_weight=float(self.args.ik_orientation_weight),
        )
        dq = np.asarray(q_next, dtype=float) - np.asarray(q_deg, dtype=float)
        max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
        limit = abs(float(self.args.max_command_delta_deg))
        if max_abs > limit and max_abs > 1e-9:
            dq = dq * (limit / max_abs)
        return np.asarray(q_deg, dtype=float) + dq

    def _send_q_target(self, q_target_deg: np.ndarray) -> None:
        self.last_positions = self._read_positions()
        action_target = dict(self.last_positions)
        for idx, joint in enumerate(self.ik_joint_names):
            action_target[joint] = float(q_target_deg[idx])
        self._send_positions(action_target)

    def _axis_unit(self, direction: str, cm_sign: float = 1.0) -> np.ndarray:
        axis = self._direction_axis_unit(direction)
        if float(cm_sign) < 0:
            axis = -axis
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            raise RuntimeError("Invalid direction mapping.")
        return axis / norm

    def _pair_lead_direction(self, direction: str) -> str:
        if direction in {"up", "down"}:
            return "up"
        if direction in {"left", "right"}:
            return "left"
        if direction in {"forward", "back"}:
            return "forward"
        raise ValueError(f"Unknown direction: {direction}")

    def _set_direction_vector(self, direction: str, axis_vector: np.ndarray) -> None:
        lead = self._pair_lead_direction(direction)
        axis = self._normalize_unit_vector(axis_vector)
        if direction != lead:
            axis = -axis
        for other_lead in ("up", "left", "forward"):
            if other_lead == lead:
                continue
            other_axis = self._direction_axis_unit(other_lead)
            if abs(float(np.dot(axis, other_axis))) >= 0.99:
                raise ValueError(
                    "Direction mapping must stay axis-orthogonal. "
                    f"{lead} conflicts with {other_lead}; choose another axis."
                )
        opposite = DIRECTION_OPPOSITES[lead]
        self.direction_vectors[lead] = axis
        self.direction_vectors[opposite] = -axis

    def _flip_direction_pair(self, direction: str) -> None:
        lead = self._pair_lead_direction(direction)
        opposite = DIRECTION_OPPOSITES[lead]
        self.direction_vectors[lead] = -self._direction_axis_unit(lead)
        self.direction_vectors[opposite] = -self._direction_axis_unit(opposite)

    def _axis_token_from_vector(self, vec: np.ndarray) -> str:
        axis = self._normalize_unit_vector(vec)
        best_token = "x+"
        best_score = -1.0
        for token, token_vec in AXIS_TOKEN_VECTORS.items():
            score = float(np.dot(axis, token_vec))
            if score > best_score:
                best_score = score
                best_token = token
        return best_token

    def _format_direction_map(self) -> str:
        parts = []
        for direction in CARTESIAN_DIRECTIONS:
            token = self._axis_token_from_vector(self._direction_axis_unit(direction))
            parts.append(f"{direction}={token}")
        return " ".join(parts)

    def _load_direction_map_if_exists(self) -> None:
        path = self.direction_map_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_directions = payload.get("directions", {})
            for direction in CARTESIAN_DIRECTIONS:
                if direction not in raw_directions:
                    continue
                vec = np.asarray(raw_directions[direction], dtype=float)
                self.direction_vectors[direction] = self._normalize_unit_vector(vec)
            # Enforce opposite-pair consistency if one side was missing.
            for lead in ("up", "left", "forward"):
                opposite = DIRECTION_OPPOSITES[lead]
                if lead in raw_directions and opposite not in raw_directions:
                    self.direction_vectors[opposite] = -self._direction_axis_unit(lead)
                if opposite in raw_directions and lead not in raw_directions:
                    self.direction_vectors[lead] = -self._direction_axis_unit(opposite)
            print(f"loaded direction map: {path} | {self._format_direction_map()}")
        except Exception as exc:  # noqa: BLE001
            print(f"warn: failed to load direction map {path}: {exc}")

    def _save_direction_map(self) -> None:
        payload = {
            "version": 1,
            "directions": {
                direction: self._direction_axis_unit(direction).tolist() for direction in CARTESIAN_DIRECTIONS
            },
        }
        path = self.direction_map_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved direction map: {path}")

    def connect(self) -> None:
        if self.args.dry_run:
            print("[dry-run] robot connect skipped")
            self.last_positions = {joint: 0.0 for joint in JOINT_NAMES}
            self.home_positions = dict(self.last_positions)
            if self.args.control_mode == "ik":
                urdf_path = Path(self.args.urdf_path)
                if not urdf_path.is_file():
                    raise FileNotFoundError(f"URDF not found: {urdf_path}")
                self.ik_joint_names = [joint for joint in JOINT_NAMES if joint != "gripper"]
                self.kinematics = RobotKinematics(
                    urdf_path=str(urdf_path),
                    target_frame_name=self.args.target_frame_name,
                    joint_names=self.ik_joint_names,
                )
                print(f"frame map: {self._format_direction_map()}")
            return

        max_relative_target: float | dict[str, float]
        if self.args.gripper_max_relative_target is None:
            max_relative_target = self.args.max_relative_target_deg
        else:
            max_relative_target = {joint: self.args.max_relative_target_deg for joint in JOINT_NAMES}
            max_relative_target["gripper"] = self.args.gripper_max_relative_target

        cfg = SO101FollowerConfig(
            port=self.args.robot_port,
            id=self.args.robot_id,
            calibration_dir=Path(self.args.calibration_dir),
            disable_torque_on_disconnect=True,
            max_relative_target=max_relative_target,
            cameras={},
            use_degrees=True,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=self.args.calibrate_on_connect)
        self.last_positions = self._read_positions()
        self.home_positions = dict(self.last_positions)
        if self.args.control_mode == "ik":
            urdf_path = Path(self.args.urdf_path)
            if not urdf_path.is_file():
                raise FileNotFoundError(f"URDF not found: {urdf_path}")
            # IK controls the arm joints only; the gripper is handled separately.
            self.ik_joint_names = [joint for joint in JOINT_NAMES if joint != "gripper"]
            self.kinematics = RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=self.args.target_frame_name,
                joint_names=self.ik_joint_names,
            )
        print(f"connected: {self.args.robot_port} | joints(deg/percent)={self._format_positions(self.last_positions)}")
        if self.args.control_mode == "ik":
            print(f"frame map: {self._format_direction_map()}")
            print(f"tcp offset xyz (m): {_fmt_xyz(self.tcp_offset_xyz.tolist())}")

    def disconnect(self) -> None:
        if self.args.dry_run:
            print("[dry-run] robot disconnect skipped")
            return
        if self.robot is not None and self.robot.is_connected:
            self.robot.disconnect()
            print("robot disconnected")

    def _read_positions(self) -> dict[str, float]:
        if self.args.dry_run:
            return dict(self.last_positions)
        assert self.robot is not None
        obs = self.robot.get_observation()
        positions: dict[str, float] = {}
        for joint in JOINT_NAMES:
            key = f"{joint}.pos"
            if key not in obs:
                raise RuntimeError(f"Missing observation key: {key}")
            positions[joint] = float(obs[key])
        return positions

    def _send_positions(self, target_positions: dict[str, float]) -> dict[str, float]:
        action = {f"{joint}.pos": float(target_positions[joint]) for joint in JOINT_NAMES}
        if self.args.dry_run:
            self.last_positions = dict(target_positions)
            print(f"[dry-run] send {self._format_positions(self.last_positions)}")
            return dict(action)
        assert self.robot is not None
        sent = self.robot.send_action(action)
        if self.args.settle_s > 0:
            time.sleep(self.args.settle_s)
        self.last_positions = {joint: float(sent[f"{joint}.pos"]) for joint in JOINT_NAMES}
        return sent

    def _format_positions(self, positions: dict[str, float]) -> str:
        return " ".join(f"{joint}={positions[joint]:.2f}" for joint in JOINT_NAMES)

    def _clamp_delta(self, delta_deg: float) -> float:
        limit = abs(self.args.max_command_delta_deg)
        if delta_deg > limit:
            return limit
        if delta_deg < -limit:
            return -limit
        return delta_deg

    def _apply_joint_delta(self, joint: str, delta_deg: float) -> None:
        if joint not in JOINT_NAMES:
            raise ValueError(f"Unknown joint: {joint}")
        self.last_positions = self._read_positions()
        bounded = self._clamp_delta(delta_deg)
        target = dict(self.last_positions)
        target[joint] = target[joint] + bounded
        if joint == "gripper":
            target[joint] = min(max(target[joint], 0.0), 100.0)
        sent = self._send_positions(target)
        print(f"ok {joint} delta={bounded:.2f} -> {sent[f'{joint}.pos']:.2f}")

    def _apply_joint_set(self, joint: str, value: float) -> None:
        if joint not in JOINT_NAMES:
            raise ValueError(f"Unknown joint: {joint}")
        self.last_positions = self._read_positions()
        target = dict(self.last_positions)
        target[joint] = float(value)
        if joint == "gripper":
            target[joint] = min(max(target[joint], 0.0), 100.0)
        sent = self._send_positions(target)
        print(f"ok {joint} set={target[joint]:.2f} -> {sent[f'{joint}.pos']:.2f}")

    def _apply_axis_move(self, direction: str, cm: float) -> None:
        if direction not in self.axis_bindings:
            raise ValueError(f"Unknown direction: {direction}")
        if self.args.control_mode == "ik":
            # User-facing axis commands in IK mode should be closed-loop for stable direction and distance.
            self._apply_axis_move_exact_ik(direction, cm)
            return
        binding = self.axis_bindings[direction]
        delta_deg = cm * self.args.cm_to_deg * binding.sign
        self._apply_joint_delta(binding.joint, delta_deg)

    def _run_axis_move_exact_ik(
        self,
        direction: str,
        cm: float,
        *,
        verbose: bool = True,
        allow_auto_flip: bool | None = None,
    ) -> IKMoveResult:
        if self.kinematics is None:
            raise RuntimeError("IK mode requested but kinematics is not initialized.")
        target_m = abs(float(cm)) * 0.01
        if target_m <= 0:
            raise ValueError("exact move requires non-zero distance")

        direction_sign = 1.0 if float(cm) >= 0 else -1.0
        requested_axis = self._axis_unit(direction, cm_sign=direction_sign)
        solver_axis = np.array(requested_axis, copy=True)
        auto_flip_enabled = self.args.ik_auto_flip_axis if allow_auto_flip is None else bool(allow_auto_flip)
        verbose_steps = bool(verbose and self.args.ik_verbose_steps)
        min_progress_m = float(self.args.ik_min_step_progress_mm) / 1000.0
        target_tol_m = float(self.args.ik_target_tol_mm) / 1000.0
        stuck_steps_limit = max(int(self.args.ik_stuck_steps), 1)

        self.last_positions = self._read_positions()
        start_xyz = self._ee_xyz_from_positions(self.last_positions)
        iterations = 0
        flipped_count = 0
        low_progress_steps = 0
        stalled = False
        converged = False

        for iterations in range(1, int(self.args.ik_max_iters) + 1):
            current_positions = self._read_positions()
            current_xyz = self._ee_xyz_from_positions(current_positions)
            achieved_m = float(np.dot(current_xyz - start_xyz, requested_axis))
            remaining_m = target_m - achieved_m
            if remaining_m <= target_tol_m:
                converged = True
                break

            step_m = min(float(self.args.ik_exact_step_m), remaining_m)
            step_delta = solver_axis * step_m
            q_current = self._q_from_positions(current_positions)
            if self.args.ik_solver == "placo":
                q_target = self._ik_pose_step(q_current, step_delta)
            else:
                q_target = self._ik_dls_step(q_current, step_delta)
            before_xyz = self._fk_xyz_from_q(q_current)
            self._send_q_target(q_target)
            after_positions = self._read_positions()
            q_after = self._q_from_positions(after_positions)
            after_xyz = self._fk_xyz_from_q(q_after)
            step_vec = after_xyz - before_xyz
            step_progress_solver = float(np.dot(step_vec, solver_axis))
            step_progress_requested = float(np.dot(step_vec, requested_axis))
            lateral_m = float(np.linalg.norm(step_vec - (step_progress_requested * requested_axis)))

            if verbose_steps:
                print(
                    "ok ik(dls) "
                    f"{direction} {step_m*100.0:g}cm | "
                    f"ee_xyz_before={_fmt_xyz(before_xyz.tolist())} ee_xyz_after={_fmt_xyz(after_xyz.tolist())} | "
                    f"step_proj_cmd_cm={step_progress_requested*100.0:.2f} "
                    f"step_proj_solver_cm={step_progress_solver*100.0:.2f} "
                    f"step_lateral_cm={lateral_m*100.0:.2f}"
                )

            if step_progress_requested < min_progress_m:
                low_progress_steps += 1
            else:
                low_progress_steps = 0

            if (
                step_progress_requested < 0.0
                and auto_flip_enabled
                and flipped_count < int(self.args.ik_max_auto_flips)
            ):
                solver_axis = -solver_axis
                flipped_count += 1
                low_progress_steps = 0
                if verbose:
                    print("warn: auto-flip IK solver axis due reverse progress")

            if low_progress_steps >= stuck_steps_limit:
                stalled = True
                if verbose:
                    print(
                        "warn: IK stalled "
                        f"(low progress {low_progress_steps} steps, threshold={min_progress_m*100.0:.2f}cm/step)"
                    )
                break

        end_positions = self._read_positions()
        end_xyz = self._ee_xyz_from_positions(end_positions)
        achieved_final = float(np.dot(end_xyz - start_xyz, requested_axis))
        result = IKMoveResult(
            direction=direction,
            target_m=target_m,
            achieved_m=achieved_final,
            iterations=iterations,
            flipped_count=flipped_count,
            stalled=stalled,
            converged=converged,
            start_xyz=start_xyz,
            end_xyz=end_xyz,
        )
        if verbose:
            status = "converged" if result.converged else ("stalled" if result.stalled else "partial")
            print(
                "ok ik exact "
                f"{direction} {cm:g}cm | achieved={result.achieved_m*100.0:.2f}cm "
                f"target={result.target_m*100.0:.2f}cm iters={result.iterations} "
                f"flips={result.flipped_count} status={status} "
                f"start={_fmt_xyz(result.start_xyz.tolist())} end={_fmt_xyz(result.end_xyz.tolist())}"
            )
        return result

    def _apply_axis_move_ik(self, direction: str, cm: float) -> None:
        self._run_axis_move_exact_ik(direction, cm, verbose=True)

    def _apply_axis_move_exact_ik(self, direction: str, cm: float) -> None:
        self._run_axis_move_exact_ik(direction, cm, verbose=True)

    def _calibrate_direction_signs(self, probe_cm: float) -> None:
        if self.args.control_mode != "ik":
            raise ValueError("`calibrate` requires --control-mode ik")
        if probe_cm <= 0:
            raise ValueError("calibrate probe distance must be > 0")

        min_good_m = float(self.args.calibration_min_progress_mm) / 1000.0
        improve_margin_m = float(self.args.calibration_improve_margin_mm) / 1000.0
        print(
            "calibrate start "
            f"probe={probe_cm:g}cm min_good={min_good_m*100.0:.2f}cm "
            f"map={self._format_direction_map()}"
        )

        for lead in ("up", "left", "forward"):
            opposite = DIRECTION_OPPOSITES[lead]
            before = self._run_axis_move_exact_ik(lead, probe_cm, verbose=False, allow_auto_flip=False)
            # Try to go back close to starting pose before evaluating flipped sign.
            self._run_axis_move_exact_ik(opposite, probe_cm, verbose=False, allow_auto_flip=False)

            self._flip_direction_pair(lead)
            after = self._run_axis_move_exact_ik(lead, probe_cm, verbose=False, allow_auto_flip=False)
            self._run_axis_move_exact_ik(opposite, probe_cm, verbose=False, allow_auto_flip=False)

            keep_flipped = (
                after.achieved_m >= min_good_m and after.achieved_m > (before.achieved_m + improve_margin_m)
            )
            if not keep_flipped:
                self._flip_direction_pair(lead)

            decision = "flip" if keep_flipped else "keep"
            print(
                f"calibrate {lead}: before={before.achieved_m*100.0:.2f}cm "
                f"after={after.achieved_m*100.0:.2f}cm -> {decision}"
            )

        if self.args.calibration_auto_save:
            self._save_direction_map()
        print(f"calibrate done map={self._format_direction_map()}")

    def _handle_frame_command(self, tokens: list[str]) -> None:
        if len(tokens) == 1 or tokens[1].lower() in {"show", "ls"}:
            print(f"frame map: {self._format_direction_map()}")
            return

        sub = tokens[1].lower()
        if sub == "reset":
            self.direction_vectors = self._default_direction_vectors()
            print(f"frame map reset: {self._format_direction_map()}")
            return
        if sub == "save":
            self._save_direction_map()
            print(f"frame map: {self._format_direction_map()}")
            return
        if sub == "load":
            self.direction_vectors = self._default_direction_vectors()
            self._load_direction_map_if_exists()
            print(f"frame map: {self._format_direction_map()}")
            return
        if sub == "set":
            if len(tokens) != 4:
                raise ValueError("Usage: frame set <direction> <x+|x-|y+|y-|z+|z->")
            direction = tokens[2].lower()
            if direction not in CARTESIAN_DIRECTIONS:
                raise ValueError(f"Unknown direction: {direction}")
            axis_token = _normalize_axis_token(tokens[3])
            self._set_direction_vector(direction, AXIS_TOKEN_VECTORS[axis_token])
            print(f"frame map updated: {self._format_direction_map()}")
            return
        raise ValueError("Usage: frame [show|set <direction> <axis>|reset|save|load]")

    def _set_gripper_open(self) -> None:
        self._apply_joint_set("gripper", self.args.gripper_open_value)

    def _set_gripper_close(self) -> None:
        self._apply_joint_set("gripper", self.args.gripper_close_value)

    def _go_home(self) -> None:
        if self.home_positions is None:
            raise RuntimeError("Home position unavailable.")
        self._send_positions(dict(self.home_positions))
        print("ok home")

    def execute(self, command_line: str) -> bool:
        line = command_line.strip()
        if not line:
            return True
        tokens = shlex.split(line)
        cmd = tokens[0].lower()

        if cmd in {"q", "quit", "exit"}:
            return False
        if cmd in {"h", "help"}:
            print(HELP_TEXT)
            return True
        if cmd in {"obs", "state"}:
            self.last_positions = self._read_positions()
            print(self._format_positions(self.last_positions))
            return True
        if cmd == "frame":
            self._handle_frame_command(tokens)
            return True
        if cmd == "calibrate":
            probe_cm = float(tokens[1]) if len(tokens) > 1 else float(self.args.calibration_probe_cm)
            self._calibrate_direction_signs(probe_cm)
            return True
        if cmd == "home":
            self._go_home()
            return True
        if cmd == "open":
            self._set_gripper_open()
            return True
        if cmd == "close":
            self._set_gripper_close()
            return True
        if cmd.endswith("_exact"):
            if self.args.control_mode != "ik":
                raise ValueError("`*_exact` commands require --control-mode ik")
            direction = cmd.removesuffix("_exact")
            if direction not in CARTESIAN_DIRECTIONS:
                raise ValueError(f"Unknown exact direction: {direction}")
            cm = float(tokens[1]) if len(tokens) > 1 else self.args.default_cm
            self._apply_axis_move_exact_ik(direction, cm)
            return True
        if cmd in CARTESIAN_DIRECTIONS:
            cm = float(tokens[1]) if len(tokens) > 1 else self.args.default_cm
            self._apply_axis_move(cmd, cm)
            return True
        if cmd == "j":
            if len(tokens) != 3:
                raise ValueError("Usage: j <joint> <delta_deg>")
            joint = _normalize_joint(tokens[1])
            delta_deg = float(tokens[2])
            self._apply_joint_delta(joint, delta_deg)
            return True
        if cmd == "set":
            if len(tokens) != 3:
                raise ValueError("Usage: set <joint> <value>")
            joint = _normalize_joint(tokens[1])
            value = float(tokens[2])
            self._apply_joint_set(joint, value)
            return True
        if cmd == "sleep":
            if len(tokens) != 2:
                raise ValueError("Usage: sleep <seconds>")
            time.sleep(float(tokens[1]))
            return True

        raise ValueError(f"Unknown command: {cmd}")

    def run(self) -> None:
        self.connect()
        try:
            if self.args.cmd:
                command_items = [item.strip() for item in self.args.cmd.split(";") if item.strip()]
                for item in command_items:
                    try:
                        keep_running = self.execute(item)
                    except Exception as exc:  # noqa: BLE001
                        print(f"error: {exc}")
                        break
                    if not keep_running:
                        break
                return

            print(HELP_TEXT)
            while True:
                try:
                    line = input("so101> ")
                except EOFError:
                    break
                try:
                    keep_running = self.execute(line)
                    if not keep_running:
                        break
                except Exception as exc:  # noqa: BLE001
                    print(f"error: {exc}")
        finally:
            self.disconnect()


HELP_TEXT = """Commands:
  obs | state
  frame [show|set <direction> <x+|x-|y+|y-|z+|z->|reset|save|load]
  calibrate [cm]                     # ik mode only, auto-fix direction signs
  up [cm] | down [cm] | left [cm] | right [cm] | forward [cm] | back [cm]
  up_exact [cm] | down_exact [cm] | left_exact [cm] | right_exact [cm]
  forward_exact [cm] | back_exact [cm]    # ik mode only, closed-loop
  open | close
  j <joint> <delta_deg>            # joint delta command
  set <joint> <value>              # absolute joint target
  home
  sleep <seconds>
  help
  quit | exit

Joint names:
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Notes:
  - with --control-mode joint, up/left/... are approximate joint deltas.
  - with --control-mode ik, up/left/... run closed-loop IK in Cartesian space.
  - `--tcp-offset-xyz` lets you shift control from the URDF frame to the real tool center point.
  - `frame set` remaps user directions to Cartesian axes.
  - `frame save` persists mapping to --direction-map-file.
  - Safety:
    * per-command delta is clamped by --max-command-delta-deg
    * robot-level relative limit uses --max-relative-target-deg
"""


def _normalize_joint(raw: str) -> str:
    joint = raw.strip().lower()
    aliases = {
        "pan": "shoulder_pan",
        "lift": "shoulder_lift",
        "elbow": "elbow_flex",
        "wflex": "wrist_flex",
        "wroll": "wrist_roll",
        "grip": "gripper",
    }
    return aliases.get(joint, joint)


def _normalize_axis_token(raw: str) -> str:
    token = raw.strip().lower()
    if token not in AXIS_TOKEN_ALIASES:
        valid = ", ".join(sorted(AXIS_TOKEN_VECTORS))
        raise ValueError(f"Unknown axis token: {raw}. Expected one of: {valid}")
    return AXIS_TOKEN_ALIASES[token]


def _parse_vector3(value: str) -> list[float]:
    pieces = [piece for piece in value.replace("[", "").replace("]", "").replace(",", " ").split() if piece]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 numeric values, got {value!r}")
    try:
        return [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected numeric values, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SO101 manual command console (safe, incremental control).")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument(
        "--calibration-dir",
        default="/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower",
    )
    parser.add_argument("--calibrate-on-connect", action="store_true")
    parser.add_argument("--control-mode", choices=("joint", "ik"), default="joint")
    parser.add_argument("--ik-solver", choices=("placo", "dls"), default="placo")
    parser.add_argument("--urdf-path", default=str(REPO_ROOT / "so101_new_calib.urdf"))
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument(
        "--tcp-offset-xyz",
        type=_parse_vector3,
        default=[0.0, 0.0, 0.0],
        help="Optional TCP offset in target frame coordinates, in meters.",
    )
    parser.add_argument("--ik-position-weight", type=float, default=1.0)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.0)
    parser.add_argument(
        "--direction-map-file",
        default=str(REPO_ROOT / "tfj_envs" / "so101_control_pause_20260319" / "runtime" / "so101_direction_map.json"),
        help="JSON file storing user direction -> Cartesian axis mapping.",
    )
    parser.add_argument("--ik-exact-step-m", type=float, default=0.004, help="Per-iteration Cartesian step for *_exact.")
    parser.add_argument("--ik-target-tol-mm", type=float, default=1.0, help="Stop threshold for *_exact commands.")
    parser.add_argument("--ik-max-iters", type=int, default=60, help="Max IK iterations for *_exact commands.")
    parser.add_argument("--ik-jacobian-eps-deg", type=float, default=0.5, help="Finite-difference epsilon for Jacobian.")
    parser.add_argument("--ik-damping", type=float, default=0.002, help="Damping factor for DLS Jacobian solver.")
    parser.add_argument("--ik-verbose-steps", action="store_true", help="Print per-step IK progress logs.")
    parser.add_argument(
        "--ik-min-step-progress-mm",
        type=float,
        default=0.05,
        help="Per-step projected progress threshold below which the move is considered low-progress.",
    )
    parser.add_argument(
        "--ik-stuck-steps",
        type=int,
        default=4,
        help="Abort exact IK move if projected progress is below threshold for N consecutive steps.",
    )
    parser.add_argument(
        "--ik-max-auto-flips",
        type=int,
        default=1,
        help="Maximum automatic solver-axis flips when reverse progress is detected.",
    )
    parser.add_argument(
        "--ik-auto-flip-axis",
        dest="ik_auto_flip_axis",
        action="store_true",
        help="Allow automatic solver-axis flip when reverse progress is detected.",
    )
    parser.add_argument(
        "--no-ik-auto-flip-axis",
        dest="ik_auto_flip_axis",
        action="store_false",
        help="Disable automatic solver-axis flip (recommended for deterministic user direction semantics).",
    )
    parser.set_defaults(ik_auto_flip_axis=False)

    parser.add_argument("--calibration-probe-cm", type=float, default=0.6)
    parser.add_argument("--calibration-min-progress-mm", type=float, default=0.5)
    parser.add_argument("--calibration-improve-margin-mm", type=float, default=0.3)
    parser.add_argument(
        "--calibration-auto-save",
        action="store_true",
        help="Automatically save direction map after `calibrate` command.",
    )

    parser.add_argument("--cm-to-deg", type=float, default=3.0, help="Approx conversion gain for axis commands.")
    parser.add_argument("--default-cm", type=float, default=1.0)
    parser.add_argument("--max-command-delta-deg", type=float, default=8.0)
    parser.add_argument(
        "--max-relative-target-deg",
        type=float,
        default=8.0,
        help="Robot-level safety clamp passed to SO101FollowerConfig.max_relative_target.",
    )
    parser.add_argument(
        "--gripper-max-relative-target",
        type=float,
        default=40.0,
        help="Optional gripper-specific safety clamp. Set to 0 or negative to disable override.",
    )
    parser.add_argument("--settle-s", type=float, default=0.15, help="Wait time after each command send.")

    parser.add_argument("--sign-up", type=float, default=1.0)
    parser.add_argument("--sign-left", type=float, default=1.0)
    parser.add_argument("--sign-forward", type=float, default=1.0)
    parser.add_argument("--sign-cartesian-x", type=float, default=1.0)
    parser.add_argument("--sign-cartesian-y", type=float, default=1.0)
    parser.add_argument("--sign-cartesian-z", type=float, default=1.0)

    parser.add_argument("--gripper-open-value", type=float, default=100.0)
    parser.add_argument("--gripper-close-value", type=float, default=0.0)

    parser.add_argument("--cmd", type=str, default="", help="One-shot commands separated by ';'")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gripper_max_relative_target is not None and args.gripper_max_relative_target <= 0:
        args.gripper_max_relative_target = None
    console = ManualConsole(args)
    console.run()
    return 0


def _fmt_xyz(xyz: list[float]) -> str:
    return f"[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]"


if __name__ == "__main__":
    raise SystemExit(main())
