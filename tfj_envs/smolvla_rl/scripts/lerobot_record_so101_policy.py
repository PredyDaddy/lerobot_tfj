#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SO101 policy-on-robot recording entrypoint.

This is a thin SO101-focused wrapper around `lerobot_record.py`.

It keeps the same LeRobot recording loop, but replaces the large generic CLI
surface with a smaller interface tailored for:

- `so101_follower`
- top + wrist OpenCV cameras
- optional `so101_leader`
- policy-controlled execution with optional eval dataset recording

Typical usage:

```bash
python src/lerobot/scripts/lerobot_record_so101_policy.py \
  --policy.path=/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --robot_port=/dev/ttyACM0 \
  --task="Put the block in the bin" \
  --dataset_root=./outputs/eval_smolvla_so101 \
  --num_episodes=1 \
  --episode_time_s=300
```

If `--policy.path` points to a SmolVLA checkpoint, the wrapper automatically
applies the image-key rename map:

- `observation.images.top -> observation.images.camera1`
- `observation.images.wrist -> observation.images.camera2`
"""

import json
import logging
import math
import os
import shutil
import sys
import time
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    for parent in Path(__file__).resolve().parents:
        repo_src = parent / "src"
        if (repo_src / "lerobot").is_dir():
            repo_src_str = str(repo_src)
            if repo_src_str not in sys.path:
                sys.path.insert(0, repo_src_str)
            break

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    record as base_record,
    record_loop,
)
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging, log_say

try:
    from lerobot.runtime.so101_pickplace.schemas import GuardResult
except ModuleNotFoundError:
    class _LocalGuardDecision(Enum):
        ACCEPT = "accept"
        CLAMP_AND_ACCEPT = "clamp_and_accept"
        REJECT = "reject"
        HALT = "halt"

    @dataclass(frozen=True)
    class GuardResult:  # fallback for environments without so101_pickplace runtime package
        decision: _LocalGuardDecision
        action: dict[str, Any] | None = None
        error_code: str | None = None
        reason: str | None = None
        fail_safe_action: dict[str, Any] | None = None
        details: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def accept(cls, action: dict[str, Any], *, details: dict[str, Any] | None = None) -> "GuardResult":
            return cls(decision=_LocalGuardDecision.ACCEPT, action=action, details=details or {})

        @classmethod
        def reject(
            cls,
            *,
            error_code: str,
            reason: str,
            fail_safe_action: dict[str, Any] | None = None,
            details: dict[str, Any] | None = None,
        ) -> "GuardResult":
            return cls(
                decision=_LocalGuardDecision.REJECT,
                error_code=error_code,
                reason=reason,
                fail_safe_action=fail_safe_action,
                details=details or {},
            )

        @classmethod
        def halt(
            cls,
            *,
            error_code: str,
            reason: str,
            fail_safe_action: dict[str, Any] | None = None,
            details: dict[str, Any] | None = None,
        ) -> "GuardResult":
            return cls(
                decision=_LocalGuardDecision.HALT,
                error_code=error_code,
                reason=reason,
                fail_safe_action=fail_safe_action,
                details=details or {},
            )


SAFETY_PROFILE_OFF = "off"
SAFETY_PROFILE_DEFAULT = "default"
SAFETY_PROFILE_STRICT = "strict"
SUPPORTED_SAFETY_PROFILES = {
    SAFETY_PROFILE_OFF,
    SAFETY_PROFILE_DEFAULT,
    SAFETY_PROFILE_STRICT,
}


@dataclass
class BridgeDecision:
    action_delta: dict[str, float] = field(default_factory=dict)
    task_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SO101RuntimeState:
    step_index: int = 0
    consecutive_rejects: int = 0
    halted: bool = False
    last_task: str = ""
    last_reason: str | None = None
    intent: dict[str, Any] = field(default_factory=dict)
    safety_profile: str = SAFETY_PROFILE_OFF


@dataclass
class SO101PickPlaceRuntime:
    task: str
    intent: dict[str, Any]
    bridge: Any | None
    guard: Any | None
    logger: Any | None
    state: SO101RuntimeState | None


class IntentBridge:
    def __init__(self, intent: dict[str, Any], fallback_task: str):
        self.intent = intent
        self.fallback_task = fallback_task

    def decide(self, **_: Any) -> BridgeDecision:
        action_delta = self.intent.get("action_delta", {})
        if not isinstance(action_delta, dict):
            action_delta = {}

        task_override = _resolve_task_text(self.intent, self.fallback_task)
        if task_override == self.fallback_task:
            task_override = None

        return BridgeDecision(
            action_delta={key: float(value) for key, value in action_delta.items()},
            task_override=task_override,
            metadata={"intent": self.intent},
        )


class ProfileSafetyGuard:
    _PROFILE_LIMITS = {
        SAFETY_PROFILE_DEFAULT: {
            "max_abs_action": 180.0,
            "max_step_delta": 45.0,
            # Dataset-informed per-joint envelope from
            # /home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1.
            # These values are based on the observed max |action - observation|
            # per joint, rounded up with a small margin where needed.
            "max_step_delta_overrides": {
                "shoulder_pan.pos": 45.0,
                "shoulder_lift.pos": 45.0,
                "elbow_flex.pos": 45.0,
                "wrist_flex.pos": 45.0,
                "wrist_roll.pos": 50.0,
                "gripper.pos": 80.0,
            },
            "max_consecutive_rejects": 3,
        },
        SAFETY_PROFILE_STRICT: {
            "max_abs_action": 120.0,
            "max_step_delta": 25.0,
            "max_consecutive_rejects": 2,
        },
    }

    def __init__(self, profile: str):
        self.profile = _normalize_safety_profile(profile)
        if self.profile == SAFETY_PROFILE_OFF:
            raise ValueError("ProfileSafetyGuard cannot be created with safety_profile=off.")
        self.limits = self._PROFILE_LIMITS[self.profile]

    def validate(self, action: dict[str, Any], obs: dict[str, Any], runtime_state: Any | None = None, **_: Any):
        reject_reason = None
        reject_metadata: dict[str, Any] = {}
        halt = False
        fail_safe_action = _hold_position_action(obs)

        for key, value in action.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                reject_reason = "non_finite_action"
                reject_metadata = {"joint": key, "value": value}
                halt = True
                break

            if abs(float(value)) > self.limits["max_abs_action"]:
                reject_reason = "action_limit_exceeded"
                reject_metadata = {
                    "joint": key,
                    "value": float(value),
                    "limit": self.limits["max_abs_action"],
                }
                halt = True
                break

            obs_value = obs.get(key)
            if isinstance(obs_value, (int, float)) and math.isfinite(float(obs_value)):
                delta = abs(float(value) - float(obs_value))
                max_step_delta = self.limits.get("max_step_delta_overrides", {}).get(key, self.limits["max_step_delta"])
                if delta > max_step_delta:
                    reject_reason = "step_delta_exceeded"
                    reject_metadata = {
                        "joint": key,
                        "value": float(value),
                        "obs_value": float(obs_value),
                        "delta": delta,
                        "limit": max_step_delta,
                    }
                    break

        if reject_reason is None:
            if runtime_state is not None:
                _set_runtime_state_value(runtime_state, "consecutive_rejects", 0)
                _set_runtime_state_value(runtime_state, "last_reason", None)
            return GuardResult.accept(dict(action), details={"profile": self.profile})

        consecutive_rejects = 1
        if runtime_state is not None:
            consecutive_rejects = _get_runtime_state_value(runtime_state, "consecutive_rejects", 0) + 1
            _set_runtime_state_value(runtime_state, "consecutive_rejects", consecutive_rejects)
            _set_runtime_state_value(runtime_state, "last_reason", reject_reason)

        if consecutive_rejects >= self.limits["max_consecutive_rejects"]:
            halt = True
            reject_metadata["consecutive_rejects"] = consecutive_rejects

        details = {"profile": self.profile, **reject_metadata}
        if halt:
            return GuardResult.halt(
                error_code=reject_reason or "guard_halt",
                reason=reject_reason or "Guard halted action",
                fail_safe_action=fail_safe_action,
                details=details,
            )
        return GuardResult.reject(
            error_code=reject_reason or "guard_reject",
            reason=reject_reason or "Guard rejected action",
            fail_safe_action=fail_safe_action,
            details=details,
        )


class JsonlStepLogger:
    def __init__(self, path: Path | None, base_context: dict[str, Any]):
        self.path = path
        self.base_context = dict(base_context)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, **payload: Any) -> None:
        if self.path is None:
            return
        record = {
            "schema_version": "so101_pickplace.v1",
            "event": event,
            "timestamp": time.time(),
            **self.base_context,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_event(self, event: str, **payload: Any) -> None:
        self._write(event, **payload)

    def log_run_start(self, **payload: Any) -> None:
        self._write("run_start", **payload)

    def log_run_end(self, **payload: Any) -> None:
        self._write("run_end", **payload)

    def log_step(self, **payload: Any) -> None:
        self._write("step", **payload)

    def log_guard_reject(self, **payload: Any) -> None:
        self._write("guard_reject", **payload)


class _DryRunRobot(Robot):
    config_class = RobotConfig
    name = "so101_dry_run"

    def __init__(self, calibration_dir: Path):
        super().__init__(RobotConfig(id="dry_run_robot", calibration_dir=calibration_dir))
        self._is_connected = False
        self.cameras: dict[str, Any] = {}
        self._state = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
        }

    @property
    def observation_features(self) -> dict[str, type]:
        return {key: float for key in self._state}

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._state}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        return dict(self._state)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for key, value in action.items():
            if isinstance(value, (int, float)):
                self._state[key] = float(value)
        return dict(action)

    def disconnect(self) -> None:
        self._is_connected = False


class _DryRunTeleop(Teleoperator):
    config_class = TeleoperatorConfig
    name = "so101_dry_run_teleop"

    def __init__(self, calibration_dir: Path, action_template: dict[str, float] | None = None):
        super().__init__(TeleoperatorConfig(id="dry_run_teleop", calibration_dir=calibration_dir))
        self._is_connected = False
        template = action_template or {}
        self._action = {
            "joint_1.pos": float(template.get("joint_1.pos", 0.0)),
            "joint_2.pos": float(template.get("joint_2.pos", 0.0)),
            "joint_3.pos": float(template.get("joint_3.pos", 0.0)),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._action}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {key: float for key in self._action}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, Any]:
        return dict(self._action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        self._is_connected = False


def _default_robot_calibration_dir() -> Path:
    return Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")


def _default_leader_calibration_dir() -> Path:
    return Path("/home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader")


def _default_smolvla_rename_map() -> dict[str, str]:
    return {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }


def _normalize_safety_profile(profile: str) -> str:
    normalized = (profile or SAFETY_PROFILE_OFF).strip().lower()
    if normalized not in SUPPORTED_SAFETY_PROFILES:
        raise ValueError(
            f"Unsupported safety_profile={profile!r}. Expected one of {sorted(SUPPORTED_SAFETY_PROFILES)}."
        )
    return normalized


def _parse_intent_payload(cfg: "SO101PolicyRecordConfig") -> dict[str, Any]:
    payload: dict[str, Any]
    if cfg.intent_json:
        payload = json.loads(cfg.intent_json)
        if not isinstance(payload, dict):
            raise ValueError("`intent_json` must decode to a JSON object.")
    elif cfg.intent_text:
        payload = {"task_text": cfg.intent_text}
    else:
        payload = {"task_text": cfg.task}

    if cfg.intent_text and "task_text" not in payload:
        payload["task_text"] = cfg.intent_text
    if cfg.task and "task_text" not in payload:
        payload["task_text"] = cfg.task
    return payload


def _resolve_task_text(intent: dict[str, Any], fallback_task: str) -> str:
    task_text = intent.get("task") or intent.get("task_text") or fallback_task
    return str(task_text)


def _get_runtime_state_value(runtime_state: Any, key: str, default: Any = None) -> Any:
    if runtime_state is None:
        return default
    if isinstance(runtime_state, dict):
        return runtime_state.get(key, default)
    return getattr(runtime_state, key, default)


def _set_runtime_state_value(runtime_state: Any, key: str, value: Any) -> None:
    if runtime_state is None:
        return
    if isinstance(runtime_state, dict):
        runtime_state[key] = value
    else:
        setattr(runtime_state, key, value)


def _hold_position_action(obs: dict[str, Any]) -> dict[str, float] | None:
    hold_action = {
        key: float(value)
        for key, value in obs.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    return hold_action or None


@dataclass
class SO101PolicyRecordConfig:
    policy: PreTrainedConfig | None = None

    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "my_so101"
    robot_calibration_dir: Path = field(default_factory=_default_robot_calibration_dir)
    robot_max_relative_target: float | dict[str, float] | None = None
    robot_disable_torque_on_disconnect: bool = True
    robot_use_degrees: bool = False

    top_camera_index: int = 4
    wrist_camera_index: int = 6
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_warmup_s: int = 1

    task: str = "Put the block in the bin"
    intent_text: str = ""
    intent_json: str = ""
    enable_perception_bridge: bool = False
    safety_profile: str = SAFETY_PROFILE_OFF
    events_jsonl_path: Path | None = None
    dry_run: bool = False
    dataset_repo_id: str = "local/eval_so101_policy"
    dataset_root: Path = Path("./outputs/eval_so101_policy")
    dataset_fps: int = 30
    episode_time_s: float = 300.0
    reset_time_s: float = 15.0
    num_episodes: int = 1
    dataset_video: bool = True
    dataset_push_to_hub: bool = False
    dataset_private: bool = False
    dataset_tags: list[str] | None = None
    dataset_num_image_writer_processes: int = 0
    dataset_num_image_writer_threads_per_camera: int = 4
    dataset_video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)
    auto_rename_for_smolvla: bool = True

    leader_port: str | None = None
    leader_id: str = "so101_leader"
    leader_calibration_dir: Path = field(default_factory=_default_leader_calibration_dir)
    leader_use_degrees: bool = False

    display_data: bool = False
    play_sounds: bool = False
    resume: bool = False
    clear_dataset_root: bool = False
    save_dataset: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None and self.leader_port is None and not self.dry_run:
            raise ValueError("Provide `--policy.path=...` or `--leader_port=...` (or both).")

        if self.resume and self.clear_dataset_root:
            raise ValueError("`resume=true` and `clear_dataset_root=true` cannot be used together.")

        if self.policy is not None and self.auto_rename_for_smolvla and not self.rename_map:
            if self.policy.type == "smolvla":
                self.rename_map = _default_smolvla_rename_map()

        if self.dataset_repo_id == "local/eval_so101_policy" and self.policy is not None:
            self.dataset_repo_id = f"local/eval_{self.policy.type}_so101"

        if self.dataset_root == Path("./outputs/eval_so101_policy") and self.policy is not None:
            self.dataset_root = Path(f"./outputs/eval_{self.policy.type}_so101")

        self.safety_profile = _normalize_safety_profile(self.safety_profile)
        if self.events_jsonl_path is not None:
            self.events_jsonl_path = Path(self.events_jsonl_path)
        elif self.save_dataset:
            self.events_jsonl_path = self.dataset_root.parent / f"{self.dataset_root.name}.events.jsonl"
        else:
            self.events_jsonl_path = None

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _build_robot_config(cfg: SO101PolicyRecordConfig) -> SO101FollowerConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=cfg.top_camera_index,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
            warmup_s=cfg.camera_warmup_s,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=cfg.wrist_camera_index,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
            warmup_s=cfg.camera_warmup_s,
        ),
    }
    return SO101FollowerConfig(
        port=cfg.robot_port,
        id=cfg.robot_id,
        calibration_dir=cfg.robot_calibration_dir,
        disable_torque_on_disconnect=cfg.robot_disable_torque_on_disconnect,
        max_relative_target=cfg.robot_max_relative_target,
        cameras=cameras,
        use_degrees=cfg.robot_use_degrees,
    )


def _build_dataset_config(cfg: SO101PolicyRecordConfig, task_text: str) -> DatasetRecordConfig:
    return DatasetRecordConfig(
        repo_id=cfg.dataset_repo_id,
        single_task=task_text,
        root=cfg.dataset_root,
        fps=cfg.dataset_fps,
        episode_time_s=cfg.episode_time_s,
        reset_time_s=cfg.reset_time_s,
        num_episodes=cfg.num_episodes,
        video=cfg.dataset_video,
        push_to_hub=cfg.dataset_push_to_hub,
        private=cfg.dataset_private,
        tags=cfg.dataset_tags,
        num_image_writer_processes=cfg.dataset_num_image_writer_processes,
        num_image_writer_threads_per_camera=cfg.dataset_num_image_writer_threads_per_camera,
        video_encoding_batch_size=cfg.dataset_video_encoding_batch_size,
        rename_map=cfg.rename_map,
    )


def _build_teleop_config(cfg: SO101PolicyRecordConfig) -> SO101LeaderConfig | None:
    if cfg.leader_port is None:
        return None
    return SO101LeaderConfig(
        port=cfg.leader_port,
        id=cfg.leader_id,
        calibration_dir=cfg.leader_calibration_dir,
        use_degrees=cfg.leader_use_degrees,
    )


def _build_record_config(cfg: SO101PolicyRecordConfig, task_text: str) -> RecordConfig:
    return RecordConfig(
        robot=_build_robot_config(cfg),
        dataset=_build_dataset_config(cfg, task_text),
        teleop=_build_teleop_config(cfg),
        policy=cfg.policy,
        display_data=cfg.display_data,
        play_sounds=cfg.play_sounds,
        resume=cfg.resume,
    )


def build_so101_pickplace_runtime(
    cfg: SO101PolicyRecordConfig,
    *,
    require_guard: bool = False,
) -> SO101PickPlaceRuntime:
    intent = _parse_intent_payload(cfg)
    task_text = _resolve_task_text(intent, cfg.task)

    if require_guard and not cfg.dry_run and cfg.safety_profile == SAFETY_PROFILE_OFF:
        raise ValueError("Real-robot pick-place runs require `safety_profile` to be `default` or `strict`.")

    bridge = IntentBridge(intent, task_text) if cfg.enable_perception_bridge else None
    guard = None if cfg.safety_profile == SAFETY_PROFILE_OFF else ProfileSafetyGuard(cfg.safety_profile)
    logger = None
    if cfg.events_jsonl_path is not None:
        logger = JsonlStepLogger(
            cfg.events_jsonl_path,
            {
                "dry_run": cfg.dry_run,
                "safety_profile": cfg.safety_profile,
            },
        )
    state = None
    if cfg.dry_run or bridge is not None or guard is not None or logger is not None:
        state = SO101RuntimeState(
            last_task=task_text,
            intent=intent,
            safety_profile=cfg.safety_profile,
        )

    return SO101PickPlaceRuntime(
        task=task_text,
        intent=intent,
        bridge=bridge,
        guard=guard,
        logger=logger,
        state=state,
    )


def _run_dry_run(cfg: SO101PolicyRecordConfig, runtime: SO101PickPlaceRuntime) -> dict[str, Any]:
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dry_run_root = cfg.dataset_root / ".dry_run"
    robot = _DryRunRobot(dry_run_root / "robot")
    dry_run_action = runtime.intent.get("dry_run_action")
    action_template = dry_run_action if isinstance(dry_run_action, dict) else None
    teleop = _DryRunTeleop(dry_run_root / "teleop", action_template=action_template)
    events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
    }

    robot.connect()
    teleop.connect()
    try:
        for episode_index in range(cfg.num_episodes):
            if events["stop_recording"]:
                break
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset_fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=None,
                control_time_s=cfg.episode_time_s,
                single_task=runtime.task,
                display_data=cfg.display_data,
                perception_bridge=runtime.bridge,
                safety_guard=runtime.guard,
                step_event_logger=runtime.logger,
                runtime_state=runtime.state,
            )
            if runtime.logger is not None:
                runtime.logger.log_event(
                    "episode_end",
                    episode_index=episode_index,
                    halted=events["stop_recording"],
                    step_index=0 if runtime.state is None else runtime.state.step_index,
                )
        return {
            "dry_run": True,
            "halted": events["stop_recording"],
            "step_index": 0 if runtime.state is None else runtime.state.step_index,
            "events_jsonl_path": None if cfg.events_jsonl_path is None else str(cfg.events_jsonl_path),
        }
    finally:
        teleop.disconnect()
        robot.disconnect()


def _run_live_inference_no_dataset(cfg: SO101PolicyRecordConfig, runtime: SO101PickPlaceRuntime) -> dict[str, Any]:
    """Run policy/teleop on real robot without persisting dataset frames."""
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = make_robot_from_config(_build_robot_config(cfg))
    teleop_cfg = _build_teleop_config(cfg)
    teleop = make_teleoperator_from_config(teleop_cfg) if teleop_cfg is not None else None

    policy = None
    preprocessor = None
    postprocessor = None
    policy_observation_features = None
    policy_action_features = None

    if cfg.policy is not None:
        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
        )
        policy.to(cfg.policy.device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )
        policy_observation_features = aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        )
        policy_action_features = aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        )

    listener, events = init_keyboard_listener()
    robot.connect()
    if teleop is not None:
        teleop.connect()

    try:
        for episode_index in range(cfg.num_episodes):
            if events["stop_recording"]:
                break

            log_say(f"Inference episode {episode_index + 1}/{cfg.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset_fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=None,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                policy_observation_features=policy_observation_features,
                policy_action_features=policy_action_features,
                control_time_s=cfg.episode_time_s,
                single_task=runtime.task,
                display_data=cfg.display_data,
                perception_bridge=runtime.bridge,
                safety_guard=runtime.guard,
                step_event_logger=runtime.logger,
                runtime_state=runtime.state,
            )

            if not events["stop_recording"] and episode_index < cfg.num_episodes - 1 and teleop is not None:
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset_fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=None,
                    teleop=teleop,
                    control_time_s=cfg.reset_time_s,
                    single_task=runtime.task,
                    display_data=cfg.display_data,
                    perception_bridge=runtime.bridge,
                    safety_guard=runtime.guard,
                    step_event_logger=runtime.logger,
                    runtime_state=runtime.state,
                )

        return {
            "inference_only": True,
            "halted": events["stop_recording"],
            "step_index": 0 if runtime.state is None else runtime.state.step_index,
            "events_jsonl_path": None if cfg.events_jsonl_path is None else str(cfg.events_jsonl_path),
        }
    finally:
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


def run_recording(
    cfg: SO101PolicyRecordConfig,
    *,
    require_guard: bool = False,
):
    init_logging()
    logging.info(cfg)

    if cfg.save_dataset and cfg.clear_dataset_root and not cfg.resume and cfg.dataset_root.exists():
        shutil.rmtree(cfg.dataset_root)

    runtime = build_so101_pickplace_runtime(cfg, require_guard=require_guard)
    if runtime.logger is not None:
        runtime.logger.log_run_start(task=runtime.task, intent=runtime.intent)

    try:
        if cfg.dry_run:
            return _run_dry_run(cfg, runtime)

        if not cfg.save_dataset:
            return _run_live_inference_no_dataset(cfg, runtime)

        record_cfg = _build_record_config(cfg, runtime.task)
        return base_record(
            record_cfg,
            perception_bridge=runtime.bridge,
            safety_guard=runtime.guard,
            step_event_logger=runtime.logger,
            runtime_state=runtime.state,
        )
    finally:
        if runtime.logger is not None:
            runtime.logger.log_run_end(
                task=runtime.task,
                halted=False if runtime.state is None else runtime.state.halted,
                step_index=0 if runtime.state is None else runtime.state.step_index,
                last_reason=None if runtime.state is None else runtime.state.last_reason,
            )


@parser.wrap()
def run(cfg: SO101PolicyRecordConfig):
    return run_recording(cfg, require_guard=False)


def main():
    register_third_party_devices()
    run()


if __name__ == "__main__":
    main()
