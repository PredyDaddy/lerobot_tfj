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

"""Direct no-save SO101 policy runtime.

This entrypoint runs a policy directly on a SO101 follower robot without
creating a LeRobot dataset. It reuses the existing SO101 bridge / safety /
logging runtime and the shared `record_loop` control loop, but passes
`dataset=None` and supplies policy-only feature schemas.
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

if __package__ is None or __package__ == "":
    repo_src = Path(__file__).resolve().parents[2]
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)

from lerobot.configs import parser
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.scripts.lerobot_record_so101_policy import (
    SAFETY_PROFILE_DEFAULT,
    SO101PolicyRecordConfig,
    _build_robot_config,
    _get_runtime_state_value,
    build_so101_pickplace_runtime,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun


def _has_graphical_display() -> bool:
    return any(os.environ.get(key) for key in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"))


def _init_runtime_events() -> tuple[object | None, dict[str, bool]]:
    events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
    }
    if not _has_graphical_display():
        logging.warning("Headless environment detected. Keyboard stop controls will not be available.")
        return None, events
    return init_keyboard_listener()


def _load_policy_bundle(cfg: "SO101PickPlaceConfig"):
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("Direct policy inference requires `--policy.path=...`.")

    policy_cfg = cfg.policy
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)
    policy.to(policy_cfg.device)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": policy_cfg.device},
    }
    if cfg.rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    return policy_cfg, policy, preprocessor, postprocessor


def _build_policy_features(robot, robot_action_processor, robot_observation_processor) -> tuple[dict[str, dict], dict[str, dict]]:
    policy_observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    )
    policy_action_features = aggregate_pipeline_dataset_features(
        pipeline=robot_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=False,
    )
    return policy_observation_features, policy_action_features


@dataclass
class SO101PickPlaceConfig(SO101PolicyRecordConfig):
    task: str = "Pick and place the target object."
    enable_perception_bridge: bool = True
    safety_profile: str = SAFETY_PROFILE_DEFAULT
    events_jsonl_path: Path | None = None
    fps: int = 30
    run_time_s: float = 300.0

    def __post_init__(self):
        super().__post_init__()
        if parser.parse_arg("events_jsonl_path") is None:
            self.events_jsonl_path = None
        if self.policy is None:
            raise ValueError("Direct policy inference requires `--policy.path=...`.")
        if self.fps <= 0:
            raise ValueError("`fps` must be strictly positive.")
        if self.display_data and not _has_graphical_display():
            logging.warning("No graphical display detected. Forcing `display_data=false` for direct inference.")
            self.display_data = False


@parser.wrap()
def run(cfg: SO101PickPlaceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="so101_pickplace_direct")

    runtime = build_so101_pickplace_runtime(cfg, require_guard=True)
    if runtime.logger is not None:
        runtime.logger.log_run_start(task=runtime.task, intent=runtime.intent)

    listener = None
    robot = None
    try:
        policy_cfg, policy, preprocessor, postprocessor = _load_policy_bundle(cfg)

        if cfg.dry_run:
            return {
                "dry_run": True,
                "policy_type": policy_cfg.type,
                "policy_path": str(policy_cfg.pretrained_path),
                "task": runtime.task,
                "events_jsonl_path": None if cfg.events_jsonl_path is None else str(cfg.events_jsonl_path),
            }

        robot = make_robot_from_config(_build_robot_config(cfg))
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        policy_observation_features, policy_action_features = _build_policy_features(
            robot,
            robot_action_processor,
            robot_observation_processor,
        )
        listener, events = _init_runtime_events()

        robot.connect()
        logging.info(
            "Direct policy runtime loaded from %s on %s. Press Right Arrow or Esc to stop early.",
            policy_cfg.pretrained_path,
            policy_cfg.device,
        )

        record_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=None,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_observation_features=policy_observation_features,
            policy_action_features=policy_action_features,
            control_time_s=float("inf") if cfg.run_time_s <= 0 else cfg.run_time_s,
            single_task=runtime.task,
            display_data=cfg.display_data,
            perception_bridge=runtime.bridge,
            safety_guard=runtime.guard,
            step_event_logger=runtime.logger,
            runtime_state=runtime.state,
        )
        return {
            "task": runtime.task,
            "halted": False if runtime.state is None else runtime.state.halted,
            "step_index": _get_runtime_state_value(runtime.state, "step_index", 0),
            "events_jsonl_path": None if cfg.events_jsonl_path is None else str(cfg.events_jsonl_path),
        }
    finally:
        if robot is not None and getattr(robot, "is_connected", False):
            robot.disconnect()
        if listener is not None:
            listener.stop()
        if runtime.logger is not None:
            runtime.logger.log_run_end(
                task=runtime.task,
                halted=False if runtime.state is None else runtime.state.halted,
                step_index=_get_runtime_state_value(runtime.state, "step_index", 0),
                last_reason=None if runtime.state is None else runtime.state.last_reason,
            )


def main():
    register_third_party_devices()
    run()


if __name__ == "__main__":
    main()
