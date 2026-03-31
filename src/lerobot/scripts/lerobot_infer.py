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

"""
Run a pretrained policy directly on a real robot without recording or saving a dataset.

This is a lightweight inference-only wrapper around the policy path used in
`lerobot_record.py`: robot observation -> preprocessing -> policy inference ->
postprocessing -> send action.

Examples:

```shell
lerobot-infer \
    --policy.path=pi_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=black \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Pick the block and place it in the bin" \
    --fps=10

lerobot-infer \
    --policy.path=pi_model/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=black \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --rename_map='{\"observation.images.front\": \"observation.images.top\", \"observation.images.gripper\": \"observation.images.wrist\"}' \
    --task="Pick the block and place it in the bin"
```
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from huggingface_hub.constants import CONFIG_NAME

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun


def _resolve_pretrained_policy_path(policy_path: str | Path) -> str | Path:
    path = Path(policy_path)
    if not path.exists():
        return policy_path

    if path.is_dir() and (path / CONFIG_NAME).is_file():
        return path

    nested_pretrained = path / "pretrained_model"
    if nested_pretrained.is_dir() and (nested_pretrained / CONFIG_NAME).is_file():
        return nested_pretrained

    return path


def load_policy_runtime(
    policy_cfg: PreTrainedConfig,
    rename_map: dict[str, str] | None = None,
) -> tuple[
    PreTrainedPolicy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    policy_path = _resolve_pretrained_policy_path(policy_cfg.pretrained_path)
    policy_cfg.pretrained_path = policy_path

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=policy_path, config=policy_cfg)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map or {}},
        },
    )

    return policy, preprocessor, postprocessor


def build_policy_runtime_features(robot, robot_action_processor, robot_observation_processor) -> tuple[dict, dict]:
    policy_action_features = aggregate_pipeline_dataset_features(
        pipeline=robot_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=True,
    )
    policy_observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    )
    return policy_action_features, policy_observation_features


@dataclass
class InferConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    fps: int = 10
    task: str = ""
    run_time_s: float | None = None
    display_data: bool = False
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            policy_path = _resolve_pretrained_policy_path(policy_path)
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.policy is not None and getattr(self.policy, "pretrained_path", None):
            self.policy.pretrained_path = _resolve_pretrained_policy_path(self.policy.pretrained_path)

        if self.policy is None:
            raise ValueError("Policy path is required. Use --policy.path=/path/to/model or --policy.path=pi_model.")

        if getattr(self.policy, "pretrained_path", None) is None:
            raise ValueError("Policy must define a pretrained_path so weights and processors can be loaded.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@parser.wrap()
def infer(cfg: InferConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if not cfg.task:
        logging.warning("Task is empty. Language-conditioned policies like pi0/pi05 usually need --task.")

    if cfg.display_data:
        init_rerun(session_name="policy_inference")

    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    policy_action_features, policy_observation_features = build_policy_runtime_features(
        robot=robot,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    if not cfg.rename_map:
        runtime_features = dataset_to_policy_features({**policy_observation_features, **policy_action_features})
        validate_visual_features_consistency(cfg.policy, runtime_features)

    policy, preprocessor, postprocessor = load_policy_runtime(cfg.policy, rename_map=cfg.rename_map)

    robot.connect()
    listener, events = init_keyboard_listener()

    try:
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=None,
            teleop=None,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_observation_features=policy_observation_features,
            policy_action_features=policy_action_features,
            control_time_s=float("inf") if cfg.run_time_s is None else cfg.run_time_s,
            single_task=cfg.task,
            display_data=cfg.display_data,
        )
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user.")
    finally:
        robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


def main():
    register_third_party_devices()
    infer()


if __name__ == "__main__":
    main()
