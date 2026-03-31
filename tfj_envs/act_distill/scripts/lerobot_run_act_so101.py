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

"""Run a distilled ACT policy directly on a SO101 follower robot.

This is a lightweight on-robot inference entrypoint derived from
`lerobot_record.py`, but it does not require dataset recording.

Accepted policy path forms:

- a `pretrained_model` directory
- a checkpoint directory such as `.../checkpoints/018000`
- an output directory containing `checkpoints/`
- `config.json` or `train_config.json`
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from pprint import pformat

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    repo_src = Path(__file__).resolve().parents[3] / "src"
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_POLICY_PATH = Path(
    "/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740"
)
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")


def _is_pretrained_policy_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def resolve_policy_pretrained_path(policy_path: str | Path) -> Path:
    policy_path = Path(policy_path).expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    if policy_path.is_file():
        if policy_path.name in {"config.json", "train_config.json"} and _is_pretrained_policy_dir(policy_path.parent):
            return policy_path.parent
        raise FileNotFoundError(
            "Policy path must point to `pretrained_model`, a checkpoint directory, an output directory, "
            f"or `config.json`/`train_config.json`. Got file: {policy_path}"
        )

    if _is_pretrained_policy_dir(policy_path):
        return policy_path

    pretrained_dir = policy_path / "pretrained_model"
    if _is_pretrained_policy_dir(pretrained_dir):
        return pretrained_dir

    checkpoints_dir = policy_path / "checkpoints"
    if checkpoints_dir.is_dir():
        last_checkpoint = checkpoints_dir / "last"
        if last_checkpoint.exists():
            resolved_last = last_checkpoint.resolve()
            if _is_pretrained_policy_dir(resolved_last):
                return resolved_last
            resolved_last_pretrained = resolved_last / "pretrained_model"
            if _is_pretrained_policy_dir(resolved_last_pretrained):
                return resolved_last_pretrained

        numeric_checkpoints = sorted(
            [candidate for candidate in checkpoints_dir.iterdir() if candidate.is_dir() and candidate.name.isdigit()]
        )
        for checkpoint_dir in reversed(numeric_checkpoints):
            checkpoint_pretrained_dir = checkpoint_dir / "pretrained_model"
            if _is_pretrained_policy_dir(checkpoint_pretrained_dir):
                return checkpoint_pretrained_dir

    raise FileNotFoundError(
        "Could not resolve a pretrained policy from path "
        f"`{policy_path}`. Expected a directory containing `config.json` and `model.safetensors`."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a distilled ACT policy on a SO101 follower robot.")
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")

    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-max-relative-target", type=float, default=8.0)
    parser.add_argument("--robot-use-degrees", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=1)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--run-time-s", type=float, default=300.0)
    parser.add_argument("--task", default="Put the block in the bin")
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)

    return parser


def build_robot_config(args: argparse.Namespace) -> SO101FollowerConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
        ),
    }
    return SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=Path(args.robot_calibration_dir),
        disable_torque_on_disconnect=True,
        max_relative_target=args.robot_max_relative_target,
        cameras=cameras,
        use_degrees=args.robot_use_degrees,
    )


def load_policy_bundle(policy_path: Path, policy_device: str):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = policy_device

    if policy_cfg.type != "act":
        raise ValueError(f"This runtime only supports ACT policies. Got `{policy_cfg.type}`.")

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
        },
    )
    return policy_cfg, policy, preprocessor, postprocessor


def run_policy_loop(
    *,
    robot,
    policy,
    preprocessor,
    postprocessor,
    policy_observation_features: dict[str, dict],
    policy_action_features: dict[str, dict],
    robot_action_processor,
    robot_observation_processor,
    fps: int,
    run_time_s: float,
    task: str,
    display_data: bool,
):
    listener, events = init_keyboard_listener()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    try:
        start_t = time.perf_counter()
        while run_time_s <= 0 or (time.perf_counter() - start_t) < run_time_s:
            if events["stop_recording"] or events["exit_early"]:
                break

            loop_start_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(policy_observation_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            robot_action = make_robot_action(action_values, policy_action_features)
            robot_action_to_send = robot_action_processor((robot_action, obs))
            robot.send_action(robot_action_to_send)

            if display_data:
                log_rerun_data(observation=obs_processed, action=robot_action)

            dt_s = time.perf_counter() - loop_start_t
            precise_sleep(max(0.0, 1.0 / fps - dt_s))
    finally:
        robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


def main() -> int:
    register_third_party_devices()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be strictly positive.")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be strictly positive.")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera-width and --camera-height must be strictly positive.")

    init_logging()

    resolved_policy_path = resolve_policy_pretrained_path(args.policy_path)
    robot_cfg = build_robot_config(args)
    logging.info(
        "Running ACT on SO101 with config:\n%s",
        pformat(
            {
                "policy_path": str(resolved_policy_path),
                "policy_device": args.policy_device,
                "robot_id": args.robot_id,
                "robot_port": args.robot_port,
                "robot_calibration_dir": args.robot_calibration_dir,
                "top_cam_index": args.top_cam_index,
                "wrist_cam_index": args.wrist_cam_index,
                "camera_width": args.camera_width,
                "camera_height": args.camera_height,
                "camera_fps": args.camera_fps,
                "fps": args.fps,
                "run_time_s": args.run_time_s,
                "task": args.task,
                "display_data": args.display_data,
            }
        ),
    )

    policy_cfg, policy, preprocessor, postprocessor = load_policy_bundle(
        resolved_policy_path,
        policy_device=args.policy_device,
    )

    robot = make_robot_from_config(robot_cfg)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

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

    if args.display_data:
        init_rerun(session_name="act_so101_run")

    robot.connect()
    logging.info(
        "ACT policy loaded from %s on %s. Press Right Arrow or Esc to stop early.",
        resolved_policy_path,
        policy_cfg.device,
    )

    try:
        run_policy_loop(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_observation_features=policy_observation_features,
            policy_action_features=policy_action_features,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            fps=args.fps,
            run_time_s=args.run_time_s,
            task=args.task,
            display_data=args.display_data,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
