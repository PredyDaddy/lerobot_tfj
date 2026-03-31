#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import traceback
from pathlib import Path

from act_trt_paths import REPO_ROOT, SRC_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config, so101_follower  # noqa: F401
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    record_loop,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from trt_act_policy import TrtActPolicyAdapter, init_keyboard_listener


DEFAULT_POLICY_PATH = REPO_ROOT / "outputs" / "act_grasp_block_in_bin1" / "checkpoints" / "last" / "pretrained_model"
DEFAULT_TRT_PATH = DEFAULT_POLICY_PATH / "act_core_b1_fp32.plan"
DEFAULT_TRT_METADATA_PATH = DEFAULT_POLICY_PATH / "act_core_b1.metadata.json"
DEFAULT_DATASET_ROOT = Path("/home/cqy/.cache/huggingface/lerobot/admin123/eval_grasp_block_in_bin2")
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")
DEFAULT_XAUTHORITY = Path("/run/user/1003/gdm/Xauthority")


def stage(message: str) -> None:
    print(f"[STAGE] {message}", flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def resolve_resume_mode(dataset_root: Path, mode: str) -> bool:
    lowered = mode.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered != "auto":
        raise ValueError(f"Unsupported resume mode: {mode}")
    return (dataset_root / "meta" / "info.json").is_file()


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing Python module `{module_name}` in current env.\n"
            f"Install hint: {install_hint}\n"
            f"Current python: {sys.executable}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct ACT TensorRT real-robot eval runner for SO101.")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration-dir", default=str(DEFAULT_CALIB_DIR))

    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--dataset-repo-id", default="admin123/eval_grasp_block_in_bin1")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-push-to-hub", type=parse_bool, default=False)
    parser.add_argument("--dataset-num-episodes", type=int, default=5)
    parser.add_argument("--dataset-episode-time-s", type=int, default=300)
    parser.add_argument("--dataset-reset-time-s", type=int, default=10)
    parser.add_argument("--dataset-single-task", default="grasp block in bin")

    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")

    parser.add_argument("--trt-path", default=str(DEFAULT_TRT_PATH))
    parser.add_argument("--trt-metadata-path", default=str(DEFAULT_TRT_METADATA_PATH))
    parser.add_argument("--trt-device", default="cuda:0")

    parser.add_argument("--display-data", type=parse_bool, default=False)
    parser.add_argument("--play-sounds", type=parse_bool, default=False)
    parser.add_argument("--resume", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--xauthority", default=os.environ.get("XAUTHORITY", str(DEFAULT_XAUTHORITY)))

    parser.add_argument("--skip-camera-preflight", action="store_true")
    parser.add_argument("--skip-keyboard-preflight", action="store_true")
    parser.add_argument("--skip-trt-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def apply_runtime_env(args: argparse.Namespace) -> None:
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.xauthority:
        os.environ["XAUTHORITY"] = args.xauthority
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    policy_path = Path(args.policy_path).expanduser().resolve()
    trt_path = Path(args.trt_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()
    metadata_path = Path(args.trt_metadata_path).expanduser().resolve() if args.trt_metadata_path else None

    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not trt_path.is_file():
        raise FileNotFoundError(f"TensorRT engine path not found: {trt_path}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"TensorRT metadata path not found: {metadata_path}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    return policy_path, trt_path, dataset_root, calib_dir


def is_empty_dataset_skeleton(dataset_root: Path) -> bool:
    if not dataset_root.exists():
        return False
    files = sorted(path.relative_to(dataset_root).as_posix() for path in dataset_root.rglob("*") if path.is_file())
    return files == ["meta/info.json"]


def build_robot_config(args: argparse.Namespace, calib_dir: Path) -> SO101FollowerConfig:
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
        cameras=cameras,
    )


def build_dataset_config(args: argparse.Namespace, dataset_root: Path) -> DatasetRecordConfig:
    return DatasetRecordConfig(
        repo_id=args.dataset_repo_id,
        single_task=args.dataset_single_task,
        root=dataset_root,
        fps=args.camera_fps,
        episode_time_s=args.dataset_episode_time_s,
        reset_time_s=args.dataset_reset_time_s,
        num_episodes=args.dataset_num_episodes,
        push_to_hub=args.dataset_push_to_hub,
    )


def print_config_summary(
    args: argparse.Namespace,
    policy_path: Path,
    trt_path: Path,
    metadata_path: Path | None,
    dataset_root: Path,
    resume_value: bool,
) -> None:
    info(f"Python: {sys.executable}")
    info(f"Policy path: {policy_path}")
    info(f"TRT engine: {trt_path}")
    info(f"TRT metadata: {metadata_path if metadata_path else '<none>'}")
    info(f"Dataset repo_id: {args.dataset_repo_id}")
    info(f"Dataset root: {dataset_root}")
    info(f"Resume: {resume_value} (requested: {args.resume})")
    info(f"Robot port: {args.robot_port}")
    info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    info(f"Display: {os.environ.get('DISPLAY', '')}")
    info(f"XAUTHORITY: {os.environ.get('XAUTHORITY', '')}")
    if dataset_root.exists() and args.dataset_repo_id.split("/")[-1] not in str(dataset_root):
        warn("dataset.repo_id and dataset.root name do not match. Verify this is intentional.")


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


def preflight_keyboard() -> None:
    listener, events = init_keyboard_listener()
    try:
        info(f"Keyboard listener OK: events={sorted(events.keys())}")
    finally:
        if listener is not None:
            listener.stop()


def load_policy_config(policy_path: Path, policy_device: str) -> PreTrainedConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = str(policy_path)
    policy_cfg.device = policy_device
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type}")
    return policy_cfg


def preflight_trt(
    policy_cfg: PreTrainedConfig,
    trt_path: Path,
    metadata_path: Path | None,
    trt_device: str,
) -> TrtActPolicyAdapter:
    trt_policy = TrtActPolicyAdapter(
        policy_cfg,
        engine_path=trt_path,
        metadata_path=metadata_path,
        trt_device=trt_device,
    )
    info(f"TRT policy OK: device={trt_policy.runner.device}")
    info(f"TRT state input: {trt_policy.io_mapping.state_input_name}")
    info(f"TRT camera inputs: {trt_policy.io_mapping.camera_input_names}")
    info(f"TRT output: {trt_policy.io_mapping.output_name}")
    return trt_policy


def main() -> int:
    args = build_parser().parse_args()
    apply_runtime_env(args)
    register_third_party_devices()
    init_logging()

    stage("Validate environment")
    require_module("tensorrt", "conda run -n lerobot pip install tensorrt==10.13.0.35")
    policy_path, trt_path, dataset_root, calib_dir = validate_paths(args)
    metadata_path = Path(args.trt_metadata_path).expanduser().resolve() if args.trt_metadata_path else None
    resume_value = resolve_resume_mode(dataset_root, args.resume)
    if not args.dry_run and not args.preflight_only and resume_value and args.resume == "auto" and is_empty_dataset_skeleton(dataset_root):
        warn(f"Dataset root is only an empty skeleton: {dataset_root}")
        warn("Removing the empty skeleton so recording can recreate the dataset cleanly.")
        shutil.rmtree(dataset_root)
        resume_value = False
    if dataset_root.exists() and not resume_value and any(dataset_root.iterdir()):
        raise FileExistsError(
            "Dataset root already exists and is non-empty. "
            "Use --resume=true or point --dataset-root to a new directory."
        )
    print_config_summary(args, policy_path, trt_path, metadata_path, dataset_root, resume_value)

    if args.dry_run:
        info("Dry run only. Exiting before any preflight or hardware access.")
        return 0

    stage("Build configs")
    robot_cfg = build_robot_config(args, calib_dir)
    dataset_cfg = build_dataset_config(args, dataset_root)
    policy_cfg = load_policy_config(policy_path, args.policy_device)

    stage("Preflight checks")
    if not args.skip_camera_preflight:
        preflight_cameras(args)
    if not args.skip_keyboard_preflight:
        preflight_keyboard()
    trt_policy = None
    if not args.skip_trt_preflight:
        trt_policy = preflight_trt(policy_cfg, trt_path, metadata_path, args.trt_device)

    if args.preflight_only:
        info("Preflight completed. Exiting before robot connect.")
        return 0

    robot = None
    listener = None
    try:
        stage("Create robot and dataset pipeline")
        robot = make_robot_from_config(robot_cfg)
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=dataset_cfg.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=dataset_cfg.video,
            ),
        )

        stage("Open or create dataset")
        if resume_value:
            dataset = LeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=dataset_cfg.num_image_writer_processes,
                    num_threads=dataset_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, dataset_cfg.fps, dataset_features)
        else:
            sanity_check_dataset_name(dataset_cfg.repo_id, policy_cfg)
            dataset = LeRobotDataset.create(
                dataset_cfg.repo_id,
                dataset_cfg.fps,
                root=dataset_cfg.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=dataset_cfg.video,
                image_writer_processes=dataset_cfg.num_image_writer_processes,
                image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
            )

        stage("Load processors and TRT policy")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, dataset_cfg.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": dataset_cfg.rename_map},
            },
        )
        if trt_policy is None:
            trt_policy = preflight_trt(policy_cfg, trt_path, metadata_path, args.trt_device)

        if args.display_data:
            stage("Init rerun display")
            init_rerun(session_name="recording_act_trt")

        stage("Connect robot")
        robot.connect()
        listener, events = init_keyboard_listener()
        info("Robot connected and keyboard listener armed.")

        stage("Start record loop")
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < dataset_cfg.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", args.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=dataset_cfg.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=None,
                    policy=trt_policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=dataset_cfg.episode_time_s,
                    single_task=dataset_cfg.single_task,
                    display_data=args.display_data,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < dataset_cfg.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", args.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=dataset_cfg.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=None,
                        control_time_s=dataset_cfg.reset_time_s,
                        single_task=dataset_cfg.single_task,
                        display_data=args.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", args.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

        stage("Finished")
        log_say("Exiting", args.play_sounds)
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if robot is not None and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
