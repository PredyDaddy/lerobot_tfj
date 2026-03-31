#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

from act_trt_paths import REPO_ROOT, SRC_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot import policies  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots import make_robot_from_config, so101_follower  # noqa: F401
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

from trt_act_policy import TrtActPolicyAdapter


DEFAULT_POLICY_PATH = (
    REPO_ROOT / "outputs" / "act_grasp_block_in_bin1" / "checkpoints" / "016000" / "pretrained_model"
)
DEFAULT_TRT_PATH = (
    REPO_ROOT
    / "outputs"
    / "deploy"
    / "act_trt"
    / "act_grasp_block_in_bin1"
    / "016000"
    / "act_single_fp32.plan"
)
DEFAULT_TRT_METADATA_PATH = (
    REPO_ROOT
    / "outputs"
    / "deploy"
    / "act_trt"
    / "act_grasp_block_in_bin1"
    / "016000"
    / "export_metadata.json"
)
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")


def stage(message: str) -> None:
    print(f"[STAGE] {message}", flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(value)


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing Python module `{module_name}` in current env.\n"
            f"Install hint: {install_hint}\n"
            f"Current python: {sys.executable}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure ACT TensorRT inference loop for the SO101 follower robot."
    )
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration-dir", default=str(DEFAULT_CALIB_DIR))

    parser.add_argument("--top-cam-index", type=int, default=4)
    parser.add_argument("--wrist-cam-index", type=int, default=6)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-n-action-steps", type=parse_optional_int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=parse_optional_float, default=None)

    parser.add_argument("--trt-path", default=str(DEFAULT_TRT_PATH))
    parser.add_argument("--trt-metadata-path", default=str(DEFAULT_TRT_METADATA_PATH))
    parser.add_argument("--trt-device", default="cuda:0")

    parser.add_argument("--task", default="grasp block in bin")
    parser.add_argument("--run-time-s", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=30)

    parser.add_argument("--skip-camera-preflight", action="store_true")
    parser.add_argument("--skip-trt-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    policy_path = Path(args.policy_path).expanduser().resolve()
    trt_path = Path(args.trt_path).expanduser().resolve()
    metadata_path = Path(args.trt_metadata_path).expanduser().resolve() if args.trt_metadata_path else None
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()

    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not trt_path.is_file():
        raise FileNotFoundError(f"TensorRT engine path not found: {trt_path}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"TensorRT metadata path not found: {metadata_path}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    return policy_path, trt_path, metadata_path, calib_dir


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


def apply_act_runtime_overrides(
    policy_cfg: PreTrainedConfig,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")

    chunk_size = int(policy_cfg.chunk_size)

    if policy_n_action_steps is not None:
        if not 1 <= policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = policy_n_action_steps

    if policy_temporal_ensemble_coeff is not None:
        if policy_cfg.n_action_steps != 1:
            raise ValueError(
                "ACT temporal ensembling requires n_action_steps == 1. "
                f"Current value: {policy_cfg.n_action_steps}"
            )
        policy_cfg.temporal_ensemble_coeff = policy_temporal_ensemble_coeff


def load_policy_config(args: argparse.Namespace, policy_path: Path) -> PreTrainedConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = str(policy_path)
    policy_cfg.device = args.policy_device
    apply_act_runtime_overrides(
        policy_cfg=policy_cfg,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )
    return policy_cfg


def load_pre_post_processors(
    policy_path: Path,
    policy_device: str,
) -> tuple[
    PolicyProcessorPipeline[dict[str, object], dict[str, object]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": policy_device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def print_summary(
    args: argparse.Namespace,
    policy_path: Path,
    trt_path: Path,
    metadata_path: Path | None,
    policy_cfg: PreTrainedConfig,
) -> None:
    info(f"Python: {sys.executable}")
    info(f"Policy path: {policy_path}")
    info(f"TRT engine: {trt_path}")
    info(f"TRT metadata: {metadata_path if metadata_path else '<none>'}")
    info(f"Policy device: {policy_cfg.device}")
    info(
        "ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )
    info(f"Robot port: {args.robot_port}")
    info(f"Cameras: top={args.top_cam_index}, wrist={args.wrist_cam_index}")
    info(f"Task: {args.task}")
    info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")


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
    for tensor in trt_policy.runner.describe():
        info(f"TRT tensor: name={tensor.name} mode={tensor.mode} dtype={tensor.dtype} shape={tensor.shape}")
    return trt_policy


def main() -> int:
    args = build_parser().parse_args()
    register_third_party_devices()
    init_logging()

    stage("Validate environment")
    require_module("tensorrt", "conda run -n lerobot pip install tensorrt==10.13.0.35")
    policy_path, trt_path, metadata_path, calib_dir = validate_paths(args)
    policy_cfg = load_policy_config(args, policy_path)
    print_summary(args, policy_path, trt_path, metadata_path, policy_cfg)

    if args.dry_run:
        info("Dry run only. Exiting before hardware access.")
        return 0

    stage("Preflight checks")
    if not args.skip_camera_preflight:
        preflight_cameras(args)
    trt_policy = None
    if not args.skip_trt_preflight:
        trt_policy = preflight_trt(policy_cfg, trt_path, metadata_path, args.trt_device)

    if args.preflight_only:
        info("Preflight completed. Exiting before robot connect.")
        return 0

    stage("Build robot and processors")
    robot_cfg = build_robot_config(args, calib_dir)
    robot = make_robot_from_config(robot_cfg)
    preprocessor, postprocessor = load_pre_post_processors(policy_path, str(policy_cfg.device))
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if trt_policy is None:
        stage("Load TRT policy")
        trt_policy = preflight_trt(policy_cfg, trt_path, metadata_path, args.trt_device)

    trt_policy.eval()

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        stage("Connect robot")
        robot.connect()
        trt_policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        info("Robot connected. Starting pure TRT inference loop.")

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                info("Reached requested run_time_s. Exiting inference loop.")
                break

            loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=trt_policy,
                device=get_safe_torch_device(trt_policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=trt_policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_processor((action_dict, obs))
            robot.send_action(robot_action_to_send)

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                info(f"Step {step} | elapsed={elapsed:.2f}s")

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.camera_fps - dt_s, 0.0))
    except KeyboardInterrupt:
        info("KeyboardInterrupt received. Stopping inference.")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception:
                pass
        info("Inference finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
