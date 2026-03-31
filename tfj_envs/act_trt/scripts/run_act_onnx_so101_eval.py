#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from act_trt_paths import REPO_ROOT

TARGET_SCRIPT = REPO_ROOT / "doc" / "lerobot_record_act_onnx.py"

DEFAULT_POLICY_PATH = REPO_ROOT / "outputs" / "act_grasp_block_in_bin1" / "checkpoints" / "last" / "pretrained_model"
DEFAULT_ONNX_PATH = DEFAULT_POLICY_PATH / "act_core.onnx"
DEFAULT_ONNX_METADATA_PATH = DEFAULT_POLICY_PATH / "act_core.metadata.json"
DEFAULT_DATASET_ROOT = Path("/home/cqy/.cache/huggingface/lerobot/admin123/eval_grasp_block_in_bin2")
DEFAULT_CALIB_DIR = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower")
DEFAULT_XAUTHORITY = Path("/run/user/1003/gdm/Xauthority")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACT ONNX real-robot eval with baked-in SO101 defaults.")
    parser.add_argument("--robot-id", default="my_so101")
    parser.add_argument("--robot-type", default="so101_follower")
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

    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--onnx-metadata-path", default=str(DEFAULT_ONNX_METADATA_PATH))
    parser.add_argument("--onnx-provider", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--display-data", type=parse_bool, default=False)
    parser.add_argument("--play-sounds", type=parse_bool, default=False)
    parser.add_argument("--resume", choices=["auto", "true", "false"], default="auto")

    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--xauthority", default=os.environ.get("XAUTHORITY", str(DEFAULT_XAUTHORITY)))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    target_script = TARGET_SCRIPT.resolve()
    policy_path = Path(args.policy_path).expanduser().resolve()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    calib_dir = Path(args.robot_calibration_dir).expanduser().resolve()

    metadata_path = Path(args.onnx_metadata_path).expanduser().resolve() if args.onnx_metadata_path else None

    if not target_script.is_file():
        raise FileNotFoundError(f"Target real-robot script not found: {target_script}")
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX path not found: {onnx_path}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"ONNX metadata path not found: {metadata_path}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    return target_script, policy_path, onnx_path, dataset_root, calib_dir


def build_cameras_json(args: argparse.Namespace) -> str:
    payload = {
        "top": {
            "type": "opencv",
            "index_or_path": args.top_cam_index,
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
        },
        "wrist": {
            "type": "opencv",
            "index_or_path": args.wrist_cam_index,
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def build_command(
    args: argparse.Namespace,
    target_script: Path,
    policy_path: Path,
    onnx_path: Path,
    dataset_root: Path,
    calib_dir: Path,
    resume_value: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(target_script),
        f"--robot.id={args.robot_id}",
        f"--robot.type={args.robot_type}",
        f"--robot.calibration_dir={calib_dir}",
        f"--robot.port={args.robot_port}",
        f"--robot.cameras={build_cameras_json(args)}",
        f"--display_data={str(args.display_data).lower()}",
        f"--play_sounds={str(args.play_sounds).lower()}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.push_to_hub={str(args.dataset_push_to_hub).lower()}",
        f"--dataset.num_episodes={args.dataset_num_episodes}",
        f"--dataset.episode_time_s={args.dataset_episode_time_s}",
        f"--dataset.reset_time_s={args.dataset_reset_time_s}",
        f"--dataset.single_task={args.dataset_single_task}",
        f"--policy.path={policy_path}",
        f"--policy.device={args.policy_device}",
        f"--onnx.path={onnx_path}",
        f"--onnx.provider={args.onnx_provider}",
        f"--resume={str(resume_value).lower()}",
    ]
    if args.onnx_metadata_path:
        command.append(f"--onnx.metadata_path={Path(args.onnx_metadata_path).expanduser().resolve()}")
    return command


def prepare_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.display:
        env["DISPLAY"] = args.display
    if args.xauthority:
        env["XAUTHORITY"] = args.xauthority
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def print_summary(args: argparse.Namespace, dataset_root: Path, resume_value: bool, command: list[str]) -> None:
    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Policy path: {Path(args.policy_path).expanduser().resolve()}")
    print(f"[INFO] ONNX path: {Path(args.onnx_path).expanduser().resolve()}")
    print(f"[INFO] Dataset repo_id: {args.dataset_repo_id}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Resume: {resume_value} (requested: {args.resume})")
    print(f"[INFO] Display: {args.display}")
    print(f"[INFO] XAUTHORITY: {args.xauthority}")
    if dataset_root.exists() and args.dataset_repo_id.split("/")[-1] not in str(dataset_root):
        print("[WARN] dataset.repo_id and dataset.root name do not match. This is allowed, but verify this is intended.")
    print("[INFO] Command:")
    print("  " + " ".join(shlex.quote(part) for part in command))


def main() -> int:
    args = build_parser().parse_args()
    target_script, policy_path, onnx_path, dataset_root, calib_dir = validate_paths(args)

    resume_value = resolve_resume_mode(dataset_root, args.resume)
    if dataset_root.exists() and not resume_value and any(dataset_root.iterdir()):
        raise FileExistsError(
            "Dataset root already exists and is non-empty. "
            "Use --resume=true or point --dataset-root to a new directory."
        )

    command = build_command(
        args=args,
        target_script=target_script,
        policy_path=policy_path,
        onnx_path=onnx_path,
        dataset_root=dataset_root,
        calib_dir=calib_dir,
        resume_value=resume_value,
    )
    env = prepare_env(args)
    print_summary(args, dataset_root, resume_value, command)

    if args.dry_run:
        print("[INFO] Dry run only. Exiting without launching the robot loop.")
        return 0

    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
