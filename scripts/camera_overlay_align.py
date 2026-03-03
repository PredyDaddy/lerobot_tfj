#!/usr/bin/env python3
"""Compare a reference frame with live camera preview for environment alignment."""

import argparse
import sys
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_PATH = (
    "/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1/"
    "videos/observation.images.top/chunk-000/file-000.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a reference frame and compare it with live camera preview."
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help="Path to reference video used for first-frame overlay.",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="Use an existing reference image directly. If set, skip video extraction.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera index passed to cv2.VideoCapture.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend mode only: reference-frame opacity in [0.0, 1.0].",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        default="side_by_side",
        choices=("side_by_side", "blend"),
        help="Display mode: side_by_side (no transparency) or blend.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional camera width. Also used to resize reference frame.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional camera height. Also used to resize reference frame.",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="outputs/overlay_snapshot.png",
        help="Base path for snapshots. Timestamp is appended before suffix.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Camera Overlay Align",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--reference-output",
        type=str,
        default="outputs/reference_first_frame_swapped.png",
        help="Where to save the extracted+swapped first frame image.",
    )
    parser.add_argument(
        "--no-channel-swap",
        action="store_true",
        help="Disable channel swap. By default, BGR<->RGB channel order is swapped.",
    )
    return parser.parse_args()


def ensure_alpha(alpha: float) -> float:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"--alpha must be in [0.0, 1.0], got {alpha}")
    return alpha


def normalize_reference_frame(frame: np.ndarray, swap_channels: bool) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        pass
    else:
        raise RuntimeError(f"不支持的参考帧通道格式: shape={frame.shape}")

    if swap_channels:
        frame = frame[:, :, ::-1]
    return frame.copy()


def extract_first_frame_with_ffmpeg(video_path: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(prefix="ref_first_frame_", suffix=".png", delete=False) as tmp:
        tmp_path_str = tmp.name
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-update",
            "1",
            tmp_path_str,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 ffmpeg 可执行文件") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"ffmpeg 提取第一帧失败: {stderr or 'unknown error'}")

        frame = cv2.imread(tmp_path_str, cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError("ffmpeg 提取完成但读取输出图片失败")
        return frame
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


def extract_first_frame_with_opencv(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"OpenCV 无法读取视频第一帧: {video_path}")
    return frame


def load_reference_frame(
    video_path: str,
    width: int | None,
    height: int | None,
    swap_channels: bool,
    reference_output: str | None,
) -> np.ndarray:
    frame: np.ndarray | None = None
    backend = ""
    errors: list[str] = []
    try:
        frame = extract_first_frame_with_ffmpeg(video_path)
        backend = "ffmpeg"
    except RuntimeError as exc:
        errors.append(str(exc))

    if frame is None:
        try:
            frame = extract_first_frame_with_opencv(video_path)
            backend = "opencv"
        except RuntimeError as exc:
            errors.append(str(exc))
            joined_errors = " | ".join(errors)
            raise RuntimeError(f"无法读取视频第一帧: {video_path}. 详情: {joined_errors}") from exc

    frame = normalize_reference_frame(frame, swap_channels)

    if width is not None and height is not None:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    if reference_output:
        out_path = Path(reference_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"无法写出参考帧图像: {out_path}")
        print(f"[INFO] 参考帧已保存: {out_path} (backend={backend}, swap={swap_channels})")
    else:
        print(f"[INFO] 参考帧加载成功 (backend={backend}, swap={swap_channels})")

    return frame


def load_reference_image(
    image_path: str,
    width: int | None,
    height: int | None,
    swap_channels: bool,
) -> np.ndarray:
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"无法读取参考图像: {image_path}")
    frame = normalize_reference_frame(frame, swap_channels)
    if width is not None and height is not None:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    print(f"[INFO] 参考图像加载成功: {image_path} (swap={swap_channels})")
    return frame


def build_snapshot_path(base_path: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = Path(base_path)
    if base.suffix:
        return base.with_name(f"{base.stem}_{timestamp}{base.suffix}")
    return base.with_name(f"{base.name}_{timestamp}.png")


def main() -> int:
    args = parse_args()
    alpha = args.alpha
    if args.display_mode == "blend":
        try:
            alpha = ensure_alpha(args.alpha)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    try:
        if args.reference_image:
            reference_frame = load_reference_image(
                args.reference_image,
                args.width,
                args.height,
                swap_channels=not args.no_channel_swap,
            )
        else:
            reference_frame = load_reference_frame(
                args.video_path,
                args.width,
                args.height,
                swap_channels=not args.no_channel_swap,
                reference_output=args.reference_output,
            )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera_id)
    if not camera.isOpened():
        print(f"[ERROR] 无法打开摄像头: camera-id={args.camera_id}", file=sys.stderr)
        return 1

    if args.width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    flip_preview = False
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    if args.display_mode == "blend":
        print("按键说明: q/ESC 退出, +/- 调整 alpha, r 重载参考帧, f 镜像翻转, s 保存截图")
    else:
        print("按键说明: q/ESC 退出, r 重载参考帧, f 镜像翻转, s 保存截图")

    try:
        while True:
            ok, cam_frame = camera.read()
            if not ok or cam_frame is None:
                print("[ERROR] 读取摄像头帧失败，退出。", file=sys.stderr)
                break

            if flip_preview:
                cam_frame = cv2.flip(cam_frame, 1)

            if reference_frame.shape[:2] != cam_frame.shape[:2]:
                ref_for_display = cv2.resize(
                    reference_frame,
                    (cam_frame.shape[1], cam_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                ref_for_display = reference_frame

            if args.display_mode == "blend":
                display_frame = cv2.addWeighted(cam_frame, 1.0 - alpha, ref_for_display, alpha, 0.0)
                status = f"mode=blend | alpha={alpha:.2f} | flip={'ON' if flip_preview else 'OFF'}"
            else:
                display_frame = np.hstack((cam_frame, ref_for_display))
                status = f"mode=side_by_side | flip={'ON' if flip_preview else 'OFF'}"

            cv2.putText(
                display_frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window_name, display_frame)

            key = cv2.waitKeyEx(1)
            key_low = key & 0xFF
            if key_low in (ord("q"), 27):
                break
            if args.display_mode == "blend" and (key_low in (ord("+"), ord("=")) or key in (65451, 107)):
                alpha = min(1.0, alpha + 0.05)
                print(f"[INFO] alpha -> {alpha:.2f}")
                continue
            if args.display_mode == "blend" and (key_low in (ord("-"), ord("_")) or key in (65453, 109)):
                alpha = max(0.0, alpha - 0.05)
                print(f"[INFO] alpha -> {alpha:.2f}")
                continue
            if key_low == ord("r"):
                try:
                    if args.reference_image:
                        reference_frame = load_reference_image(
                            args.reference_image,
                            args.width,
                            args.height,
                            swap_channels=not args.no_channel_swap,
                        )
                    else:
                        reference_frame = load_reference_frame(
                            args.video_path,
                            args.width,
                            args.height,
                            swap_channels=not args.no_channel_swap,
                            reference_output=args.reference_output,
                        )
                    print("[INFO] 参考帧已重载。")
                except RuntimeError as exc:
                    print(f"[ERROR] 重载失败: {exc}", file=sys.stderr)
                continue
            if key_low == ord("f"):
                flip_preview = not flip_preview
                print(f"[INFO] 水平翻转 -> {'ON' if flip_preview else 'OFF'}")
                continue
            if key_low == ord("s"):
                try:
                    out_path = build_snapshot_path(args.output_image)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    saved = cv2.imwrite(str(out_path), display_frame)
                    if saved:
                        print(f"[INFO] 已保存: {out_path}")
                    else:
                        print(f"[ERROR] 保存失败: {out_path}", file=sys.stderr)
                except Exception as exc:
                    print(f"[ERROR] 保存截图异常: {exc}", file=sys.stderr)
                continue
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
