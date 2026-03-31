#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from act_trt_paths import REPO_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent
ACT_EXPORT_SCRIPT = SCRIPT_DIR / "export_act_onnx.py"
ACT_BUILD_SCRIPT = SCRIPT_DIR / "build_act_trt_engine.py"
ACT_VERIFY_SCRIPT = SCRIPT_DIR / "verify_act_torch_onnx_trt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a LeRobot ACT checkpoint to ONNX and TensorRT.\n"
            "This wrapper runs ONNX export, writes export_metadata.json, builds the TensorRT engine,\n"
            "and optionally verifies Torch vs ONNX vs TensorRT."
        ),
        epilog=(
            "Beginner example:\n"
            "  conda run --live-stream -n lerobot_flex python "
            "/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \\\n"
            "    --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \\\n"
            "    --precision fp32 \\\n"
            "    --device cpu \\\n"
            "    --trt-device cuda:0\n\n"
            "What this command creates:\n"
            "  - act_single.onnx\n"
            "  - export_metadata.json\n"
            "  - act_single_fp32.plan\n"
            "  - trt_build_summary_fp32.json\n"
            "  - consistency_report_act_single_fp32.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model checkpoint directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact output directory. Defaults to outputs/deploy/act_trt/<run_name>/<checkpoint_step>.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for ONNX export.")
    parser.add_argument("--trt-device", default="cuda:0", help="CUDA device string used by TensorRT runtime inspection.")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--timing-cache", default=None, help="Optional TensorRT timing cache path.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing act_single.onnx and metadata.")
    parser.add_argument(
        "--verify-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run exporter Torch-vs-ONNX verification.",
    )
    parser.add_argument(
        "--verify-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Torch-vs-ONNX-vs-TensorRT verification after engine build.",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX simplification in the reference exporter.",
    )
    parser.add_argument("--dynamo", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    return checkpoint


def resolve_output_dir(checkpoint: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    try:
        run_name = checkpoint.parents[2].name
        checkpoint_step = checkpoint.parent.name
    except IndexError as exc:
        raise ValueError(f"Unexpected checkpoint layout: {checkpoint}") from exc

    return (REPO_ROOT / "outputs" / "deploy" / "act_trt" / run_name / checkpoint_step).resolve()


def require_source_scripts() -> None:
    missing = [str(path) for path in (ACT_EXPORT_SCRIPT, ACT_BUILD_SCRIPT, ACT_VERIFY_SCRIPT) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required ACT TRT scripts are missing:\n" + "\n".join(missing))


def require_export_dependencies(skip_export: bool) -> None:
    if skip_export:
        return
    if importlib.util.find_spec("onnx") is not None:
        return

    raise ModuleNotFoundError(
        "当前 Python 环境缺少 `onnx`，无法执行 ONNX 导出。\n"
        "建议直接使用 `lerobot_flex` 环境运行：\n"
        "conda run --live-stream -n lerobot_flex python "
        "/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py "
        "--checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model\n"
        "如果你坚持在当前环境导出，请先安装 onnx。"
    )


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    require_source_scripts()
    require_export_dependencies(args.skip_export)

    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = resolve_output_dir(checkpoint, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "act_single.onnx"
    metadata_path = output_dir / "export_metadata.json"
    engine_path = output_dir / f"act_single_{args.precision}.plan"
    report_path = output_dir / f"trt_build_summary_{args.precision}.json"
    verify_report_path = output_dir / f"consistency_report_act_single_{args.precision}.json"

    python_exe = sys.executable

    if not args.skip_export:
        export_command = [
            python_exe,
            str(ACT_EXPORT_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--opset",
            str(args.opset),
            "--device",
            args.device,
        ]
        if args.dynamo:
            export_command.append("--dynamo")
        export_command.append("--verify" if args.verify_export else "--no-verify")
        export_command.append("--simplify" if args.simplify else "--no-simplify")
        run_command(export_command)
    else:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"--skip-export was set but ONNX file is missing: {onnx_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"--skip-export was set but metadata file is missing: {metadata_path}")

    build_command = [
        python_exe,
        str(ACT_BUILD_SCRIPT),
        "--onnx",
        str(onnx_path),
        "--metadata",
        str(metadata_path),
        "--engine",
        str(engine_path),
        "--precision",
        args.precision,
        "--workspace-gb",
        str(args.workspace_gb),
        "--opt-level",
        str(args.opt_level),
        "--report",
        str(report_path),
        "--device",
        args.trt_device,
    ]
    if args.timing_cache:
        build_command.extend(["--timing-cache", str(Path(args.timing_cache).expanduser().resolve())])
    run_command(build_command)

    if args.verify_engine:
        verify_command = [
            python_exe,
            str(ACT_VERIFY_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--onnx",
            str(onnx_path),
            "--engine",
            str(engine_path),
            "--report",
            str(verify_report_path),
        ]
        run_command(verify_command)

    build_summary = read_json(report_path)
    print(f"[ARTIFACT] onnx={onnx_path}")
    print(f"[ARTIFACT] metadata={metadata_path}")
    print(f"[ARTIFACT] engine={engine_path}")
    print(f"[ARTIFACT] build_report={report_path}")
    if args.verify_engine and verify_report_path.is_file():
        print(f"[ARTIFACT] verify_report={verify_report_path}")

    tensors = build_summary.get("engine_tensors", [])
    if tensors:
        print("[ENGINE I/O]")
        for tensor in tensors:
            print(
                f"  - {tensor['name']}: mode={tensor['mode']} dtype={tensor['dtype']} "
                f"shape={tuple(tensor['shape'])}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
