#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONDA_ENV,
    DEFAULT_DEVICE,
    REPO_ROOT,
    build_conda_python_cmd,
    default_run_dir,
    default_stage_report_path,
    ensure_dir,
    expected_onnx_contracts,
    resolve_policy_dir,
    run_command,
    validate_onnx_contracts,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: export the 7 fixed GROOT ONNX subgraphs with local scripts.")
    parser.add_argument("--policy-path", required=True, help="Checkpoint root or pretrained_model/ path.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory used to place ONNX outputs, logs, and the stage report.",
    )
    parser.add_argument(
        "--onnx-dir",
        default=None,
        help="Explicit ONNX output directory. Default: <run-dir>/gr00t_onnx",
    )
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda env used for export scripts.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda"])
    parser.add_argument("--seq-len", type=int, default=296, help="Tracing sequence length for ONNX export.")
    parser.add_argument("--video-views", type=int, default=1, help="Tracing view count for ViT ONNX export.")
    parser.add_argument("--state-horizon", type=int, default=1, help="State horizon for action-head export.")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version.")
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16"])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If all 7 expected ONNX files already exist, skip export commands and only validate contracts.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit JSON report path. Default: <run-dir>/stage2_export_onnx.json",
    )
    return parser


def all_expected_onnx_exist(onnx_dir: Path) -> bool:
    return all((onnx_dir / rel_path).is_file() for rel_path in expected_onnx_contracts())


def collect_onnx_files(onnx_dir: Path) -> dict[str, Any]:
    files = {}
    for rel_path in expected_onnx_contracts():
        path = onnx_dir / rel_path
        files[rel_path] = {
            "path": path.as_posix(),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return files


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir()
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else (run_dir / "gr00t_onnx")
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else default_stage_report_path(
        run_dir, "stage2_export_onnx"
    )
    logs_dir = ensure_dir(run_dir / "logs")

    policy_dir = resolve_policy_dir(args.policy_path)
    export_backbone_script = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "export_backbone_onnx_local.py"
    export_action_head_script = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "export_action_head_onnx_local.py"

    commands = []
    skipped = False
    if args.skip_existing and all_expected_onnx_exist(onnx_dir):
        skipped = True
    else:
        ensure_dir(onnx_dir)

        backbone_cmd = build_conda_python_cmd(
            args.conda_env,
            export_backbone_script,
            [
                "--policy-path",
                policy_dir.as_posix(),
                "--onnx-out-dir",
                onnx_dir.as_posix(),
                "--seq-len",
                str(args.seq_len),
                "--video-views",
                str(args.video_views),
                "--vit-dtype",
                args.vit_dtype,
                "--llm-dtype",
                args.llm_dtype,
                "--opset",
                str(args.opset),
                "--device",
                args.device,
            ],
        )
        commands.append(
            run_command(
                backbone_cmd,
                log_path=logs_dir / "export_backbone.log",
                cwd=REPO_ROOT,
            )
        )

        action_head_cmd = build_conda_python_cmd(
            args.conda_env,
            export_action_head_script,
            [
                "--policy-path",
                policy_dir.as_posix(),
                "--onnx-out-dir",
                onnx_dir.as_posix(),
                "--seq-len",
                str(args.seq_len),
                "--state-horizon",
                str(args.state_horizon),
                "--opset",
                str(args.opset),
                "--device",
                args.device,
            ],
        )
        commands.append(
            run_command(
                action_head_cmd,
                log_path=logs_dir / "export_action_head.log",
                cwd=REPO_ROOT,
            )
        )

    contract_report = validate_onnx_contracts(onnx_dir)
    report = {
        "stage": "export_onnx",
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "skipped_export": skipped,
        "export_args": {
            "conda_env": args.conda_env,
            "device": args.device,
            "seq_len": args.seq_len,
            "video_views": args.video_views,
            "state_horizon": args.state_horizon,
            "opset": args.opset,
            "vit_dtype": args.vit_dtype,
            "llm_dtype": args.llm_dtype,
        },
        "commands": commands,
        "onnx_files": collect_onnx_files(onnx_dir),
        "contract_validation": contract_report,
    }
    write_json(report_path, report)
    print(f"[OK] Stage 2 report written to: {report_path}")


if __name__ == "__main__":
    main()
