#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONDA_ENV,
    DEFAULT_DEVICE,
    DEFAULT_SEQ_LEN_1,
    DEFAULT_SEQ_LEN_2,
    DEFAULT_VIDEO_VIEWS_1,
    DEFAULT_VIDEO_VIEWS_2,
    REPO_ROOT,
    build_conda_python_cmd,
    default_run_dir,
    default_stage_report_path,
    ensure_dir,
    resolve_policy_dir,
    run_command,
    summarize_onnx_compare,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: run Torch vs ONNX compare for 1-view and 2-view synthetic batches."
    )
    parser.add_argument("--policy-path", required=True, help="Checkpoint root or pretrained_model/ path.")
    parser.add_argument("--onnx-dir", default=None, help="Directory containing gr00t_onnx/. Default: <run-dir>/gr00t_onnx")
    parser.add_argument("--run-dir", default=None, help="Run directory used to place compare outputs and logs.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda"])
    parser.add_argument("--vit-dtype", default="fp16", choices=["fp16", "fp8"])
    parser.add_argument("--llm-dtype", default="fp16", choices=["fp16", "fp8", "nvfp4", "nvfp4_full"])
    parser.add_argument("--dit-dtype", default="fp16", help="DiT ONNX suffix for compare_torch_onnx.py")
    parser.add_argument("--seed", type=int, default=20260303)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse compare JSON outputs if they already exist and only rebuild the stage summary.",
    )
    parser.add_argument("--report-path", default=None, help="Optional explicit JSON report path.")
    parser.add_argument("--min-llm-from-vit-cosine", type=float, default=None, help="Optional warning threshold.")
    parser.add_argument("--min-denoising-cosine", type=float, default=None, help="Optional warning threshold.")
    return parser


def build_warning(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    warnings = []
    if summary.get("missing"):
        warnings.append(f"missing artifacts: {summary['missing']}")
    llm_cos = float(summary["llm_from_vit_cosine"])
    denoise_cos = float(summary["denoising_cosine"])
    if args.min_llm_from_vit_cosine is not None and llm_cos < float(args.min_llm_from_vit_cosine):
        warnings.append(
            f"llm_from_vit_pipeline cosine {llm_cos:.6f} < requested threshold {float(args.min_llm_from_vit_cosine):.6f}"
        )
    if args.min_denoising_cosine is not None and denoise_cos < float(args.min_denoising_cosine):
        warnings.append(
            f"action_denoising_pipeline cosine {denoise_cos:.6f} < requested threshold {float(args.min_denoising_cosine):.6f}"
        )
    return warnings


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir()
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else (run_dir / "gr00t_onnx")
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else default_stage_report_path(
        run_dir, "stage3_verify_onnx"
    )
    logs_dir = ensure_dir(run_dir / "logs")
    policy_dir = resolve_policy_dir(args.policy_path)
    compare_script = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "compare_torch_onnx_local.py"

    compare_jobs = [
        ("1view", DEFAULT_SEQ_LEN_1, DEFAULT_VIDEO_VIEWS_1, onnx_dir / "compare_metrics_1view.json"),
        ("2view", DEFAULT_SEQ_LEN_2, DEFAULT_VIDEO_VIEWS_2, onnx_dir / "compare_metrics_2view.json"),
    ]

    commands = []
    summaries = {}
    warnings = {}

    for label, seq_len, video_views, json_out in compare_jobs:
        if not (args.skip_existing and json_out.is_file()):
            cmd = build_conda_python_cmd(
                args.conda_env,
                compare_script,
                [
                    "--policy-path",
                    policy_dir.as_posix(),
                    "--onnx-dir",
                    onnx_dir.as_posix(),
                    "--seq-len",
                    str(seq_len),
                    "--video-views",
                    str(video_views),
                    "--vit-dtype",
                    args.vit_dtype,
                    "--llm-dtype",
                    args.llm_dtype,
                    "--dit-dtype",
                    args.dit_dtype,
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--json-out",
                    json_out.as_posix(),
                ],
            )
            commands.append(
                run_command(
                    cmd,
                    log_path=logs_dir / f"compare_onnx_{label}.log",
                    cwd=REPO_ROOT,
                )
            )

        summary = summarize_onnx_compare(json_out)
        raw = json.loads(json_out.read_text())
        summaries[label] = {
            **summary,
            "provider_names": raw.get("providers"),
            "seed": raw.get("seed"),
            "seq_len": raw.get("seq_len"),
            "video_views": raw.get("video_views"),
        }
        warnings[label] = build_warning(summary, args)

    report = {
        "stage": "verify_onnx",
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "compare_args": {
            "conda_env": args.conda_env,
            "device": args.device,
            "vit_dtype": args.vit_dtype,
            "llm_dtype": args.llm_dtype,
            "dit_dtype": args.dit_dtype,
            "seed": args.seed,
        },
        "commands": commands,
        "summaries": summaries,
        "warnings": warnings,
    }
    write_json(report_path, report)
    print(f"[OK] Stage 3 report written to: {report_path}")


if __name__ == "__main__":
    main()
