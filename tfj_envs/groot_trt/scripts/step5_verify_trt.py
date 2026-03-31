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
    resolve_tensorrt_py_dir,
    resolve_tmpdir,
    resolve_policy_dir,
    run_command,
    summarize_trt_compare,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 5: run Torch vs TensorRT compare for 1-view and 2-view synthetic batches."
    )
    parser.add_argument("--policy-path", required=True, help="Checkpoint root or pretrained_model/ path.")
    parser.add_argument("--engine-dir", default=None, help="Directory containing the 7 TensorRT engine files.")
    parser.add_argument("--run-dir", default=None, help="Run directory used to place compare outputs and logs.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda"])
    parser.add_argument("--seed", type=int, default=20260303)
    parser.add_argument("--tensorrt-py-dir", default=None, help="Default: $TENSORRT_PY_DIR if set, otherwise leave unset.")
    parser.add_argument("--tmpdir", default=None, help="Default: $TMPDIR if set, else <run-dir>/.tmp")
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
    engine_dir = Path(args.engine_dir).expanduser().resolve() if args.engine_dir else (run_dir / "gr00t_engine_api_trt1013")
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else default_stage_report_path(
        run_dir, "stage5_verify_trt"
    )
    logs_dir = ensure_dir(run_dir / "logs")
    tensorrt_py_dir = resolve_tensorrt_py_dir(args.tensorrt_py_dir)
    tmpdir = resolve_tmpdir(args.tmpdir, run_dir)
    policy_dir = resolve_policy_dir(args.policy_path)
    compare_script = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "compare_torch_trt_local.py"

    compare_jobs = [
        ("1view", DEFAULT_SEQ_LEN_1, DEFAULT_VIDEO_VIEWS_1, engine_dir / "compare_metrics_trt_1view.json"),
        ("2view", DEFAULT_SEQ_LEN_2, DEFAULT_VIDEO_VIEWS_2, engine_dir / "compare_metrics_trt_2view.json"),
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
                    "--engine-dir",
                    engine_dir.as_posix(),
                    "--seq-len",
                    str(seq_len),
                    "--video-views",
                    str(video_views),
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
                    log_path=logs_dir / f"compare_trt_{label}.log",
                    env_extra={
                        "TMPDIR": tmpdir,
                        "TENSORRT_PY_DIR": tensorrt_py_dir,
                    },
                    cwd=REPO_ROOT,
                )
            )

        summary = summarize_trt_compare(json_out)
        raw = json.loads(json_out.read_text())
        summaries[label] = {
            **summary,
            "seed": raw.get("seed"),
            "seq_len": raw.get("seq_len"),
            "video_views": raw.get("video_views"),
        }
        warnings[label] = build_warning(summary, args)

    report = {
        "stage": "verify_trt",
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "compare_args": {
            "conda_env": args.conda_env,
            "device": args.device,
            "seed": args.seed,
            "tensorrt_py_dir": tensorrt_py_dir,
            "tmpdir": tmpdir,
        },
        "commands": commands,
        "summaries": summaries,
        "warnings": warnings,
    }
    write_json(report_path, report)
    print(f"[OK] Stage 5 report written to: {report_path}")


if __name__ == "__main__":
    main()
