#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_CONDA_ENV,
    DEFAULT_MAX_SEQ_LEN_1,
    DEFAULT_MAX_SEQ_LEN_2,
    DEFAULT_MIN_SEQ_LEN,
    DEFAULT_SEQ_LEN_1,
    DEFAULT_SEQ_LEN_2,
    DEFAULT_VIDEO_VIEWS_1,
    DEFAULT_VIDEO_VIEWS_2,
    REPO_ROOT,
    build_conda_python_cmd,
    default_run_dir,
    default_stage_report_path,
    ensure_dir,
    expected_engine_files,
    resolve_tensorrt_py_dir,
    resolve_tmpdir,
    run_command,
    summarize_build_report,
    validate_engine_dir,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4: build the 7 fixed TensorRT engines from the exported ONNX subgraphs."
    )
    parser.add_argument("--run-dir", default=None, help="Run directory that contains gr00t_onnx/ and will receive engines.")
    parser.add_argument("--onnx-dir", default=None, help="Explicit ONNX directory. Default: <run-dir>/gr00t_onnx")
    parser.add_argument(
        "--engine-dir",
        default=None,
        help="Explicit engine output directory. Default: <run-dir>/gr00t_engine_api_trt1013",
    )
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument(
        "--video-views",
        type=int,
        default=DEFAULT_VIDEO_VIEWS_2,
        choices=[DEFAULT_VIDEO_VIEWS_1, DEFAULT_VIDEO_VIEWS_2],
        help="Used to pick the default sequence-length profile when explicit values are not given.",
    )
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--vit-opt-batch", type=int, default=None)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    parser.add_argument("--opt-seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--tensorrt-py-dir", default=None, help="Default: $TENSORRT_PY_DIR if set, otherwise leave unset.")
    parser.add_argument("--tmpdir", default=None, help="Default: $TMPDIR if set, else <run-dir>/.tmp")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If all 7 engine files and build_report.json already exist, skip the build command and only validate outputs.",
    )
    parser.add_argument("--report-path", default=None, help="Optional explicit JSON report path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir()
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else (run_dir / "gr00t_onnx")
    engine_dir = Path(args.engine_dir).expanduser().resolve() if args.engine_dir else (run_dir / "gr00t_engine_api_trt1013")
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else default_stage_report_path(
        run_dir, "stage4_build_engines"
    )
    logs_dir = ensure_dir(run_dir / "logs")
    tensorrt_py_dir = resolve_tensorrt_py_dir(args.tensorrt_py_dir)
    tmpdir = resolve_tmpdir(args.tmpdir, run_dir)

    build_script = REPO_ROOT / "tfj_envs" / "groot_trt" / "scripts" / "build_groot_engines_local.py"
    build_report_path = engine_dir / "build_report.json"

    opt_seq_len = args.opt_seq_len
    max_seq_len = args.max_seq_len
    if opt_seq_len is None:
        opt_seq_len = DEFAULT_SEQ_LEN_2 if args.video_views == DEFAULT_VIDEO_VIEWS_2 else DEFAULT_SEQ_LEN_1
    if max_seq_len is None:
        max_seq_len = DEFAULT_MAX_SEQ_LEN_2 if args.video_views == DEFAULT_VIDEO_VIEWS_2 else DEFAULT_MAX_SEQ_LEN_1
    vit_opt_batch = args.vit_opt_batch if args.vit_opt_batch is not None else args.video_views

    commands = []
    skipped = False
    if args.skip_existing and build_report_path.is_file():
        try:
            validate_engine_dir(engine_dir)
            skipped = True
        except FileNotFoundError:
            skipped = False

    if not skipped:
        cmd = build_conda_python_cmd(
            args.conda_env,
            build_script,
            [
                "--onnx-dir",
                onnx_dir.as_posix(),
                "--engine-out-dir",
                engine_dir.as_posix(),
                "--max-batch",
                str(args.max_batch),
                "--vit-opt-batch",
                str(vit_opt_batch),
                "--opt-batch",
                str(args.opt_batch),
                "--min-seq-len",
                str(args.min_seq_len),
                "--opt-seq-len",
                str(opt_seq_len),
                "--max-seq-len",
                str(max_seq_len),
                "--workspace-gb",
                str(args.workspace_gb),
                *(["--verbose"] if args.verbose else []),
            ],
        )
        commands.append(
            run_command(
                cmd,
                log_path=logs_dir / "build_trt_engine.log",
                env_extra={
                    "TMPDIR": tmpdir,
                    "TENSORRT_PY_DIR": tensorrt_py_dir,
                },
                cwd=REPO_ROOT,
            )
        )

    engine_validation = validate_engine_dir(engine_dir)
    build_summary = summarize_build_report(build_report_path)

    report = {
        "stage": "build_engines",
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "skipped_build": skipped,
        "build_args": {
            "conda_env": args.conda_env,
            "video_views": args.video_views,
            "max_batch": args.max_batch,
            "vit_opt_batch": vit_opt_batch,
            "opt_batch": args.opt_batch,
            "min_seq_len": args.min_seq_len,
            "opt_seq_len": opt_seq_len,
            "max_seq_len": max_seq_len,
            "workspace_gb": args.workspace_gb,
            "tensorrt_py_dir": tensorrt_py_dir,
            "tmpdir": tmpdir,
            "verbose": args.verbose,
        },
        "commands": commands,
        "build_report": build_summary,
        "engine_validation": engine_validation,
        "engine_contract_ok": sorted(build_summary["engine_names"]) == sorted(expected_engine_files()),
    }
    write_json(report_path, report)
    print(f"[OK] Stage 4 report written to: {report_path}")


if __name__ == "__main__":
    main()
