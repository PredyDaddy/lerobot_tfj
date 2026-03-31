#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    DEFAULT_CONDA_ENV,
    DEFAULT_DEVICE,
    resolve_tensorrt_py_dir,
    resolve_tmpdir,
    run_command,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified numerical compare: safetensors-loaded PyTorch vs ONNX vs TensorRT."
    )
    parser.add_argument("--policy-path", required=True, help="Checkpoint root or pretrained_model/ path.")
    parser.add_argument("--run-dir", required=True, help="Run directory that already contains gr00t_onnx/ and engines.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda"])
    parser.add_argument("--seed", type=int, default=20260303)
    parser.add_argument("--tensorrt-py-dir", default=None, help="Default: $TENSORRT_PY_DIR if set, otherwise leave unset.")
    parser.add_argument("--tmpdir", default=None, help="Default: $TMPDIR if set, else <run-dir>/.tmp")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-llm-from-vit-cosine", type=float, default=None)
    parser.add_argument("--min-denoising-cosine", type=float, default=None)
    parser.add_argument("--report-path", default=None, help="Default: <run-dir>/compare_safetensor_onnx_trt.json")
    return parser


def stage_script(name: str) -> Path:
    return SCRIPT_DIR / name


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    onnx_dir = run_dir / "gr00t_onnx"
    engine_dir = run_dir / "gr00t_engine_api_trt1013"
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (run_dir / "compare_safetensor_onnx_trt.json")
    )
    tensorrt_py_dir = resolve_tensorrt_py_dir(args.tensorrt_py_dir)
    tmpdir = resolve_tmpdir(args.tmpdir, run_dir)

    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"ONNX directory is missing: {onnx_dir}")
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"Engine directory is missing: {engine_dir}")

    stage_jobs = [
        (
            "stage3_verify_onnx",
            stage_script("step3_verify_onnx.py"),
            [
                "--policy-path",
                args.policy_path,
                "--run-dir",
                run_dir.as_posix(),
                "--conda-env",
                args.conda_env,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                *(["--skip-existing"] if args.skip_existing else []),
                *(
                    ["--min-llm-from-vit-cosine", str(args.min_llm_from_vit_cosine)]
                    if args.min_llm_from_vit_cosine is not None
                    else []
                ),
                *(
                    ["--min-denoising-cosine", str(args.min_denoising_cosine)]
                    if args.min_denoising_cosine is not None
                    else []
                ),
            ],
        ),
        (
            "stage5_verify_trt",
            stage_script("step5_verify_trt.py"),
            [
                "--policy-path",
                args.policy_path,
                "--run-dir",
                run_dir.as_posix(),
                "--conda-env",
                args.conda_env,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--tensorrt-py-dir",
                tensorrt_py_dir or "",
                "--tmpdir",
                tmpdir,
                *(["--skip-existing"] if args.skip_existing else []),
                *(
                    ["--min-llm-from-vit-cosine", str(args.min_llm_from_vit_cosine)]
                    if args.min_llm_from_vit_cosine is not None
                    else []
                ),
                *(
                    ["--min-denoising-cosine", str(args.min_denoising_cosine)]
                    if args.min_denoising_cosine is not None
                    else []
                ),
            ],
        ),
    ]

    stage_runs = []
    for name, script_path, stage_args in stage_jobs:
        result = run_command(
            [sys.executable, script_path.as_posix(), *stage_args],
            log_path=(run_dir / "logs" / f"{name}.log"),
            cwd=SCRIPT_DIR,
        )
        stage_runs.append(
            {
                "name": name,
                "script": script_path.as_posix(),
                "result": result,
            }
        )

    stage3 = json.loads((run_dir / "stage3_verify_onnx.json").read_text())
    stage5 = json.loads((run_dir / "stage5_verify_trt.json").read_text())

    combined = {
        "policy_path": args.policy_path,
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "seed": args.seed,
        "environment": {
            "conda_env": args.conda_env,
            "device": args.device,
            "tensorrt_py_dir": tensorrt_py_dir,
            "tmpdir": tmpdir,
        },
        "artifacts": {
            "stage3_report": (run_dir / "stage3_verify_onnx.json").as_posix(),
            "stage5_report": (run_dir / "stage5_verify_trt.json").as_posix(),
            "onnx_compare_1view": (onnx_dir / "compare_metrics_1view.json").as_posix(),
            "onnx_compare_2view": (onnx_dir / "compare_metrics_2view.json").as_posix(),
            "trt_compare_1view": (engine_dir / "compare_metrics_trt_1view.json").as_posix(),
            "trt_compare_2view": (engine_dir / "compare_metrics_trt_2view.json").as_posix(),
        },
        "views": {
            "1view": {
                "onnx": stage3["summaries"]["1view"],
                "trt": stage5["summaries"]["1view"],
                "warnings": {
                    "onnx": stage3["warnings"]["1view"],
                    "trt": stage5["warnings"]["1view"],
                },
            },
            "2view": {
                "onnx": stage3["summaries"]["2view"],
                "trt": stage5["summaries"]["2view"],
                "warnings": {
                    "onnx": stage3["warnings"]["2view"],
                    "trt": stage5["warnings"]["2view"],
                },
            },
        },
        "stage_runs": stage_runs,
    }
    write_json(report_path, combined)

    print("[OK] Unified compare finished.")
    for view in ["1view", "2view"]:
        onnx_summary = combined["views"][view]["onnx"]
        trt_summary = combined["views"][view]["trt"]
        print(
            f"{view} ONNX: worst={onnx_summary['worst_cosine_key']}:{onnx_summary['worst_cosine']:.9f} "
            f"llm_from_vit={onnx_summary['llm_from_vit_cosine']:.9f} "
            f"denoising={onnx_summary['denoising_cosine']:.9f}"
        )
        print(
            f"{view} TRT : worst={trt_summary['worst_cosine_key']}:{trt_summary['worst_cosine']:.9f} "
            f"llm_from_vit={trt_summary['llm_from_vit_cosine']:.9f} "
            f"denoising={trt_summary['denoising_cosine']:.9f}"
        )
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
