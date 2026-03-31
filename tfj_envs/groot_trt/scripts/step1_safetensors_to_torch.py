#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from common import (
    DEFAULT_CONDA_ENV,
    DEFAULT_DEVICE,
    build_conda_python_cmd,
    default_run_dir,
    default_stage_report_path,
    ensure_dir,
    load_policy,
    load_pre_post_processors,
    policy_summary,
    repo_env,
    required_checkpoint_files,
    resolve_tensorrt_py_dir,
    resolve_tmpdir,
    resolve_policy_dir,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: resolve pretrained_model/, load safetensors into current-checkout PyTorch policy, and verify processors."
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Checkpoint root or pretrained_model/ path. The script resolves the real pretrained_model/ directory.",
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help="Conda env used for the safetensors -> PyTorch load stage.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda"], help="PyTorch device for policy load.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory used to place the stage report. Default: outputs/trt/groot_stepwise_<ts>/",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit JSON report path. Default: <run-dir>/stage1_safetensors_to_torch.json",
    )
    parser.add_argument(
        "--torch-state-out",
        default=None,
        help="Optional .pt path. When set, dump the loaded policy state_dict() for inspection.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Attempt a strict=True load first and record the result before doing the repo-default non-strict load.",
    )
    parser.add_argument(
        "--strict-fatal",
        action="store_true",
        help="Fail the stage if the strict=True probe does not load successfully.",
    )
    parser.add_argument(
        "--tensorrt-py-dir",
        default=None,
        help="Optional TensorRT pip --target dir used for environment probing. Default: $TENSORRT_PY_DIR if set.",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="TMPDIR value to record in the report for later TensorRT stages. Default: $TMPDIR if set, else <run-dir>/.tmp",
    )
    parser.add_argument("--_already-in-conda", action="store_true", help=argparse.SUPPRESS)
    return parser


def probe_tensorrt(tensorrt_py_dir: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_tensorrt_py_dir": tensorrt_py_dir,
        "import_ok": False,
        "version": None,
        "error": None,
    }

    if tensorrt_py_dir:
        candidate = Path(tensorrt_py_dir).expanduser().resolve()
        if candidate.as_posix() not in sys.path:
            sys.path.insert(0, candidate.as_posix())
        result["resolved_tensorrt_py_dir"] = candidate.as_posix()
        result["path_exists"] = candidate.is_dir()

    try:
        import tensorrt as trt  # type: ignore

        result["import_ok"] = True
        result["version"] = getattr(trt, "__version__", None)
    except Exception as exc:  # pragma: no cover - import errors are part of the report
        result["error"] = repr(exc)

    return result


def dump_state_dict(path: Path, policy: Any, cfg_type: str) -> dict[str, Any]:
    ensure_dir(path.parent)
    state_dict = policy.state_dict()
    torch.save(
        {
            "policy_type": cfg_type,
            "state_dict": state_dict,
        },
        path,
    )
    return {
        "path": path.as_posix(),
        "num_tensors": len(state_dict),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = build_parser().parse_args()

    current_conda_env = os.getenv("CONDA_DEFAULT_ENV")
    if args.conda_env and current_conda_env != args.conda_env and not args._already_in_conda:
        forwarded_args = [*sys.argv[1:], "--_already-in-conda"]
        cmd = build_conda_python_cmd(args.conda_env, Path(__file__).resolve(), forwarded_args)
        subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).resolve().parent.as_posix(),
            env=repo_env(),
        )
        return

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else default_run_dir()
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else default_stage_report_path(
        run_dir, "stage1_safetensors_to_torch"
    )
    tensorrt_py_dir = resolve_tensorrt_py_dir(args.tensorrt_py_dir)
    tmpdir = resolve_tmpdir(args.tmpdir, run_dir)

    policy_dir = resolve_policy_dir(args.policy_path)
    checkpoint_files = required_checkpoint_files(policy_dir)
    missing = [name for name, ok in checkpoint_files.items() if not ok]
    if missing:
        raise FileNotFoundError("Missing required checkpoint files:\n" + "\n".join(f"  - {name}" for name in missing))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GROOT TensorRT export/compare requires CUDA.")

    strict_probe = {
        "attempted": bool(args.strict),
        "passed": None,
        "error": None,
    }
    if args.strict:
        try:
            strict_cfg, strict_policy_cls, strict_policy = load_policy(policy_dir, device=args.device, strict=True)
            strict_probe["passed"] = True
            cfg, policy_cls, policy = strict_cfg, strict_policy_cls, strict_policy
        except Exception as exc:  # pragma: no cover - strict failure is a runtime finding
            strict_probe["passed"] = False
            strict_probe["error"] = repr(exc)
            if args.strict_fatal:
                raise
            cfg, policy_cls, policy = load_policy(policy_dir, device=args.device, strict=False)
    else:
        cfg, policy_cls, policy = load_policy(policy_dir, device=args.device, strict=False)

    preprocessor, postprocessor = load_pre_post_processors(policy_dir)

    param = next(policy.parameters())
    report: dict[str, Any] = {
        "stage": "safetensors_to_torch",
        "policy_dir": policy_dir.as_posix(),
        "resolved_from": Path(args.policy_path).expanduser().resolve().as_posix(),
        "checkpoint_files": checkpoint_files,
        "strict_probe": strict_probe,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "cwd": Path.cwd().as_posix(),
            "conda_default_env": os.getenv("CONDA_DEFAULT_ENV"),
            "tmpdir": tmpdir,
            "tensorrt_probe": probe_tensorrt(tensorrt_py_dir),
        },
        "cuda": {
            "is_available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "policy": policy_summary(cfg, policy_cls, policy),
        "load_checks": {
            "policy_eval_mode": not policy.training,
            "first_parameter_device": str(param.device),
            "first_parameter_dtype": str(param.dtype),
            "effective_load_strict": bool(strict_probe["attempted"] and strict_probe["passed"]),
            "action_queue_maxlen": getattr(policy, "_action_queue").maxlen,
            "preprocessor_type": type(preprocessor).__name__,
            "postprocessor_type": type(postprocessor).__name__,
            "preprocessor_name": getattr(preprocessor, "name", None),
            "postprocessor_name": getattr(postprocessor, "name", None),
            "preprocessor_steps": len(getattr(preprocessor, "steps", [])),
            "postprocessor_steps": len(getattr(postprocessor, "steps", [])),
        },
    }

    if args.torch_state_out:
        state_info = dump_state_dict(Path(args.torch_state_out).expanduser().resolve(), policy, cfg.type)
        report["torch_state_export"] = state_info

    write_json(report_path, report)
    print(f"[OK] Stage 1 report written to: {report_path}")


if __name__ == "__main__":
    main()
