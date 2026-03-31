#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act_trt_paths import DOCS_DIR, REPO_ROOT, SRC_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from act_model_utils import (
    build_case_catalog,
    load_act_policy,
    load_model_spec,
    make_case_inputs,
    save_json,
    tensor_to_numpy,
    torch_core_forward,
)
from trt_runtime import TensorRTRunner


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing Python module `{module_name}` in current env.\n"
            f"Install hint: {install_hint}\n"
            f"Current python: {sys.executable}"
        )


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_f = np.asarray(lhs, dtype=np.float32)
    rhs_f = np.asarray(rhs, dtype=np.float32)
    diff = np.abs(lhs_f - rhs_f)
    denom = np.maximum(np.abs(lhs_f), 1e-8)
    rel = diff / denom
    flat_lhs = lhs_f.reshape(-1)
    flat_rhs = rhs_f.reshape(-1)
    cosine = float(np.dot(flat_lhs, flat_rhs) / (np.linalg.norm(flat_lhs) * np.linalg.norm(flat_rhs) + 1e-12))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "cosine_similarity": cosine,
    }


def merge_summary(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in metrics_list),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in metrics_list),
        "max_rel_diff": max(item["max_rel_diff"] for item in metrics_list),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics_list),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ACT Torch vs ONNX vs TensorRT consistency.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--threshold-max-abs-diff", type=float, default=1e-4)
    parser.add_argument(
        "--report",
        type=Path,
        default=DOCS_DIR / "ACT_VERIFY_TORCH_ONNX_TRT.json",
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    require_module("onnxruntime", "conda run -n lerobot pip install onnxruntime-gpu or onnxruntime")

    checkpoint = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    engine_path = args.engine.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    import onnxruntime as ort

    spec = load_model_spec(checkpoint)
    policy = load_act_policy(checkpoint, device="cpu")

    # Keep CPUExecutionProvider to avoid any ORT CUDA setup mismatch.
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    trt_runner = TensorRTRunner(engine_path=engine_path, device=args.device)

    case_reports: list[dict[str, Any]] = []
    torch_onnx_metrics: list[dict[str, float]] = []
    torch_trt_metrics: list[dict[str, float]] = []
    onnx_trt_metrics: list[dict[str, float]] = []

    for case_info in build_case_catalog(args.random_cases):
        obs_state, img0, img1 = make_case_inputs(spec=spec, seed=case_info["seed"], case=case_info["case"])

        with torch.inference_mode():
            torch_output = tensor_to_numpy(torch_core_forward(policy, obs_state, img0, img1))

        feed_dict = {
            "obs_state_norm": obs_state.numpy(),
            "img0_norm": img0.numpy(),
            "img1_norm": img1.numpy(),
        }

        onnx_output = np.asarray(ort_session.run(["actions_norm"], feed_dict)[0], dtype=np.float32)
        trt_output = tensor_to_numpy(trt_runner.infer(feed_dict)["actions_norm"])

        torch_onnx = compare_arrays(torch_output, onnx_output)
        torch_trt = compare_arrays(torch_output, trt_output)
        onnx_trt = compare_arrays(onnx_output, trt_output)
        torch_onnx_metrics.append(torch_onnx)
        torch_trt_metrics.append(torch_trt)
        onnx_trt_metrics.append(onnx_trt)

        case_reports.append(
            {
                "case": case_info["name"],
                "seed": case_info["seed"],
                "kind": case_info["case"],
                "torch_vs_onnx": torch_onnx,
                "torch_vs_trt": torch_trt,
                "onnx_vs_trt": onnx_trt,
            }
        )

    summary = {
        "torch_vs_onnx": merge_summary(torch_onnx_metrics),
        "torch_vs_trt": merge_summary(torch_trt_metrics),
        "onnx_vs_trt": merge_summary(onnx_trt_metrics),
    }
    passed = (
        summary["torch_vs_onnx"]["max_abs_diff"] <= args.threshold_max_abs_diff
        and summary["torch_vs_trt"]["max_abs_diff"] <= args.threshold_max_abs_diff
        and summary["onnx_vs_trt"]["max_abs_diff"] <= args.threshold_max_abs_diff
    )

    payload = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "device": args.device,
        "threshold_max_abs_diff": float(args.threshold_max_abs_diff),
        "spec": {
            "visual_keys": spec.visual_keys,
            "state_dim": spec.state_dim,
            "image_height": spec.image_height,
            "image_width": spec.image_width,
            "action_dim": spec.action_dim,
            "chunk_size": spec.chunk_size,
        },
        "summary": summary,
        "passed": bool(passed),
        "cases": case_reports,
    }

    save_json(report_path, payload)
    print(report_path)
    print(payload["summary"])
    print(f"passed={passed}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())

