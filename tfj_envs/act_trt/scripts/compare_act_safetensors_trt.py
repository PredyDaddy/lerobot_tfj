#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "act_grasp_block_in_bin1" / "checkpoints" / "last" / "pretrained_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ACT checkpoint outputs restored from safetensors against a TensorRT engine.\n"
            "This script loads the same checkpoint in PyTorch, runs several synthetic test cases,\n"
            "and checks whether the TensorRT engine stays within the configured error threshold."
        ),
        epilog=(
            "Beginner example:\n"
            "  conda run --live-stream -n lerobot python "
            "/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py \\\n"
            "    --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \\\n"
            "    --engine /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \\\n"
            "    --metadata /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json \\\n"
            "    --trt-device cuda:0 \\\n"
            "    --threshold-max-abs-diff 1e-4 \\\n"
            "    --report-json /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_SAFETENSORS_VS_TRT.json\n\n"
            "Success criteria:\n"
            "  - all_within_threshold = true\n"
            "  - max_abs_diff is usually around 1e-6 to 1e-5 for a good fp32 engine"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the checkpoint/pretrained_model directory that contains model.safetensors and config.json.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="Path to the TensorRT engine (.plan or .engine). If omitted, the script tries <checkpoint>/act_core_b1_fp32.plan.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to export_metadata.json. Recommended when your engine uses img0_norm/img1_norm names.",
    )
    parser.add_argument("--trt-device", type=str, default="cuda:0", help="CUDA device used to run TensorRT inference.")
    parser.add_argument(
        "--threshold-max-abs-diff",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute difference between PyTorch output and TensorRT output.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Where to save the comparison report JSON. If omitted, a default file is written under tfj_envs/act_trt/docs/.",
    )
    return parser.parse_args()


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
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
        "exact_equal": bool(np.array_equal(lhs_f, rhs_f)),
        "torch_head": lhs_f.reshape(-1)[:10].tolist(),
        "trt_head": rhs_f.reshape(-1)[:10].tolist(),
        "diff_head": diff.reshape(-1)[:10].tolist(),
    }


def build_cases() -> list[tuple[str, str, int | None]]:
    return [
        ("random_seed_0", "random", 0),
        ("random_seed_1", "random", 1),
        ("random_seed_2", "random", 2),
        ("zeros", "zeros", None),
        ("ones", "ones", None),
        ("linspace", "linspace", None),
    ]


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_default_report_path(engine_path: Path) -> Path:
    return DOCS_DIR / f"compare_{engine_path.stem}_vs_safetensors.json"


def main() -> int:
    args = parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from act_model_utils import (
        load_act_policy,
        load_model_spec,
        make_case_inputs,
        tensor_to_numpy,
        torch_core_forward,
    )
    from trt_act_policy import TrtActPolicyAdapter

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    engine_path = args.engine.expanduser().resolve() if args.engine else (checkpoint / "act_core_b1_fp32.plan").resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else (checkpoint / "act_core_b1.metadata.json").resolve()
    )
    if not engine_path.is_file():
        raise FileNotFoundError(f"TRT engine file not found: {engine_path}")
    if metadata_path is not None and metadata_path.exists() and not metadata_path.is_file():
        raise FileNotFoundError(f"TRT metadata path is not a file: {metadata_path}")

    report_path = (
        args.report_json.expanduser().resolve() if args.report_json else resolve_default_report_path(engine_path)
    )

    spec = load_model_spec(checkpoint)
    policy = load_act_policy(checkpoint, device="cpu")
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.device = "cuda"
    trt_policy = TrtActPolicyAdapter(
        policy_cfg,
        engine_path=engine_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        trt_device=args.trt_device,
    )
    metadata = load_json(metadata_path if metadata_path.is_file() else None)

    case_reports: list[dict[str, Any]] = []
    max_abs_list: list[float] = []
    mean_abs_list: list[float] = []
    max_rel_list: list[float] = []
    cosine_list: list[float] = []
    all_exact = True
    all_within_threshold = True

    for case_name, kind, seed in build_cases():
        obs_state, img0, img1 = make_case_inputs(spec=spec, seed=seed, case=kind)
        batch = {
            "observation.state": obs_state,
            "observation.images.top": img0,
            "observation.images.wrist": img1,
        }
        with torch.inference_mode():
            torch_out = tensor_to_numpy(torch_core_forward(policy, obs_state, img0, img1))
            trt_out = tensor_to_numpy(trt_policy.predict_action_chunk(batch))

        metrics = compare_arrays(torch_out, trt_out)
        max_abs_list.append(metrics["max_abs_diff"])
        mean_abs_list.append(metrics["mean_abs_diff"])
        max_rel_list.append(metrics["max_rel_diff"])
        cosine_list.append(metrics["cosine_similarity"])
        all_exact = all_exact and metrics["exact_equal"]
        all_within_threshold = all_within_threshold and (
            metrics["max_abs_diff"] <= args.threshold_max_abs_diff
        )
        case_reports.append(
            {
                "case": case_name,
                "kind": kind,
                "seed": seed,
                **metrics,
            }
        )

    payload = {
        "checkpoint": str(checkpoint),
        "resolved_checkpoint": str(checkpoint),
        "engine": str(engine_path),
        "metadata": str(metadata_path) if metadata_path.is_file() else None,
        "engine_checkpoint_from_metadata": metadata.get("checkpoint_dir") if metadata else None,
        "environment": Path(sys.executable).parent.parent.name,
        "trt_device": args.trt_device,
        "threshold_max_abs_diff": float(args.threshold_max_abs_diff),
        "spec": {
            "visual_keys": spec.visual_keys,
            "obs_state_shape": list(spec.obs_state_shape),
            "image_shape": list(spec.image_shape),
            "action_shape": list(spec.action_shape),
        },
        "engine_io_mapping": {
            "state_input_name": trt_policy.io_mapping.state_input_name,
            "camera_input_names": trt_policy.io_mapping.camera_input_names,
            "output_name": trt_policy.io_mapping.output_name,
        },
        "summary": {
            "num_cases": len(case_reports),
            "all_exact_equal": all_exact,
            "all_within_threshold": all_within_threshold,
            "max_abs_diff": max(max_abs_list),
            "max_mean_abs_diff": max(mean_abs_list),
            "max_rel_diff": max(max_rel_list),
            "min_cosine_similarity": min(cosine_list),
        },
        "cases": case_reports,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(report_path)
    print(json.dumps(payload["summary"], indent=2))
    print("TRT checkpoint owner:", payload["engine_checkpoint_from_metadata"] or "<unknown>")
    return 0 if payload["summary"]["all_within_threshold"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
