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

from lerobot.configs.policies import PreTrainedConfig
from act_model_utils import load_act_policy, load_model_spec, make_case_inputs, tensor_to_numpy, torch_core_forward
from trt_act_policy import TrtActPolicyAdapter


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "outputs" / "act_grasp_block_in_bin1" / "checkpoints" / "016000" / "pretrained_model"
)
DEFAULT_ENGINE = (
    REPO_ROOT
    / "outputs"
    / "deploy"
    / "act_trt"
    / "act_grasp_block_in_bin1"
    / "016000"
    / "act_single_fp32.plan"
)
DEFAULT_METADATA = (
    REPO_ROOT
    / "outputs"
    / "deploy"
    / "act_trt"
    / "act_grasp_block_in_bin1"
    / "016000"
    / "export_metadata.json"
)
DEFAULT_REPORT_JSON = DOCS_DIR / "ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json"
DEFAULT_REPORT_MD = DOCS_DIR / "ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ACT 016000 model.safetensors against my exported 016000 TensorRT engine."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--trt-device", type=str, default="cuda:0")
    parser.add_argument("--threshold-max-abs-diff", type=float, default=1e-4)
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
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


def build_cases(random_cases: int) -> list[tuple[str, str, int | None]]:
    cases: list[tuple[str, str, int | None]] = []
    for seed in range(int(random_cases)):
        cases.append((f"random_seed_{seed}", "random", seed))
    cases.extend(
        [
            ("zeros", "zeros", None),
            ("ones", "ones", None),
            ("linspace", "linspace", None),
        ]
    )
    return cases


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metadata_checkpoint(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    for key in ("checkpoint", "checkpoint_dir"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def write_markdown_report(report_path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    worst_case = max(payload["cases"], key=lambda case: case["max_abs_diff"])
    text = "\n".join(
        [
            "# ACT Compare: 016000 Safetensors vs My Exported TRT",
            "",
            "## Artifacts",
            "",
            f"- checkpoint: `{payload['checkpoint']}`",
            f"- engine: `{payload['engine']}`",
            f"- metadata: `{payload['metadata']}`",
            f"- trt_device: `{payload['trt_device']}`",
            "",
            "## Summary",
            "",
            f"- num_cases: `{summary['num_cases']}`",
            f"- all_within_threshold: `{summary['all_within_threshold']}`",
            f"- threshold_max_abs_diff: `{payload['threshold_max_abs_diff']}`",
            f"- max_abs_diff: `{summary['max_abs_diff']}`",
            f"- max_mean_abs_diff: `{summary['max_mean_abs_diff']}`",
            f"- max_rel_diff: `{summary['max_rel_diff']}`",
            f"- min_cosine_similarity: `{summary['min_cosine_similarity']}`",
            "",
            "## Engine IO",
            "",
            f"- state_input_name: `{payload['engine_io_mapping']['state_input_name']}`",
            f"- camera_input_names: `{payload['engine_io_mapping']['camera_input_names']}`",
            f"- output_name: `{payload['engine_io_mapping']['output_name']}`",
            "",
            "## Worst Case",
            "",
            f"- case: `{worst_case['case']}`",
            f"- max_abs_diff: `{worst_case['max_abs_diff']}`",
            f"- mean_abs_diff: `{worst_case['mean_abs_diff']}`",
            f"- cosine_similarity: `{worst_case['cosine_similarity']}`",
            "",
            "## Conclusion",
            "",
            (
                "The 016000 checkpoint restored from `model.safetensors` is numerically aligned with "
                "my exported 016000 TensorRT engine."
                if summary["all_within_threshold"]
                else "The compared TensorRT engine does not match the 016000 checkpoint within the configured threshold."
            ),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    engine_path = args.engine.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve() if args.metadata else None
    report_json_path = args.report_json.expanduser().resolve()
    report_md_path = args.report_md.expanduser().resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"TensorRT metadata file not found: {metadata_path}")

    spec = load_model_spec(checkpoint)
    torch_policy = load_act_policy(checkpoint, device="cpu")
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.device = "cuda"
    trt_policy = TrtActPolicyAdapter(
        policy_cfg,
        engine_path=engine_path,
        metadata_path=metadata_path,
        trt_device=args.trt_device,
    )
    metadata = load_json(metadata_path)

    cases_payload: list[dict[str, Any]] = []
    max_abs_list: list[float] = []
    mean_abs_list: list[float] = []
    max_rel_list: list[float] = []
    cosine_list: list[float] = []
    all_exact = True
    all_within_threshold = True

    for case_name, case_kind, seed in build_cases(args.random_cases):
        obs_state, img0, img1 = make_case_inputs(spec=spec, seed=seed, case=case_kind)
        trt_batch = {
            "observation.state": obs_state,
            "observation.images.top": img0,
            "observation.images.wrist": img1,
        }
        with torch.inference_mode():
            torch_out = tensor_to_numpy(torch_core_forward(torch_policy, obs_state, img0, img1))
            trt_out = tensor_to_numpy(trt_policy.predict_action_chunk(trt_batch))

        metrics = compare_arrays(torch_out, trt_out)
        max_abs_list.append(metrics["max_abs_diff"])
        mean_abs_list.append(metrics["mean_abs_diff"])
        max_rel_list.append(metrics["max_rel_diff"])
        cosine_list.append(metrics["cosine_similarity"])
        all_exact = all_exact and metrics["exact_equal"]
        all_within_threshold = all_within_threshold and (
            metrics["max_abs_diff"] <= args.threshold_max_abs_diff
        )
        cases_payload.append(
            {
                "case": case_name,
                "kind": case_kind,
                "seed": seed,
                **metrics,
            }
        )

    payload = {
        "comparison_name": "act_016000_safetensors_vs_my_exported_trt",
        "checkpoint": str(checkpoint),
        "engine": str(engine_path),
        "metadata": str(metadata_path) if metadata_path else None,
        "engine_checkpoint_from_metadata": resolve_metadata_checkpoint(metadata),
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
            "num_cases": len(cases_payload),
            "all_exact_equal": all_exact,
            "all_within_threshold": all_within_threshold,
            "max_abs_diff": max(max_abs_list),
            "max_mean_abs_diff": max(mean_abs_list),
            "max_rel_diff": max(max_rel_list),
            "min_cosine_similarity": min(cosine_list),
        },
        "cases": cases_payload,
    }

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_report(report_md_path, payload)

    print(f"json_report={report_json_path}")
    print(f"md_report={report_md_path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if payload["summary"]["all_within_threshold"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
