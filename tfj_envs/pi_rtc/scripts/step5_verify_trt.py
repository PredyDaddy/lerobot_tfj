#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from common import build_metadata_skeleton, metadata_path, prepare_run_layout, read_json, resolve_checkpoint_dir, write_json
from export_wrappers import (
    Pi05DenoiseStepExportWrapper,
    Pi05PrefixCacheExportWrapper,
    Pi05VisionEncoderExportWrapper,
    cache_tensor_names,
)
from pi_compare_common import (
    build_runtime_context,
    compare_arrays,
    feed_value_to_numpy,
    lazy_import_pi05_modules,
    metadata_note_payload,
    run_onnx_with_fallback,
    summarize_metric_map,
    tensor_to_numpy,
    write_markdown,
)
from trt_runtime import TensorRTRunner


PAIRWISE_COMPARISONS = ("torch_vs_onnx", "torch_vs_trt", "onnx_vs_trt")
SUBGRAPH_FILES = {
    "vision_encoder": {
        "onnx": "pi_shared_vision_encoder.onnx",
        "engine": "pi_shared_vision_encoder.engine",
    },
    "prefix_cache": {
        "onnx": "pi_shared_prefix_cache.onnx",
        "engine": "pi_shared_prefix_cache.engine",
    },
    "denoise_step": {
        "onnx": "pi05_denoise_step.onnx",
        "engine": "pi05_denoise_step.engine",
    },
}
DEFAULT_SUBGRAPHS = ["vision_encoder", "prefix_cache", "denoise_step", "pipeline"]
ONNX_COMPARE_PROFILES = {
    "export_fidelity": {
        "display_name": "Export Fidelity ONNX",
        "purpose": (
            "Primary ONNX baseline for Stage 5 metrics. Align to the Stage 2 export boundary "
            "without conflating Torch/ONNX/TRT comparisons with CUDA runtime Torch drift."
        ),
        "subgraphs": {
            "vision_encoder": {
                "provider_candidates": [["CPUExecutionProvider"]],
                "optimization_order": ["all"],
            },
            "prefix_cache": {
                "provider_candidates": [["CPUExecutionProvider"]],
                "optimization_order": ["disable"],
            },
            "denoise_step": {
                "provider_candidates": [
                    ["CPUExecutionProvider"],
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ],
                "optimization_order": ["disable", "basic", "all"],
            },
        },
    },
    "runtime_oriented": {
        "display_name": "Runtime-Oriented ONNX",
        "purpose": (
            "Optional runtime-oriented ONNX execution profile. Keep CUDA-preferred optimized ORT "
            "separate from the primary export-fidelity metric."
        ),
        "subgraphs": {
            "vision_encoder": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
            },
            "prefix_cache": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
            },
            "denoise_step": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
            },
        },
    },
}
PRIMARY_ONNX_COMPARE_PROFILE = "export_fidelity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5: compare PI05 Torch / ONNX / TensorRT subgraphs.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument("--engine-dir", default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--subgraphs",
        nargs="+",
        choices=DEFAULT_SUBGRAPHS,
        default=DEFAULT_SUBGRAPHS,
        help="Subset of verify jobs to run. 'pipeline' means the one-step vision->prefix->denoise chain.",
    )
    parser.add_argument("--max-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-4)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after the first execution error.",
    )
    return parser


def _resolve_artifact_paths(
    *,
    run_dir: Path,
    onnx_dir: Path,
    engine_dir: Path,
    onnx_dir_explicit: bool,
    engine_dir_explicit: bool,
) -> dict[str, dict[str, Path]]:
    metadata_file = metadata_path(run_dir)
    metadata = read_json(metadata_file) if metadata_file.is_file() else {}
    metadata_onnx = metadata.get("onnx_paths", {})
    metadata_engines = metadata.get("engine_paths", {})

    artifacts: dict[str, dict[str, Path]] = {}
    for subgraph, names in SUBGRAPH_FILES.items():
        if onnx_dir_explicit:
            resolved_onnx = (onnx_dir / names["onnx"]).resolve()
        else:
            onnx_candidate = Path(metadata_onnx.get(subgraph, onnx_dir / names["onnx"]))
            resolved_onnx = onnx_candidate if onnx_candidate.is_absolute() else (onnx_dir / onnx_candidate).resolve()

        if engine_dir_explicit:
            resolved_engine = (engine_dir / names["engine"]).resolve()
        else:
            engine_candidate = Path(metadata_engines.get(subgraph, engine_dir / names["engine"]))
            resolved_engine = (
                engine_candidate if engine_candidate.is_absolute() else (engine_dir / engine_candidate).resolve()
            )

        artifacts[subgraph] = {
            "onnx": resolved_onnx,
            "engine": resolved_engine,
        }
    return artifacts


def _tensor_spec(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = [int(dim) for dim in value.shape]
        dtype = str(value.dtype)
    else:
        raise TypeError(f"Unsupported tensor-like value: {type(value)}")
    return {
        "shape": shape,
        "dtype": dtype.replace("torch.", ""),
    }


def _onnx_profile_request(profile_key: str, subgraph_name: str) -> dict[str, Any]:
    profile = ONNX_COMPARE_PROFILES[profile_key]
    subgraph_profile = profile["subgraphs"][subgraph_name]
    return {
        "profile_key": profile_key,
        "profile_display_name": profile["display_name"],
        "subgraph_name": subgraph_name,
        "provider_candidates": [list(candidate) for candidate in subgraph_profile["provider_candidates"]],
        "optimization_order": list(subgraph_profile["optimization_order"]),
    }


def _run_onnx_with_profile(
    *,
    onnx_path: Path,
    output_names: list[str],
    input_feed: dict[str, Any],
    profile_key: str,
    subgraph_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = _onnx_profile_request(profile_key, subgraph_name)
    outputs, runtime_info = run_onnx_with_fallback(
        onnx_path,
        output_names,
        input_feed,
        provider_candidates_override=request["provider_candidates"],
        optimization_order=request["optimization_order"],
    )
    runtime_info["compare_profile"] = request
    return outputs, runtime_info


def _runtime_summary(runtime_info: dict[str, Any] | None) -> str:
    if not runtime_info:
        return "not_available"
    return (
        f"active_providers={runtime_info.get('active_providers', [])}, "
        f"graph_optimization_level={runtime_info.get('graph_optimization_level', 'unknown')}"
    )


def _status_from_metrics(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    reasons = []
    if not all(math.isfinite(value) for value in metrics.values()):
        reasons.append("contains_non_finite_metric")
    if metrics["max_abs_diff"] > thresholds["max_abs_diff"]:
        reasons.append(
            f"max_abs_diff {metrics['max_abs_diff']:.6g} > threshold {thresholds['max_abs_diff']:.6g}"
        )
    if metrics["mean_abs_diff"] > thresholds["mean_abs_diff"]:
        reasons.append(
            f"mean_abs_diff {metrics['mean_abs_diff']:.6g} > threshold {thresholds['mean_abs_diff']:.6g}"
        )
    cosine_similarity = metrics.get("cosine_similarity", metrics.get("min_cosine_similarity"))
    if cosine_similarity is None:
        reasons.append("missing_cosine_similarity_metric")
    elif cosine_similarity < thresholds["min_cosine_similarity"]:
        reasons.append(
            "cosine_similarity "
            f"{cosine_similarity:.6g} < threshold {thresholds['min_cosine_similarity']:.6g}"
        )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "thresholds": thresholds,
    }


def _build_output_metrics(
    *,
    outputs: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    per_output: dict[str, dict[str, Any]] = {}
    for output_name, frameworks in outputs.items():
        pairwise: dict[str, Any] = {}
        torch_value = frameworks.get("torch")
        onnx_value = frameworks.get("onnx")
        trt_value = frameworks.get("trt")

        if torch_value is not None and onnx_value is not None:
            metrics = compare_arrays(torch_value, onnx_value)
            pairwise["torch_vs_onnx"] = {
                "metrics": metrics,
                "status": _status_from_metrics(metrics, thresholds),
            }
        if torch_value is not None and trt_value is not None:
            metrics = compare_arrays(torch_value, trt_value)
            pairwise["torch_vs_trt"] = {
                "metrics": metrics,
                "status": _status_from_metrics(metrics, thresholds),
            }
        if onnx_value is not None and trt_value is not None:
            metrics = compare_arrays(onnx_value, trt_value)
            pairwise["onnx_vs_trt"] = {
                "metrics": metrics,
                "status": _status_from_metrics(metrics, thresholds),
            }

        per_output[output_name] = pairwise

    summary: dict[str, dict[str, float]] = {}
    pairwise_status: dict[str, dict[str, Any]] = {}
    for pair_name in PAIRWISE_COMPARISONS:
        available = {
            output_name: pairwise[pair_name]["metrics"]
            for output_name, pairwise in per_output.items()
            if pair_name in pairwise
        }
        if not available:
            continue
        summary[pair_name] = summarize_metric_map(available)
        pairwise_status[pair_name] = _status_from_metrics(summary[pair_name], thresholds)

    return {
        "outputs": per_output,
        "summary": summary,
        "pairwise_status": pairwise_status,
        "missing_pairs": [pair for pair in PAIRWISE_COMPARISONS if pair not in summary],
    }


def _subgraph_status(
    *,
    errors: list[str],
    metric_payload: dict[str, Any],
) -> str:
    if errors or metric_payload["missing_pairs"]:
        return "error"
    if any(not info["pass"] for info in metric_payload["pairwise_status"].values()):
        return "fail"
    return "pass"


def _overall_status(results: dict[str, dict[str, Any]]) -> str:
    statuses = [entry["status"] for entry in results.values()]
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PI05 Torch / ONNX / TensorRT Report",
        "",
        f"- policy: `{report['policy_dir']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- variant: `{report.get('variant')}`",
        f"- checkpoint_dir: `{report.get('checkpoint_dir')}`",
        f"- requested_precision: `{report.get('trt_artifact_provenance', {}).get('requested_precision')}`",
        f"- stage4_report_path: `{report.get('trt_artifact_provenance', {}).get('stage4_report_path')}`",
        f"- torch_reference_mode: `{report['torch_reference_mode']['name']}`",
        "- stage5_scope: `export-boundary single-step correctness gate`",
        "",
        "## ONNX Compare Profiles",
        "",
        f"- primary_onnx_compare_profile: `{report['primary_onnx_compare_profile']}`",
    ]
    for profile_key, profile in report["onnx_compare_profiles"].items():
        lines.append(f"- {profile['display_name']} (`{profile_key}`)")
        lines.append(f"  - purpose: `{profile['purpose']}`")
        for subgraph_name in ("vision_encoder", "prefix_cache", "denoise_step"):
            subgraph_profile = profile["subgraphs"][subgraph_name]
            lines.append(
                "  - "
                f"`{subgraph_name}` providers={subgraph_profile['provider_candidates']} "
                f"optimization_order={subgraph_profile['optimization_order']}"
            )

    lines.extend(
        [
            "",
            "## Subgraphs",
            "",
        ]
    )
    for name in report["requested_subgraphs"]:
        subgraph = report["results"].get(name)
        if subgraph is None:
            continue
        lines.append(f"- {name}: `{subgraph['status']}`")
        if subgraph.get("onnx_compare_profile") is not None:
            lines.append(f"  - onnx_compare_profile: `{subgraph['onnx_compare_profile']}`")
        if subgraph["comparison"]["summary"]:
            for pair_name, metrics in subgraph["comparison"]["summary"].items():
                lines.append(f"  - {pair_name}: {metrics}")
        onnx_runtime = subgraph.get("onnx_runtime")
        if isinstance(onnx_runtime, dict):
            if name == "vision_encoder":
                for label in ("top", "wrist"):
                    if label in onnx_runtime:
                        lines.append(f"  - onnx_runtime[{label}]: `{_runtime_summary(onnx_runtime[label])}`")
            elif name == "pipeline":
                for label in ("vision_top", "vision_wrist", "prefix_cache", "denoise_step"):
                    if label in onnx_runtime:
                        lines.append(f"  - onnx_runtime[{label}]: `{_runtime_summary(onnx_runtime[label])}`")
            elif "active_providers" in onnx_runtime:
                lines.append(f"  - onnx_runtime: `{_runtime_summary(onnx_runtime)}`")
        if subgraph["errors"]:
            lines.append(f"  - errors: {subgraph['errors']}")
    lines.append("")
    return "\n".join(lines)


def _resolve_metadata_stage_report(run_dir: Path, metadata: dict[str, Any], key: str) -> tuple[Path | None, dict[str, Any] | None]:
    artifact_paths = metadata.get("artifacts", {}) if isinstance(metadata, dict) else {}
    candidates: list[Path] = []
    raw_path = artifact_paths.get(key) if isinstance(artifact_paths, dict) else None
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            candidates.append(candidate.resolve(strict=False))
        else:
            candidates.append((run_dir / candidate).resolve(strict=False))
    candidates.append((run_dir / f"{key}.json").resolve(strict=False))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            payload = read_json(candidate)
            if isinstance(payload, dict):
                return candidate, payload
    return None, None


def _resolve_stage4_provenance(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    stage4_report_path, stage4_report = _resolve_metadata_stage_report(run_dir, metadata, "stage4_build_engines")
    requested_precision = None
    if isinstance(stage4_report, dict):
        build_settings = stage4_report.get("build_settings", {})
        if isinstance(build_settings, dict) and build_settings.get("precision") is not None:
            requested_precision = str(build_settings["precision"])
    if requested_precision is None:
        if metadata.get("requested_trt_precision") is not None:
            requested_precision = str(metadata["requested_trt_precision"])
        else:
            engine_build_settings = metadata.get("engine_build_settings", {})
            if isinstance(engine_build_settings, dict) and engine_build_settings.get("precision") is not None:
                requested_precision = str(engine_build_settings["precision"])

    return {
        "variant": metadata.get("variant"),
        "checkpoint_dir": metadata.get("checkpoint_dir"),
        "requested_precision": requested_precision,
        "stage4_report_path": stage4_report_path.as_posix() if stage4_report_path is not None else None,
        "stage4_overall_status": stage4_report.get("overall_status") if isinstance(stage4_report, dict) else None,
        "stage4_build_settings": stage4_report.get("build_settings") if isinstance(stage4_report, dict) else None,
        "effective_precision_evidence": metadata.get("trt_effective_precision_evidence"),
        "note": (
            "Stage 5 is an export-boundary single-step correctness gate. "
            "It does not by itself prove full runtime correctness or multi-step chunk stability."
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    layout = prepare_run_layout(args.run_dir, prefix="pi05_trt")
    run_dir = layout["run_dir"]
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else layout["onnx_dir"]
    engine_dir = Path(args.engine_dir).expanduser().resolve() if args.engine_dir else layout["engines_dir"]
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (run_dir / "stage5_verify_trt.json")
    )
    markdown_path = (
        Path(args.markdown_path).expanduser().resolve()
        if args.markdown_path
        else (run_dir / "stage5_verify_trt.md")
    )
    thresholds = {
        "max_abs_diff": float(args.max_abs_threshold),
        "mean_abs_diff": float(args.mean_abs_threshold),
        "min_cosine_similarity": float(args.min_cosine_similarity),
    }

    policy_dir = resolve_checkpoint_dir(args.policy_path)
    metadata_file = metadata_path(run_dir)
    if metadata_file.is_file():
        metadata = read_json(metadata_file)
    else:
        metadata = build_metadata_skeleton(run_dir=run_dir, variant="pi05", checkpoint_dir=policy_dir)
    artifacts = _resolve_artifact_paths(
        run_dir=run_dir,
        onnx_dir=onnx_dir,
        engine_dir=engine_dir,
        onnx_dir_explicit=bool(args.onnx_dir),
        engine_dir_explicit=bool(args.engine_dir),
    )
    primary_onnx_profile = PRIMARY_ONNX_COMPARE_PROFILE
    stage4_provenance = _resolve_stage4_provenance(run_dir, metadata)

    modules = lazy_import_pi05_modules()
    context = build_runtime_context(policy_dir, strict=args.strict_load)
    num_layers = int(context.policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)
    prefix_output_names = ["prefix_pad_masks", *cache_tensor_names(num_layers)]

    context.policy.modeling_make_att_2d_masks = modules["make_att_2d_masks"]
    export_top_image = context.top_image.detach().cpu()
    export_wrist_image = context.wrist_image.detach().cpu()
    export_image_mask_top = context.image_mask_top.detach().cpu()
    export_image_mask_wrist = context.image_mask_wrist.detach().cpu()
    export_tokens = context.tokens.detach().cpu()
    export_token_attention_mask = context.token_attention_mask.detach().cpu()
    export_x_t = context.x_t.detach().cpu()
    export_timestep = context.timestep.detach().cpu()

    context.policy.cpu()
    context.policy.float()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vision_wrapper = Pi05VisionEncoderExportWrapper(context.policy, use_autocast=False).eval()
    prefix_wrapper = Pi05PrefixCacheExportWrapper(
        context.policy,
        num_layers=num_layers,
        use_autocast=False,
    ).eval()
    denoise_wrapper = Pi05DenoiseStepExportWrapper(
        context.policy,
        num_layers=num_layers,
        dynamic_cache_cls=modules["DynamicCache"],
        use_autocast=False,
    ).eval()

    with torch.no_grad():
        top_torch_tensor = vision_wrapper(export_top_image)
        wrist_torch_tensor = vision_wrapper(export_wrist_image)
        prefix_torch_outputs = prefix_wrapper(
            top_torch_tensor,
            wrist_torch_tensor,
            export_image_mask_top,
            export_image_mask_wrist,
            export_tokens,
            export_token_attention_mask,
        )
        denoise_torch_tensor = denoise_wrapper(
            export_x_t,
            export_timestep,
            prefix_torch_outputs[0],
            *prefix_torch_outputs[1:],
        )

    top_torch = tensor_to_numpy(top_torch_tensor)
    wrist_torch = tensor_to_numpy(wrist_torch_tensor)
    prefix_torch_map = {
        name: tensor_to_numpy(tensor)
        for name, tensor in zip(prefix_output_names, prefix_torch_outputs, strict=True)
    }
    denoise_torch = tensor_to_numpy(denoise_torch_tensor)

    canonical_prefix_input_torch = {
        "image_embs_top": top_torch_tensor,
        "image_embs_wrist": wrist_torch_tensor,
        "image_mask_top": export_image_mask_top,
        "image_mask_wrist": export_image_mask_wrist,
        "tokens": export_tokens,
        "token_attention_mask": export_token_attention_mask,
    }
    canonical_prefix_input_onnx = {
        "image_embs_top": top_torch,
        "image_embs_wrist": wrist_torch,
        "image_mask_top": feed_value_to_numpy(export_image_mask_top),
        "image_mask_wrist": feed_value_to_numpy(export_image_mask_wrist),
        "tokens": feed_value_to_numpy(export_tokens),
        "token_attention_mask": feed_value_to_numpy(export_token_attention_mask),
    }
    canonical_denoise_input_torch = {
        "x_t": export_x_t,
        "timestep": export_timestep,
        "prefix_pad_masks": prefix_torch_outputs[0],
        **{
            name: tensor
            for name, tensor in zip(cache_tensor_names(num_layers), prefix_torch_outputs[1:], strict=True)
        },
    }
    canonical_denoise_input_onnx = {
        "x_t": feed_value_to_numpy(export_x_t),
        "timestep": feed_value_to_numpy(export_timestep),
        "prefix_pad_masks": prefix_torch_map["prefix_pad_masks"],
        **{
            name: prefix_torch_map[name]
            for name in cache_tensor_names(num_layers)
        },
    }

    requested_subgraphs = list(dict.fromkeys(args.subgraphs))
    results: dict[str, dict[str, Any]] = {}

    if "vision_encoder" in requested_subgraphs:
        errors: list[str] = []
        onnx_outputs: dict[str, Any] = {}
        onnx_runtime: dict[str, Any] = {}
        trt_outputs: dict[str, Any] = {}
        trt_runner_summary = None

        vision_onnx_path = artifacts["vision_encoder"]["onnx"]
        vision_engine_path = artifacts["vision_encoder"]["engine"]

        if vision_onnx_path.is_file():
            try:
                onnx_outputs["top"], onnx_runtime["top"] = _run_onnx_with_profile(
                    onnx_path=vision_onnx_path,
                    output_names=["image_embs"],
                    input_feed={"image": feed_value_to_numpy(export_top_image)},
                    profile_key=primary_onnx_profile,
                    subgraph_name="vision_encoder",
                )
                onnx_outputs["wrist"], onnx_runtime["wrist"] = _run_onnx_with_profile(
                    onnx_path=vision_onnx_path,
                    output_names=["image_embs"],
                    input_feed={"image": feed_value_to_numpy(export_wrist_image)},
                    profile_key=primary_onnx_profile,
                    subgraph_name="vision_encoder",
                )
            except Exception as exc:
                errors.append(f"ONNX vision_encoder failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing ONNX artifact: {vision_onnx_path}")

        if vision_engine_path.is_file():
            try:
                with TensorRTRunner(vision_engine_path, device=args.device) as vision_trt:
                    trt_runner_summary = vision_trt.engine_summary()
                    trt_outputs["top"] = tensor_to_numpy(vision_trt.infer({"image": export_top_image})["image_embs"])
                    trt_outputs["wrist"] = tensor_to_numpy(
                        vision_trt.infer({"image": export_wrist_image})["image_embs"]
                    )
            except Exception as exc:
                errors.append(f"TensorRT vision_encoder failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing engine artifact: {vision_engine_path}")

        comparison = _build_output_metrics(
            outputs={
                "top": {
                    "torch": top_torch,
                    "onnx": None if "top" not in onnx_outputs else onnx_outputs["top"]["image_embs"],
                    "trt": trt_outputs.get("top"),
                },
                "wrist": {
                    "torch": wrist_torch,
                    "onnx": None if "wrist" not in onnx_outputs else onnx_outputs["wrist"]["image_embs"],
                    "trt": trt_outputs.get("wrist"),
                },
            },
            thresholds=thresholds,
        )
        results["vision_encoder"] = {
            "status": _subgraph_status(errors=errors, metric_payload=comparison),
            "comparison_basis": (
                "same export-reference image tensors fed to Torch, ONNX, and TensorRT vision encoder"
            ),
            "artifacts": {
                "onnx": vision_onnx_path.as_posix(),
                "engine": vision_engine_path.as_posix(),
            },
            "onnx_compare_profile": primary_onnx_profile,
            "input_specs": {
                "top_image": _tensor_spec(export_top_image),
                "wrist_image": _tensor_spec(export_wrist_image),
            },
            "onnx_runtime": onnx_runtime,
            "trt_engine": trt_runner_summary,
            "comparison": comparison,
            "errors": errors,
        }

    if "prefix_cache" in requested_subgraphs:
        errors = []
        prefix_onnx_outputs = None
        prefix_onnx_runtime = None
        prefix_trt_outputs = None
        prefix_trt_summary = None

        prefix_onnx_path = artifacts["prefix_cache"]["onnx"]
        prefix_engine_path = artifacts["prefix_cache"]["engine"]

        if prefix_onnx_path.is_file():
            try:
                prefix_onnx_outputs, prefix_onnx_runtime = _run_onnx_with_profile(
                    onnx_path=prefix_onnx_path,
                    output_names=prefix_output_names,
                    input_feed=canonical_prefix_input_onnx,
                    profile_key=primary_onnx_profile,
                    subgraph_name="prefix_cache",
                )
            except Exception as exc:
                errors.append(f"ONNX prefix_cache failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing ONNX artifact: {prefix_onnx_path}")

        if prefix_engine_path.is_file():
            try:
                with TensorRTRunner(prefix_engine_path, device=args.device) as prefix_trt:
                    prefix_trt_summary = prefix_trt.engine_summary()
                    prefix_trt_outputs = {
                        name: tensor_to_numpy(value)
                        for name, value in prefix_trt.infer(canonical_prefix_input_torch).items()
                    }
            except Exception as exc:
                errors.append(f"TensorRT prefix_cache failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing engine artifact: {prefix_engine_path}")

        comparison = _build_output_metrics(
            outputs={
                name: {
                    "torch": prefix_torch_map[name],
                    "onnx": None if prefix_onnx_outputs is None else prefix_onnx_outputs.get(name),
                    "trt": None if prefix_trt_outputs is None else prefix_trt_outputs.get(name),
                }
                for name in prefix_output_names
            },
            thresholds=thresholds,
        )
        results["prefix_cache"] = {
            "status": _subgraph_status(errors=errors, metric_payload=comparison),
            "comparison_basis": (
                "same export-reference prefix inputs: export-reference Torch vision outputs + "
                "export-reference masks/tokens fed to all frameworks"
            ),
            "artifacts": {
                "onnx": prefix_onnx_path.as_posix(),
                "engine": prefix_engine_path.as_posix(),
            },
            "onnx_compare_profile": primary_onnx_profile,
            "input_specs": {name: _tensor_spec(value) for name, value in canonical_prefix_input_torch.items()},
            "onnx_runtime": prefix_onnx_runtime,
            "trt_engine": prefix_trt_summary,
            "comparison": comparison,
            "errors": errors,
        }

    if "denoise_step" in requested_subgraphs:
        errors = []
        denoise_onnx_outputs = None
        denoise_onnx_runtime = None
        denoise_trt_outputs = None
        denoise_trt_summary = None

        denoise_onnx_path = artifacts["denoise_step"]["onnx"]
        denoise_engine_path = artifacts["denoise_step"]["engine"]

        if denoise_onnx_path.is_file():
            try:
                denoise_onnx_outputs, denoise_onnx_runtime = _run_onnx_with_profile(
                    onnx_path=denoise_onnx_path,
                    output_names=["v_t"],
                    input_feed=canonical_denoise_input_onnx,
                    profile_key=primary_onnx_profile,
                    subgraph_name="denoise_step",
                )
            except Exception as exc:
                errors.append(f"ONNX denoise_step failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing ONNX artifact: {denoise_onnx_path}")

        if denoise_engine_path.is_file():
            try:
                with TensorRTRunner(denoise_engine_path, device=args.device) as denoise_trt:
                    denoise_trt_summary = denoise_trt.engine_summary()
                    denoise_trt_outputs = {
                        name: tensor_to_numpy(value)
                        for name, value in denoise_trt.infer(canonical_denoise_input_torch).items()
                    }
            except Exception as exc:
                errors.append(f"TensorRT denoise_step failed: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    raise
        else:
            errors.append(f"Missing engine artifact: {denoise_engine_path}")

        comparison = _build_output_metrics(
            outputs={
                "v_t": {
                    "torch": denoise_torch,
                    "onnx": None if denoise_onnx_outputs is None else denoise_onnx_outputs.get("v_t"),
                    "trt": None if denoise_trt_outputs is None else denoise_trt_outputs.get("v_t"),
                }
            },
            thresholds=thresholds,
        )
        results["denoise_step"] = {
            "status": _subgraph_status(errors=errors, metric_payload=comparison),
            "comparison_basis": (
                "same export-reference denoise inputs: export-reference x_t, timestep, and "
                "export-reference Torch prefix-cache outputs"
            ),
            "artifacts": {
                "onnx": denoise_onnx_path.as_posix(),
                "engine": denoise_engine_path.as_posix(),
            },
            "onnx_compare_profile": primary_onnx_profile,
            "input_specs": {name: _tensor_spec(value) for name, value in canonical_denoise_input_torch.items()},
            "onnx_runtime": denoise_onnx_runtime,
            "trt_engine": denoise_trt_summary,
            "comparison": comparison,
            "errors": errors,
        }

    if "pipeline" in requested_subgraphs:
        errors = []
        pipeline_onnx_runtime: dict[str, Any] = {}
        pipeline_trt_summary: dict[str, Any] = {}
        onnx_pipeline_v_t = None
        trt_pipeline_v_t = None

        try:
            vision_top_onnx, pipeline_onnx_runtime["vision_top"] = _run_onnx_with_profile(
                onnx_path=artifacts["vision_encoder"]["onnx"],
                output_names=["image_embs"],
                input_feed={"image": feed_value_to_numpy(export_top_image)},
                profile_key=primary_onnx_profile,
                subgraph_name="vision_encoder",
            )
            vision_wrist_onnx, pipeline_onnx_runtime["vision_wrist"] = _run_onnx_with_profile(
                onnx_path=artifacts["vision_encoder"]["onnx"],
                output_names=["image_embs"],
                input_feed={"image": feed_value_to_numpy(export_wrist_image)},
                profile_key=primary_onnx_profile,
                subgraph_name="vision_encoder",
            )
            pipeline_prefix_feed = {
                "image_embs_top": vision_top_onnx["image_embs"],
                "image_embs_wrist": vision_wrist_onnx["image_embs"],
                "image_mask_top": feed_value_to_numpy(export_image_mask_top),
                "image_mask_wrist": feed_value_to_numpy(export_image_mask_wrist),
                "tokens": feed_value_to_numpy(export_tokens),
                "token_attention_mask": feed_value_to_numpy(export_token_attention_mask),
            }
            prefix_onnx_outputs, pipeline_onnx_runtime["prefix_cache"] = _run_onnx_with_profile(
                onnx_path=artifacts["prefix_cache"]["onnx"],
                output_names=prefix_output_names,
                input_feed=pipeline_prefix_feed,
                profile_key=primary_onnx_profile,
                subgraph_name="prefix_cache",
            )
            pipeline_denoise_feed = {
                "x_t": feed_value_to_numpy(export_x_t),
                "timestep": feed_value_to_numpy(export_timestep),
                "prefix_pad_masks": prefix_onnx_outputs["prefix_pad_masks"],
                **{name: prefix_onnx_outputs[name] for name in cache_tensor_names(num_layers)},
            }
            denoise_onnx_outputs, pipeline_onnx_runtime["denoise_step"] = _run_onnx_with_profile(
                onnx_path=artifacts["denoise_step"]["onnx"],
                output_names=["v_t"],
                input_feed=pipeline_denoise_feed,
                profile_key=primary_onnx_profile,
                subgraph_name="denoise_step",
            )
            onnx_pipeline_v_t = denoise_onnx_outputs["v_t"]
        except Exception as exc:
            errors.append(f"ONNX pipeline failed: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

        try:
            with TensorRTRunner(artifacts["vision_encoder"]["engine"], device=args.device) as vision_trt:
                pipeline_trt_summary["vision_encoder"] = vision_trt.engine_summary()
                vision_top_trt = vision_trt.infer({"image": export_top_image})["image_embs"]
                vision_wrist_trt = vision_trt.infer({"image": export_wrist_image})["image_embs"]

            with TensorRTRunner(artifacts["prefix_cache"]["engine"], device=args.device) as prefix_trt:
                pipeline_trt_summary["prefix_cache"] = prefix_trt.engine_summary()
                prefix_trt_outputs = prefix_trt.infer(
                    {
                        "image_embs_top": vision_top_trt,
                        "image_embs_wrist": vision_wrist_trt,
                        "image_mask_top": export_image_mask_top,
                        "image_mask_wrist": export_image_mask_wrist,
                        "tokens": export_tokens,
                        "token_attention_mask": export_token_attention_mask,
                    }
                )

            with TensorRTRunner(artifacts["denoise_step"]["engine"], device=args.device) as denoise_trt:
                pipeline_trt_summary["denoise_step"] = denoise_trt.engine_summary()
                denoise_trt_outputs = denoise_trt.infer(
                    {
                        "x_t": export_x_t,
                        "timestep": export_timestep,
                        "prefix_pad_masks": prefix_trt_outputs["prefix_pad_masks"],
                        **{name: prefix_trt_outputs[name] for name in cache_tensor_names(num_layers)},
                    }
                )
                trt_pipeline_v_t = tensor_to_numpy(denoise_trt_outputs["v_t"])
        except Exception as exc:
            errors.append(f"TensorRT pipeline failed: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

        comparison = _build_output_metrics(
            outputs={
                "v_t": {
                    "torch": denoise_torch,
                    "onnx": onnx_pipeline_v_t,
                    "trt": trt_pipeline_v_t,
                }
            },
            thresholds=thresholds,
        )
        results["pipeline"] = {
            "status": _subgraph_status(errors=errors, metric_payload=comparison),
            "comparison_basis": (
                "same export-reference observation tensors and export-reference x_t/timestep through "
                "the full one-step vision->prefix->denoise pipeline"
            ),
            "artifacts": {
                name: {
                    "onnx": artifacts[name]["onnx"].as_posix(),
                    "engine": artifacts[name]["engine"].as_posix(),
                }
                for name in SUBGRAPH_FILES
            },
            "onnx_compare_profile": primary_onnx_profile,
            "input_specs": {
                "top_image": _tensor_spec(export_top_image),
                "wrist_image": _tensor_spec(export_wrist_image),
                "image_mask_top": _tensor_spec(export_image_mask_top),
                "image_mask_wrist": _tensor_spec(export_image_mask_wrist),
                "tokens": _tensor_spec(export_tokens),
                "token_attention_mask": _tensor_spec(export_token_attention_mask),
                "x_t": _tensor_spec(export_x_t),
                "timestep": _tensor_spec(export_timestep),
            },
            "onnx_runtime": pipeline_onnx_runtime,
            "trt_engine": pipeline_trt_summary,
            "comparison": comparison,
            "errors": errors,
        }

    overall_status = _overall_status(results)
    report = {
        "stage": "stage5_verify_trt",
        "overall_status": overall_status,
        "variant": metadata.get("variant"),
        "checkpoint_dir": metadata.get("checkpoint_dir"),
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "device": args.device,
        "strict_load": bool(args.strict_load),
        "requested_subgraphs": requested_subgraphs,
        "thresholds": thresholds,
        "torch_reference_mode": {
            "name": "export_reference_torch",
            "policy_device": "cpu",
            "policy_dtype": "float32",
            "use_autocast": False,
            "inputs_device": "cpu",
            "purpose": (
                "Primary Stage 5 Torch baseline. Align Torch/ONNX/TRT comparisons to the Stage 2 "
                "export-fidelity boundary instead of the runtime autocast path."
            ),
        },
        "acceptance_criteria": {
            "primary_onnx_compare_profile": primary_onnx_profile,
            "numeric_threshold_failures_are_hard_failures": True,
            "exit_code_follows": "overall_status",
            "stage5_scope": "export_boundary_single_step",
        },
        "trt_artifact_provenance": stage4_provenance,
        "metadata_notes": metadata_note_payload(context),
        "artifact_paths": {
            name: {
                "onnx": paths["onnx"].as_posix(),
                "engine": paths["engine"].as_posix(),
            }
            for name, paths in artifacts.items()
        },
        "onnx_compare_profiles": ONNX_COMPARE_PROFILES,
        "primary_onnx_compare_profile": primary_onnx_profile,
        "results": results,
    }
    write_json(report_path, report)
    write_markdown(markdown_path, _format_markdown(report))

    metadata.setdefault("artifacts", {})["stage5_verify_trt"] = report_path.as_posix()
    metadata.setdefault("stage_status", {})["stage5_verify_trt"] = overall_status
    metadata["trt_compare_summary"] = {
        name: result["comparison"]["summary"] for name, result in results.items()
    }
    metadata["verified_trt_artifact_provenance"] = stage4_provenance
    metadata["verified_trt_requested_precision"] = stage4_provenance.get("requested_precision")
    metadata.setdefault("validation_gates", {})["stage5_verify_trt"] = {
        "status": overall_status,
        "hard_fail": overall_status in {"fail", "error"},
        "failed_checks": [
            name
            for name, result in results.items()
            if result.get("status") != "pass"
        ],
        "report_path": report_path.as_posix(),
    }
    metadata["trt_compare_reference_mode"] = report["torch_reference_mode"]
    if overall_status == "pass":
        metadata["last_completed_stage"] = "stage5_verify_trt"
    write_json(metadata_file, metadata)

    print(f"[{overall_status.upper()}] Stage 5 report written to: {report_path}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
