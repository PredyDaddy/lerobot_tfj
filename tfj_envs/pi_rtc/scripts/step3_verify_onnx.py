#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from common import build_metadata_skeleton, metadata_path, read_json, resolve_checkpoint_dir, write_json
from export_wrappers import (
    Pi05DenoiseStepExportWrapper,
    Pi05PrefixCacheExportWrapper,
    Pi05VisionEncoderExportWrapper,
    cache_tensor_names,
)
from pi_compare_common import (
    build_runtime_context,
    compare_arrays,
    evaluate_summary,
    feed_value_to_numpy,
    lazy_import_pi05_modules,
    metadata_note_payload,
    run_onnx_with_fallback,
    summarize_metric_map,
    tensor_to_numpy,
    write_markdown,
)


SUBGRAPH_FILES = {
    "vision_encoder": "pi_shared_vision_encoder.onnx",
    "prefix_cache": "pi_shared_prefix_cache.onnx",
    "denoise_step": "pi05_denoise_step.onnx",
}
DEFAULT_THRESHOLDS = {
    "max_abs_diff": 5e-2,
    "mean_abs_diff": 5e-3,
    "min_cosine_similarity": 0.999,
}
COMPARE_PROFILES = {
    "export_reference_vs_onnx": {
        "report_label": "export_fidelity_compare",
        "display_name": "Export Fidelity Compare",
        "torch_mode": "policy.cpu().float() with use_autocast=False",
        "scope": (
            "compare ONNX outputs against export-mode Torch reference, aligned to the Stage 2 "
            "export boundary instead of the CUDA runtime Torch path"
        ),
        "stage2_alignment": (
            "Stage 2 immediate compare keeps CPU ORT for vision_encoder/prefix_cache and allows "
            "denoise_step to fall back to CUDAExecutionProvider if CPUExecutionProvider cannot execute it."
        ),
        "onnx_execution_profiles": {
            "vision_encoder": {
                "provider_candidates": [["CPUExecutionProvider"]],
                "optimization_order": ["all"],
                "note": "Stage 2-aligned vision compare: CPU ORT with default optimizations.",
            },
            "prefix_cache": {
                "provider_candidates": [["CPUExecutionProvider"]],
                "optimization_order": ["disable"],
                "note": "Stage 2-aligned transformer compare: CPU ORT with optimizations disabled.",
            },
            "denoise_step": {
                "provider_candidates": [
                    ["CPUExecutionProvider"],
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ],
                "optimization_order": ["disable", "basic", "all"],
                "note": (
                    "Stage 2-aligned transformer compare: prefer CPU ORT with optimizations disabled, "
                    "but fall back to CUDAExecutionProvider when CPU kernels are unavailable."
                ),
            },
        },
    },
    "runtime_reference_vs_onnx": {
        "report_label": "runtime_oriented_compare",
        "display_name": "Runtime-Oriented Compare",
        "torch_mode": "policy on runtime device with use_autocast=True",
        "scope": (
            "compare ONNX outputs against runtime Torch reference using a CUDA-preferred, "
            "optimized ORT path that reflects deployment-oriented execution more closely"
        ),
        "runtime_note": (
            "Runtime-oriented compare keeps the CUDA-preferred ORT path and allows optimization "
            "fallbacks if the preferred runtime configuration does not execute."
        ),
        "onnx_execution_profiles": {
            "vision_encoder": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
                "note": "Prefer CUDAExecutionProvider with optimized graph execution.",
            },
            "prefix_cache": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
                "note": "Prefer CUDAExecutionProvider; fall back only if the optimized runtime path fails.",
            },
            "denoise_step": {
                "provider_candidates": [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ],
                "optimization_order": ["all", "basic", "disable"],
                "note": "Prefer CUDAExecutionProvider; keep timestep as a live input during runtime validation.",
            },
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: verify PI05 ONNX artifacts against both runtime-reference Torch "
            "and export-reference Torch, with local-subgraph and chained-pipeline reports."
        )
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--max-abs-threshold", type=float, default=DEFAULT_THRESHOLDS["max_abs_diff"])
    parser.add_argument("--mean-abs-threshold", type=float, default=DEFAULT_THRESHOLDS["mean_abs_diff"])
    parser.add_argument(
        "--min-cosine-similarity",
        type=float,
        default=DEFAULT_THRESHOLDS["min_cosine_similarity"],
    )
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    return parser


def _resolve_onnx_artifacts(run_dir: Path, onnx_dir_arg: str | None) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    explicit_onnx_root = Path(onnx_dir_arg).expanduser().resolve() if onnx_dir_arg else None
    stage2_candidates = [run_dir / "stage2_export_onnx.json"]
    if explicit_onnx_root is not None:
        stage2_candidates.append(explicit_onnx_root.parent / "stage2_export_onnx.json")
    stage2_candidates.append(run_dir.parent / "stage2_export_onnx.json")

    stage2_payload: dict[str, Any] = {}
    stage2_path = next((candidate for candidate in stage2_candidates if candidate.is_file()), None)
    if stage2_path is not None:
        stage2_payload = read_json(stage2_path)

    if explicit_onnx_root is not None:
        onnx_root = explicit_onnx_root
    else:
        onnx_root = Path(stage2_payload.get("onnx_dir", run_dir / "artifacts" / "onnx")).expanduser().resolve()

    stage2_paths = stage2_payload.get("onnx_paths", {})
    resolved: dict[str, Path] = {}
    for subgraph, filename in SUBGRAPH_FILES.items():
        if onnx_dir_arg:
            path = onnx_root / filename
        else:
            candidate = Path(stage2_paths.get(subgraph, filename))
            path = candidate if candidate.is_absolute() else (onnx_root / candidate)
        resolved[subgraph] = path.resolve()

    if stage2_path is not None:
        stage2_payload.setdefault("resolved_stage2_report_path", stage2_path.as_posix())

    return onnx_root, resolved, stage2_payload


def _policy_contract(policy: Any, *, label: str, use_autocast: bool, inputs_device: str) -> dict[str, Any]:
    first_param = next(policy.parameters())
    return {
        "label": label,
        "policy_device": str(first_param.device),
        "policy_dtype": str(first_param.dtype).replace("torch.", ""),
        "inputs_device": inputs_device,
        "use_autocast": bool(use_autocast),
    }


def _compute_reference_outputs(
    *,
    policy: Any,
    num_layers: int,
    modules: dict[str, Any],
    top_image: torch.Tensor,
    wrist_image: torch.Tensor,
    image_mask_top: torch.Tensor,
    image_mask_wrist: torch.Tensor,
    tokens: torch.Tensor,
    token_attention_mask: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    use_autocast: bool,
    reference_label: str,
) -> dict[str, Any]:
    policy.modeling_make_att_2d_masks = modules["make_att_2d_masks"]

    prefix_names = ["prefix_pad_masks", *cache_tensor_names(num_layers)]
    vision_wrapper = Pi05VisionEncoderExportWrapper(policy, use_autocast=use_autocast).eval()
    prefix_wrapper = Pi05PrefixCacheExportWrapper(
        policy,
        num_layers=num_layers,
        use_autocast=use_autocast,
    ).eval()
    denoise_wrapper = Pi05DenoiseStepExportWrapper(
        policy,
        num_layers=num_layers,
        dynamic_cache_cls=modules["DynamicCache"],
        use_autocast=use_autocast,
    ).eval()

    with torch.no_grad():
        vision_top_tensor = vision_wrapper(top_image)
        vision_wrist_tensor = vision_wrapper(wrist_image)
        prefix_tensors = prefix_wrapper(
            vision_top_tensor,
            vision_wrist_tensor,
            image_mask_top,
            image_mask_wrist,
            tokens,
            token_attention_mask,
        )
        denoise_tensor = denoise_wrapper(
            x_t,
            timestep,
            prefix_tensors[0],
            *prefix_tensors[1:],
        )

    vision_outputs = {
        "top": tensor_to_numpy(vision_top_tensor),
        "wrist": tensor_to_numpy(vision_wrist_tensor),
    }
    prefix_outputs = {
        name: tensor_to_numpy(tensor)
        for name, tensor in zip(prefix_names, prefix_tensors, strict=True)
    }
    denoise_outputs = {
        "v_t": tensor_to_numpy(denoise_tensor),
    }

    static_prefix_inputs = {
        "image_mask_top": feed_value_to_numpy(image_mask_top),
        "image_mask_wrist": feed_value_to_numpy(image_mask_wrist),
        "tokens": feed_value_to_numpy(tokens),
        "token_attention_mask": feed_value_to_numpy(token_attention_mask),
    }
    static_denoise_inputs = {
        "x_t": feed_value_to_numpy(x_t),
        "timestep": feed_value_to_numpy(timestep),
    }

    return {
        "contract": _policy_contract(
            policy,
            label=reference_label,
            use_autocast=use_autocast,
            inputs_device=str(top_image.device),
        ),
        "outputs": {
            "vision_encoder": vision_outputs,
            "prefix_cache": prefix_outputs,
            "denoise_step": denoise_outputs,
        },
        "onnx_feeds": {
            "prefix_cache": {
                "image_embs_top": vision_outputs["top"],
                "image_embs_wrist": vision_outputs["wrist"],
                **static_prefix_inputs,
            },
            "denoise_step": {
                **static_denoise_inputs,
                "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
                **{
                    name: prefix_outputs[name]
                    for name in cache_tensor_names(num_layers)
                },
            },
        },
    }


def _execution_profile_request(compare_key: str, subgraph_name: str) -> dict[str, Any]:
    compare_profile = COMPARE_PROFILES[compare_key]
    subgraph_profile = compare_profile["onnx_execution_profiles"][subgraph_name]
    return {
        "compare_key": compare_key,
        "compare_label": compare_profile["report_label"],
        "compare_display_name": compare_profile["display_name"],
        "subgraph_name": subgraph_name,
        "provider_candidates": [list(candidate) for candidate in subgraph_profile["provider_candidates"]],
        "optimization_order": list(subgraph_profile["optimization_order"]),
        "note": subgraph_profile["note"],
    }


def _execute_onnx_case(
    onnx_path: Path,
    output_names: list[str],
    input_feed: dict[str, Any],
    *,
    execution_profile: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        outputs, runtime_info = run_onnx_with_fallback(
            onnx_path,
            output_names,
            input_feed,
            provider_candidates_override=execution_profile["provider_candidates"],
            optimization_order=execution_profile["optimization_order"],
        )
    except Exception as exc:
        return None, {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "onnx_path": onnx_path.as_posix(),
            "requested_outputs": output_names,
            "execution_profile": execution_profile,
        }

    return outputs, {
        "status": "ok",
        "onnx_path": onnx_path.as_posix(),
        "requested_outputs": output_names,
        "execution_profile": execution_profile,
        "runtime": runtime_info,
    }


def _skipped_execution(
    *,
    onnx_path: Path,
    output_names: list[str],
    reason: str,
    execution_profile: dict[str, Any],
) -> tuple[None, dict[str, Any]]:
    return None, {
        "status": "skipped",
        "reason": reason,
        "onnx_path": onnx_path.as_posix(),
        "requested_outputs": output_names,
        "execution_profile": execution_profile,
    }


def _timestep_live_input_check(execution_entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        "status": execution_entry["status"],
        "session_has_timestep": False,
        "dropped_inputs": [],
        "consumed_as_live_input": False,
    }
    if execution_entry["status"] != "ok":
        return result

    runtime_info = execution_entry["runtime"]
    session_inputs = list(runtime_info.get("session_input_names", []))
    dropped_inputs = list(runtime_info.get("dropped_inputs", []))
    result["session_input_names"] = session_inputs
    result["dropped_inputs"] = dropped_inputs
    result["session_has_timestep"] = "timestep" in session_inputs
    result["consumed_as_live_input"] = result["session_has_timestep"] and "timestep" not in dropped_inputs
    return result


def _runtime_summary(execution_entry: dict[str, Any]) -> str:
    if execution_entry["status"] != "ok":
        return f"status={execution_entry['status']}"

    runtime_info = execution_entry["runtime"]
    return (
        f"active_providers={runtime_info.get('active_providers', [])}, "
        f"graph_optimization_level={runtime_info.get('graph_optimization_level', 'unknown')}"
    )


def _comparison_status(
    *,
    summary: dict[str, float] | None,
    execution_entries: list[dict[str, Any]],
    missing_pairs: list[str],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, Any] | None]:
    check = evaluate_summary(summary, thresholds) if summary is not None else None
    if any(entry["status"] != "ok" for entry in execution_entries) or missing_pairs:
        return "error", check
    if check is not None and not check["passed"]:
        return "warn", check
    return "pass", check


def _finalize_compare_block(
    *,
    metrics: dict[str, dict[str, float]],
    expected_pairs: list[str],
    pair_prefix: str,
    execution_entries: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    compared_pairs = [f"{pair_prefix}.{name}" for name in expected_pairs if name in metrics]
    missing_pairs = [f"{pair_prefix}.{name}" for name in expected_pairs if name not in metrics]
    summary = summarize_metric_map(metrics) if metrics else None
    status, check = _comparison_status(
        summary=summary,
        execution_entries=execution_entries,
        missing_pairs=missing_pairs,
        thresholds=thresholds,
    )
    return {
        "metrics": metrics,
        "summary": summary,
        "check": check,
        "compared_pairs": compared_pairs,
        "missing_pairs": missing_pairs,
        "status": status,
    }


def _aggregate_section_status(section_entries: list[dict[str, Any]]) -> str:
    statuses = [entry["status"] for entry in section_entries]
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _status_rank(status: str) -> int:
    return {
        "pass": 0,
        "warn": 1,
        "error": 2,
        "fail": 3,
    }.get(status, 3)


def _max_status(*statuses: str) -> str:
    normalized = [status for status in statuses if status]
    if not normalized:
        return "pass"
    return max(normalized, key=_status_rank)


def _acceptance_check(
    *,
    name: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "required": True,
        "status": "pass" if passed else "fail",
        "message": message,
    }
    if details:
        payload.update(details)
    return payload


def _build_stage3_acceptance(
    *,
    results: dict[str, Any],
    denoise_timestep_live_input_checks: dict[str, Any],
) -> dict[str, Any]:
    export_local = results["local_subgraph_compare"]["export_reference_vs_onnx"]
    export_chained = results["chained_compare"]["export_reference_vs_onnx"]

    failed_timestep_checks: list[str] = []
    for scope_name in ("local_subgraph_compare", "chained_compare"):
        check = denoise_timestep_live_input_checks[scope_name]["export_reference_vs_onnx"]
        if check.get("status") != "ok" or not check.get("consumed_as_live_input", False):
            failed_timestep_checks.append(f"{scope_name}.export_reference_vs_onnx")

    checks = {
        "local_export_fidelity_compare": _acceptance_check(
            name="local_export_fidelity_compare",
            passed=export_local["status"] == "pass",
            message=(
                "Local export-fidelity ONNX compare passed."
                if export_local["status"] == "pass"
                else "Local export-fidelity ONNX compare did not meet acceptance."
            ),
            details={
                "observed_status": export_local["status"],
                "missing_pairs": list(export_local["missing_pairs"]),
            },
        ),
        "chained_export_fidelity_compare": _acceptance_check(
            name="chained_export_fidelity_compare",
            passed=export_chained["status"] == "pass",
            message=(
                "Chained export-fidelity ONNX compare passed."
                if export_chained["status"] == "pass"
                else "Chained export-fidelity ONNX compare did not meet acceptance."
            ),
            details={
                "observed_status": export_chained["status"],
                "missing_pairs": list(export_chained["missing_pairs"]),
            },
        ),
        "denoise_timestep_live_input": _acceptance_check(
            name="denoise_timestep_live_input",
            passed=not failed_timestep_checks,
            message=(
                "The denoise timestep remained a live ONNX session input for every Stage 3 execution path."
                if not failed_timestep_checks
                else "The denoise timestep contract regressed in one or more Stage 3 execution paths."
            ),
            details={
                "failed_checks": failed_timestep_checks,
            },
        ),
    }
    hard_fail = any(check["status"] == "fail" and check.get("required") for check in checks.values())
    return {
        "checks": checks,
        "hard_fail": hard_fail,
        "status": "pass" if not hard_fail else "fail",
        "failed_checks": [
            name
            for name, check in checks.items()
            if check["status"] != "pass"
        ],
    }


def _assemble_local_compare(
    *,
    reference_key: str,
    reference_payload: dict[str, Any],
    compare_definition: dict[str, Any],
    prefix_output_names: list[str],
    vision_outputs: dict[str, dict[str, Any]] | None,
    vision_execution: dict[str, dict[str, Any]],
    prefix_outputs: dict[str, Any] | None,
    prefix_execution: dict[str, Any],
    denoise_outputs: dict[str, Any] | None,
    denoise_execution: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    vision_metrics: dict[str, dict[str, float]] = {}
    if vision_outputs is not None:
        for image_name in ("top", "wrist"):
            vision_metrics[image_name] = compare_arrays(
                reference_payload["outputs"]["vision_encoder"][image_name],
                vision_outputs[image_name]["image_embs"],
            )
    vision_block = _finalize_compare_block(
        metrics=vision_metrics,
        expected_pairs=["top", "wrist"],
        pair_prefix=f"local_subgraph_compare.{reference_key}.vision_encoder",
        execution_entries=[vision_execution["top"], vision_execution["wrist"]],
        thresholds=thresholds,
    )

    prefix_metrics: dict[str, dict[str, float]] = {}
    if prefix_outputs is not None:
        for name in prefix_output_names:
            prefix_metrics[name] = compare_arrays(
                reference_payload["outputs"]["prefix_cache"][name],
                prefix_outputs[name],
            )
    prefix_block = _finalize_compare_block(
        metrics=prefix_metrics,
        expected_pairs=prefix_output_names,
        pair_prefix=f"local_subgraph_compare.{reference_key}.prefix_cache",
        execution_entries=[prefix_execution],
        thresholds=thresholds,
    )

    denoise_metrics: dict[str, dict[str, float]] = {}
    if denoise_outputs is not None:
        denoise_metrics["v_t"] = compare_arrays(
            reference_payload["outputs"]["denoise_step"]["v_t"],
            denoise_outputs["v_t"],
        )
    denoise_block = _finalize_compare_block(
        metrics=denoise_metrics,
        expected_pairs=["v_t"],
        pair_prefix=f"local_subgraph_compare.{reference_key}.denoise_step",
        execution_entries=[denoise_execution],
        thresholds=thresholds,
    )

    return {
        "reference_contract": reference_payload["contract"],
        "compare_label": compare_definition["report_label"],
        "compare_display_name": compare_definition["display_name"],
        "onnx_execution_profile": compare_definition["onnx_execution_profiles"],
        "status": _aggregate_section_status([vision_block, prefix_block, denoise_block]),
        "subgraphs": {
            "vision_encoder": {
                "comparison_basis": (
                    "local subgraph compare: same processed image tensor into ONNX vision_encoder; "
                    "compare ONNX outputs against this Torch reference mode."
                ),
                "execution": vision_execution,
                **vision_block,
            },
            "prefix_cache": {
                "comparison_basis": (
                    "local subgraph compare: same Torch-produced vision embeddings for this reference mode "
                    "+ same masks/tokens fed to ONNX prefix_cache."
                ),
                "execution": prefix_execution,
                **prefix_block,
            },
            "denoise_step": {
                "comparison_basis": (
                    "local subgraph compare: same Torch-produced prefix tensors for this reference mode "
                    "+ same x_t/timestep values fed to ONNX denoise_step. "
                    "Feed is filtered to actual ONNX session inputs."
                ),
                "execution": denoise_execution,
                **denoise_block,
            },
        },
        "compared_pairs": [
            *vision_block["compared_pairs"],
            *prefix_block["compared_pairs"],
            *denoise_block["compared_pairs"],
        ],
        "missing_pairs": [
            *vision_block["missing_pairs"],
            *prefix_block["missing_pairs"],
            *denoise_block["missing_pairs"],
        ],
    }


def _assemble_chained_compare(
    *,
    reference_key: str,
    reference_payload: dict[str, Any],
    compare_definition: dict[str, Any],
    pipeline_output: dict[str, Any] | None,
    pipeline_execution: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    pipeline_metrics: dict[str, dict[str, float]] = {}
    if pipeline_output is not None:
        pipeline_metrics["v_t"] = compare_arrays(
            reference_payload["outputs"]["denoise_step"]["v_t"],
            pipeline_output["v_t"],
        )
    pipeline_block = _finalize_compare_block(
        metrics=pipeline_metrics,
        expected_pairs=["v_t"],
        pair_prefix=f"chained_compare.{reference_key}.pipeline",
        execution_entries=[
            pipeline_execution["vision_top"],
            pipeline_execution["vision_wrist"],
            pipeline_execution["prefix_cache"],
            pipeline_execution["denoise_step"],
        ],
        thresholds=thresholds,
    )

    return {
        "reference_contract": reference_payload["contract"],
        "compare_label": compare_definition["report_label"],
        "compare_display_name": compare_definition["display_name"],
        "onnx_execution_profile": compare_definition["onnx_execution_profiles"],
        "status": pipeline_block["status"],
        "pipeline": {
            "comparison_basis": (
                "chained compare: raw images -> ONNX vision_encoder -> ONNX prefix_cache -> ONNX denoise_step, "
                "then compare final v_t against this Torch reference mode."
            ),
            "execution": pipeline_execution,
            **pipeline_block,
        },
        "compared_pairs": list(pipeline_block["compared_pairs"]),
        "missing_pairs": list(pipeline_block["missing_pairs"]),
    }


def _collect_pair_inventory(results: dict[str, Any]) -> dict[str, Any]:
    by_section: dict[str, dict[str, Any]] = {}
    compared_pairs: list[str] = []
    missing_pairs: list[str] = []

    for scope_name, scope_payload in results.items():
        by_section[scope_name] = {}
        for reference_key, entry in scope_payload.items():
            section_key = f"{scope_name}.{reference_key}"
            section_compared = list(entry["compared_pairs"])
            section_missing = list(entry["missing_pairs"])
            compared_pairs.extend(section_compared)
            missing_pairs.extend(section_missing)
            by_section[scope_name][reference_key] = {
                "status": entry["status"],
                "compared_pairs": section_compared,
                "missing_pairs": section_missing,
                "compared_count": len(section_compared),
                "missing_count": len(section_missing),
            }

    return {
        "compared_pairs": compared_pairs,
        "missing_pairs": missing_pairs,
        "compared_count": len(compared_pairs),
        "missing_count": len(missing_pairs),
        "by_section": by_section,
    }


def _overall_status(results: dict[str, Any]) -> str:
    statuses = []
    for scope_payload in results.values():
        for entry in scope_payload.values():
            statuses.append(entry["status"])
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _format_summary_line(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "not_available"
    return (
        f"max_abs={summary['max_abs_diff']:.6g}, "
        f"mean_abs={summary['mean_abs_diff']:.6g}, "
        f"min_cos={summary['min_cosine_similarity']:.6g}"
    )


def _append_profile_plan(lines: list[str], *, profile_key: str, profile_definition: dict[str, Any]) -> None:
    lines.append(f"- {profile_definition['display_name']} (`{profile_key}`):")
    lines.append(f"  - torch_mode: `{profile_definition['torch_mode']}`")
    lines.append(f"  - scope: `{profile_definition['scope']}`")
    note = profile_definition.get("stage2_alignment") or profile_definition.get("runtime_note")
    if note:
        lines.append(f"  - note: `{note}`")
    lines.append("  - onnx_execution_plan:")
    for subgraph_name in ("vision_encoder", "prefix_cache", "denoise_step"):
        subgraph_profile = profile_definition["onnx_execution_profiles"][subgraph_name]
        lines.append(
            "    - "
            f"`{subgraph_name}` providers={subgraph_profile['provider_candidates']} "
            f"optimization_order={subgraph_profile['optimization_order']}"
        )


def _append_execution_details(
    lines: list[str],
    *,
    heading: str,
    execution_entries: list[tuple[str, dict[str, Any]]],
) -> None:
    lines.append(f"- {heading}:")
    for label, execution_entry in execution_entries:
        lines.append(f"  - `{label}` {_runtime_summary(execution_entry)}")


def _format_markdown(report: dict[str, Any]) -> str:
    profiles = report["comparison_profiles"]
    local_runtime = report["results"]["local_subgraph_compare"]["runtime_reference_vs_onnx"]
    local_export = report["results"]["local_subgraph_compare"]["export_reference_vs_onnx"]
    chained_runtime = report["results"]["chained_compare"]["runtime_reference_vs_onnx"]
    chained_export = report["results"]["chained_compare"]["export_reference_vs_onnx"]
    timestep_checks = report["denoise_timestep_live_input_checks"]

    runtime_denoise_exec = local_runtime["subgraphs"]["denoise_step"]["execution"]
    export_denoise_exec = local_export["subgraphs"]["denoise_step"]["execution"]
    pipeline_runtime_denoise_exec = chained_runtime["pipeline"]["execution"]["denoise_step"]
    pipeline_export_denoise_exec = chained_export["pipeline"]["execution"]["denoise_step"]

    lines = [
        "# PI05 Stage 3 ONNX Verification",
        "",
        f"- policy: `{report['policy_dir']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- onnx_dir: `{report['onnx_dir']}`",
        f"- overall_status: `{report['overall_status']}`",
        f"- stage3_acceptance: `{report['stage3_acceptance']['status']}`",
        "- report_goal: `export-fidelity compare` and `runtime-oriented compare` are separated so CPU ORT export fidelity is not conflated with CUDA ORT runtime drift.",
        "- compare_scopes: `local_subgraph_compare` means Torch intermediates are used as ONNX inputs; `chained_compare` means ONNX vision -> ONNX prefix -> ONNX denoise.",
        "",
        "## Acceptance",
        "",
    ]

    for check_name, check in report["stage3_acceptance"]["checks"].items():
        lines.append(f"- {check_name}: `{check['status']}`")
        lines.append(f"  - message: `{check['message']}`")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- compared_pairs: `{report['pair_inventory']['compared_count']}`",
            f"- missing_pairs: `{report['pair_inventory']['missing_count']}`",
        ]
    )

    if report["pair_inventory"]["missing_pairs"]:
        lines.append("- missing_pair_list:")
        for pair_name in report["pair_inventory"]["missing_pairs"]:
            lines.append(f"  - `{pair_name}`")
    else:
        lines.append("- missing_pair_list: `none`")

    lines.extend(
        [
            "",
            "## Compare Profiles",
            "",
        ]
    )
    _append_profile_plan(lines, profile_key="export_reference_vs_onnx", profile_definition=profiles["export_reference_vs_onnx"])
    _append_profile_plan(lines, profile_key="runtime_reference_vs_onnx", profile_definition=profiles["runtime_reference_vs_onnx"])

    lines.extend(
        [
            "",
            "## Export Fidelity Compare",
            "",
            f"- local_subgraph_compare: `{local_export['status']}`",
            f"- local vision summary: `{_format_summary_line(local_export['subgraphs']['vision_encoder']['summary'])}`",
            f"- local prefix summary: `{_format_summary_line(local_export['subgraphs']['prefix_cache']['summary'])}`",
            f"- local denoise summary: `{_format_summary_line(local_export['subgraphs']['denoise_step']['summary'])}`",
            f"- chained_compare: `{chained_export['status']}`",
            f"- chained pipeline summary: `{_format_summary_line(chained_export['pipeline']['summary'])}`",
        ]
    )
    _append_execution_details(
        lines,
        heading="local_subgraph_execution",
        execution_entries=[
            ("vision_encoder.top", local_export["subgraphs"]["vision_encoder"]["execution"]["top"]),
            ("vision_encoder.wrist", local_export["subgraphs"]["vision_encoder"]["execution"]["wrist"]),
            ("prefix_cache", local_export["subgraphs"]["prefix_cache"]["execution"]),
            ("denoise_step", export_denoise_exec),
        ],
    )
    _append_execution_details(
        lines,
        heading="chained_execution",
        execution_entries=[
            ("vision_encoder.top", chained_export["pipeline"]["execution"]["vision_top"]),
            ("vision_encoder.wrist", chained_export["pipeline"]["execution"]["vision_wrist"]),
            ("prefix_cache", chained_export["pipeline"]["execution"]["prefix_cache"]),
            ("denoise_step", pipeline_export_denoise_exec),
        ],
    )

    lines.extend(
        [
            "",
            "## Runtime-Oriented Compare",
            "",
            f"- local_subgraph_compare: `{local_runtime['status']}`",
            f"- local vision summary: `{_format_summary_line(local_runtime['subgraphs']['vision_encoder']['summary'])}`",
            f"- local prefix summary: `{_format_summary_line(local_runtime['subgraphs']['prefix_cache']['summary'])}`",
            f"- local denoise summary: `{_format_summary_line(local_runtime['subgraphs']['denoise_step']['summary'])}`",
            f"- chained_compare: `{chained_runtime['status']}`",
            f"- chained pipeline summary: `{_format_summary_line(chained_runtime['pipeline']['summary'])}`",
        ]
    )
    _append_execution_details(
        lines,
        heading="local_subgraph_execution",
        execution_entries=[
            ("vision_encoder.top", local_runtime["subgraphs"]["vision_encoder"]["execution"]["top"]),
            ("vision_encoder.wrist", local_runtime["subgraphs"]["vision_encoder"]["execution"]["wrist"]),
            ("prefix_cache", local_runtime["subgraphs"]["prefix_cache"]["execution"]),
            ("denoise_step", runtime_denoise_exec),
        ],
    )
    _append_execution_details(
        lines,
        heading="chained_execution",
        execution_entries=[
            ("vision_encoder.top", chained_runtime["pipeline"]["execution"]["vision_top"]),
            ("vision_encoder.wrist", chained_runtime["pipeline"]["execution"]["vision_wrist"]),
            ("prefix_cache", chained_runtime["pipeline"]["execution"]["prefix_cache"]),
            ("denoise_step", pipeline_runtime_denoise_exec),
        ],
    )

    lines.extend(
        [
            "",
            "## Denoise Timestep Live Input",
            "",
            f"- local export-fidelity consumed: `{timestep_checks['local_subgraph_compare']['export_reference_vs_onnx']['consumed_as_live_input']}`",
            f"- local export-fidelity session_input_names: `{timestep_checks['local_subgraph_compare']['export_reference_vs_onnx'].get('session_input_names', [])}`",
            f"- local export-fidelity dropped_inputs: `{timestep_checks['local_subgraph_compare']['export_reference_vs_onnx'].get('dropped_inputs', [])}`",
            f"- local runtime-oriented consumed: `{timestep_checks['local_subgraph_compare']['runtime_reference_vs_onnx']['consumed_as_live_input']}`",
            f"- local runtime-oriented session_input_names: `{timestep_checks['local_subgraph_compare']['runtime_reference_vs_onnx'].get('session_input_names', [])}`",
            f"- local runtime-oriented dropped_inputs: `{timestep_checks['local_subgraph_compare']['runtime_reference_vs_onnx'].get('dropped_inputs', [])}`",
            f"- chained export-fidelity consumed: `{timestep_checks['chained_compare']['export_reference_vs_onnx']['consumed_as_live_input']}`",
            f"- chained export-fidelity session_input_names: `{timestep_checks['chained_compare']['export_reference_vs_onnx'].get('session_input_names', [])}`",
            f"- chained export-fidelity dropped_inputs: `{timestep_checks['chained_compare']['export_reference_vs_onnx'].get('dropped_inputs', [])}`",
            f"- chained runtime-oriented consumed: `{timestep_checks['chained_compare']['runtime_reference_vs_onnx']['consumed_as_live_input']}`",
            f"- chained runtime-oriented session_input_names: `{timestep_checks['chained_compare']['runtime_reference_vs_onnx'].get('session_input_names', [])}`",
            f"- chained runtime-oriented dropped_inputs: `{timestep_checks['chained_compare']['runtime_reference_vs_onnx'].get('dropped_inputs', [])}`",
        ]
    )

    return "\n".join(lines) + "\n"




def main() -> int:
    args = build_parser().parse_args()
    thresholds = {
        "max_abs_diff": float(args.max_abs_threshold),
        "mean_abs_diff": float(args.mean_abs_threshold),
        "min_cosine_similarity": float(args.min_cosine_similarity),
    }

    policy_dir = resolve_checkpoint_dir(args.policy_path)
    run_dir = Path(args.run_dir).expanduser().resolve()
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (run_dir / "stage3_verify_onnx.json")
    )
    markdown_path = (
        Path(args.markdown_path).expanduser().resolve()
        if args.markdown_path
        else (run_dir / "stage3_verify_onnx.md")
    )

    onnx_dir, onnx_artifacts, stage2_payload = _resolve_onnx_artifacts(run_dir, args.onnx_dir)
    modules = lazy_import_pi05_modules()
    context = build_runtime_context(policy_dir, strict=args.strict_load)
    num_layers = int(context.policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)
    prefix_output_names = ["prefix_pad_masks", *cache_tensor_names(num_layers)]

    runtime_reference = _compute_reference_outputs(
        policy=context.policy,
        num_layers=num_layers,
        modules=modules,
        top_image=context.top_image,
        wrist_image=context.wrist_image,
        image_mask_top=context.image_mask_top,
        image_mask_wrist=context.image_mask_wrist,
        tokens=context.tokens,
        token_attention_mask=context.token_attention_mask,
        x_t=context.x_t,
        timestep=context.timestep,
        use_autocast=True,
        reference_label="runtime_reference_torch",
    )

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

    export_reference = _compute_reference_outputs(
        policy=context.policy,
        num_layers=num_layers,
        modules=modules,
        top_image=export_top_image,
        wrist_image=export_wrist_image,
        image_mask_top=export_image_mask_top,
        image_mask_wrist=export_image_mask_wrist,
        tokens=export_tokens,
        token_attention_mask=export_token_attention_mask,
        x_t=export_x_t,
        timestep=export_timestep,
        use_autocast=False,
        reference_label="export_reference_torch",
    )

    raw_vision_feeds = {
        "top": {"image": feed_value_to_numpy(context.top_image)},
        "wrist": {"image": feed_value_to_numpy(context.wrist_image)},
    }
    execution_profiles = {
        compare_key: {
            subgraph_name: _execution_profile_request(compare_key, subgraph_name)
            for subgraph_name in SUBGRAPH_FILES
        }
        for compare_key in COMPARE_PROFILES
    }

    runtime_vision_top_outputs, runtime_vision_top_execution = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["top"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["vision_encoder"],
    )
    runtime_vision_wrist_outputs, runtime_vision_wrist_execution = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["wrist"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["vision_encoder"],
    )
    runtime_vision_outputs = None
    if runtime_vision_top_outputs is not None and runtime_vision_wrist_outputs is not None:
        runtime_vision_outputs = {
            "top": runtime_vision_top_outputs,
            "wrist": runtime_vision_wrist_outputs,
        }
    runtime_vision_execution = {
        "top": runtime_vision_top_execution,
        "wrist": runtime_vision_wrist_execution,
    }

    export_vision_top_outputs, export_vision_top_execution = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["top"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["vision_encoder"],
    )
    export_vision_wrist_outputs, export_vision_wrist_execution = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["wrist"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["vision_encoder"],
    )
    export_vision_outputs = None
    if export_vision_top_outputs is not None and export_vision_wrist_outputs is not None:
        export_vision_outputs = {
            "top": export_vision_top_outputs,
            "wrist": export_vision_wrist_outputs,
        }
    export_vision_execution = {
        "top": export_vision_top_execution,
        "wrist": export_vision_wrist_execution,
    }

    prefix_runtime_outputs, prefix_runtime_execution = _execute_onnx_case(
        onnx_artifacts["prefix_cache"],
        prefix_output_names,
        runtime_reference["onnx_feeds"]["prefix_cache"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["prefix_cache"],
    )
    prefix_export_outputs, prefix_export_execution = _execute_onnx_case(
        onnx_artifacts["prefix_cache"],
        prefix_output_names,
        export_reference["onnx_feeds"]["prefix_cache"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["prefix_cache"],
    )

    denoise_runtime_outputs, denoise_runtime_execution = _execute_onnx_case(
        onnx_artifacts["denoise_step"],
        ["v_t"],
        runtime_reference["onnx_feeds"]["denoise_step"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["denoise_step"],
    )
    denoise_export_outputs, denoise_export_execution = _execute_onnx_case(
        onnx_artifacts["denoise_step"],
        ["v_t"],
        export_reference["onnx_feeds"]["denoise_step"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["denoise_step"],
    )

    runtime_pipeline_execution: dict[str, dict[str, Any]] = {}
    runtime_pipeline_vision_top, runtime_pipeline_execution["vision_top"] = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["top"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["vision_encoder"],
    )
    runtime_pipeline_vision_wrist, runtime_pipeline_execution["vision_wrist"] = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["wrist"],
        execution_profile=execution_profiles["runtime_reference_vs_onnx"]["vision_encoder"],
    )
    if runtime_pipeline_vision_top is not None and runtime_pipeline_vision_wrist is not None:
        runtime_pipeline_prefix_feed = {
            "image_embs_top": runtime_pipeline_vision_top["image_embs"],
            "image_embs_wrist": runtime_pipeline_vision_wrist["image_embs"],
            "image_mask_top": feed_value_to_numpy(context.image_mask_top),
            "image_mask_wrist": feed_value_to_numpy(context.image_mask_wrist),
            "tokens": feed_value_to_numpy(context.tokens),
            "token_attention_mask": feed_value_to_numpy(context.token_attention_mask),
        }
        runtime_pipeline_prefix_outputs, runtime_pipeline_execution["prefix_cache"] = _execute_onnx_case(
            onnx_artifacts["prefix_cache"],
            prefix_output_names,
            runtime_pipeline_prefix_feed,
            execution_profile=execution_profiles["runtime_reference_vs_onnx"]["prefix_cache"],
        )
    else:
        runtime_pipeline_prefix_outputs, runtime_pipeline_execution["prefix_cache"] = _skipped_execution(
            onnx_path=onnx_artifacts["prefix_cache"],
            output_names=prefix_output_names,
            reason="vision_encoder_failed",
            execution_profile=execution_profiles["runtime_reference_vs_onnx"]["prefix_cache"],
        )

    if runtime_pipeline_prefix_outputs is not None:
        runtime_pipeline_denoise_feed = {
            "x_t": feed_value_to_numpy(context.x_t),
            "timestep": feed_value_to_numpy(context.timestep),
            "prefix_pad_masks": runtime_pipeline_prefix_outputs["prefix_pad_masks"],
            **{
                name: runtime_pipeline_prefix_outputs[name]
                for name in cache_tensor_names(num_layers)
            },
        }
        runtime_pipeline_denoise_outputs, runtime_pipeline_execution["denoise_step"] = _execute_onnx_case(
            onnx_artifacts["denoise_step"],
            ["v_t"],
            runtime_pipeline_denoise_feed,
            execution_profile=execution_profiles["runtime_reference_vs_onnx"]["denoise_step"],
        )
    else:
        runtime_pipeline_denoise_outputs, runtime_pipeline_execution["denoise_step"] = _skipped_execution(
            onnx_path=onnx_artifacts["denoise_step"],
            output_names=["v_t"],
            reason="prefix_cache_failed",
            execution_profile=execution_profiles["runtime_reference_vs_onnx"]["denoise_step"],
        )

    export_pipeline_execution: dict[str, dict[str, Any]] = {}
    export_pipeline_vision_top, export_pipeline_execution["vision_top"] = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["top"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["vision_encoder"],
    )
    export_pipeline_vision_wrist, export_pipeline_execution["vision_wrist"] = _execute_onnx_case(
        onnx_artifacts["vision_encoder"],
        ["image_embs"],
        raw_vision_feeds["wrist"],
        execution_profile=execution_profiles["export_reference_vs_onnx"]["vision_encoder"],
    )
    if export_pipeline_vision_top is not None and export_pipeline_vision_wrist is not None:
        export_pipeline_prefix_feed = {
            "image_embs_top": export_pipeline_vision_top["image_embs"],
            "image_embs_wrist": export_pipeline_vision_wrist["image_embs"],
            "image_mask_top": feed_value_to_numpy(context.image_mask_top),
            "image_mask_wrist": feed_value_to_numpy(context.image_mask_wrist),
            "tokens": feed_value_to_numpy(context.tokens),
            "token_attention_mask": feed_value_to_numpy(context.token_attention_mask),
        }
        export_pipeline_prefix_outputs, export_pipeline_execution["prefix_cache"] = _execute_onnx_case(
            onnx_artifacts["prefix_cache"],
            prefix_output_names,
            export_pipeline_prefix_feed,
            execution_profile=execution_profiles["export_reference_vs_onnx"]["prefix_cache"],
        )
    else:
        export_pipeline_prefix_outputs, export_pipeline_execution["prefix_cache"] = _skipped_execution(
            onnx_path=onnx_artifacts["prefix_cache"],
            output_names=prefix_output_names,
            reason="vision_encoder_failed",
            execution_profile=execution_profiles["export_reference_vs_onnx"]["prefix_cache"],
        )

    if export_pipeline_prefix_outputs is not None:
        export_pipeline_denoise_feed = {
            "x_t": feed_value_to_numpy(context.x_t),
            "timestep": feed_value_to_numpy(context.timestep),
            "prefix_pad_masks": export_pipeline_prefix_outputs["prefix_pad_masks"],
            **{
                name: export_pipeline_prefix_outputs[name]
                for name in cache_tensor_names(num_layers)
            },
        }
        export_pipeline_denoise_outputs, export_pipeline_execution["denoise_step"] = _execute_onnx_case(
            onnx_artifacts["denoise_step"],
            ["v_t"],
            export_pipeline_denoise_feed,
            execution_profile=execution_profiles["export_reference_vs_onnx"]["denoise_step"],
        )
    else:
        export_pipeline_denoise_outputs, export_pipeline_execution["denoise_step"] = _skipped_execution(
            onnx_path=onnx_artifacts["denoise_step"],
            output_names=["v_t"],
            reason="prefix_cache_failed",
            execution_profile=execution_profiles["export_reference_vs_onnx"]["denoise_step"],
        )

    results = {
        "local_subgraph_compare": {
            "runtime_reference_vs_onnx": _assemble_local_compare(
                reference_key="runtime_reference_vs_onnx",
                reference_payload=runtime_reference,
                compare_definition=COMPARE_PROFILES["runtime_reference_vs_onnx"],
                prefix_output_names=prefix_output_names,
                vision_outputs=runtime_vision_outputs,
                vision_execution=runtime_vision_execution,
                prefix_outputs=prefix_runtime_outputs,
                prefix_execution=prefix_runtime_execution,
                denoise_outputs=denoise_runtime_outputs,
                denoise_execution=denoise_runtime_execution,
                thresholds=thresholds,
            ),
            "export_reference_vs_onnx": _assemble_local_compare(
                reference_key="export_reference_vs_onnx",
                reference_payload=export_reference,
                compare_definition=COMPARE_PROFILES["export_reference_vs_onnx"],
                prefix_output_names=prefix_output_names,
                vision_outputs=export_vision_outputs,
                vision_execution=export_vision_execution,
                prefix_outputs=prefix_export_outputs,
                prefix_execution=prefix_export_execution,
                denoise_outputs=denoise_export_outputs,
                denoise_execution=denoise_export_execution,
                thresholds=thresholds,
            ),
        },
        "chained_compare": {
            "runtime_reference_vs_onnx": _assemble_chained_compare(
                reference_key="runtime_reference_vs_onnx",
                reference_payload=runtime_reference,
                compare_definition=COMPARE_PROFILES["runtime_reference_vs_onnx"],
                pipeline_output=runtime_pipeline_denoise_outputs,
                pipeline_execution=runtime_pipeline_execution,
                thresholds=thresholds,
            ),
            "export_reference_vs_onnx": _assemble_chained_compare(
                reference_key="export_reference_vs_onnx",
                reference_payload=export_reference,
                compare_definition=COMPARE_PROFILES["export_reference_vs_onnx"],
                pipeline_output=export_pipeline_denoise_outputs,
                pipeline_execution=export_pipeline_execution,
                thresholds=thresholds,
            ),
        },
    }
    denoise_timestep_live_input_checks = {
        "local_subgraph_compare": {
            "runtime_reference_vs_onnx": _timestep_live_input_check(denoise_runtime_execution),
            "export_reference_vs_onnx": _timestep_live_input_check(denoise_export_execution),
        },
        "chained_compare": {
            "runtime_reference_vs_onnx": _timestep_live_input_check(runtime_pipeline_execution["denoise_step"]),
            "export_reference_vs_onnx": _timestep_live_input_check(export_pipeline_execution["denoise_step"]),
        },
    }
    stage3_acceptance = _build_stage3_acceptance(
        results=results,
        denoise_timestep_live_input_checks=denoise_timestep_live_input_checks,
    )
    result_status = _overall_status(results)
    overall_status = _max_status(result_status, stage3_acceptance["status"])

    report = {
        "stage": "stage3_verify_onnx",
        "overall_status": overall_status,
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "strict_load": bool(args.strict_load),
        "thresholds": thresholds,
        "artifact_paths": {
            name: path.as_posix() for name, path in onnx_artifacts.items()
        },
        "comparison_profiles": COMPARE_PROFILES,
        "reference_definitions": {
            "runtime_reference_vs_onnx": {
                "report_label": COMPARE_PROFILES["runtime_reference_vs_onnx"]["report_label"],
                "display_name": COMPARE_PROFILES["runtime_reference_vs_onnx"]["display_name"],
                "torch_mode": COMPARE_PROFILES["runtime_reference_vs_onnx"]["torch_mode"],
                "scope": COMPARE_PROFILES["runtime_reference_vs_onnx"]["scope"],
                "onnx_execution_profiles": COMPARE_PROFILES["runtime_reference_vs_onnx"]["onnx_execution_profiles"],
            },
            "export_reference_vs_onnx": {
                "report_label": COMPARE_PROFILES["export_reference_vs_onnx"]["report_label"],
                "display_name": COMPARE_PROFILES["export_reference_vs_onnx"]["display_name"],
                "torch_mode": COMPARE_PROFILES["export_reference_vs_onnx"]["torch_mode"],
                "scope": COMPARE_PROFILES["export_reference_vs_onnx"]["scope"],
                "onnx_execution_profiles": COMPARE_PROFILES["export_reference_vs_onnx"]["onnx_execution_profiles"],
            },
            "compare_scopes": {
                "local_subgraph_compare": (
                    "Each ONNX subgraph is compared independently. Downstream ONNX inputs come from "
                    "Torch reference intermediates, not from previous ONNX stages."
                ),
                "chained_compare": (
                    "A true ONNX chain: ONNX vision_encoder -> ONNX prefix_cache -> ONNX denoise_step, "
                    "then compare the final ONNX v_t against Torch references."
                ),
            },
        },
        "stage2_context": {
            "stage2_report_path": stage2_payload.get(
                "resolved_stage2_report_path",
                (run_dir / "stage2_export_onnx.json").as_posix(),
            ),
            "stage2_onnx_paths": stage2_payload.get("onnx_paths", {}),
            "stage2_immediate_onnx_compare_execution": stage2_payload.get("immediate_onnx_compare", {}).get("execution", {}),
            "stage2_immediate_onnx_compare_providers": stage2_payload.get("immediate_onnx_compare", {}).get("providers", {}),
        },
        "metadata_notes": metadata_note_payload(context),
        "results": results,
        "pair_inventory": _collect_pair_inventory(results),
        "denoise_timestep_live_input_checks": denoise_timestep_live_input_checks,
        "stage3_acceptance": stage3_acceptance,
        "acceptance_criteria": {
            "required_checks": [
                "local_export_fidelity_compare",
                "chained_export_fidelity_compare",
                "denoise_timestep_live_input",
            ],
            "report_only_sections": [
                "local_subgraph_compare.runtime_reference_vs_onnx",
                "chained_compare.runtime_reference_vs_onnx",
            ],
            "exit_code_follows": "stage3_acceptance.status",
        },
    }

    write_json(report_path, report)
    write_markdown(markdown_path, _format_markdown(report))

    metadata_file = metadata_path(run_dir)
    if metadata_file.is_file():
        metadata = read_json(metadata_file)
    else:
        metadata = build_metadata_skeleton(run_dir=run_dir, variant="pi05", checkpoint_dir=policy_dir)
    metadata["onnx_paths"] = {
        name: path.as_posix() for name, path in onnx_artifacts.items()
    }
    metadata["stage3_acceptance"] = stage3_acceptance
    metadata["stage3_pair_inventory"] = report["pair_inventory"]
    metadata["denoise_timestep_live_input_checks"] = denoise_timestep_live_input_checks
    metadata.setdefault("stage_status", {})["stage3_verify_onnx"] = stage3_acceptance["status"]
    metadata.setdefault("validation_gates", {})["stage3_verify_onnx"] = {
        "status": stage3_acceptance["status"],
        "hard_fail": bool(stage3_acceptance["hard_fail"]),
        "failed_checks": list(stage3_acceptance["failed_checks"]),
        "report_path": report_path.as_posix(),
    }
    if stage3_acceptance["status"] == "pass":
        metadata["last_completed_stage"] = "stage3_verify_onnx"
    write_json(metadata_file, metadata)

    print(f"[{report['overall_status'].upper()}] Stage 3 report written to: {report_path}")
    print(f"[{stage3_acceptance['status'].upper()}] Stage 3 gate status written to metadata: {metadata_file}")
    print(f"[OK] Stage 3 markdown written to: {markdown_path}")
    return 0 if stage3_acceptance["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
