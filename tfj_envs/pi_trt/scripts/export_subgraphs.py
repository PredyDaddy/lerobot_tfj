#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from export_wrappers import (
    Pi05DenoiseStepExportWrapper,
    Pi05PrefixCacheExportWrapper,
    Pi05VisionEncoderExportWrapper,
    cache_tensor_names,
    describe_execution_mode,
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
)


def export_onnx_model(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    exporter: str = "legacy",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export_kwargs = {
            "export_params": True,
            "opset_version": opset,
            "do_constant_folding": True,
            "input_names": input_names,
            "output_names": output_names,
        }
        if exporter == "dynamo":
            export_kwargs.update(
                {
                    "dynamo": True,
                    "external_data": True,
                    "verify": False,
                    "fallback": False,
                    "artifacts_dir": output_path.parent.as_posix(),
                }
            )
        else:
            export_kwargs["dynamo"] = False
        torch.onnx.export(
            module,
            inputs,
            output_path.as_posix(),
            **export_kwargs,
        )
    onnx.checker.check_model(output_path.as_posix())


def sanitize_onnx_for_trt(onnx_path: str | Path) -> dict[str, Any]:
    target = Path(onnx_path)
    model = onnx.load(target.as_posix(), load_external_data=True)
    rewritten_initializers: list[str] = []
    rewritten_cast_nodes: list[str] = []
    rewritten_value_info: list[str] = []

    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type != onnx.TensorProto.DOUBLE:
            continue
        array = numpy_helper.to_array(initializer).astype(np.float32)
        rewritten = numpy_helper.from_array(array, name=initializer.name)
        model.graph.initializer[index].CopyFrom(rewritten)
        rewritten_initializers.append(initializer.name)

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.DOUBLE:
                attribute.i = onnx.TensorProto.FLOAT
                rewritten_cast_nodes.append(node.name or "<unnamed_cast>")

    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            tensor_type.elem_type = onnx.TensorProto.FLOAT
            rewritten_value_info.append(value_info.name)

    if rewritten_initializers or rewritten_cast_nodes or rewritten_value_info:
        onnx.save_model(
            model,
            target.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{target.name}.data",
            size_threshold=4096,
            convert_attribute=False,
        )
        onnx.checker.check_model(target.as_posix())

    return {
        "rewritten_initializers": rewritten_initializers,
        "rewritten_cast_nodes": rewritten_cast_nodes,
        "rewritten_value_info": rewritten_value_info,
        "rewritten_count": len(rewritten_initializers),
        "rewritten_cast_count": len(rewritten_cast_nodes),
        "rewritten_value_info_count": len(rewritten_value_info),
    }


def _onnx_contract(onnx_path: str | Path) -> dict[str, Any]:
    model = onnx.load(Path(onnx_path).as_posix(), load_external_data=False)
    return {
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def _filter_feed_for_onnx(session: Any, input_feed: dict[str, Any]) -> dict[str, Any]:
    session_inputs = {item.name for item in session.get_inputs()}
    return {
        name: value
        for name, value in input_feed.items()
        if name in session_inputs
    }


def _vision_contract(output: torch.Tensor, *, mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_names": ["image"],
        "output_names": ["image_embs"],
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype).replace("torch.", ""),
    }


def _prefix_contract(outputs: tuple[torch.Tensor, ...], *, num_layers: int, mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_names": [
            "image_embs_top",
            "image_embs_wrist",
            "image_mask_top",
            "image_mask_wrist",
            "tokens",
            "token_attention_mask",
        ],
        "output_names": ["prefix_pad_masks", *cache_tensor_names(num_layers)],
        "prefix_pad_masks_shape": list(outputs[0].shape),
        "prefix_pad_masks_dtype": str(outputs[0].dtype).replace("torch.", ""),
        "cache_tensor_shape": list(outputs[1].shape),
        "cache_tensor_dtype": str(outputs[1].dtype).replace("torch.", ""),
    }


def _denoise_contract(output: torch.Tensor, *, num_layers: int, mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_names": [
            "x_t",
            "timestep",
            "prefix_pad_masks",
            *cache_tensor_names(num_layers),
        ],
        "output_names": ["v_t"],
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype).replace("torch.", ""),
    }


def _build_required_check(
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


def _contract_checks(onnx_contract: dict[str, Any], denoise_execution: dict[str, Any]) -> dict[str, Any]:
    denoise_inputs = onnx_contract.get("denoise_step", {}).get("inputs", [])
    session_input_names = list(denoise_execution.get("session_input_names", []))
    dropped_inputs = list(denoise_execution.get("dropped_inputs", []))

    timestep_is_graph_input = "timestep" in denoise_inputs
    timestep_is_session_input = denoise_execution.get("status") == "ok" and "timestep" in session_input_names
    timestep_consumed_as_live_input = timestep_is_session_input and "timestep" not in dropped_inputs

    checks = {
        "denoise_step_timestep_is_graph_input": _build_required_check(
            name="denoise_step_timestep_is_graph_input",
            passed=timestep_is_graph_input,
            message=(
                "The denoise_step ONNX graph must expose timestep as a declared graph input."
                if timestep_is_graph_input
                else "The denoise_step ONNX graph is missing timestep as a declared graph input."
            ),
            details={
                "graph_inputs": denoise_inputs,
            },
        ),
        "denoise_step_timestep_consumed_as_live_input": _build_required_check(
            name="denoise_step_timestep_consumed_as_live_input",
            passed=timestep_consumed_as_live_input,
            message=(
                "The denoise_step ONNX session consumed timestep as a live runtime input."
                if timestep_consumed_as_live_input
                else "The denoise_step ONNX session did not consume timestep as a live runtime input."
            ),
            details={
                "execution_status": denoise_execution.get("status"),
                "session_input_names": session_input_names,
                "dropped_inputs": dropped_inputs,
            },
        ),
    }
    hard_fail = any(check["status"] == "fail" and check.get("required") for check in checks.values())
    return {
        "checks": checks,
        "hard_fail": hard_fail,
        "status": "pass" if not hard_fail else "fail",
    }


def _stage2_acceptance(
    *,
    execution: dict[str, Any],
    export_reference_vs_onnx_summary: dict[str, Any],
    contract_checks: dict[str, Any],
) -> dict[str, Any]:
    execution_failures = [
        name
        for name, entry in execution.items()
        if entry.get("status") != "ok"
    ]
    compare_checks = {
        name: (evaluate_summary(summary) if summary is not None else None)
        for name, summary in export_reference_vs_onnx_summary.items()
    }
    compare_failures = [
        name
        for name, check in compare_checks.items()
        if check is None or not check["passed"]
    ]

    checks = {
        "contract_checks": _build_required_check(
            name="contract_checks",
            passed=not contract_checks["hard_fail"],
            message=(
                "All required ONNX contract checks passed."
                if not contract_checks["hard_fail"]
                else "One or more required ONNX contract checks failed."
            ),
            details={
                "contract_status": contract_checks["status"],
                "failed_checks": [
                    name
                    for name, check in contract_checks["checks"].items()
                    if check["status"] != "pass"
                ],
            },
        ),
        "immediate_export_fidelity_compare": _build_required_check(
            name="immediate_export_fidelity_compare",
            passed=not execution_failures and not compare_failures,
            message=(
                "Immediate export-fidelity ONNX compare passed for all Stage 2 subgraphs."
                if not execution_failures and not compare_failures
                else "Immediate export-fidelity ONNX compare failed Stage 2 acceptance."
            ),
            details={
                "execution_failures": execution_failures,
                "compare_failures": compare_failures,
                "per_subgraph_checks": compare_checks,
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


def compare_exported_subgraphs(
    *,
    context: Any,
    num_layers: int,
    onnx_paths: dict[str, str],
    runtime_outputs: dict[str, Any],
    export_outputs: dict[str, Any],
) -> dict[str, Any]:
    providers: dict[str, list[str]] = {}
    execution: dict[str, Any] = {}
    onnx_contract = {
        name: _onnx_contract(path)
        for name, path in onnx_paths.items()
    }
    execution_profiles = {
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
    }

    def _execute_onnx_case(
        *,
        onnx_path: str,
        output_names: list[str],
        input_feed: dict[str, Any],
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
                "providers": [],
            }

        return outputs, {
            "status": "ok",
            "providers": list(runtime_info.get("active_providers", [])),
            "session_input_names": list(runtime_info.get("session_input_names", [])),
            "filtered_input_names": sorted(runtime_info.get("filtered_input_names", [])),
            "dropped_inputs": sorted(runtime_info.get("dropped_inputs", [])),
            "graph_optimization_level": runtime_info.get("graph_optimization_level"),
        }

    vision_top_onnx = None
    vision_wrist_onnx = None
    vision_top_feed = {"image": feed_value_to_numpy(context.top_image)}
    vision_wrist_feed = {"image": feed_value_to_numpy(context.wrist_image)}
    vision_top_outputs, vision_top_execution = _execute_onnx_case(
        onnx_path=onnx_paths["vision_encoder"],
        output_names=["image_embs"],
        input_feed=vision_top_feed,
        execution_profile=execution_profiles["vision_encoder"],
    )
    vision_wrist_outputs, vision_wrist_execution = _execute_onnx_case(
        onnx_path=onnx_paths["vision_encoder"],
        output_names=["image_embs"],
        input_feed=vision_wrist_feed,
        execution_profile=execution_profiles["vision_encoder"],
    )
    if vision_top_outputs is not None and vision_wrist_outputs is not None:
        vision_top_onnx = vision_top_outputs["image_embs"]
        vision_wrist_onnx = vision_wrist_outputs["image_embs"]
        providers["vision_encoder"] = list(vision_top_execution["providers"])
        execution["vision_encoder"] = {
            "status": "ok",
            "providers": providers["vision_encoder"],
            "session_input_names": vision_top_execution["session_input_names"],
            "filtered_input_names": vision_top_execution["filtered_input_names"],
            "dropped_inputs": vision_top_execution["dropped_inputs"],
            "graph_optimization_level": vision_top_execution.get("graph_optimization_level"),
        }
    else:
        failed_vision_execution = (
            vision_top_execution if vision_top_execution["status"] != "ok" else vision_wrist_execution
        )
        execution["vision_encoder"] = {
            "status": "error",
            "error_type": failed_vision_execution.get("error_type", "VisionEncoderExecutionError"),
            "error": failed_vision_execution.get("error", "vision_encoder ONNX execution failed"),
            "providers": list(failed_vision_execution.get("providers", [])),
        }

    prefix_names = ["prefix_pad_masks", *cache_tensor_names(num_layers)]
    prefix_onnx_outputs = None
    prefix_input_feed = {
        "image_embs_top": tensor_to_numpy(export_outputs["vision_top"]),
        "image_embs_wrist": tensor_to_numpy(export_outputs["vision_wrist"]),
        "image_mask_top": feed_value_to_numpy(context.image_mask_top),
        "image_mask_wrist": feed_value_to_numpy(context.image_mask_wrist),
        "tokens": feed_value_to_numpy(context.tokens),
        "token_attention_mask": feed_value_to_numpy(context.token_attention_mask),
    }
    prefix_outputs_local, prefix_execution_local = _execute_onnx_case(
        onnx_path=onnx_paths["prefix_cache"],
        output_names=prefix_names,
        input_feed=prefix_input_feed,
        execution_profile=execution_profiles["prefix_cache"],
    )
    prefix_pipeline_outputs = None
    prefix_pipeline_execution = None
    if vision_top_onnx is not None and vision_wrist_onnx is not None:
        prefix_pipeline_feed = {
            "image_embs_top": vision_top_onnx,
            "image_embs_wrist": vision_wrist_onnx,
            "image_mask_top": feed_value_to_numpy(context.image_mask_top),
            "image_mask_wrist": feed_value_to_numpy(context.image_mask_wrist),
            "tokens": feed_value_to_numpy(context.tokens),
            "token_attention_mask": feed_value_to_numpy(context.token_attention_mask),
        }
        prefix_pipeline_outputs, prefix_pipeline_execution = _execute_onnx_case(
            onnx_path=onnx_paths["prefix_cache"],
            output_names=prefix_names,
            input_feed=prefix_pipeline_feed,
            execution_profile=execution_profiles["prefix_cache"],
        )
    if prefix_outputs_local is not None and (
        prefix_pipeline_execution is None or prefix_pipeline_outputs is not None
    ):
        prefix_onnx_outputs = [prefix_outputs_local[name] for name in prefix_names]
        providers["prefix_cache"] = list(prefix_execution_local["providers"])
        execution["prefix_cache"] = {
            "status": "ok",
            "providers": providers["prefix_cache"],
            "session_input_names": prefix_execution_local["session_input_names"],
            "filtered_input_names": prefix_execution_local["filtered_input_names"],
            "dropped_inputs": prefix_execution_local["dropped_inputs"],
            "graph_optimization_level": prefix_execution_local.get("graph_optimization_level"),
        }
    else:
        prefix_pipeline_outputs = None
        failed_prefix_execution = prefix_execution_local
        if prefix_pipeline_execution is not None and prefix_pipeline_outputs is None:
            failed_prefix_execution = prefix_pipeline_execution
        execution["prefix_cache"] = {
            "status": "error",
            "error_type": failed_prefix_execution.get("error_type", "PrefixCacheExecutionError"),
            "error": failed_prefix_execution.get("error", "prefix_cache ONNX execution failed"),
            "providers": list(failed_prefix_execution.get("providers", [])),
        }

    denoise_onnx = None
    denoise_pipeline_onnx = None
    if prefix_onnx_outputs is not None:
        denoise_input_feed = {
            "x_t": feed_value_to_numpy(context.x_t),
            "timestep": feed_value_to_numpy(context.timestep),
            "prefix_pad_masks": prefix_onnx_outputs[0],
            **{
                name: value
                for name, value in zip(cache_tensor_names(num_layers), prefix_onnx_outputs[1:], strict=True)
            },
        }
        denoise_outputs_local, denoise_execution_local = _execute_onnx_case(
            onnx_path=onnx_paths["denoise_step"],
            output_names=["v_t"],
            input_feed=denoise_input_feed,
            execution_profile=execution_profiles["denoise_step"],
        )
        denoise_pipeline_execution = None
        if prefix_pipeline_outputs is not None:
            denoise_pipeline_feed = {
                "x_t": feed_value_to_numpy(context.x_t),
                "timestep": feed_value_to_numpy(context.timestep),
                "prefix_pad_masks": prefix_pipeline_outputs["prefix_pad_masks"],
                **{
                    name: prefix_pipeline_outputs[name]
                    for name in cache_tensor_names(num_layers)
                },
            }
            denoise_pipeline_outputs, denoise_pipeline_execution = _execute_onnx_case(
                onnx_path=onnx_paths["denoise_step"],
                output_names=["v_t"],
                input_feed=denoise_pipeline_feed,
                execution_profile=execution_profiles["denoise_step"],
            )
        if denoise_outputs_local is not None and (
            denoise_pipeline_execution is None or denoise_pipeline_outputs is not None
        ):
            denoise_onnx = denoise_outputs_local["v_t"]
            if denoise_pipeline_outputs is not None:
                denoise_pipeline_onnx = denoise_pipeline_outputs["v_t"]
            providers["denoise_step"] = list(denoise_execution_local["providers"])
            execution["denoise_step"] = {
                "status": "ok",
                "providers": providers["denoise_step"],
                "session_input_names": denoise_execution_local["session_input_names"],
                "filtered_input_names": denoise_execution_local["filtered_input_names"],
                "dropped_inputs": denoise_execution_local["dropped_inputs"],
                "graph_optimization_level": denoise_execution_local.get("graph_optimization_level"),
            }
        else:
            failed_denoise_execution = denoise_execution_local
            if denoise_pipeline_execution is not None and denoise_pipeline_outputs is None:
                failed_denoise_execution = denoise_pipeline_execution
            execution["denoise_step"] = {
                "status": "error",
                "error_type": failed_denoise_execution.get("error_type", "DenoiseStepExecutionError"),
                "error": failed_denoise_execution.get("error", "denoise_step ONNX execution failed"),
                "providers": list(failed_denoise_execution.get("providers", [])),
            }
    else:
        execution["denoise_step"] = {
            "status": "skipped",
            "reason": "prefix_cache_onnx_unavailable",
        }

    def _metric_map_from_pairs(pairs: list[tuple[str, torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, float]]:
        return {
            name: compare_arrays(tensor_to_numpy(lhs), tensor_to_numpy(rhs))
            for name, lhs, rhs in pairs
        }

    runtime_vs_export_reference = {
        "vision_encoder": _metric_map_from_pairs(
            [
                ("top", runtime_outputs["vision_top"], export_outputs["vision_top"]),
                ("wrist", runtime_outputs["vision_wrist"], export_outputs["vision_wrist"]),
            ]
        ),
        "prefix_cache": _metric_map_from_pairs(
            [
                (name, runtime_tensor, export_tensor)
                for name, runtime_tensor, export_tensor in zip(
                    prefix_names,
                    runtime_outputs["prefix_outputs"],
                    export_outputs["prefix_outputs"],
                    strict=True,
                )
            ]
        ),
        "denoise_step": {
            "v_t": compare_arrays(
                tensor_to_numpy(runtime_outputs["denoise_output"]),
                tensor_to_numpy(export_outputs["denoise_output"]),
            )
        },
    }

    export_reference_vs_onnx = {
        "vision_encoder": (
            {
                "top": compare_arrays(tensor_to_numpy(export_outputs["vision_top"]), vision_top_onnx),
                "wrist": compare_arrays(tensor_to_numpy(export_outputs["vision_wrist"]), vision_wrist_onnx),
            }
            if vision_top_onnx is not None and vision_wrist_onnx is not None
            else {}
        ),
        "prefix_cache": (
            {
                name: compare_arrays(tensor_to_numpy(export_tensor), onnx_value)
                for name, export_tensor, onnx_value in zip(
                    prefix_names,
                    export_outputs["prefix_outputs"],
                    prefix_onnx_outputs,
                    strict=True,
                )
            }
            if prefix_onnx_outputs is not None
            else {}
        ),
        "denoise_step": (
            {
                "v_t": compare_arrays(
                    tensor_to_numpy(export_outputs["denoise_output"]),
                    denoise_onnx,
                )
            }
            if denoise_onnx is not None
            else {}
        ),
    }

    runtime_reference_vs_onnx = {
        "vision_encoder": (
            {
                "top": compare_arrays(tensor_to_numpy(runtime_outputs["vision_top"]), vision_top_onnx),
                "wrist": compare_arrays(tensor_to_numpy(runtime_outputs["vision_wrist"]), vision_wrist_onnx),
            }
            if vision_top_onnx is not None and vision_wrist_onnx is not None
            else {}
        ),
        "prefix_cache": (
            {
                name: compare_arrays(tensor_to_numpy(runtime_tensor), onnx_value)
                for name, runtime_tensor, onnx_value in zip(
                    prefix_names,
                    runtime_outputs["prefix_outputs"],
                    prefix_onnx_outputs,
                    strict=True,
                )
            }
            if prefix_onnx_outputs is not None
            else {}
        ),
        "denoise_step": (
            {
                "v_t": compare_arrays(
                    tensor_to_numpy(runtime_outputs["denoise_output"]),
                    denoise_onnx,
                )
            }
            if denoise_onnx is not None
            else {}
        ),
    }

    pipeline_compare = (
        {
            "runtime_reference_vs_onnx": {
                "v_t": compare_arrays(
                    tensor_to_numpy(runtime_outputs["denoise_output"]),
                    denoise_pipeline_onnx,
                )
            },
            "export_reference_vs_onnx": {
                "v_t": compare_arrays(
                    tensor_to_numpy(export_outputs["denoise_output"]),
                    denoise_pipeline_onnx,
                )
            },
        }
        if denoise_pipeline_onnx is not None
        else {}
    )

    def _summaries(metric_payload: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
        return {
            name: (summarize_metric_map(metrics) if metrics else None)
            for name, metrics in metric_payload.items()
        }

    runtime_vs_export_summary = _summaries(runtime_vs_export_reference)
    export_reference_vs_onnx_summary = _summaries(export_reference_vs_onnx)
    runtime_reference_vs_onnx_summary = _summaries(runtime_reference_vs_onnx)
    pipeline_summary = _summaries(pipeline_compare) if pipeline_compare else {}
    contract_checks = _contract_checks(onnx_contract, execution.get("denoise_step", {}))
    stage2_acceptance = _stage2_acceptance(
        execution=execution,
        export_reference_vs_onnx_summary=export_reference_vs_onnx_summary,
        contract_checks=contract_checks,
    )

    return {
        "providers": providers,
        "onnx_contract": onnx_contract,
        "contract_checks": contract_checks,
        "stage2_acceptance": stage2_acceptance,
        "execution": execution,
        "runtime_vs_export_reference": {
            "metrics": runtime_vs_export_reference,
            "summary": runtime_vs_export_summary,
            "checks": {
                name: (evaluate_summary(summary) if summary is not None else None)
                for name, summary in runtime_vs_export_summary.items()
            },
        },
        "export_reference_vs_onnx": {
            "metrics": export_reference_vs_onnx,
            "summary": export_reference_vs_onnx_summary,
            "checks": {
                name: (evaluate_summary(summary) if summary is not None else None)
                for name, summary in export_reference_vs_onnx_summary.items()
            },
        },
        "runtime_reference_vs_onnx": {
            "metrics": runtime_reference_vs_onnx,
            "summary": runtime_reference_vs_onnx_summary,
            "checks": {
                name: (evaluate_summary(summary) if summary is not None else None)
                for name, summary in runtime_reference_vs_onnx_summary.items()
            },
        },
        "pipeline_compare": {
            "metrics": pipeline_compare,
            "summary": pipeline_summary,
            "checks": {
                name: (evaluate_summary(summary) if summary is not None else None)
                for name, summary in pipeline_summary.items()
            },
        },
        "passed": stage2_acceptance["status"] == "pass",
    }


def export_all_subgraphs(
    *,
    policy_path: str | Path,
    onnx_dir: str | Path,
    opset: int = 19,
    strict: bool = False,
) -> dict[str, Any]:
    context = build_runtime_context(policy_path, strict=strict)
    modules = lazy_import_pi05_modules()
    num_layers = int(context.policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)
    onnx_root = Path(onnx_dir).expanduser().resolve()
    onnx_root.mkdir(parents=True, exist_ok=True)

    context.policy.modeling_make_att_2d_masks = modules["make_att_2d_masks"]
    runtime_mode = describe_execution_mode(context.policy, use_autocast=True)

    reference_vision_wrapper = Pi05VisionEncoderExportWrapper(context.policy, use_autocast=True).eval()
    reference_prefix_wrapper = Pi05PrefixCacheExportWrapper(
        context.policy,
        num_layers=num_layers,
        use_autocast=True,
    ).eval()
    reference_denoise_wrapper = Pi05DenoiseStepExportWrapper(
        context.policy,
        num_layers=num_layers,
        dynamic_cache_cls=modules["DynamicCache"],
        use_autocast=True,
    ).eval()

    vision_output = reference_vision_wrapper(context.top_image)
    wrist_output = reference_vision_wrapper(context.wrist_image)
    prefix_outputs = reference_prefix_wrapper(
        vision_output,
        wrist_output,
        context.image_mask_top,
        context.image_mask_wrist,
        context.tokens,
        context.token_attention_mask,
    )
    denoise_output = reference_denoise_wrapper(
        context.x_t,
        context.timestep,
        prefix_outputs[0],
        *prefix_outputs[1:],
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
    export_mode = describe_execution_mode(context.policy, use_autocast=False)

    export_vision_wrapper = Pi05VisionEncoderExportWrapper(context.policy, use_autocast=False).eval()
    export_prefix_wrapper = Pi05PrefixCacheExportWrapper(
        context.policy,
        num_layers=num_layers,
        use_autocast=False,
    ).eval()
    export_denoise_wrapper = Pi05DenoiseStepExportWrapper(
        context.policy,
        num_layers=num_layers,
        dynamic_cache_cls=modules["DynamicCache"],
        use_autocast=False,
    ).eval()

    vision_path = onnx_root / "pi_shared_vision_encoder.onnx"
    prefix_path = onnx_root / "pi_shared_prefix_cache.onnx"
    denoise_path = onnx_root / "pi05_denoise_step.onnx"

    export_vision_output = export_vision_wrapper(export_top_image)
    export_wrist_output = export_vision_wrapper(export_wrist_image)
    export_prefix_outputs = export_prefix_wrapper(
        export_vision_output,
        export_wrist_output,
        export_image_mask_top,
        export_image_mask_wrist,
        export_tokens,
        export_token_attention_mask,
    )
    denoise_export_output = export_denoise_wrapper(
        export_x_t,
        export_timestep,
        export_prefix_outputs[0],
        *export_prefix_outputs[1:],
    )

    denoise_exporter = "dynamo" if importlib.util.find_spec("onnxscript") is not None else "legacy"

    export_onnx_model(
        export_vision_wrapper,
        (export_top_image,),
        vision_path,
        input_names=["image"],
        output_names=["image_embs"],
        opset=opset,
        exporter="legacy",
    )
    export_onnx_model(
        export_prefix_wrapper,
        (
            export_vision_output,
            export_wrist_output,
            export_image_mask_top,
            export_image_mask_wrist,
            export_tokens,
            export_token_attention_mask,
        ),
        prefix_path,
        input_names=[
            "image_embs_top",
            "image_embs_wrist",
            "image_mask_top",
            "image_mask_wrist",
            "tokens",
            "token_attention_mask",
        ],
        output_names=["prefix_pad_masks", *cache_tensor_names(num_layers)],
        opset=opset,
        exporter="legacy",
    )
    export_onnx_model(
        export_denoise_wrapper,
        (
            export_x_t,
            export_timestep,
            export_prefix_outputs[0],
            *export_prefix_outputs[1:],
        ),
        denoise_path,
        input_names=[
            "x_t",
            "timestep",
            "prefix_pad_masks",
            *cache_tensor_names(num_layers),
        ],
        output_names=["v_t"],
        opset=opset,
        exporter=denoise_exporter,
    )
    denoise_trt_sanitization = sanitize_onnx_for_trt(denoise_path)

    onnx_paths = {
        "vision_encoder": vision_path.as_posix(),
        "prefix_cache": prefix_path.as_posix(),
        "denoise_step": denoise_path.as_posix(),
    }
    immediate_compare = compare_exported_subgraphs(
        context=context,
        num_layers=num_layers,
        onnx_paths=onnx_paths,
        runtime_outputs={
            "vision_top": vision_output,
            "vision_wrist": wrist_output,
            "prefix_outputs": prefix_outputs,
            "denoise_output": denoise_output,
        },
        export_outputs={
            "vision_top": export_vision_output,
            "vision_wrist": export_wrist_output,
            "prefix_outputs": export_prefix_outputs,
            "denoise_output": denoise_export_output,
        },
    )
    onnx_contract = immediate_compare["onnx_contract"]
    contract_checks = immediate_compare["contract_checks"]
    runtime_reference_contract = {
        "vision_encoder": _vision_contract(vision_output, mode=runtime_mode),
        "prefix_cache": _prefix_contract(prefix_outputs, num_layers=num_layers, mode=runtime_mode),
        "denoise_step": _denoise_contract(denoise_output, num_layers=num_layers, mode=runtime_mode),
    }
    export_reference_contract = {
        "vision_encoder": _vision_contract(export_vision_output, mode=export_mode),
        "prefix_cache": _prefix_contract(export_prefix_outputs, num_layers=num_layers, mode=export_mode),
        "denoise_step": _denoise_contract(denoise_export_output, num_layers=num_layers, mode=export_mode),
    }

    return {
        "context": context,
        "num_layers": num_layers,
        "onnx_paths": onnx_paths,
        "export_routes": {
            "vision_encoder": "legacy",
            "prefix_cache": "legacy",
            "denoise_step": denoise_exporter,
        },
        "trt_sanitization": {
            "vision_encoder": {"rewritten_initializers": [], "rewritten_count": 0},
            "prefix_cache": {"rewritten_initializers": [], "rewritten_count": 0},
            "denoise_step": denoise_trt_sanitization,
        },
        "onnx_file_sizes": {
            "vision_encoder": vision_path.stat().st_size,
            "prefix_cache": prefix_path.stat().st_size,
            "denoise_step": denoise_path.stat().st_size,
        },
        "runtime_reference_contract": runtime_reference_contract,
        "export_reference_contract": export_reference_contract,
        "onnx_contract": onnx_contract,
        "contract_checks": contract_checks,
        "compare_basis": {
            "primary_export_fidelity_metric": "export_reference_vs_onnx",
            "runtime_reference_note": (
                "runtime_reference_vs_onnx is reported separately because the current runtime path uses "
                "autocast/bfloat16 when CUDA is available, while export_reference stays float32/no-autocast."
            ),
            "pipeline_note": (
                "pipeline_compare is only present when all three ONNX subgraphs execute successfully in a chained run."
            ),
            "denoise_exporter_rationale": (
                "denoise_step prefers dynamo export when onnxscript is installed because legacy export can drop "
                "timestep from the ONNX graph inputs."
            ),
            "export_fidelity_execution_note": (
                "Stage 2 export-fidelity compare keeps CPUExecutionProvider for vision/prefix and allows "
                "denoise_step to fall back to CUDAExecutionProvider when the CPU provider lacks required kernels."
            ),
            "trt_sanitization_note": (
                "Post-export ONNX sanitization rewrites FLOAT64 initializers to FLOAT32 and inlines small "
                "constants so TensorRT can parse the denoise_step graph reliably."
            ),
        },
        "analysis_notes": [
            "Current main mismatch source is the runtime/export mode split, not the ONNX conversion itself.",
            "A passing export_reference_vs_onnx check means the exported graph matches the float32/no-autocast export boundary.",
            (
                "A failing runtime_reference_vs_onnx check can still be expected if the runtime path remains "
                "GPU autocast/bfloat16."
            ),
            (
                "Stage 2 acceptance now requires the denoise_step ONNX to keep timestep as a live runtime "
                "input and requires the immediate export-fidelity compare to pass."
            ),
        ],
        "torch_reference": runtime_reference_contract,
        "immediate_onnx_compare": immediate_compare,
        "stage2_acceptance": immediate_compare["stage2_acceptance"],
        "metadata_notes": metadata_note_payload(context),
    }
