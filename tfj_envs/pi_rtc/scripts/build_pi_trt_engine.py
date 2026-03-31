#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import numpy_helper

from trt_runtime import TensorRTRunner, get_trt_logger


def _dims_to_list(dims: Any) -> list[int]:
    return [int(dim) for dim in dims]

def _network_tensor_summary(network: trt.INetworkDefinition) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        tensors.append(
            {
                "name": tensor.name,
                "mode": "input",
                "dtype": str(tensor.dtype),
                "shape": _dims_to_list(tensor.shape),
                "is_shape_tensor": bool(tensor.is_shape_tensor),
            }
        )
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        tensors.append(
            {
                "name": tensor.name,
                "mode": "output",
                "dtype": str(tensor.dtype),
                "shape": _dims_to_list(tensor.shape),
                "is_shape_tensor": bool(tensor.is_shape_tensor),
            }
        )
    return tensors


def _validate_static_shapes(network: trt.INetworkDefinition) -> None:
    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = _dims_to_list(tensor.shape)
        if any(dim < 0 for dim in shape):
            dynamic_inputs.append({"name": tensor.name, "shape": shape})
    if dynamic_inputs:
        raise RuntimeError(
            "Dynamic input shapes were found in the parsed ONNX network. "
            "The current PI stage-4 builder expects static-export ONNX files. "
            f"dynamic_inputs={dynamic_inputs}"
        )


def _parser_errors(parser: trt.OnnxParser) -> list[str]:
    return [str(parser.get_error(index)) for index in range(parser.num_errors)]


def _flag_state(config: trt.IBuilderConfig, flag: Any) -> bool | None:
    if flag is None:
        return None
    try:
        return bool(config.get_flag(flag))
    except Exception:
        return None


def _builder_flag_summary(config: trt.IBuilderConfig) -> dict[str, bool | None]:
    return {
        "FP16": _flag_state(config, getattr(trt.BuilderFlag, "FP16", None)),
        "BF16": _flag_state(config, getattr(trt.BuilderFlag, "BF16", None)),
        "TF32": _flag_state(config, getattr(trt.BuilderFlag, "TF32", None)),
        "OBEY_PRECISION_CONSTRAINTS": _flag_state(
            config,
            getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None),
        ),
        "PREFER_PRECISION_CONSTRAINTS": _flag_state(
            config,
            getattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS", None),
        ),
    }


def _engine_io_dtype_summary(engine_summary: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for tensor in engine_summary.get("tensors", []):
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        mode = tensor.get("mode")
        if not isinstance(name, str) or dtype is None:
            continue
        if mode == "input":
            inputs[name] = str(dtype)
        elif mode == "output":
            outputs[name] = str(dtype)
    return {
        "inputs": inputs,
        "outputs": outputs,
    }


def _constant_node_dtypes(graph: onnx.GraphProto) -> dict[str, dict[str, Any]]:
    dtypes: dict[str, dict[str, Any]] = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            value = numpy_helper.to_array(attr.t)
            value_arr = np.asarray(value)
            for output_name in node.output:
                dtypes[output_name] = {
                    "dtype": str(value_arr.dtype),
                    "shape": list(value_arr.shape),
                }
    return dtypes


def _onnx_trt_compatibility_hints(onnx_path: Path) -> list[str]:
    try:
        model = onnx.load(onnx_path.as_posix(), load_external_data=True)
    except Exception as exc:
        return [f"compatibility inspection skipped: {type(exc).__name__}: {exc}"]

    hints: list[str] = []
    constant_dtypes = _constant_node_dtypes(model.graph)
    supported_cumsum_dtypes = {"float16", "float32", "int32", "int64", "bfloat16"}
    for node in model.graph.node:
        if node.op_type != "CumSum" or not node.input:
            continue
        info = constant_dtypes.get(node.input[0])
        if info is None:
            hints.append(
                f"CumSum node {node.name or '<unnamed>'} consumes non-constant input {node.input[0]}; "
                "inspect exporter-side dtype before this node."
            )
            continue
        dtype = str(info["dtype"])
        if dtype not in supported_cumsum_dtypes:
            hints.append(
                f"CumSum node {node.name or '<unnamed>'} consumes constant input {node.input[0]} "
                f"with dtype={dtype}, shape={info['shape']}. TensorRT 10.13 only accepts "
                "Float/Half/BFloat16/Int32/Int64 for cumulative layers. "
                "Exporter should cast the data input before CumSum."
            )
    return hints


def _resolve_layer_type(name: str) -> trt.LayerType:
    try:
        return getattr(trt.LayerType, name.upper())
    except AttributeError as exc:
        raise ValueError(f"Unsupported TensorRT layer type name: {name}") from exc


def _apply_precision_constraints(
    network: trt.INetworkDefinition,
    *,
    force_fp32_layer_types: list[str],
) -> dict[str, Any]:
    if not force_fp32_layer_types:
        return {
            "forced_layer_types": [],
            "matched_layers": [],
            "matched_count": 0,
        }

    target_types = {_resolve_layer_type(name): name.upper() for name in force_fp32_layer_types}
    matched_layers: list[dict[str, Any]] = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type not in target_types:
            continue
        layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
            layer.set_output_type(output_index, trt.float32)
        matched_layers.append(
            {
                "index": int(index),
                "name": str(layer.name),
                "layer_type": target_types[layer.type],
                "num_outputs": int(layer.num_outputs),
            }
        )

    return {
        "forced_layer_types": [name.upper() for name in force_fp32_layer_types],
        "matched_layers": matched_layers,
        "matched_count": len(matched_layers),
    }


def build_engine_from_onnx(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp32",
    allow_tf32: bool = False,
    workspace_gb: float = 8.0,
    opt_level: int = 3,
    device: str = "cuda:0",
    timing_cache_path: str | Path | None = None,
    force_fp32_layer_types: list[str] | None = None,
    log_severity: int = trt.Logger.WARNING,
) -> dict[str, Any]:
    resolved_onnx = Path(onnx_path).expanduser().resolve()
    resolved_engine = Path(engine_path).expanduser().resolve()
    resolved_timing_cache = (
        Path(timing_cache_path).expanduser().resolve() if timing_cache_path is not None else None
    )

    if not resolved_onnx.is_file():
        raise FileNotFoundError(f"ONNX file not found: {resolved_onnx}")

    logger = get_trt_logger(log_severity)
    trt.init_libnvinfer_plugins(logger, "")

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError(f"TensorRT build requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    with torch.cuda.device(cuda_device):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        parsed = parser.parse_from_file(resolved_onnx.as_posix())
        parser_errors = _parser_errors(parser)
        if not parsed:
            compatibility_hints = _onnx_trt_compatibility_hints(resolved_onnx)
            hint_block = ""
            if compatibility_hints:
                hint_block = "\nCompatibility hints:\n- " + "\n- ".join(compatibility_hints)
            raise RuntimeError(
                "Failed to parse ONNX file with TensorRT.\n"
                + "\n".join(parser_errors or [f"unknown parser failure for {resolved_onnx}"])
                + hint_block
            )

        _validate_static_shapes(network)

        force_fp32_layer_types = list(force_fp32_layer_types or [])
        precision_constraints = _apply_precision_constraints(
            network,
            force_fp32_layer_types=force_fp32_layer_types,
        )

        config = builder.create_builder_config()
        config.builder_optimization_level = int(opt_level)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(float(workspace_gb) * (1024**3)))

        if hasattr(trt.BuilderFlag, "TF32") and not allow_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)

        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("Requested fp16 build but platform_has_fast_fp16 is False")
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "bf16":
            bf16_flag = getattr(trt.BuilderFlag, "BF16", None)
            if bf16_flag is None:
                raise RuntimeError("Requested bf16 build but this TensorRT build exposes no BF16 flag")
            config.set_flag(bf16_flag)
        elif precision != "fp32":
            raise ValueError(f"Unsupported precision: {precision}")

        if precision_constraints["matched_count"] > 0:
            obey_constraints = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None)
            if obey_constraints is not None:
                config.set_flag(obey_constraints)
            else:
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        if resolved_timing_cache is not None:
            if resolved_timing_cache.is_file():
                cache = config.create_timing_cache(resolved_timing_cache.read_bytes())
            else:
                cache = config.create_timing_cache(b"")
            config.set_timing_cache(cache, ignore_mismatch=False)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")

    resolved_engine.parent.mkdir(parents=True, exist_ok=True)
    resolved_engine.write_bytes(bytes(serialized_engine))

    if resolved_timing_cache is not None:
        resolved_timing_cache.parent.mkdir(parents=True, exist_ok=True)
        resolved_timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))

    runner = TensorRTRunner(engine_path=resolved_engine, device=device)
    external_data_files = [
        path.as_posix()
        for path in sorted(resolved_onnx.parent.glob(f"{resolved_onnx.name}*"))
        if path.resolve() != resolved_onnx
    ]

    try:
        builder_flags = _builder_flag_summary(config)
        engine_summary = runner.engine_summary()
        return {
            "onnx": resolved_onnx.as_posix(),
            "onnx_size_bytes": int(resolved_onnx.stat().st_size),
            "onnx_external_data_files": external_data_files,
            "engine": resolved_engine.as_posix(),
            "engine_size_bytes": int(resolved_engine.stat().st_size),
            "variant": "pi05",
            "requested_precision": precision,
            "precision": precision,
            "precision_constraints": precision_constraints,
            "allow_tf32": bool(allow_tf32),
            "workspace_gb": float(workspace_gb),
            "opt_level": int(opt_level),
            "device": device,
            "timing_cache": resolved_timing_cache.as_posix() if resolved_timing_cache is not None else None,
            "tensorrt_version": str(trt.__version__),
            "builder_flags": builder_flags,
            "builder_capabilities": {
                "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
                "platform_has_tf32": bool(getattr(builder, "platform_has_tf32", False)),
            },
            "parser_errors": parser_errors,
            "network_tensors": _network_tensor_summary(network),
            "engine_summary": engine_summary,
            "effective_precision_evidence": {
                "builder_flags": builder_flags,
                "forced_fp32_layer_types": precision_constraints.get("forced_layer_types", []),
                "precision_constraint_matched_count": precision_constraints.get("matched_count", 0),
                "engine_io_dtypes": _engine_io_dtype_summary(engine_summary),
                "note": (
                    "This report proves requested precision, TensorRT builder flag state, "
                    "forced-fp32 constraints, and visible engine I/O dtypes. "
                    "It does not guarantee per-layer effective execution precision."
                ),
            },
        }
    finally:
        runner.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one TensorRT engine from one PI ONNX file.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 math during fp32 fallback paths. Disabled by default to reduce drift.",
    )
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timing-cache", type=Path, default=None)
    parser.add_argument(
        "--force-fp32-layer-types",
        nargs="*",
        default=None,
        help="Optional TensorRT layer types to force to fp32 while building lower-precision engines.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_engine_from_onnx(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        allow_tf32=bool(args.allow_tf32),
        workspace_gb=args.workspace_gb,
        opt_level=args.opt_level,
        device=args.device,
        timing_cache_path=args.timing_cache,
        force_fp32_layer_types=args.force_fp32_layer_types,
    )

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if report["timing_cache"] is not None:
        print(report["timing_cache"])
    print(report["engine"])
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
