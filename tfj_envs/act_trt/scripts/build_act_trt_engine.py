#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt

from trt_runtime import TensorRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ACT ONNX.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 math during fp32 builds. Disabled by default to reduce numeric drift.",
    )
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timing-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    engine_path = args.engine.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    timing_cache_path = args.timing_cache.expanduser().resolve() if args.timing_cache else None

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.builder_optimization_level = int(args.opt_level)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024**3)))
    if args.precision == "fp32" and hasattr(trt.BuilderFlag, "TF32") and not args.allow_tf32:
        # TensorRT may use TF32 math on supported GPUs by default. That is usually fine for throughput,
        # but it increases drift versus the original PyTorch/ONNX fp32 path.
        config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Requested fp16 but platform_has_fast_fp16 is False")
        config.set_flag(trt.BuilderFlag.FP16)

    if timing_cache_path:
        if timing_cache_path.is_file():
            cache = config.create_timing_cache(timing_cache_path.read_bytes())
        else:
            cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))

    if timing_cache_path:
        timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))

    runner = TensorRTRunner(engine_path=engine_path, device=args.device)
    report = {
        "onnx": str(onnx_path),
        "metadata": str(metadata_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "allow_tf32": bool(args.allow_tf32),
        "workspace_gb": args.workspace_gb,
        "opt_level": args.opt_level,
        "device": args.device,
        "timing_cache": str(timing_cache_path) if timing_cache_path else None,
        "engine_tensors": [tensor.__dict__ for tensor in runner.describe()],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if timing_cache_path:
        print(timing_cache_path)
    print(engine_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
