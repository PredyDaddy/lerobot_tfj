#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorrt as trt

from common import ensure_dir


@dataclass
class BuildProfile:
    max_batch: int = 2
    vit_opt_batch: int = 2
    opt_batch: int = 1
    min_seq_len: int = 80
    opt_seq_len: int = 568
    max_seq_len: int = 600
    workspace_gb: float = 8.0
    strongly_typed: bool = True
    verbose: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GROOT TensorRT engines from locally exported ONNX files.")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--engine-out-dir", required=True)
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--vit-opt-batch", type=int, default=2)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--min-seq-len", type=int, default=80)
    parser.add_argument("--opt-seq-len", type=int, default=568)
    parser.add_argument("--max-seq-len", type=int, default=600)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def serialize_trt_buffer(buffer: Any) -> bytes:
    try:
        return bytes(memoryview(buffer))
    except TypeError:
        with buffer as managed:
            return bytes(managed)


def dim_value(dim: int) -> int | None:
    return int(dim) if dim >= 0 else None


def profile_shape(
    tensor_name: str,
    shape: tuple[int, ...],
    profile: BuildProfile,
    *,
    is_vit: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    min_shape = []
    opt_shape = []
    max_shape = []
    for idx, dim in enumerate(shape):
        if dim >= 0:
            min_shape.append(int(dim))
            opt_shape.append(int(dim))
            max_shape.append(int(dim))
            continue

        if idx == 0:
            min_shape.append(1)
            opt_shape.append(profile.vit_opt_batch if is_vit else profile.opt_batch)
            max_shape.append(profile.max_batch)
            continue

        if idx == 1 and tensor_name in {"inputs_embeds", "attention_mask", "backbone_features", "vl_embs"}:
            min_shape.append(profile.min_seq_len)
            opt_shape.append(profile.opt_seq_len)
            max_shape.append(profile.max_seq_len)
            continue

        raise ValueError(f"Unhandled dynamic dim for tensor {tensor_name}: shape={shape}")

    return tuple(min_shape), tuple(opt_shape), tuple(max_shape)


def build_one_engine(
    onnx_path: Path,
    engine_path: Path,
    profile: BuildProfile,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.INFO if profile.verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if profile.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    with trt.Builder(logger) as builder, builder.create_network(flags) as network, trt.OnnxParser(
        network, logger
    ) as parser:
        ok = parser.parse_from_file(onnx_path.as_posix())
        if not ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT parser failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(profile.workspace_gb * (1 << 30)))
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = 3
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
        if not profile.strongly_typed and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)

        trt_profile = builder.create_optimization_profile()
        is_vit = onnx_path.name == "vit_fp16.onnx"
        inputs = []
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shape = tuple(int(dim) for dim in tensor.shape)
            min_shape, opt_shape, max_shape = profile_shape(tensor.name, shape, profile, is_vit=is_vit)
            trt_profile.set_shape(tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
            inputs.append(
                {
                    "name": tensor.name,
                    "dtype": str(tensor.dtype),
                    "shape": [dim_value(dim) for dim in shape],
                    "profile": {
                        "min": list(min_shape),
                        "opt": list(opt_shape),
                        "max": list(max_shape),
                    },
                }
            )
        config.add_optimization_profile(trt_profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT build_serialized_network returned None for {onnx_path.name}")
        engine_path.write_bytes(serialize_trt_buffer(serialized))

        outputs = []
        for index in range(network.num_outputs):
            tensor = network.get_output(index)
            outputs.append(
                {
                    "name": tensor.name,
                    "dtype": str(tensor.dtype),
                    "shape": [dim_value(int(dim)) for dim in tensor.shape],
                }
            )

    return {
        "onnx": onnx_path.as_posix(),
        "engine": engine_path.as_posix(),
        "inputs": inputs,
        "outputs": outputs,
    }


def main() -> None:
    args = build_parser().parse_args()

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    engine_dir = ensure_dir(Path(args.engine_out_dir).expanduser().resolve())
    profile = BuildProfile(
        max_batch=int(args.max_batch),
        vit_opt_batch=int(args.vit_opt_batch),
        opt_batch=int(args.opt_batch),
        min_seq_len=int(args.min_seq_len),
        opt_seq_len=int(args.opt_seq_len),
        max_seq_len=int(args.max_seq_len),
        workspace_gb=float(args.workspace_gb),
        strongly_typed=True,
        verbose=bool(args.verbose),
    )

    planned = [
        (onnx_dir / "eagle2" / "vit_fp16.onnx", engine_dir / "vit_fp16.engine"),
        (onnx_dir / "eagle2" / "llm_fp16.onnx", engine_dir / "llm_fp16.engine"),
        (onnx_dir / "action_head" / "vlln_vl_self_attention.onnx", engine_dir / "vlln_vl_self_attention.engine"),
        (onnx_dir / "action_head" / "state_encoder.onnx", engine_dir / "state_encoder.engine"),
        (onnx_dir / "action_head" / "action_encoder.onnx", engine_dir / "action_encoder.engine"),
        (onnx_dir / "action_head" / "DiT_fp16.onnx", engine_dir / "DiT_fp16.engine"),
        (onnx_dir / "action_head" / "action_decoder.onnx", engine_dir / "action_decoder.engine"),
    ]

    missing = [path.as_posix() for path, _ in planned if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required ONNX files:\n" + "\n".join(f"  - {path}" for path in missing))

    report = {
        "tensorrt_version": trt.__version__,
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "build_profile": asdict(profile),
        "engines": [],
    }
    for onnx_path, engine_path in planned:
        print(f"[BUILD] {onnx_path.name} -> {engine_path.name}")
        report["engines"].append(build_one_engine(onnx_path, engine_path, profile))
        print(f"[OK] Built {engine_path.name}")

    report_path = engine_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OK] Build report saved to: {report_path}")


if __name__ == "__main__":
    main()
