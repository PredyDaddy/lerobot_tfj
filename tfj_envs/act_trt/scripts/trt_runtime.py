from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorrt as trt
import torch


def trt_dtype_to_torch_dtype(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


@dataclass(frozen=True)
class TensorMeta:
    name: str
    mode: str
    dtype: str
    shape: list[int]


class TensorRTRunner:
    """Minimal TensorRT engine runner (static batch expected by default)."""

    def __init__(self, engine_path: str | Path, device: str = "cuda:0") -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Engine not found: {self.engine_path}")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRTRunner requires a CUDA device, got {self.device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.profile_index = 0
        self.stream = torch.cuda.Stream(device=self.device)

    def describe(self) -> list[TensorMeta]:
        result: list[TensorMeta] = []
        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            result.append(
                TensorMeta(
                    name=name,
                    mode="input" if mode == trt.TensorIOMode.INPUT else "output",
                    dtype=str(self.engine.get_tensor_dtype(name)),
                    shape=[int(dim) for dim in self.engine.get_tensor_shape(name)],
                )
            )
        return result

    def _coerce_input(self, name: str, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        dtype = trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(name))
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
        elif isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            raise TypeError(f"Unsupported input type for {name}: {type(value)}")

        tensor = tensor.contiguous()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=False)
        return tensor.contiguous()

    def infer(self, feed_dict: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in feed_dict]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}")

        active_tensors: dict[str, torch.Tensor] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            stream = self.stream
            if self.engine.num_optimization_profiles > 0:
                self.context.set_optimization_profile_async(self.profile_index, stream.cuda_stream)

            for name in self.input_names:
                tensor = self._coerce_input(name, feed_dict[name])
                engine_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
                if any(dim < 0 for dim in engine_shape):
                    self.context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
                active_tensors[name] = tensor
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

            for name in self.output_names:
                output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in output_shape):
                    raise RuntimeError(f"Unresolved output shape for {name}: {output_shape}")
                output_dtype = trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(name))
                output = torch.empty(output_shape, dtype=output_dtype, device=self.device)
                active_tensors[name] = output
                self.context.set_tensor_address(name, int(output.data_ptr()))

            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            stream.synchronize()

        return {name: active_tensors[name] for name in self.output_names}

