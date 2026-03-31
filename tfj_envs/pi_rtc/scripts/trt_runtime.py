from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorrt as trt
import torch

_TRT_LOGGER_CACHE: dict[int, trt.Logger] = {}


def _dims_to_list(dims: Sequence[int]) -> list[int]:
    return [int(dim) for dim in dims]


def get_trt_logger(severity: int = trt.Logger.WARNING) -> trt.Logger:
    logger = _TRT_LOGGER_CACHE.get(int(severity))
    if logger is None:
        logger = trt.Logger(severity)
        _TRT_LOGGER_CACHE[int(severity)] = logger
    return logger


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
    bf16_dtype = getattr(trt, "bfloat16", None)
    if bf16_dtype is not None:
        mapping[bf16_dtype] = torch.bfloat16
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


@dataclass(frozen=True)
class ProfileShape:
    min: list[int]
    opt: list[int]
    max: list[int]


@dataclass(frozen=True)
class TensorMeta:
    name: str
    mode: str
    dtype: str
    shape: list[int]
    is_dynamic: bool
    profile_shape: dict[str, list[int]] | None


class TensorRTRunner:
    def __init__(
        self,
        engine_path: str | Path,
        device: str = "cuda:0",
        *,
        profile_index: int = 0,
        log_severity: int = trt.Logger.WARNING,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Engine not found: {self.engine_path}")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRTRunner requires a CUDA device, got {self.device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.logger = get_trt_logger(log_severity)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        if self.engine.num_optimization_profiles > 0 and not (0 <= profile_index < self.engine.num_optimization_profiles):
            raise ValueError(
                f"profile_index={profile_index} is out of range for {self.engine.num_optimization_profiles} profiles"
            )

        self.profile_index = int(profile_index)
        self.tensor_names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.stream = torch.cuda.Stream(device=self.device)

    def close(self) -> None:
        # Explicitly drop TensorRT objects so large engines release memory promptly between stages.
        for attr in ("context", "engine", "runtime", "stream"):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def __enter__(self) -> "TensorRTRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _profile_shape(self, name: str) -> ProfileShape | None:
        if self.engine.num_optimization_profiles <= 0:
            return None
        if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            return None
        min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, self.profile_index)
        return ProfileShape(
            min=_dims_to_list(min_shape),
            opt=_dims_to_list(opt_shape),
            max=_dims_to_list(max_shape),
        )

    def tensor_meta(self, name: str) -> TensorMeta:
        mode = self.engine.get_tensor_mode(name)
        profile_shape = self._profile_shape(name)
        shape = _dims_to_list(self.engine.get_tensor_shape(name))
        return TensorMeta(
            name=name,
            mode="input" if mode == trt.TensorIOMode.INPUT else "output",
            dtype=str(self.engine.get_tensor_dtype(name)),
            shape=shape,
            is_dynamic=any(dim < 0 for dim in shape),
            profile_shape=asdict(profile_shape) if profile_shape is not None else None,
        )

    def describe(self) -> list[TensorMeta]:
        return [self.tensor_meta(name) for name in self.tensor_names]

    def engine_summary(self) -> dict[str, Any]:
        return {
            "engine_path": self.engine_path.as_posix(),
            "device": str(self.device),
            "num_io_tensors": int(self.engine.num_io_tensors),
            "num_layers": int(self.engine.num_layers),
            "num_optimization_profiles": int(self.engine.num_optimization_profiles),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "tensors": [asdict(meta) for meta in self.describe()],
        }

    def _coerce_input(self, name: str, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        expected_dtype = trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(name))
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
        elif isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            raise TypeError(f"Unsupported input type for {name}: {type(value)}")

        tensor = tensor.contiguous()
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(dtype=expected_dtype)
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=False)
        return tensor.contiguous()

    def _validate_input_shape(self, name: str, runtime_shape: tuple[int, ...]) -> None:
        engine_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        if len(runtime_shape) != len(engine_shape):
            raise ValueError(
                f"TensorRT input rank mismatch for {name}: "
                f"engine_shape={engine_shape}, runtime_shape={runtime_shape}"
            )

        profile_shape = self._profile_shape(name)
        if profile_shape is not None:
            for axis, (value, min_dim, max_dim) in enumerate(zip(runtime_shape, profile_shape.min, profile_shape.max)):
                if value < int(min_dim) or value > int(max_dim):
                    raise ValueError(
                        f"TensorRT input shape for {name} is outside profile bounds on axis {axis}: "
                        f"value={value}, min={int(min_dim)}, max={int(max_dim)}, runtime_shape={runtime_shape}"
                    )

        if any(dim < 0 for dim in engine_shape):
            ok = self.context.set_input_shape(name, runtime_shape)
            if not ok:
                raise RuntimeError(
                    f"Failed to set dynamic input shape for {name}: "
                    f"engine_shape={engine_shape}, runtime_shape={runtime_shape}"
                )
            resolved_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in resolved_shape):
                raise RuntimeError(
                    f"TensorRT kept unresolved dynamic dims for {name}: "
                    f"engine_shape={engine_shape}, runtime_shape={runtime_shape}, resolved_shape={resolved_shape}"
                )
            if resolved_shape != runtime_shape:
                raise RuntimeError(
                    f"TensorRT resolved an unexpected input shape for {name}: "
                    f"runtime_shape={runtime_shape}, resolved_shape={resolved_shape}"
                )
            return

        if runtime_shape != engine_shape:
            raise ValueError(
                f"TensorRT static input shape mismatch for {name}: "
                f"expected={engine_shape}, got={runtime_shape}"
            )

    def infer(self, feed_dict: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in feed_dict]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}")
        unexpected = sorted(name for name in feed_dict if name not in self.input_names)
        if unexpected:
            raise KeyError(f"Unexpected TensorRT inputs: {unexpected}")

        active_tensors: dict[str, torch.Tensor] = {}
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if self.engine.num_optimization_profiles > 0:
                self.context.set_optimization_profile_async(self.profile_index, self.stream.cuda_stream)

            for name in self.input_names:
                tensor = self._coerce_input(name, feed_dict[name])
                runtime_shape = tuple(int(dim) for dim in tensor.shape)
                self._validate_input_shape(name, runtime_shape)
                active_tensors[name] = tensor
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"TensorRT shape inference is missing bindings for: {unresolved}")

            for name in self.output_names:
                output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in output_shape):
                    raise RuntimeError(f"Unresolved output shape for {name}: {output_shape}")
                output_dtype = trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(name))
                output_tensor = torch.empty(output_shape, dtype=output_dtype, device=self.device)
                active_tensors[name] = output_tensor
                self.context.set_tensor_address(name, int(output_tensor.data_ptr()))

            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            self.stream.synchronize()

        return {name: active_tensors[name] for name in self.output_names}

    def infer_numpy(self, feed_dict: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
        outputs = self.infer(feed_dict)
        result: dict[str, np.ndarray] = {}
        for name, value in outputs.items():
            detached = value.detach().cpu()
            if detached.dtype == torch.bfloat16:
                result[name] = detached.float().numpy()
            else:
                result[name] = detached.numpy()
        return result
