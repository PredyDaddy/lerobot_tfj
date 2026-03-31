from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort
import torch


_ORT_OPT_LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


@dataclass(frozen=True)
class OrtTensorMeta:
    name: str
    dtype: str
    shape: list[Any]


def resolve_ort_providers(provider: str = "auto") -> list[str]:
    normalized = provider.strip().lower()
    available = ort.get_available_providers()
    if normalized == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this onnxruntime build. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"Unsupported ORT provider mode: {provider!r}")


class OnnxRuntimeRunner:
    def __init__(
        self,
        onnx_path: str | Path,
        *,
        provider: str = "auto",
        optimization_level: str = "all",
    ) -> None:
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        if optimization_level not in _ORT_OPT_LEVELS:
            raise ValueError(
                f"Unsupported optimization_level={optimization_level!r}. "
                f"Expected one of {sorted(_ORT_OPT_LEVELS)}"
            )

        self.provider_mode = provider
        self.requested_providers = resolve_ort_providers(provider)
        self.optimization_level = optimization_level

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = _ORT_OPT_LEVELS[optimization_level]
        self.session = ort.InferenceSession(
            self.onnx_path.as_posix(),
            sess_options=session_options,
            providers=self.requested_providers,
        )

        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def close(self) -> None:
        self.session = None

    def __enter__(self) -> "OnnxRuntimeRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _tensor_meta(self, item: Any) -> OrtTensorMeta:
        return OrtTensorMeta(
            name=item.name,
            dtype=str(item.type),
            shape=list(item.shape),
        )

    def describe_inputs(self) -> list[OrtTensorMeta]:
        return [self._tensor_meta(item) for item in self.session.get_inputs()]

    def describe_outputs(self) -> list[OrtTensorMeta]:
        return [self._tensor_meta(item) for item in self.session.get_outputs()]

    def engine_summary(self) -> dict[str, Any]:
        return {
            "onnx_path": self.onnx_path.as_posix(),
            "provider_mode": self.provider_mode,
            "requested_providers": list(self.requested_providers),
            "active_providers": list(self.session.get_providers()),
            "optimization_level": self.optimization_level,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "inputs": [asdict(meta) for meta in self.describe_inputs()],
            "outputs": [asdict(meta) for meta in self.describe_outputs()],
        }

    def _coerce_input(self, value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value)

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unsupported ONNX input type: {type(value)}")

        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return np.ascontiguousarray(tensor.numpy())

    def infer_numpy(self, feed_dict: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]:
        filtered_feed = {
            name: self._coerce_input(feed_dict[name])
            for name in self.input_names
            if name in feed_dict
        }
        missing = [name for name in self.input_names if name not in filtered_feed]
        if missing:
            raise KeyError(f"Missing ONNX inputs: {missing}")

        outputs = self.session.run(self.output_names, filtered_feed)
        return {
            name: np.ascontiguousarray(value)
            for name, value in zip(self.output_names, outputs, strict=True)
        }

    def infer(self, feed_dict: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.infer_numpy(feed_dict)
        return {
            name: torch.from_numpy(value.copy())
            for name, value in outputs.items()
        }
