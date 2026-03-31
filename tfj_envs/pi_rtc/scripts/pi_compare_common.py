#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from common import discover_local_tokenizer_path as resolve_common_local_tokenizer_path  # noqa: E402
from common import resolve_checkpoint_dir, write_json  # noqa: E402


DEFAULT_COMPARE_THRESHOLDS = {
    "max_abs_diff": 5e-2,
    "mean_abs_diff": 5e-3,
    "min_cosine_similarity": 0.999,
}


@dataclass
class Pi05RuntimeContext:
    policy_path: Path
    local_tokenizer_path: str | None
    policy: Any
    preprocessor: Any
    raw_batch: dict[str, Any]
    processed_batch: dict[str, Any]
    top_image: torch.Tensor
    wrist_image: torch.Tensor
    image_mask_top: torch.Tensor
    image_mask_wrist: torch.Tensor
    tokens: torch.Tensor
    token_attention_mask: torch.Tensor
    x_t: torch.Tensor
    timestep: torch.Tensor


def install_siglip_check_shim() -> None:
    try:
        importlib.import_module("transformers.models.siglip.check")
        return
    except Exception:
        pass

    shim = ModuleType("transformers.models.siglip.check")
    shim.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules["transformers.models.siglip.check"] = shim

    siglip_pkg = importlib.import_module("transformers.models.siglip")
    setattr(siglip_pkg, "check", shim)


def install_gemma_rmsnorm_repr_shim() -> None:
    try:
        from transformers.models.gemma import modeling_gemma
    except Exception:
        return

    gemma_rmsnorm_cls = modeling_gemma.GemmaRMSNorm
    if getattr(gemma_rmsnorm_cls, "_pi_trt_extra_repr_shim_installed", False):
        return

    def _safe_extra_repr(self: Any) -> str:
        if hasattr(self, "weight") and getattr(self, "weight") is not None:
            shape = tuple(self.weight.shape)
        else:
            shape = (int(getattr(self, "dim", -1)),)

        repr_str = f"{shape}, eps={self.eps}"
        if getattr(self, "dense", None) is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str

    gemma_rmsnorm_cls.extra_repr = _safe_extra_repr
    gemma_rmsnorm_cls._pi_trt_extra_repr_shim_installed = True


def discover_local_tokenizer_path() -> str | None:
    resolved = resolve_common_local_tokenizer_path(require=False)
    return resolved.as_posix() if resolved is not None else None


def lazy_import_pi05_modules() -> dict[str, Any]:
    install_siglip_check_shim()
    install_gemma_rmsnorm_repr_shim()
    from transformers.cache_utils import DynamicCache

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
    from lerobot.processor import PolicyProcessorPipeline

    return {
        "DynamicCache": DynamicCache,
        "PI05Policy": PI05Policy,
        "PolicyProcessorPipeline": PolicyProcessorPipeline,
        "make_att_2d_masks": make_att_2d_masks,
    }


def load_pi05_policy(policy_path: str | Path, *, strict: bool = False) -> Any:
    modules = lazy_import_pi05_modules()
    policy_dir = resolve_checkpoint_dir(policy_path)
    policy = modules["PI05Policy"].from_pretrained(policy_dir, strict=strict)
    policy.eval()
    return policy


def load_pi05_preprocessor(policy_path: str | Path, *, device: str = "cpu") -> tuple[Any, str | None]:
    modules = lazy_import_pi05_modules()
    policy_dir = resolve_checkpoint_dir(policy_path)
    tokenizer_path = discover_local_tokenizer_path()
    overrides = {"device_processor": {"device": device}}
    if tokenizer_path is not None:
        overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}
    preprocessor = modules["PolicyProcessorPipeline"].from_pretrained(
        policy_dir,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )
    return preprocessor, tokenizer_path


def autocast_context() -> Any:
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def policy_compute_dtype(policy: Any) -> torch.dtype:
    return policy.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype


def _linspace_tensor(shape: tuple[int, ...], start: float, end: float) -> torch.Tensor:
    numel = math.prod(shape)
    tensor = torch.linspace(start, end, steps=numel, dtype=torch.float32)
    return tensor.view(shape)


def build_deterministic_raw_batch(policy: Any) -> dict[str, Any]:
    state_dim = int(policy.config.input_features["observation.state"].shape[0])
    action_dim = int(policy.config.output_features["action"].shape[0])
    chunk_size = int(policy.config.chunk_size)
    height = int(policy.config.input_features["observation.images.top"].shape[1])
    width = int(policy.config.input_features["observation.images.top"].shape[2])
    top = _linspace_tensor((3, height, width), 0.0, 1.0)
    wrist = _linspace_tensor((3, height, width), 0.1, 1.1).remainder(1.0)
    return {
        "observation.state": _linspace_tensor((state_dim,), -0.75, 0.75),
        "observation.images.top": top,
        "observation.images.wrist": wrist,
        "action": _linspace_tensor((chunk_size, action_dim), -0.25, 0.25),
        "task": "Pick up the red block and place it into the tray",
    }


def build_runtime_context(policy_path: str | Path, *, strict: bool = False) -> Pi05RuntimeContext:
    policy_dir = resolve_checkpoint_dir(policy_path)
    policy = load_pi05_policy(policy_dir, strict=strict)
    preprocessor, tokenizer_path = load_pi05_preprocessor(policy_dir, device="cpu")
    raw_batch = build_deterministic_raw_batch(policy)
    processed_batch = preprocessor(raw_batch)

    images, img_masks = policy._preprocess_images(processed_batch)
    device = torch.device(policy.config.device)
    images = [image.to(device) for image in images]
    img_masks = [mask.to(device) for mask in img_masks]
    tokens = processed_batch["observation.language.tokens"].to(device)
    token_attention_mask = processed_batch["observation.language.attention_mask"].to(device)

    x_t = _linspace_tensor(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        -0.5,
        0.5,
    ).to(device)
    timestep = torch.tensor([1.0], dtype=torch.float32, device=device)

    return Pi05RuntimeContext(
        policy_path=policy_dir,
        local_tokenizer_path=tokenizer_path,
        policy=policy,
        preprocessor=preprocessor,
        raw_batch=raw_batch,
        processed_batch=processed_batch,
        top_image=images[0],
        wrist_image=images[1],
        image_mask_top=img_masks[0].to(dtype=torch.int32),
        image_mask_wrist=img_masks[1].to(dtype=torch.int32),
        tokens=tokens,
        token_attention_mask=token_attention_mask.to(dtype=torch.int32),
        x_t=x_t,
        timestep=timestep,
    )


def cache_tensor_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"past_key_values.layer_{layer_idx:02d}.key")
        names.append(f"past_key_values.layer_{layer_idx:02d}.value")
    return names


def prefix_output_names(num_layers: int) -> list[str]:
    return ["prefix_pad_masks", *cache_tensor_names(num_layers)]


def flatten_dynamic_cache(cache: Any) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for layer_idx in range(len(cache)):
        key, value = cache[layer_idx]
        flat.extend([key, value])
    return flat


def dynamic_cache_from_flat_tensors(flat_tensors: list[torch.Tensor]) -> Any:
    modules = lazy_import_pi05_modules()
    legacy_cache = tuple(
        (flat_tensors[index], flat_tensors[index + 1]) for index in range(0, len(flat_tensors), 2)
    )
    return modules["DynamicCache"].from_legacy_cache(legacy_cache)


def bfloat16_numpy(tensor: torch.Tensor) -> np.ndarray:
    import ml_dtypes

    return np.asarray(tensor.detach().cpu().float().numpy(), dtype=ml_dtypes.bfloat16)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    detached = tensor.detach().cpu()
    if detached.dtype == torch.bfloat16:
        return detached.float().numpy()
    return detached.numpy()


def feed_value_to_numpy(value: torch.Tensor) -> np.ndarray:
    detached = value.detach().cpu()
    if detached.dtype == torch.bfloat16:
        return bfloat16_numpy(detached)
    return detached.numpy()


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
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
    }


def summarize_metric_map(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in metrics.values()),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in metrics.values()),
        "max_rel_diff": max(item["max_rel_diff"] for item in metrics.values()),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics.values()),
    }


def evaluate_summary(summary: dict[str, float], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_COMPARE_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    failures: list[str] = []
    if float(summary["max_abs_diff"]) > float(active_thresholds["max_abs_diff"]):
        failures.append(
            f"max_abs_diff {float(summary['max_abs_diff']):.6g} > {float(active_thresholds['max_abs_diff']):.6g}"
        )
    if float(summary["mean_abs_diff"]) > float(active_thresholds["mean_abs_diff"]):
        failures.append(
            f"mean_abs_diff {float(summary['mean_abs_diff']):.6g} > {float(active_thresholds['mean_abs_diff']):.6g}"
        )
    if float(summary["min_cosine_similarity"]) < float(active_thresholds["min_cosine_similarity"]):
        failures.append(
            "min_cosine_similarity "
            f"{float(summary['min_cosine_similarity']):.6g} < {float(active_thresholds['min_cosine_similarity']):.6g}"
        )

    return {
        "thresholds": active_thresholds,
        "passed": not failures,
        "failures": failures,
    }


def create_ort_session(
    onnx_path: str | Path,
    *,
    disable_optimizations: bool = False,
    use_cuda: bool = False,
) -> Any:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    if disable_optimizations:
        # Keep BF16 transformer subgraphs unfused. ORT's default graph optimizers
        # rewrite these PI05 graphs into kernels that are not available in the
        # current lerobot environment for prefix_cache / denoise_step.
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in providers:
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def provider_candidates(*, prefer_cuda: bool = True) -> list[list[str]]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    candidates: list[list[str]] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        candidates.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CPUExecutionProvider" in available:
        candidates.append(["CPUExecutionProvider"])
    return candidates or [["CPUExecutionProvider"]]


def run_onnx_with_fallback(
    onnx_path: str | Path,
    output_names: list[str],
    input_feed: dict[str, Any],
    *,
    prefer_cuda: bool = True,
    provider_candidates_override: list[list[str]] | None = None,
    optimization_order: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    attempts: list[dict[str, Any]] = []
    optimization_level_map = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    requested_optimization_order = list(optimization_order or ["all", "basic", "disable"])
    invalid_levels = [name for name in requested_optimization_order if name not in optimization_level_map]
    if invalid_levels:
        raise ValueError(f"Unsupported optimization levels requested: {invalid_levels}")
    optimization_levels = [
        (name, optimization_level_map[name])
        for name in requested_optimization_order
    ]
    requested_provider_candidates = [
        list(candidate)
        for candidate in (provider_candidates_override or provider_candidates(prefer_cuda=prefer_cuda))
    ]

    for providers in requested_provider_candidates:
        for optimization_name, optimization_level in optimization_levels:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = optimization_level
            try:
                session = ort.InferenceSession(
                    onnx_path.as_posix(),
                    sess_options=session_options,
                    providers=providers,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "providers": providers,
                        "optimization_level": optimization_name,
                        "stage": "load",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            session_input_names = [item.name for item in session.get_inputs()]
            filtered_input_feed = {
                name: value for name, value in input_feed.items() if name in session_input_names
            }
            try:
                outputs = session.run(output_names, filtered_input_feed)
            except Exception as exc:
                attempts.append(
                    {
                        "providers": providers,
                        "optimization_level": optimization_name,
                        "active_providers": session.get_providers(),
                        "stage": "run",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "session_input_names": session_input_names,
                        "filtered_input_names": sorted(filtered_input_feed),
                    }
                )
                continue

            runtime_info = {
                "onnx_path": onnx_path.as_posix(),
                "requested_provider_candidates": requested_provider_candidates,
                "requested_providers": providers,
                "active_providers": session.get_providers(),
                "requested_optimization_order": requested_optimization_order,
                "graph_optimization_level": optimization_name,
                "session_input_names": session_input_names,
                "filtered_input_names": sorted(filtered_input_feed),
                "dropped_inputs": sorted(set(input_feed) - set(filtered_input_feed)),
                "attempts": attempts,
            }
            return {name: value for name, value in zip(output_names, outputs, strict=True)}, runtime_info

    raise RuntimeError(
        f"Unable to execute ONNX model {onnx_path} with any provider candidate. attempts={attempts}"
    )


def run_ort_session_filtered(
    session: Any,
    output_names: list[str],
    input_feed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    session_input_names = [item.name for item in session.get_inputs()]
    filtered_input_feed = {
        name: value for name, value in input_feed.items() if name in session_input_names
    }
    outputs = session.run(output_names, filtered_input_feed)
    runtime_info = {
        "active_providers": session.get_providers(),
        "session_input_names": session_input_names,
        "dropped_inputs": sorted(set(input_feed) - set(filtered_input_feed)),
    }
    return {name: value for name, value in zip(output_names, outputs, strict=True)}, runtime_info


def write_markdown(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def dump_json(path: str | Path, payload: Any) -> Path:
    return write_json(path, payload)


def metadata_note_payload(context: Pi05RuntimeContext) -> dict[str, Any]:
    return {
        "policy_path": context.policy_path.as_posix(),
        "local_tokenizer_path": context.local_tokenizer_path,
        "processed_batch_keys": sorted(context.processed_batch.keys()),
        "config": {
            "chunk_size": int(context.policy.config.chunk_size),
            "num_inference_steps": int(context.policy.config.num_inference_steps),
            "tokenizer_max_length": int(context.policy.config.tokenizer_max_length),
            "max_action_dim": int(context.policy.config.max_action_dim),
            "image_resolution": list(context.policy.config.image_resolution),
        },
    }
