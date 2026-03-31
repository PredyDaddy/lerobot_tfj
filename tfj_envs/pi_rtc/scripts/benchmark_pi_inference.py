#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PI_TRT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common import ensure_pi_runtime_compatibility, resolve_checkpoint_dir  # noqa: E402
from export_wrappers import (  # noqa: E402
    Pi05DenoiseStepExportWrapper,
    Pi05PrefixCacheExportWrapper,
    Pi05VisionEncoderExportWrapper,
    describe_execution_mode,
)
from onnx_pi_adapter import OnnxPi05PolicyAdapter  # noqa: E402
from pi_compare_common import build_runtime_context, lazy_import_pi05_modules  # noqa: E402
from run_pi05_onnx_infer_so101 import resolve_onnx_artifacts  # noqa: E402
from run_pi05_trt_infer_so101 import assess_trt_artifact_safety, resolve_trt_artifacts  # noqa: E402
from trt_pi_adapter import TrtPi05PolicyAdapter  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402


DEFAULT_POLICY_PATH = REPO_ROOT / "pi_model"
DEFAULT_ARTIFACTS_PATH = (
    PI_TRT_ROOT / "docs" / "results" / "pi_model_consistency_20260313_182839"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sync_cuda(device: str | torch.device | None) -> None:
    if device is None or not torch.cuda.is_available():
        return
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    with torch.cuda.device(resolved):
        torch.cuda.synchronize()


def _autocast_context(device: torch.device, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    if high == low:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _summarize_times(times_s: list[float], *, warmup: int, iterations: int) -> dict[str, Any]:
    values_ms = [value * 1000.0 for value in times_s]
    sorted_ms = sorted(values_ms)
    return {
        "warmup_iterations": int(warmup),
        "measured_iterations": int(iterations),
        "mean_ms": float(statistics.fmean(values_ms)),
        "std_ms": float(statistics.pstdev(values_ms)) if len(values_ms) > 1 else 0.0,
        "min_ms": float(min(values_ms)),
        "p50_ms": float(_percentile(sorted_ms, 0.50)),
        "p95_ms": float(_percentile(sorted_ms, 0.95)),
        "max_ms": float(max(values_ms)),
        "samples_ms": [float(value) for value in values_ms],
    }


def _summarize_tensor(value: Any) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        numeric_tensor = detached
        if not (torch.is_floating_point(numeric_tensor) or torch.is_complex(numeric_tensor)):
            numeric_tensor = numeric_tensor.to(dtype=torch.float32)
        return {
            "shape": [int(dim) for dim in detached.shape],
            "dtype": str(detached.dtype).replace("torch.", ""),
            "device": str(detached.device),
            "max_abs": float(numeric_tensor.abs().max().item()),
            "mean_abs": float(numeric_tensor.abs().mean().item()),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [_summarize_tensor(item) for item in value[:2]],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "keys": list(value.keys()),
        }
    return None


def _benchmark_callable(
    label: str,
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    sync_device: str | torch.device | None,
) -> tuple[dict[str, Any], Any]:
    last_output: Any = None
    for _ in range(warmup):
        _sync_cuda(sync_device)
        last_output = fn()
        _sync_cuda(sync_device)

    timings_s: list[float] = []
    for _ in range(iterations):
        _sync_cuda(sync_device)
        start_t = time.perf_counter()
        last_output = fn()
        _sync_cuda(sync_device)
        timings_s.append(time.perf_counter() - start_t)

    summary = _summarize_times(timings_s, warmup=warmup, iterations=iterations)
    summary["label"] = label
    output_summary = _summarize_tensor(last_output)
    if output_summary is not None:
        summary["output_summary"] = output_summary
    return summary, last_output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = completed.stdout.strip()
    return output or None


def _probe_gpu() -> dict[str, Any]:
    result: dict[str, Any] = {}
    if torch.cuda.is_available():
        result["torch_cuda_available"] = True
        result["device_count"] = int(torch.cuda.device_count())
        result["devices"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": int(index),
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    else:
        result["torch_cuda_available"] = False
        result["device_count"] = 0
        result["devices"] = []

    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia_smi is not None:
        result["nvidia_smi"] = [line.strip() for line in nvidia_smi.splitlines() if line.strip()]
    return result


def _probe_versions() -> dict[str, Any]:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    try:
        import onnxruntime as ort

        versions["onnxruntime"] = ort.__version__
        versions["onnxruntime_available_providers"] = list(ort.get_available_providers())
    except Exception as exc:
        versions["onnxruntime_error"] = str(exc)

    try:
        import tensorrt as trt

        versions["tensorrt"] = trt.__version__
    except Exception as exc:
        versions["tensorrt_error"] = str(exc)

    git_commit = _run_command(["git", "-C", REPO_ROOT.as_posix(), "rev-parse", "HEAD"])
    if git_commit is not None:
        versions["git_commit"] = git_commit
    return versions


def _build_policy_batch(processed_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    keys = [
        "observation.images.top",
        "observation.images.wrist",
        "observation.language.tokens",
        "observation.language.attention_mask",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        value = processed_batch[key]
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().to(device=device).contiguous()
        else:
            result[key] = value
    return result


def _build_runtime_batch_cpu(processed_batch: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "observation.images.top",
        "observation.images.wrist",
        "observation.language.tokens",
        "observation.language.attention_mask",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        value = processed_batch[key]
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone().contiguous()
        else:
            result[key] = value
    return result


def _load_policy_config(policy_dir: Path, *, device: str) -> Any:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = str(policy_dir)
    policy_cfg.device = device
    if getattr(policy_cfg, "type", None) != "pi05":
        raise ValueError(f"Expected PI05 policy, got {getattr(policy_cfg, 'type', None)!r}")
    return policy_cfg


def _build_markdown_report(report: dict[str, Any]) -> str:
    settings = report["settings"]
    env = report["environment"]
    results = report["results"]

    lines: list[str] = []
    lines.append("# PI0.5 推理时延 Benchmark")
    lines.append("")
    lines.append("## 1. 测试对象")
    lines.append("")
    lines.append(f"- policy_path: `{report['policy_path']}`")
    lines.append(f"- onnx_path: `{report['onnx_artifacts']['onnx_dir']}`")
    lines.append(f"- trt_path: `{report['trt_artifacts']['engine_dir']}`")
    lines.append(f"- measured_at_utc: `{report['measured_at_utc']}`")
    lines.append("")
    lines.append("## 2. 测试设置")
    lines.append("")
    lines.append(f"- warmup_iterations: `{settings['warmup_iterations']}`")
    lines.append(f"- measured_iterations: `{settings['measured_iterations']}`")
    lines.append(f"- num_inference_steps: `{settings['num_inference_steps']}`")
    lines.append(f"- n_action_steps: `{settings['n_action_steps']}`")
    lines.append(f"- chunk_size: `{settings['chunk_size']}`")
    lines.append(f"- torch_device: `{settings['torch_device']}`")
    lines.append(f"- torch_use_amp: `{settings['torch_use_amp']}`")
    lines.append(f"- torch_amp_mode: `{settings['torch_amp_mode']}`")
    lines.append(f"- onnx_provider: `{settings['onnx_provider']}`")
    lines.append(f"- trt_device: `{settings['trt_device']}`")
    lines.append("")
    lines.append("## 3. TRT Provenance")
    lines.append("")
    lines.append(f"- variant: `{report['trt_artifacts'].get('variant')}`")
    lines.append(f"- requested_precision: `{report['trt_artifacts'].get('requested_precision')}`")
    lines.append(f"- metadata_path: `{report['trt_artifacts'].get('metadata_path')}`")
    lines.append(f"- checkpoint_dir: `{report['trt_artifacts'].get('checkpoint_dir')}`")
    lines.append(f"- stage4_report_path: `{report['trt_artifacts'].get('stage4_report_path')}`")
    lines.append(f"- stage4_report_status: `{report['trt_artifacts'].get('stage4_report_status')}`")
    lines.append(f"- stage5_report_path: `{report['trt_artifacts'].get('stage5_report_path')}`")
    lines.append(f"- stage5_report_status: `{report['trt_artifacts'].get('stage5_report_status')}`")
    lines.append(f"- allow_unsafe_trt_artifacts: `{report['trt_artifacts'].get('allow_unsafe_trt_artifacts')}`")
    lines.append("")
    lines.append("## 4. 环境")
    lines.append("")
    if env["versions"].get("git_commit"):
        lines.append(f"- git_commit: `{env['versions']['git_commit']}`")
    lines.append(f"- python: `{env['versions'].get('python')}`")
    lines.append(f"- torch: `{env['versions'].get('torch')}`")
    lines.append(f"- onnxruntime: `{env['versions'].get('onnxruntime', '<unavailable>')}`")
    lines.append(f"- tensorrt: `{env['versions'].get('tensorrt', '<unavailable>')}`")
    if env["gpu"].get("nvidia_smi"):
        for item in env["gpu"]["nvidia_smi"]:
            lines.append(f"- gpu: `{item}`")
    lines.append("")
    lines.append("## 5. 结果总表")
    lines.append("")
    lines.append("| Backend | Stage | mean_ms | p50_ms | p95_ms | min_ms | max_ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for backend_name, backend_payload in results.items():
        for stage_name, stage_payload in backend_payload["latency_ms"].items():
            lines.append(
                f"| {backend_name} | {stage_name} | "
                f"{stage_payload['mean_ms']:.3f} | {stage_payload['p50_ms']:.3f} | "
                f"{stage_payload['p95_ms']:.3f} | {stage_payload['min_ms']:.3f} | "
                f"{stage_payload['max_ms']:.3f} |"
            )
    lines.append("")
    lines.append("## 6. Chunk 推理的实际意义")
    lines.append("")
    for backend_name, backend_payload in results.items():
        chunk = backend_payload["latency_ms"]["pipeline_chunk"]
        amortized = backend_payload["derived"]["amortized_per_action_step_ms"]
        denoise_loop = backend_payload["derived"]["denoise_loop_total_estimate_ms"]
        lines.append(
            f"- {backend_name}: chunk_mean={chunk['mean_ms']:.3f} ms, "
            f"amortized_per_action_step={amortized:.3f} ms, "
            f"estimated_denoise_loop_total={denoise_loop:.3f} ms"
        )
    lines.append("")
    lines.append("## 7. 说明")
    lines.append("")
    lines.append("- 这是离线纯推理 benchmark，不等价于机器人闭环控制 latency。")
    lines.append("- 这次只测模型推理链，不包含相机采集、MJPG 解码、robot observation、send_action、安全限幅、sleep 控频。")
    lines.append("- 输入是 `build_runtime_context()` 生成的 deterministic baseline batch，适合做固定口径对比，不代表真实任务数据分布。")
    lines.append("- 当 `torch_use_amp=true` 时，这里的 `PyTorch AMP` 明确表示 `CUDA BF16 autocast`，不是 `Torch FP16`。")
    lines.append("- `vision_encoder_single` 表示单相机一次调用；`vision_encoder_pair` 表示 top+wrist 两次调用总和。")
    lines.append("- `denoise_step` 是单次 denoise 迭代，不是完整 chunk；`pipeline_chunk` 才是完整一次 action chunk 生成。")
    lines.append("- `amortized_per_action_step_ms = pipeline_chunk / n_action_steps`，这是均摊值，不是每个 control loop 的最坏时延。")
    lines.append("- ONNX Runtime 结果反映的是当前工程实现路径，runner 使用常规 `session.run(...)` 和 `numpy <-> torch` 边界，不是纯 GPU kernel benchmark。")
    lines.append("- PyTorch 子图拆分时延来自 export wrapper 路径，用来和 ONNX/TRT 子图做对应比较。")
    lines.append("- PyTorch 的 `pipeline_chunk` 额外给了真实 `policy.predict_action_chunk(...)` 路径，不是 wrapper 拼接模拟。")
    lines.append("- TensorRT 结论只对当前已验证通过的 static-shape、batch=1、固定 token length engine 成立。")
    if report["trt_artifacts"].get("allow_unsafe_trt_artifacts"):
        lines.append("- 本次 TensorRT benchmark 显式允许了 `unsafe` 工件，因此这些数字只能用于诊断，不可直接当作已通过正确性 gate 的部署结论。")
    return "\n".join(lines) + "\n"


def _benchmark_torch_backend(
    *,
    context: Any,
    torch_device: str,
    use_amp: bool,
    warmup: int,
    iterations: int,
    num_steps: int,
) -> dict[str, Any]:
    device = torch.device(torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA torch benchmark, but CUDA is not available.")

    context.policy.to(device)
    context.policy.eval()

    batch = _build_policy_batch(context.processed_batch, device)
    top_image = context.top_image.detach().to(device=device).contiguous()
    wrist_image = context.wrist_image.detach().to(device=device).contiguous()
    image_mask_top = context.image_mask_top.detach().to(device=device).contiguous()
    image_mask_wrist = context.image_mask_wrist.detach().to(device=device).contiguous()
    tokens = context.tokens.detach().to(device=device).contiguous()
    token_attention_mask = context.token_attention_mask.detach().to(device=device).contiguous()
    noise = context.x_t.detach().to(device=device).contiguous()
    timestep = context.timestep.detach().to(device=device).contiguous()

    num_layers = int(
        context.policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
    )
    original_action_dim = int(context.policy.config.output_features["action"].shape[0])
    batch_size = int(tokens.shape[0])
    modules = lazy_import_pi05_modules()

    def _policy_pipeline() -> torch.Tensor:
        with torch.inference_mode(), _autocast_context(device, use_amp):
            return context.policy.predict_action_chunk(
                batch,
                noise=noise.clone(),
                num_steps=num_steps,
            )

    pipeline_stats, pipeline_output = _benchmark_callable(
        "pipeline_chunk",
        _policy_pipeline,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )

    vision_wrapper = Pi05VisionEncoderExportWrapper(
        context.policy,
        use_autocast=use_amp,
    ).eval()
    prefix_wrapper = Pi05PrefixCacheExportWrapper(
        context.policy,
        num_layers=num_layers,
        use_autocast=use_amp,
    ).eval()
    denoise_wrapper = Pi05DenoiseStepExportWrapper(
        context.policy,
        num_layers=num_layers,
        dynamic_cache_cls=modules["DynamicCache"],
        use_autocast=use_amp,
    ).eval()

    with torch.inference_mode():
        top_embs = vision_wrapper(top_image)
        wrist_embs = vision_wrapper(wrist_image)
        prefix_outputs = prefix_wrapper(
            top_embs,
            wrist_embs,
            image_mask_top,
            image_mask_wrist,
            tokens,
            token_attention_mask,
        )

    def _vision_single() -> torch.Tensor:
        with torch.inference_mode():
            return vision_wrapper(top_image)

    def _vision_pair() -> torch.Tensor:
        with torch.inference_mode():
            top = vision_wrapper(top_image)
            wrist = vision_wrapper(wrist_image)
            return torch.cat([top.flatten(), wrist.flatten()], dim=0)

    def _prefix_stage() -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            return prefix_wrapper(
                top_embs,
                wrist_embs,
                image_mask_top,
                image_mask_wrist,
                tokens,
                token_attention_mask,
            )

    def _denoise_stage() -> torch.Tensor:
        with torch.inference_mode():
            return denoise_wrapper(
                noise,
                timestep,
                prefix_outputs[0],
                *prefix_outputs[1:],
            )

    def _staged_pipeline() -> torch.Tensor:
        with torch.inference_mode():
            stage_top = vision_wrapper(top_image)
            stage_wrist = vision_wrapper(wrist_image)
            stage_prefix = prefix_wrapper(
                stage_top,
                stage_wrist,
                image_mask_top,
                image_mask_wrist,
                tokens,
                token_attention_mask,
            )
            x_t = noise.clone()
            dt = torch.tensor(-1.0 / float(num_steps), dtype=torch.float32, device=device)
            timestep_values = 1.0 - (
                torch.arange(num_steps, dtype=torch.float32, device=device) / float(num_steps)
            )
            for timestep_value in timestep_values:
                v_t = denoise_wrapper(
                    x_t,
                    timestep_value.expand(batch_size),
                    stage_prefix[0],
                    *stage_prefix[1:],
                )
                x_t = (x_t + dt * v_t).contiguous()
            return x_t[:, :, :original_action_dim].to(dtype=torch.float32).contiguous()

    vision_single_stats, _ = _benchmark_callable(
        "vision_encoder_single",
        _vision_single,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )
    vision_pair_stats, _ = _benchmark_callable(
        "vision_encoder_pair",
        _vision_pair,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )
    prefix_stats, _ = _benchmark_callable(
        "prefix_cache",
        _prefix_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )
    denoise_stats, _ = _benchmark_callable(
        "denoise_step",
        _denoise_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )
    staged_pipeline_stats, _ = _benchmark_callable(
        "pipeline_staged",
        _staged_pipeline,
        warmup=warmup,
        iterations=iterations,
        sync_device=device,
    )

    return {
        "backend": "pytorch",
        "execution": describe_execution_mode(context.policy, use_autocast=use_amp),
        "latency_ms": {
            "vision_encoder_single": vision_single_stats,
            "vision_encoder_pair": vision_pair_stats,
            "prefix_cache": prefix_stats,
            "denoise_step": denoise_stats,
            "pipeline_staged": staged_pipeline_stats,
            "pipeline_chunk": pipeline_stats,
        },
        "derived": {
            "amortized_per_action_step_ms": float(
                pipeline_stats["mean_ms"] / float(context.policy.config.n_action_steps)
            ),
            "denoise_loop_total_estimate_ms": float(denoise_stats["mean_ms"] * float(num_steps)),
        },
        "output_summary": _summarize_tensor(pipeline_output),
    }


def _benchmark_onnx_backend(
    *,
    policy_dir: Path,
    runtime_batch_cpu: dict[str, Any],
    noise_cpu: torch.Tensor,
    timestep_cpu: torch.Tensor,
    onnx_path: str,
    stage2_report_path: str | None,
    onnx_provider: str,
    warmup: int,
    iterations: int,
    num_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_cfg = _load_policy_config(policy_dir, device="cpu")
    policy_cfg.num_inference_steps = int(num_steps)
    artifacts, _ = resolve_onnx_artifacts(onnx_path, stage2_report_path)

    onnx_policy = OnnxPi05PolicyAdapter(
        policy_cfg,
        artifacts=artifacts,
        onnx_provider=onnx_provider,
    )
    onnx_policy.eval()

    runtime_inputs = onnx_policy._extract_runtime_inputs(runtime_batch_cpu)
    top_image = runtime_inputs["top_image"]
    wrist_image = runtime_inputs["wrist_image"]
    prefix_feed = {
        "image_embs_top": onnx_policy.vision_runner.infer({"image": top_image})["image_embs"],
        "image_embs_wrist": onnx_policy.vision_runner.infer({"image": wrist_image})["image_embs"],
        "image_mask_top": runtime_inputs["image_mask_top"],
        "image_mask_wrist": runtime_inputs["image_mask_wrist"],
        "tokens": runtime_inputs["tokens"],
        "token_attention_mask": runtime_inputs["token_attention_mask"],
    }
    prefix_outputs = onnx_policy.prefix_runner.infer(prefix_feed)
    denoise_feed = {
        "x_t": noise_cpu.clone(),
        "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
        **{name: prefix_outputs[name] for name in onnx_policy.cache_output_names},
    }
    if onnx_policy.denoise_accepts_timestep:
        denoise_feed["timestep"] = timestep_cpu.clone()

    sync_device = "cuda:0" if onnx_provider != "cpu" and torch.cuda.is_available() else None

    def _vision_single() -> torch.Tensor:
        return onnx_policy.vision_runner.infer({"image": top_image})["image_embs"]

    def _vision_pair() -> torch.Tensor:
        top = onnx_policy.vision_runner.infer({"image": top_image})["image_embs"]
        wrist = onnx_policy.vision_runner.infer({"image": wrist_image})["image_embs"]
        return torch.cat([top.flatten(), wrist.flatten()], dim=0)

    def _prefix_stage() -> dict[str, torch.Tensor]:
        return onnx_policy.prefix_runner.infer(prefix_feed)

    def _denoise_stage() -> torch.Tensor:
        return onnx_policy.denoise_runner.infer(denoise_feed)["v_t"]

    def _pipeline() -> torch.Tensor:
        return onnx_policy.predict_action_chunk(
            runtime_batch_cpu,
            noise=noise_cpu.clone(),
            num_inference_steps=num_steps,
        )

    vision_single_stats, _ = _benchmark_callable(
        "vision_encoder_single",
        _vision_single,
        warmup=warmup,
        iterations=iterations,
        sync_device=sync_device,
    )
    vision_pair_stats, _ = _benchmark_callable(
        "vision_encoder_pair",
        _vision_pair,
        warmup=warmup,
        iterations=iterations,
        sync_device=sync_device,
    )
    prefix_stats, _ = _benchmark_callable(
        "prefix_cache",
        _prefix_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=sync_device,
    )
    denoise_stats, _ = _benchmark_callable(
        "denoise_step",
        _denoise_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=sync_device,
    )
    pipeline_stats, pipeline_output = _benchmark_callable(
        "pipeline_chunk",
        _pipeline,
        warmup=warmup,
        iterations=iterations,
        sync_device=sync_device,
    )

    result = {
        "backend": "onnx",
        "execution": onnx_policy.runtime_summary(),
        "latency_ms": {
            "vision_encoder_single": vision_single_stats,
            "vision_encoder_pair": vision_pair_stats,
            "prefix_cache": prefix_stats,
            "denoise_step": denoise_stats,
            "pipeline_chunk": pipeline_stats,
        },
        "derived": {
            "amortized_per_action_step_ms": float(
                pipeline_stats["mean_ms"] / float(policy_cfg.n_action_steps)
            ),
            "denoise_loop_total_estimate_ms": float(denoise_stats["mean_ms"] * float(num_steps)),
        },
        "output_summary": _summarize_tensor(pipeline_output),
    }
    artifact_summary = {
        "onnx_dir": artifacts.onnx_dir.as_posix(),
        "vision_onnx": artifacts.vision_onnx.as_posix(),
        "prefix_onnx": artifacts.prefix_onnx.as_posix(),
        "denoise_onnx": artifacts.denoise_onnx.as_posix(),
        "stage2_report_path": (
            artifacts.stage2_report_path.as_posix()
            if artifacts.stage2_report_path is not None
            else None
        ),
    }
    onnx_policy.close()
    return result, artifact_summary


def _benchmark_trt_backend(
    *,
    policy_dir: Path,
    runtime_batch_cpu: dict[str, Any],
    noise_cpu: torch.Tensor,
    timestep_cpu: torch.Tensor,
    trt_path: str,
    metadata_path: str | None,
    trt_device: str,
    allow_unsafe_trt_artifacts: bool,
    warmup: int,
    iterations: int,
    num_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_cfg = _load_policy_config(policy_dir, device=trt_device)
    policy_cfg.num_inference_steps = int(num_steps)
    artifacts, metadata_checkpoint_dir = resolve_trt_artifacts(trt_path, metadata_path)
    artifact_safety = assess_trt_artifact_safety(
        policy_dir=policy_dir,
        artifacts=artifacts,
        metadata_checkpoint_dir=metadata_checkpoint_dir,
        allow_unsafe_override=allow_unsafe_trt_artifacts,
    )
    if not artifact_safety.is_safe and not allow_unsafe_trt_artifacts:
        reasons = "; ".join(artifact_safety.blocking_reasons)
        raise RuntimeError(
            "Refusing to benchmark incoherent TensorRT artifacts. "
            f"Reasons: {reasons}"
        )

    trt_policy = TrtPi05PolicyAdapter(
        policy_cfg,
        artifacts=artifacts,
        trt_device=trt_device,
    )
    trt_policy.eval()

    runtime_inputs = trt_policy._extract_runtime_inputs(runtime_batch_cpu)
    top_image = runtime_inputs["top_image"]
    wrist_image = runtime_inputs["wrist_image"]
    prefix_feed = {
        "image_embs_top": trt_policy.vision_runner.infer({"image": top_image})["image_embs"],
        "image_embs_wrist": trt_policy.vision_runner.infer({"image": wrist_image})["image_embs"],
        "image_mask_top": runtime_inputs["image_mask_top"],
        "image_mask_wrist": runtime_inputs["image_mask_wrist"],
        "tokens": runtime_inputs["tokens"],
        "token_attention_mask": runtime_inputs["token_attention_mask"],
    }
    prefix_outputs = trt_policy.prefix_runner.infer(prefix_feed)
    denoise_feed = {
        "x_t": noise_cpu.clone().to(device=trt_policy.device),
        "timestep": timestep_cpu.clone().to(device=trt_policy.device),
        "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
        **{name: prefix_outputs[name] for name in trt_policy.cache_output_names},
    }

    def _vision_single() -> torch.Tensor:
        return trt_policy.vision_runner.infer({"image": top_image})["image_embs"]

    def _vision_pair() -> torch.Tensor:
        top = trt_policy.vision_runner.infer({"image": top_image})["image_embs"]
        wrist = trt_policy.vision_runner.infer({"image": wrist_image})["image_embs"]
        return torch.cat([top.flatten(), wrist.flatten()], dim=0)

    def _prefix_stage() -> dict[str, torch.Tensor]:
        return trt_policy.prefix_runner.infer(prefix_feed)

    def _denoise_stage() -> torch.Tensor:
        return trt_policy.denoise_runner.infer(denoise_feed)["v_t"]

    def _pipeline() -> torch.Tensor:
        return trt_policy.predict_action_chunk(
            runtime_batch_cpu,
            noise=noise_cpu.clone().to(device=trt_policy.device),
            num_inference_steps=num_steps,
        )

    vision_single_stats, _ = _benchmark_callable(
        "vision_encoder_single",
        _vision_single,
        warmup=warmup,
        iterations=iterations,
        sync_device=trt_policy.device,
    )
    vision_pair_stats, _ = _benchmark_callable(
        "vision_encoder_pair",
        _vision_pair,
        warmup=warmup,
        iterations=iterations,
        sync_device=trt_policy.device,
    )
    prefix_stats, _ = _benchmark_callable(
        "prefix_cache",
        _prefix_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=trt_policy.device,
    )
    denoise_stats, _ = _benchmark_callable(
        "denoise_step",
        _denoise_stage,
        warmup=warmup,
        iterations=iterations,
        sync_device=trt_policy.device,
    )
    pipeline_stats, pipeline_output = _benchmark_callable(
        "pipeline_chunk",
        _pipeline,
        warmup=warmup,
        iterations=iterations,
        sync_device=trt_policy.device,
    )

    result = {
        "backend": "tensorrt",
        "execution": trt_policy.runtime_summary(),
        "latency_ms": {
            "vision_encoder_single": vision_single_stats,
            "vision_encoder_pair": vision_pair_stats,
            "prefix_cache": prefix_stats,
            "denoise_step": denoise_stats,
            "pipeline_chunk": pipeline_stats,
        },
        "derived": {
            "amortized_per_action_step_ms": float(
                pipeline_stats["mean_ms"] / float(policy_cfg.n_action_steps)
            ),
            "denoise_loop_total_estimate_ms": float(denoise_stats["mean_ms"] * float(num_steps)),
        },
        "output_summary": _summarize_tensor(pipeline_output),
    }
    artifact_summary = {
        "engine_dir": artifacts.engine_dir.as_posix(),
        "vision_engine": artifacts.vision_engine.as_posix(),
        "prefix_engine": artifacts.prefix_engine.as_posix(),
        "denoise_engine": artifacts.denoise_engine.as_posix(),
        "checkpoint_dir": metadata_checkpoint_dir.as_posix() if metadata_checkpoint_dir is not None else None,
        "variant": artifact_safety.resolved_variant,
        "requested_precision": artifact_safety.resolved_requested_precision,
        "metadata_path": (
            artifacts.metadata_path.as_posix()
            if artifacts.metadata_path is not None
            else None
        ),
        "stage4_report_path": (
            artifact_safety.stage4_report_path.as_posix()
            if artifact_safety.stage4_report_path is not None
            else None
        ),
        "stage4_report_status": artifact_safety.stage4_report_status,
        "stage5_report_path": (
            artifact_safety.stage5_report_path.as_posix()
            if artifact_safety.stage5_report_path is not None
            else None
        ),
        "stage5_report_status": artifact_safety.stage5_report_status,
        "metadata_stage_status": artifact_safety.metadata_stage_status,
        "allow_unsafe_trt_artifacts": bool(allow_unsafe_trt_artifacts),
        "blocking_reasons": list(artifact_safety.blocking_reasons),
        "notes": list(artifact_safety.notes),
        "effective_precision_evidence": artifact_safety.effective_precision_evidence,
    }
    trt_policy.close()
    return result, artifact_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PI0.5 PyTorch / ONNX Runtime / TensorRT inference latency on the same deterministic batch."
    )
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--onnx-path", default=str(DEFAULT_ARTIFACTS_PATH))
    parser.add_argument("--onnx-stage2-report-path", default=None)
    parser.add_argument("--onnx-provider", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--trt-path", default=str(DEFAULT_ARTIFACTS_PATH))
    parser.add_argument("--trt-metadata-path", default=None)
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--trt-device", default="cuda:0")
    parser.add_argument("--allow-unsafe-trt-artifacts", action="store_true")
    parser.add_argument("--torch-use-amp", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--policy-num-inference-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.policy_num_inference_steps is not None and args.policy_num_inference_steps <= 0:
        raise ValueError("--policy-num-inference-steps must be positive")

    compatibility = ensure_pi_runtime_compatibility(require_local_tokenizer=True)
    if not compatibility["ready"]:
        raise RuntimeError(
            "PI runtime compatibility check failed: "
            + "; ".join(compatibility["errors"])
        )

    policy_dir = resolve_checkpoint_dir(args.policy_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            PI_TRT_ROOT
            / "docs"
            / "results"
            / f"pi_inference_benchmark_{_timestamp_for_path()}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_runtime_context(policy_dir)
    runtime_batch_cpu = _build_runtime_batch_cpu(context.processed_batch)
    noise_cpu = context.x_t.detach().cpu().clone().contiguous()
    timestep_cpu = context.timestep.detach().cpu().clone().contiguous()

    num_steps = (
        int(args.policy_num_inference_steps)
        if args.policy_num_inference_steps is not None
        else int(context.policy.config.num_inference_steps)
    )

    pytorch_result = _benchmark_torch_backend(
        context=context,
        torch_device=args.torch_device,
        use_amp=bool(args.torch_use_amp),
        warmup=int(args.warmup_iterations),
        iterations=int(args.iterations),
        num_steps=num_steps,
    )

    del context
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    onnx_result, onnx_artifacts = _benchmark_onnx_backend(
        policy_dir=policy_dir,
        runtime_batch_cpu=runtime_batch_cpu,
        noise_cpu=noise_cpu,
        timestep_cpu=timestep_cpu,
        onnx_path=args.onnx_path,
        stage2_report_path=args.onnx_stage2_report_path,
        onnx_provider=args.onnx_provider,
        warmup=int(args.warmup_iterations),
        iterations=int(args.iterations),
        num_steps=num_steps,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trt_result, trt_artifacts = _benchmark_trt_backend(
        policy_dir=policy_dir,
        runtime_batch_cpu=runtime_batch_cpu,
        noise_cpu=noise_cpu,
        timestep_cpu=timestep_cpu,
        trt_path=args.trt_path,
        metadata_path=args.trt_metadata_path,
        trt_device=args.trt_device,
        allow_unsafe_trt_artifacts=bool(args.allow_unsafe_trt_artifacts),
        warmup=int(args.warmup_iterations),
        iterations=int(args.iterations),
        num_steps=num_steps,
    )

    report = {
        "measured_at_utc": _utc_now(),
        "policy_path": policy_dir.as_posix(),
        "compatibility": compatibility,
        "environment": {
            "versions": _probe_versions(),
            "gpu": _probe_gpu(),
        },
        "settings": {
            "warmup_iterations": int(args.warmup_iterations),
            "measured_iterations": int(args.iterations),
            "num_inference_steps": int(num_steps),
            "n_action_steps": int(pytorch_result["output_summary"]["shape"][1] if pytorch_result["output_summary"] else 0),
            "chunk_size": int(pytorch_result["output_summary"]["shape"][1] if pytorch_result["output_summary"] else 0),
            "torch_device": args.torch_device,
            "torch_use_amp": bool(args.torch_use_amp),
            "torch_amp_mode": "cuda_bfloat16_autocast" if args.torch_use_amp else None,
            "onnx_provider": args.onnx_provider,
            "trt_device": args.trt_device,
            "allow_unsafe_trt_artifacts": bool(args.allow_unsafe_trt_artifacts),
        },
        "onnx_artifacts": onnx_artifacts,
        "trt_artifacts": trt_artifacts,
        "results": {
            "pytorch": pytorch_result,
            "onnx": onnx_result,
            "tensorrt": trt_result,
        },
    }

    n_action_steps = report["results"]["pytorch"]["output_summary"]["shape"][1]
    chunk_size = report["results"]["pytorch"]["output_summary"]["shape"][1]
    report["settings"]["n_action_steps"] = int(
        PreTrainedConfig.from_pretrained(str(policy_dir)).n_action_steps
    )
    report["settings"]["chunk_size"] = int(
        PreTrainedConfig.from_pretrained(str(policy_dir)).chunk_size
    )
    report["settings"]["predicted_action_chunk_shape"] = report["results"]["pytorch"]["output_summary"]["shape"]
    report["settings"]["raw_output_second_dim"] = int(n_action_steps)
    report["settings"]["raw_output_chunk_dim"] = int(chunk_size)

    json_path = output_dir / "benchmark_report.json"
    md_path = output_dir / "benchmark_report.md"
    _write_json(json_path, report)
    md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
