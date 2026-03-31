#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
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
from onnx_pi_adapter import OnnxPi05PolicyAdapter  # noqa: E402
from pi_compare_common import build_runtime_context  # noqa: E402
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


def _torch_amp_context(device: torch.device, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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


def _load_policy_config(policy_dir: Path, *, device: str) -> Any:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = str(policy_dir)
    policy_cfg.device = device
    if getattr(policy_cfg, "type", None) != "pi05":
        raise ValueError(f"Expected PI05 policy, got {getattr(policy_cfg, 'type', None)!r}")
    return policy_cfg


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


def _reset_policy(policy: Any) -> None:
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _action_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    return None


def _time_select_action_loop(
    *,
    step_fn: Callable[[], Any],
    reset_fn: Callable[[], None],
    sync_device: str | torch.device | None,
    warmup_steps: int,
    steps: int,
) -> tuple[float, Any]:
    reset_fn()
    for _ in range(warmup_steps):
        step_fn()
    _sync_cuda(sync_device)

    reset_fn()
    _sync_cuda(sync_device)
    start_t = time.perf_counter()
    last_action: Any = None
    for _ in range(steps):
        last_action = step_fn()
    _sync_cuda(sync_device)
    total_time_s = time.perf_counter() - start_t
    return total_time_s, last_action


def _summarize_backend_result(
    *,
    backend: str,
    total_time_s: float,
    last_action: Any,
    steps: int,
    warmup_steps: int,
    n_action_steps: int,
    num_inference_steps: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "backend": backend,
        "total_steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "n_action_steps": int(n_action_steps),
        "expected_chunk_refreshes": int(math.ceil(float(steps) / float(n_action_steps))),
        "num_inference_steps": int(num_inference_steps),
        "total_time_s": float(total_time_s),
        "total_time_ms": float(total_time_s * 1000.0),
        "mean_per_step_ms": float(total_time_s * 1000.0 / float(steps)),
        "steps_per_s": float(float(steps) / float(total_time_s)),
        "last_action_shape": _action_shape(last_action),
    }
    if extra:
        payload.update(extra)
    return payload


def _measure_torch_backend(
    *,
    context: Any,
    torch_device: str,
    use_amp: bool,
    steps: int,
    warmup_steps: int,
    num_inference_steps: int,
) -> dict[str, Any]:
    device = torch.device(torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA torch benchmark, but CUDA is not available.")

    context.policy.to(device)
    context.policy.eval()
    context.policy.config.num_inference_steps = int(num_inference_steps)
    batch = _build_policy_batch(context.processed_batch, device)

    def _step() -> Any:
        with torch.inference_mode(), _torch_amp_context(device, use_amp):
            return context.policy.select_action(batch)

    total_time_s, last_action = _time_select_action_loop(
        step_fn=_step,
        reset_fn=lambda: _reset_policy(context.policy),
        sync_device=device,
        warmup_steps=warmup_steps,
        steps=steps,
    )
    backend_name = "pytorch_amp_bf16" if use_amp else "pytorch_fp32"
    extra = {
        "torch_use_amp": bool(use_amp),
        "torch_amp_mode": "cuda_bfloat16_autocast" if use_amp else None,
        "device": str(device),
    }
    return _summarize_backend_result(
        backend=backend_name,
        total_time_s=total_time_s,
        last_action=last_action,
        steps=steps,
        warmup_steps=warmup_steps,
        n_action_steps=int(context.policy.config.n_action_steps),
        num_inference_steps=num_inference_steps,
        extra=extra,
    )


def _measure_onnx_backend(
    *,
    policy_dir: Path,
    runtime_batch_cpu: dict[str, Any],
    onnx_path: str,
    stage2_report_path: str | None,
    onnx_provider: str,
    steps: int,
    warmup_steps: int,
    num_inference_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_cfg = _load_policy_config(policy_dir, device="cpu")
    policy_cfg.num_inference_steps = int(num_inference_steps)
    artifacts, stage2_policy_dir = resolve_onnx_artifacts(onnx_path, stage2_report_path)
    if stage2_policy_dir is not None and stage2_policy_dir != policy_dir:
        raise RuntimeError(
            "ONNX artifacts do not match the requested checkpoint. "
            f"policy={policy_dir}, stage2_report={stage2_policy_dir}"
        )
    onnx_policy = OnnxPi05PolicyAdapter(
        policy_cfg,
        artifacts=artifacts,
        onnx_provider=onnx_provider,
        num_inference_steps=num_inference_steps,
    )
    onnx_policy.eval()

    def _step() -> Any:
        return onnx_policy.select_action(runtime_batch_cpu)

    total_time_s, last_action = _time_select_action_loop(
        step_fn=_step,
        reset_fn=lambda: _reset_policy(onnx_policy),
        sync_device=None,
        warmup_steps=warmup_steps,
        steps=steps,
    )

    backend_name = "onnx_cuda_runtime" if onnx_provider != "cpu" else "onnx_cpu_runtime"
    result = _summarize_backend_result(
        backend=backend_name,
        total_time_s=total_time_s,
        last_action=last_action,
        steps=steps,
        warmup_steps=warmup_steps,
        n_action_steps=int(policy_cfg.n_action_steps),
        num_inference_steps=num_inference_steps,
        extra={
            "onnx_provider": onnx_provider,
            "runtime_summary": onnx_policy.runtime_summary(),
        },
    )
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


def _measure_trt_backend(
    *,
    policy_dir: Path,
    runtime_batch_cpu: dict[str, Any],
    trt_path: str,
    metadata_path: str | None,
    trt_device: str,
    allow_unsafe_trt_artifacts: bool,
    steps: int,
    warmup_steps: int,
    num_inference_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_cfg = _load_policy_config(policy_dir, device=trt_device)
    policy_cfg.num_inference_steps = int(num_inference_steps)
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
        num_inference_steps=num_inference_steps,
    )
    trt_policy.eval()

    def _step() -> Any:
        return trt_policy.select_action(runtime_batch_cpu)

    total_time_s, last_action = _time_select_action_loop(
        step_fn=_step,
        reset_fn=lambda: _reset_policy(trt_policy),
        sync_device=trt_policy.device,
        warmup_steps=warmup_steps,
        steps=steps,
    )

    resolved_precision = artifact_safety.resolved_requested_precision or "unknown"
    result = _summarize_backend_result(
        backend=f"tensorrt_{resolved_precision}",
        total_time_s=total_time_s,
        last_action=last_action,
        steps=steps,
        warmup_steps=warmup_steps,
        n_action_steps=int(policy_cfg.n_action_steps),
        num_inference_steps=num_inference_steps,
        extra={
            "trt_device": trt_device,
            "runtime_summary": trt_policy.runtime_summary(),
        },
    )
    artifact_summary = {
        "engine_dir": artifacts.engine_dir.as_posix(),
        "vision_engine": artifacts.vision_engine.as_posix(),
        "prefix_engine": artifacts.prefix_engine.as_posix(),
        "denoise_engine": artifacts.denoise_engine.as_posix(),
        "metadata_path": (
            artifacts.metadata_path.as_posix()
            if artifacts.metadata_path is not None
            else None
        ),
        "checkpoint_dir": metadata_checkpoint_dir.as_posix() if metadata_checkpoint_dir is not None else None,
        "variant": artifact_safety.resolved_variant,
        "requested_precision": artifact_safety.resolved_requested_precision,
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


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# PI0.5 1000-step Pure Inference Compare")
    lines.append("")
    lines.append(f"- measured_at_utc: `{report['measured_at_utc']}`")
    lines.append(f"- policy_path: `{report['policy_path']}`")
    lines.append(f"- steps: `{report['steps']}`")
    lines.append(f"- warmup_steps: `{report['warmup_steps']}`")
    lines.append(f"- n_action_steps: `{report['n_action_steps']}`")
    lines.append(f"- expected_chunk_refreshes: `{report['expected_chunk_refreshes']}`")
    lines.append(f"- num_inference_steps: `{report['num_inference_steps']}`")
    lines.append("")
    lines.append("## 1. TRT Provenance")
    lines.append("")
    lines.append(f"- variant: `{report['trt_artifacts'].get('variant')}`")
    lines.append(f"- requested_precision: `{report['trt_artifacts'].get('requested_precision')}`")
    lines.append(f"- metadata_path: `{report['trt_artifacts'].get('metadata_path')}`")
    lines.append(f"- stage4_report_path: `{report['trt_artifacts'].get('stage4_report_path')}`")
    lines.append(f"- stage5_report_path: `{report['trt_artifacts'].get('stage5_report_path')}`")
    lines.append(f"- allow_unsafe_trt_artifacts: `{report['trt_artifacts'].get('allow_unsafe_trt_artifacts')}`")
    lines.append("")
    lines.append("## 2. Results")
    lines.append("")
    lines.append("| Backend | total_time_ms | mean_per_step_ms | steps_per_s |")
    lines.append("| --- | ---: | ---: | ---: |")
    for backend_name, payload in report["results"].items():
        lines.append(
            f"| {backend_name} | {payload['total_time_ms']:.3f} | "
            f"{payload['mean_per_step_ms']:.3f} | {payload['steps_per_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 3. Notes")
    lines.append("")
    lines.append("- 这是纯 `select_action()` 推理 benchmark，不接机器人、不读串口、不下发动作。")
    lines.append("- 计时包含 chunk queue 的刷新与复用，因此反映的是均摊后的纯推理吞吐，而不是单次 chunk 刷新时延。")
    lines.append("- `PyTorch AMP` 在本报告中明确表示 `CUDA BF16 autocast`，不是 `Torch FP16`。")
    lines.append("- TensorRT 结果只对当前已验证通过的 static-shape、batch=1、固定 token length 工件成立。")
    if report["trt_artifacts"].get("allow_unsafe_trt_artifacts"):
        lines.append("- 本次 TensorRT pure benchmark 显式允许了 `unsafe` 工件，因此这些数字只能用于诊断，不可直接当作已通过正确性 gate 的部署结论。")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PI0.5 pure select_action() inference without any robot execution."
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--policy-num-inference-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.policy_num_inference_steps is not None and args.policy_num_inference_steps <= 0:
        raise ValueError("--policy-num-inference-steps must be positive")

    compatibility = ensure_pi_runtime_compatibility(require_local_tokenizer=True)
    if not compatibility["ready"]:
        raise RuntimeError(
            "PI runtime compatibility check failed: " + "; ".join(compatibility["errors"])
        )

    policy_dir = resolve_checkpoint_dir(args.policy_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            PI_TRT_ROOT
            / "docs"
            / "results"
            / f"pi_select_action_1000steps_{_timestamp_for_path()}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_runtime_context(policy_dir)
    runtime_batch_cpu = _build_runtime_batch_cpu(context.processed_batch)
    num_inference_steps = (
        int(args.policy_num_inference_steps)
        if args.policy_num_inference_steps is not None
        else int(context.policy.config.num_inference_steps)
    )
    n_action_steps = int(context.policy.config.n_action_steps)
    expected_chunk_refreshes = int(math.ceil(float(args.steps) / float(n_action_steps)))

    pytorch_fp32 = _measure_torch_backend(
        context=context,
        torch_device=args.torch_device,
        use_amp=False,
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        num_inference_steps=num_inference_steps,
    )
    pytorch_amp_bf16 = _measure_torch_backend(
        context=context,
        torch_device=args.torch_device,
        use_amp=True,
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        num_inference_steps=num_inference_steps,
    )

    del context
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    onnx_result, onnx_artifacts = _measure_onnx_backend(
        policy_dir=policy_dir,
        runtime_batch_cpu=runtime_batch_cpu,
        onnx_path=args.onnx_path,
        stage2_report_path=args.onnx_stage2_report_path,
        onnx_provider=args.onnx_provider,
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        num_inference_steps=num_inference_steps,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trt_result, trt_artifacts = _measure_trt_backend(
        policy_dir=policy_dir,
        runtime_batch_cpu=runtime_batch_cpu,
        trt_path=args.trt_path,
        metadata_path=args.trt_metadata_path,
        trt_device=args.trt_device,
        allow_unsafe_trt_artifacts=bool(args.allow_unsafe_trt_artifacts),
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        num_inference_steps=num_inference_steps,
    )

    report = {
        "measured_at_utc": _utc_now(),
        "mode": "pure_inference_select_action_only",
        "policy_path": policy_dir.as_posix(),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "n_action_steps": n_action_steps,
        "expected_chunk_refreshes": expected_chunk_refreshes,
        "num_inference_steps": int(num_inference_steps),
        "compatibility": compatibility,
        "environment": {
            "versions": _probe_versions(),
            "gpu": _probe_gpu(),
        },
        "onnx_artifacts": onnx_artifacts,
        "trt_artifacts": trt_artifacts,
        "results": {
            "pytorch_fp32": pytorch_fp32,
            "pytorch_amp_bf16": pytorch_amp_bf16,
            onnx_result["backend"]: onnx_result,
            trt_result["backend"]: trt_result,
        },
    }

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    _write_json(json_path, report)
    md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
