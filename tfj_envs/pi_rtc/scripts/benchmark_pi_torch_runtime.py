#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common import resolve_checkpoint_dir  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tfj_envs" / "pi_rtc" / "docs" / "results"
DEFAULT_TASKS = {
    "pi05": "Pick up the red block and place it into the tray",
    "pi06": "Put the block in the bin",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sync_cuda(device: str | torch.device | None) -> None:
    if device is None or not torch.cuda.is_available():
        return
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    with torch.cuda.device(resolved):
        torch.cuda.synchronize()


def _amp_context(device: torch.device, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


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


def _summarize_times(times_s: list[float]) -> dict[str, Any]:
    values_ms = [value * 1000.0 for value in times_s]
    sorted_ms = sorted(values_ms)
    mean_ms = float(statistics.fmean(values_ms))
    return {
        "iterations": int(len(values_ms)),
        "mean_ms": mean_ms,
        "std_ms": float(statistics.pstdev(values_ms)) if len(values_ms) > 1 else 0.0,
        "min_ms": float(min(values_ms)),
        "p50_ms": float(_percentile(sorted_ms, 0.50)),
        "p95_ms": float(_percentile(sorted_ms, 0.95)),
        "max_ms": float(max(values_ms)),
        "steps_per_s": float(1000.0 / mean_ms) if mean_ms > 0.0 else float("inf"),
    }


def _reset_policy(policy: Any) -> None:
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _tensor_numel(shape: tuple[int, ...]) -> int:
    return math.prod(int(dim) for dim in shape)


def _build_image(shape: tuple[int, ...], *, start: float, end: float) -> torch.Tensor:
    numel = _tensor_numel(shape)
    return torch.linspace(start, end, steps=numel, dtype=torch.float32).view(shape).remainder(1.0)


def _build_raw_batch(policy_cfg: Any, task: str) -> dict[str, Any]:
    input_features = dict(policy_cfg.input_features or {})
    output_features = dict(policy_cfg.output_features or {})
    image_keys = sorted(
        key
        for key in input_features
        if key.startswith("observation.images.") and "empty_camera_" not in key
    )
    if not image_keys:
        raise ValueError("No image features found in policy input_features.")

    raw_batch: dict[str, Any] = {"task": task}

    if "observation.state" in input_features:
        state_dim = int(input_features["observation.state"].shape[0])
        raw_batch["observation.state"] = torch.linspace(-0.75, 0.75, steps=state_dim, dtype=torch.float32)

    for index, key in enumerate(image_keys):
        shape = tuple(int(dim) for dim in input_features[key].shape)
        raw_batch[key] = _build_image(shape, start=0.1 * index, end=1.0 + 0.1 * index)

    if "action" in output_features:
        action_dim = int(output_features["action"].shape[0])
        chunk_size = int(policy_cfg.chunk_size)
        raw_batch["action"] = torch.zeros(chunk_size, action_dim, dtype=torch.float32)

    return raw_batch


def _build_runtime_batch(processed_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    runtime_batch: dict[str, Any] = {}
    for key, value in processed_batch.items():
        if isinstance(value, torch.Tensor):
            runtime_batch[key] = value.detach().to(device=device).contiguous()
        else:
            runtime_batch[key] = value
    return runtime_batch


def _summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
            }
    return summary


def _benchmark_loop(
    fn: Callable[[], Any],
    *,
    warmup_iterations: int,
    measured_iterations: int,
    sync_device: str | torch.device | None,
) -> tuple[dict[str, Any], Any]:
    last_output: Any = None
    for _ in range(warmup_iterations):
        _sync_cuda(sync_device)
        last_output = fn()
        _sync_cuda(sync_device)

    timings_s: list[float] = []
    for _ in range(measured_iterations):
        _sync_cuda(sync_device)
        start_t = time.perf_counter()
        last_output = fn()
        _sync_cuda(sync_device)
        timings_s.append(time.perf_counter() - start_t)

    return _summarize_times(timings_s), last_output


def _load_runtime(
    *,
    policy_dir: Path,
    device: torch.device,
    task: str | None,
    num_inference_steps: int | None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = str(policy_dir)
    policy_cfg.device = str(device)
    if hasattr(policy_cfg, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = False
    rtc_cfg = getattr(policy_cfg, "rtc_config", None)
    if rtc_cfg is not None:
        rtc_cfg.enabled = False
    if num_inference_steps is not None:
        policy_cfg.num_inference_steps = int(num_inference_steps)

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(
        str(policy_dir),
        config=policy_cfg,
        local_files_only=True,
        strict=False,
    )
    policy.eval().to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    resolved_task = task or DEFAULT_TASKS.get(policy_cfg.type, "Pick up the block and place it in the bin")
    raw_batch = _build_raw_batch(policy_cfg, resolved_task)
    processed_batch = preprocessor(raw_batch)
    runtime_batch = _build_runtime_batch(processed_batch, device)
    return policy_cfg, policy, raw_batch, processed_batch, runtime_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PI torch runtime with deterministic input batches."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=None)
    parser.add_argument("--policy-num-inference-steps", type=int, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--select-steps", type=int, default=80)
    parser.add_argument("--select-warmup-steps", type=int, default=8)
    parser.add_argument("--chunk-iterations", type=int, default=20)
    parser.add_argument("--chunk-warmup-iterations", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.select_steps <= 0:
        raise ValueError("--select-steps must be > 0")
    if args.select_warmup_steps < 0:
        raise ValueError("--select-warmup-steps must be >= 0")
    if args.chunk_iterations <= 0:
        raise ValueError("--chunk-iterations must be > 0")
    if args.chunk_warmup_iterations < 0:
        raise ValueError("--chunk-warmup-iterations must be >= 0")
    if args.policy_num_inference_steps is not None and args.policy_num_inference_steps <= 0:
        raise ValueError("--policy-num-inference-steps must be > 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")

    policy_dir = resolve_checkpoint_dir(args.policy_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / f"pi_torch_benchmark_{policy_dir.name}_{_timestamp_slug()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg, policy, raw_batch, processed_batch, runtime_batch = _load_runtime(
        policy_dir=policy_dir,
        device=device,
        task=args.task,
        num_inference_steps=args.policy_num_inference_steps,
    )

    def _select_step() -> Any:
        with torch.inference_mode(), _amp_context(device, bool(args.use_amp)):
            return policy.select_action(runtime_batch)

    def _chunk_step() -> Any:
        with torch.inference_mode(), _amp_context(device, bool(args.use_amp)):
            return policy.predict_action_chunk(runtime_batch)

    _reset_policy(policy)
    select_summary, _ = _benchmark_loop(
        _select_step,
        warmup_iterations=int(args.select_warmup_steps),
        measured_iterations=int(args.select_steps),
        sync_device=device,
    )

    _reset_policy(policy)
    chunk_summary, _ = _benchmark_loop(
        _chunk_step,
        warmup_iterations=int(args.chunk_warmup_iterations),
        measured_iterations=int(args.chunk_iterations),
        sync_device=device,
    )

    n_action_steps = int(policy.config.n_action_steps)
    chunk_mean_s = float(chunk_summary["mean_ms"]) / 1000.0
    chunk_summary["amortized_step_ms"] = float(chunk_summary["mean_ms"]) / float(n_action_steps)
    chunk_summary["implied_max_control_fps"] = (
        float(n_action_steps) / chunk_mean_s if chunk_mean_s > 0.0 else float("inf")
    )

    report = {
        "measured_at_utc": _utc_now(),
        "policy_path": str(policy_dir),
        "policy_type": getattr(policy_cfg, "type", "<unknown>"),
        "device": str(device),
        "torch_use_amp": bool(args.use_amp),
        "policy_num_inference_steps": int(policy.config.num_inference_steps),
        "chunk_size": int(policy.config.chunk_size),
        "n_action_steps": n_action_steps,
        "raw_batch_keys": sorted(raw_batch.keys()),
        "processed_batch": _summarize_batch(processed_batch),
        "results": {
            "select_action": select_summary,
            "predict_action_chunk": chunk_summary,
        },
    }

    report_path = output_dir / "report.json"
    _write_json(report_path, report)

    print(f"[INFO] Policy path: {policy_dir}")
    print(f"[INFO] Policy type: {report['policy_type']}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] use_amp: {bool(args.use_amp)}")
    print(
        "[INFO] Runtime config: "
        f"chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"num_inference_steps={policy.config.num_inference_steps}"
    )
    print(
        "[RESULT] select_action: "
        f"mean={select_summary['mean_ms']:.2f} ms, "
        f"p95={select_summary['p95_ms']:.2f} ms, "
        f"steps_per_s={select_summary['steps_per_s']:.2f}"
    )
    print(
        "[RESULT] predict_action_chunk: "
        f"mean={chunk_summary['mean_ms']:.2f} ms, "
        f"p95={chunk_summary['p95_ms']:.2f} ms, "
        f"amortized_step_ms={chunk_summary['amortized_step_ms']:.2f}, "
        f"implied_max_control_fps={chunk_summary['implied_max_control_fps']:.2f}"
    )
    print(f"[INFO] Report written to: {report_path}")

    del runtime_batch
    del processed_batch
    del raw_batch
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
