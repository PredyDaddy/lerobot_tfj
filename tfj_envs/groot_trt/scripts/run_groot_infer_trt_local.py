#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    DEFAULT_DEVICE,
    ensure_dir,
    load_policy,
    load_pre_post_processors,
    resolve_policy_dir,
    validate_engine_dir,
    write_json,
)
from groot_trt_adapter_local import TrtGrootPolicyAdapter
from lerobot.policies.utils import prepare_observation_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GROOT TensorRT inference without my_devs.")
    parser.add_argument("--policy-path", required=True, help="Checkpoint path or pretrained_model directory.")
    parser.add_argument("--engine-dir", required=True, help="Directory containing the 7 TensorRT engine files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for actions and reports.")
    parser.add_argument("--source", choices=["random"], default="random", help="Observation source type.")
    parser.add_argument("--num-steps", type=int, default=4, help="Number of select_action() calls to run.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for observations and denoising noise.")
    parser.add_argument("--task", default="Perform the task.", help="Language instruction passed into the processor.")
    parser.add_argument("--robot-type", default="", help="Optional robot_type string for inference preprocessing.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA device for policy glue and TensorRT runtime.")
    parser.add_argument(
        "--refresh-observation-per-step",
        action="store_true",
        help="Generate a fresh random observation on every control step. By default, one observation is reused.",
    )
    return parser.parse_args()


def feature_type_name(feature: Any) -> str | None:
    feature_type = getattr(feature, "type", None)
    if feature_type is None and isinstance(feature, dict):
        feature_type = feature.get("type")
    return getattr(feature_type, "name", feature_type)


def build_random_observation(config: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key, feature in config.input_features.items():
        shape = tuple(int(dim) for dim in feature.shape)
        feature_type = feature_type_name(feature)
        if key.startswith("observation.images.") or feature_type == "VISUAL":
            if len(shape) != 3:
                raise ValueError(f"Visual feature `{key}` must be CHW, got {shape}")
            channels, height, width = shape
            if channels != 3:
                raise ValueError(f"Visual feature `{key}` must have 3 channels, got {shape}")
            observation[key] = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
        elif key == "observation.state" or feature_type == "STATE":
            observation[key] = rng.standard_normal(size=shape).astype(np.float32)
        else:
            raise NotImplementedError(
                f"Random observation generation is not implemented for feature `{key}` ({feature_type}, {shape})."
            )
    return observation


def summarize_tensor_tree(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")

    policy_dir = resolve_policy_dir(args.policy_path)
    engine_dir = Path(args.engine_dir).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    validate_engine_dir(engine_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config, _, base_policy = load_policy(policy_dir, device=args.device, strict=False)
    preprocessor, postprocessor = load_pre_post_processors(policy_dir)
    adapter = TrtGrootPolicyAdapter(
        config,
        base_policy=base_policy,
        engine_dir=engine_dir,
        trt_device=args.device,
    )
    adapter.reset()

    rng = np.random.default_rng(args.seed)
    cached_observation = build_random_observation(config, rng)

    raw_actions: list[np.ndarray] = []
    post_actions: list[np.ndarray] = []
    step_reports: list[dict[str, Any]] = []
    batch_summary: dict[str, Any] | None = None

    for step_idx in range(args.num_steps):
        if args.refresh_observation_per_step and step_idx > 0:
            raw_observation = build_random_observation(config, rng)
        else:
            raw_observation = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in cached_observation.items()
            }

        observation = prepare_observation_for_inference(
            raw_observation,
            device=torch.device(args.device),
            task=args.task,
            robot_type=args.robot_type,
        )
        policy_batch = preprocessor(observation)
        if batch_summary is None:
            batch_summary = summarize_tensor_tree(policy_batch)

        queue_was_empty = len(adapter._action_queue) == 0
        action = adapter.select_action(policy_batch)
        action_post = postprocessor(action)

        raw_actions.append(action.detach().cpu().numpy())
        post_actions.append(action_post.detach().cpu().numpy())
        step_reports.append(
            {
                "step": step_idx,
                "generated_new_chunk": queue_was_empty,
                "queue_size_after_step": len(adapter._action_queue),
            }
        )

    np.save(out_dir / "actions_raw.npy", np.stack(raw_actions, axis=0))
    np.save(out_dir / "actions_postprocessed.npy", np.stack(post_actions, axis=0))

    report = {
        "policy_dir": policy_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "out_dir": out_dir.as_posix(),
        "source": args.source,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "task": args.task,
        "robot_type": args.robot_type,
        "refresh_observation_per_step": bool(args.refresh_observation_per_step),
        "policy_device": args.device,
        "engine_descriptions": adapter.describe_engines(),
        "policy_batch_summary": batch_summary,
        "steps": step_reports,
        "actions_raw_path": (out_dir / "actions_raw.npy").as_posix(),
        "actions_postprocessed_path": (out_dir / "actions_postprocessed.npy").as_posix(),
    }
    write_json(out_dir / "run_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
