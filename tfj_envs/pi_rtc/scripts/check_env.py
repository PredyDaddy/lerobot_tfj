#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    build_metadata_skeleton,
    build_preflight_summary,
    collect_env_probe,
    metadata_path,
    prepare_run_layout,
    resolve_checkpoint_dir,
    stage_json_path,
    validate_variant,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-0 PI TensorRT environment and checkpoint preflight for the lerobot env."
    )
    parser.add_argument(
        "--variant",
        default="pi05",
        help="Policy variant. Phase-1 default is pi05.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint root, pretrained_model dir, or training run dir.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing or new run directory. Defaults to tfj_envs/pi_trt/runs/pi_trt_<timestamp>.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit JSON output path. Defaults to <run_dir>/stage0_env_check.json.",
    )
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Only probe the environment when no checkpoint is available yet.",
    )
    parser.add_argument(
        "--local-tokenizer-path",
        default=None,
        help="Optional explicit offline tokenizer directory override.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when preflight is not ready.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the final JSON payload to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        variant = validate_variant(args.variant, phase1_only=False)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    layout = prepare_run_layout(args.run_dir, prefix=f"pi_trt_{variant}")
    run_dir = layout["run_dir"]

    checkpoint_dir = None
    checkpoint_error = None
    if args.checkpoint_path:
        try:
            checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)
        except FileNotFoundError as exc:
            checkpoint_error = str(exc)

    env_probe = collect_env_probe(local_tokenizer_path=args.local_tokenizer_path)
    payload = {
        "schema_version": 1,
        "stage": "stage0_env_check",
        "phase": "phase-1",
        "generated_at": env_probe["probed_at"],
        "run_dir": str(run_dir),
        "output_path": None,
        "metadata_path": str(metadata_path(run_dir)),
        "preflight": build_preflight_summary(
            variant=variant,
            checkpoint_dir=checkpoint_dir,
            env_probe=env_probe,
            require_checkpoint=not args.allow_missing_checkpoint,
            require_local_tokenizer=True,
        ),
    }

    if checkpoint_error is not None:
        payload["preflight"]["ready"] = False
        payload["preflight"]["errors"].append(checkpoint_error)
        payload["preflight"]["checkpoint"] = {
            "checkpoint_dir": None,
            "assets": {},
            "asset_status": {},
            "missing_assets": [],
            "all_required_present": False,
        }

    output_path = Path(args.output).expanduser().resolve() if args.output else stage_json_path(run_dir, "stage0_env_check")
    payload["output_path"] = str(output_path)

    metadata = build_metadata_skeleton(
        run_dir=run_dir,
        variant=variant,
        checkpoint_dir=checkpoint_dir,
    )
    metadata["last_completed_stage"] = "stage0_env_check"
    metadata["environment"] = {
        "generated_at": env_probe["probed_at"],
        "modules": env_probe["modules"],
        "cuda": env_probe["cuda"],
    }
    metadata["checkpoint"] = payload["preflight"]["checkpoint"]
    metadata["stage_status"] = {
        "stage0_env_check": "pass" if payload["preflight"]["ready"] else "fail",
    }

    write_json(output_path, payload)
    write_json(metadata_path(run_dir), metadata)

    if args.print_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if args.strict and not payload["preflight"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
