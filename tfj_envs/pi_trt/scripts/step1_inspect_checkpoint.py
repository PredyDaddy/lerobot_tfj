#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import re
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import (
    ensure_pi_runtime_compatibility,
    load_policy_preprocessor_from_checkpoint,
    resolve_checkpoint_dir,
    validate_variant,
)


REQUIRED_ASSETS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
OPTIONAL_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "parse_ok": False, "data": None, "error": "file is missing"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "exists": True,
            "parse_ok": False,
            "data": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "exists": True,
        "parse_ok": True,
        "data": data,
        "error": None,
    }


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _recursive_find_first(value: Any, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _recursive_find_first(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _recursive_find_first(item, key)
            if found is not None:
                return found
    return None


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _asset_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "exists": exists,
        "path": str(path),
    }
    if exists and path.is_file():
        record["size_bytes"] = path.stat().st_size
    return record


def _collect_assets(checkpoint_dir: Path) -> tuple[dict[str, Any], list[str]]:
    inventory: dict[str, Any] = {}
    missing: list[str] = []
    for name in REQUIRED_ASSETS + OPTIONAL_ASSETS:
        asset_path = checkpoint_dir / name
        inventory[name] = _asset_record(asset_path)
        if name in REQUIRED_ASSETS and not asset_path.exists():
            missing.append(name)
    return inventory, missing


def _infer_variant(config_data: dict[str, Any] | None, requested_variant: str) -> tuple[str, str]:
    if requested_variant != "auto":
        return validate_variant(requested_variant, phase1_only=False), "cli"
    if not config_data:
        return "pi05", "default_without_config"
    flattened = " ".join(token.lower() for token in _iter_strings(config_data))
    if "pi0.5" in flattened or "pi05" in flattened:
        return "pi05", "config_string_match"
    if "pi0" in flattened:
        return "pi0", "config_string_match"
    return "pi05", "default_without_variant_hint"


def _shape_to_hw(shape: Any) -> dict[str, int] | None:
    if not isinstance(shape, (list, tuple)) or len(shape) < 3:
        return None
    values = [_maybe_int(item) for item in shape]
    if any(item is None for item in values[-2:]):
        return None
    return {
        "height": int(values[-2]),
        "width": int(values[-1]),
    }


def _extract_camera_specs(config_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not config_data:
        return []
    input_features = _recursive_find_first(config_data, "input_features")
    if not isinstance(input_features, dict):
        return []

    camera_specs: list[dict[str, Any]] = []
    for name, spec in input_features.items():
        if not isinstance(name, str) or "observation.images." not in name:
            continue
        if not isinstance(spec, dict):
            spec = {}
        shape = spec.get("shape")
        camera_specs.append(
            {
                "name": name,
                "shape": list(shape) if isinstance(shape, (list, tuple)) else None,
                "resolution": _shape_to_hw(shape),
            }
        )
    return camera_specs


def _extract_contract(
    config_data: dict[str, Any] | None,
    variant: str,
    asset_inventory: dict[str, Any],
) -> dict[str, Any]:
    camera_specs = _extract_camera_specs(config_data)
    chunk_size = _maybe_int(_recursive_find_first(config_data, "chunk_size")) if config_data else None
    num_inference_steps = (
        _maybe_int(_recursive_find_first(config_data, "num_inference_steps")) if config_data else None
    )
    tokenizer_max_length = (
        _maybe_int(_recursive_find_first(config_data, "tokenizer_max_length")) if config_data else None
    )
    max_state_dim = _maybe_int(_recursive_find_first(config_data, "max_state_dim")) if config_data else None
    max_action_dim = _maybe_int(_recursive_find_first(config_data, "max_action_dim")) if config_data else None
    image_resolution = _recursive_find_first(config_data, "image_resolution") if config_data else None

    return {
        "phase1_variant": variant,
        "batch_size": 1,
        "chunk_size": chunk_size,
        "num_inference_steps": num_inference_steps,
        "tokenizer_max_length": tokenizer_max_length,
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "camera_keys": [item["name"] for item in camera_specs],
        "camera_count": len(camera_specs),
        "checkpoint_image_shape": camera_specs[0]["resolution"] if camera_specs else None,
        "configured_image_resolution": image_resolution,
        "processor_assets_present": all(asset_inventory[name]["exists"] for name in REQUIRED_ASSETS[2:]),
        "single_camera_minimal_case": {
            "camera_keys": [camera_specs[0]["name"]] if camera_specs else [],
            "note": "Use the first checkpoint camera key and keep batch_size fixed at 1.",
        },
        "multi_camera_nominal_case": {
            "camera_keys": [item["name"] for item in camera_specs],
            "note": (
                "Use all checkpoint camera keys in serialized order. "
                "If only one key exists, multi-camera coverage is checkpoint-limited."
            ),
        },
    }


def _inspect_safetensors(path: Path, sample_limit: int) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "reason": "model.safetensors is missing"}
    try:
        from safetensors import safe_open
    except Exception as exc:
        return {"available": False, "reason": f"safetensors import failed: {type(exc).__name__}: {exc}"}

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            sample_tensors = []
            for key in keys[:sample_limit]:
                tensor = handle.get_tensor(key)
                sample_tensors.append(
                    {
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                    }
                )
        return {
            "available": True,
            "tensor_count": len(keys),
            "sample_tensors": sample_tensors,
        }
    except Exception as exc:
        return {"available": False, "reason": f"safe_open failed: {type(exc).__name__}: {exc}"}


def _verify_preprocessor_load(
    checkpoint_dir: Path,
    *,
    device: str,
    local_tokenizer_path: str | None,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "attempted": True,
        "status": "error",
        "device": device,
        "local_tokenizer_path": None,
        "config_filename": "policy_preprocessor.json",
        "steps": [],
        "error": None,
    }
    try:
        pipeline, details = load_policy_preprocessor_from_checkpoint(
            checkpoint_dir,
            device=device,
            local_tokenizer_path=local_tokenizer_path,
            require_local_tokenizer=True,
        )
    except Exception as exc:
        validation["error"] = f"{type(exc).__name__}: {exc}"
        return validation

    del pipeline
    validation["status"] = "ok"
    validation["local_tokenizer_path"] = details["local_tokenizer_path"]
    validation["runtime_compatibility"] = details["runtime_compatibility"]
    validation["overrides"] = details["overrides"]
    validation["steps"] = details["steps"]
    return validation


def _verify_policy_load(
    checkpoint_dir: Path,
    *,
    local_tokenizer_path: str | None,
    strict: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "status": "error",
        "strict": strict,
        "error": None,
        "stdout_excerpt": [],
        "missing_key_count": None,
        "unexpected_key_count": None,
        "policy_device": None,
    }

    runtime = ensure_pi_runtime_compatibility(
        local_tokenizer_path=local_tokenizer_path,
        require_local_tokenizer=True,
    )
    if not runtime["ready"]:
        result["error"] = "; ".join(runtime["errors"])
        return result

    from common import install_siglip_check_shim

    install_siglip_check_shim()
    from lerobot.policies.pi05 import PI05Policy

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            policy = PI05Policy.from_pretrained(
                checkpoint_dir,
                strict=strict,
                local_files_only=True,
            )
            policy.eval()
        stdout_lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        missing_match = re.search(r"Missing keys when loading state dict: (\d+) keys", buffer.getvalue())
        unexpected_match = re.search(
            r"Unexpected keys when loading state dict: (\d+) keys",
            buffer.getvalue(),
        )
        result["status"] = "ok"
        result["stdout_excerpt"] = stdout_lines[:20]
        result["missing_key_count"] = int(missing_match.group(1)) if missing_match else 0
        result["unexpected_key_count"] = int(unexpected_match.group(1)) if unexpected_match else 0
        result["policy_device"] = str(getattr(policy.config, "device", None))
        del policy
        return result
    except Exception as exc:
        stdout_lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["stdout_excerpt"] = stdout_lines[:20]
        return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.policy_path or args.checkpoint_dir).expanduser()
    resolved_checkpoint_dir = resolve_checkpoint_dir(input_path)
    config_doc = _read_json_document(resolved_checkpoint_dir / "config.json")
    config_data = config_doc["data"] if config_doc["parse_ok"] else None
    asset_inventory, missing_assets = _collect_assets(resolved_checkpoint_dir)
    variant, variant_source = _infer_variant(config_data, args.variant)
    runtime = ensure_pi_runtime_compatibility(
        local_tokenizer_path=args.local_tokenizer_path,
        require_local_tokenizer=True,
    )
    contract = _extract_contract(config_data, variant, asset_inventory)
    phase1_supported = variant == "pi05"

    errors: list[str] = []
    limitations: list[str] = []

    if missing_assets:
        errors.append("Missing required checkpoint assets: " + ", ".join(missing_assets))
    if not config_doc["parse_ok"]:
        errors.append(f"Failed to parse config.json: {config_doc['error']}")
    if not runtime["ready"]:
        errors.extend(runtime["errors"])
    if not phase1_supported:
        limitations.append("Phase 1 only guarantees pi05. pi0 is documented here but not export-ready.")

    processor_validation = _verify_preprocessor_load(
        resolved_checkpoint_dir,
        device=args.device,
        local_tokenizer_path=args.local_tokenizer_path,
    )
    if processor_validation["status"] != "ok":
        errors.append("Checkpoint preprocessor validation failed: " + str(processor_validation["error"]))

    if args.verify_policy_load:
        policy_validation = _verify_policy_load(
            resolved_checkpoint_dir,
            local_tokenizer_path=args.local_tokenizer_path,
            strict=args.policy_strict,
        )
        if policy_validation["status"] != "ok":
            errors.append("PI05 policy load validation failed: " + str(policy_validation["error"]))
    else:
        policy_validation = {
            "attempted": False,
            "status": "skipped",
            "strict": args.policy_strict,
            "error": None,
            "stdout_excerpt": [],
            "missing_key_count": None,
            "unexpected_key_count": None,
            "policy_device": None,
        }

    report: dict[str, Any] = {
        "schema_version": "pi_trt.stage1_inspect.v2",
        "generated_at_utc": _utc_now(),
        "status": "ok" if not errors else "error",
        "policy_path_input": str(input_path),
        "resolved_checkpoint_dir": str(resolved_checkpoint_dir),
        "variant_requested": args.variant,
        "variant_inferred": variant,
        "variant_source": variant_source,
        "phase1_supported": phase1_supported,
        "missing_assets": missing_assets,
        "asset_inventory": asset_inventory,
        "runtime_compatibility": runtime,
        "config_summary": {
            "config_json_exists": config_doc["exists"],
            "config_parse_ok": config_doc["parse_ok"],
            "config_error": config_doc["error"],
            "config_top_level_keys": sorted(config_data.keys()) if isinstance(config_data, dict) else [],
        },
        "contract": contract,
        "processor_validation": processor_validation,
        "policy_validation": policy_validation,
        "limitations": limitations,
        "errors": errors,
    }

    if args.inspect_weights:
        report["weights_summary"] = _inspect_safetensors(
            resolved_checkpoint_dir / "model.safetensors",
            args.sample_limit,
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a PI checkpoint and validate that the checkpoint-backed PI05 processor can load offline."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--policy-path",
        help="Training run root, checkpoints/last path, or pretrained_model directory.",
    )
    source_group.add_argument(
        "--checkpoint-dir",
        help="Direct checkpoint directory alias for --policy-path.",
    )
    parser.add_argument(
        "--variant",
        default="auto",
        choices=("auto", "pi05", "pi0"),
        help="Variant override. Phase 1 only fully supports pi05.",
    )
    parser.add_argument(
        "--output",
        default="stage1_inspect_checkpoint.json",
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Processor device override used during checkpoint-backed preprocessor validation.",
    )
    parser.add_argument(
        "--local-tokenizer-path",
        default=None,
        help="Optional explicit offline tokenizer directory override.",
    )
    parser.add_argument(
        "--inspect-weights",
        action="store_true",
        help="Attempt lightweight safetensors metadata inspection when model.safetensors exists.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=8,
        help="Maximum number of safetensors keys to sample when --inspect-weights is enabled.",
    )
    parser.add_argument(
        "--verify-policy-load",
        action="store_true",
        help="Also instantiate the PI05 policy to exercise the runtime shim path.",
    )
    parser.add_argument(
        "--policy-strict",
        action="store_true",
        help="Use strict=True when --verify-policy-load is enabled.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when the checkpoint is not inspection-ready.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the JSON payload after writing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=False))

    if args.strict and report["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
