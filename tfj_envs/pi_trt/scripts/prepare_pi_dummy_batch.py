#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from common import (
    ensure_pi_runtime_compatibility,
    load_policy_preprocessor_from_checkpoint,
    resolve_checkpoint_dir,
    summarize_pipeline_steps,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _lazy_import_pi05_modules() -> dict[str, Any]:
    from common import install_siglip_check_shim

    install_siglip_check_shim()

    import lerobot.policies.pi0.processor_pi0 as _pi0_processor  # noqa: F401
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import PI05Config
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
        batch_to_transition,
        transition_to_batch,
    )
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
    return {
        "AddBatchDimensionProcessorStep": AddBatchDimensionProcessorStep,
        "DeviceProcessorStep": DeviceProcessorStep,
        "FeatureType": FeatureType,
        "NormalizerProcessorStep": NormalizerProcessorStep,
        "PolicyFeature": PolicyFeature,
        "PolicyProcessorPipeline": PolicyProcessorPipeline,
        "PI05Config": PI05Config,
        "Pi05PrepareStateTokenizerProcessorStep": Pi05PrepareStateTokenizerProcessorStep,
        "POLICY_PREPROCESSOR_DEFAULT_NAME": POLICY_PREPROCESSOR_DEFAULT_NAME,
        "RenameObservationsProcessorStep": RenameObservationsProcessorStep,
        "TokenizerProcessorStep": TokenizerProcessorStep,
        "batch_to_transition": batch_to_transition,
        "transition_to_batch": transition_to_batch,
    }


def _feature_type_name(feature: Any) -> str:
    value = getattr(feature, "type", None)
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def _feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not isinstance(shape, (list, tuple)):
        return tuple()
    return tuple(int(item) for item in shape)


def _camera_keys_from_features(input_features: dict[str, Any]) -> list[str]:
    camera_keys: list[str] = []
    for key, feature in input_features.items():
        type_name = _feature_type_name(feature)
        if "visual" in type_name or key.startswith("observation.images."):
            camera_keys.append(key)
    return camera_keys


def _state_key_from_features(input_features: dict[str, Any]) -> str:
    for key, feature in input_features.items():
        type_name = _feature_type_name(feature)
        if "state" in type_name or key == "observation.state":
            return key
    return "observation.state"


def _action_key_from_features(output_features: dict[str, Any]) -> str:
    for key, feature in output_features.items():
        type_name = _feature_type_name(feature)
        if "action" in type_name or key == "action":
            return key
    return "action"


def _image_hw(camera_keys: list[str], input_features: dict[str, Any], default_hw: int) -> tuple[int, int]:
    if camera_keys:
        shape = _feature_shape(input_features[camera_keys[0]])
        if len(shape) >= 3:
            return int(shape[-2]), int(shape[-1])
    return default_hw, default_hw


def _build_feature_map(
    *,
    state_dim: int,
    action_dim: int,
    image_hw: int,
    camera_keys: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    modules = _lazy_import_pi05_modules()
    FeatureType = modules["FeatureType"]
    PolicyFeature = modules["PolicyFeature"]

    input_features: dict[str, Any] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for key in camera_keys:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_hw, image_hw))

    output_features: dict[str, Any] = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    return input_features, output_features


def _stats_tensor(shape: tuple[int, ...], fill: float) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.float32)


def _build_dataset_stats(
    input_features: dict[str, Any],
    output_features: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}

    for feature_map in (input_features, output_features):
        for key, feature in feature_map.items():
            shape = _feature_shape(feature)
            if not shape:
                continue
            stats[key] = {
                "mean": _stats_tensor(shape, 0.0),
                "std": _stats_tensor(shape, 1.0),
                "min": _stats_tensor(shape, -1.0),
                "max": _stats_tensor(shape, 1.0),
                "q01": _stats_tensor(shape, -1.0),
                "q99": _stats_tensor(shape, 1.0),
            }
    return stats


def _build_constructed_pi05_preprocessor_pipeline(
    *,
    config: Any,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    device: str,
    local_tokenizer_path: str,
) -> Any:
    modules = _lazy_import_pi05_modules()
    pipeline = modules["PolicyProcessorPipeline"](
        steps=[
            modules["RenameObservationsProcessorStep"](rename_map={}),
            modules["AddBatchDimensionProcessorStep"](),
            modules["NormalizerProcessorStep"](
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            modules["Pi05PrepareStateTokenizerProcessorStep"](max_state_dim=config.max_state_dim),
            modules["TokenizerProcessorStep"](
                tokenizer_name=local_tokenizer_path,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
                truncation=True,
            ),
            modules["DeviceProcessorStep"](device=device),
        ],
        name=modules["POLICY_PREPROCESSOR_DEFAULT_NAME"],
        to_transition=modules["batch_to_transition"],
        to_output=modules["transition_to_batch"],
    )
    return pipeline


def _maybe_load_checkpoint_config(checkpoint_dir: Path | None, device: str) -> Any | None:
    if checkpoint_dir is None:
        return None
    modules = _lazy_import_pi05_modules()
    config_cls = modules["PI05Config"]
    config = config_cls.from_pretrained(checkpoint_dir)
    config.device = device
    return config


def _ensure_pi05_config(
    *,
    checkpoint_dir: Path | None,
    device: str,
    state_dim: int,
    action_dim: int,
    image_hw: int,
) -> Any:
    config = _maybe_load_checkpoint_config(checkpoint_dir, device)
    if config is None:
        modules = _lazy_import_pi05_modules()
        config = modules["PI05Config"](device=device)
    config.device = device

    input_features = dict(getattr(config, "input_features", {}) or {})
    output_features = dict(getattr(config, "output_features", {}) or {})
    camera_keys = _camera_keys_from_features(input_features)

    if not camera_keys:
        default_camera_keys = [
            "observation.images.top",
            "observation.images.wrist",
        ]
        input_features, output_features = _build_feature_map(
            state_dim=state_dim,
            action_dim=action_dim,
            image_hw=image_hw,
            camera_keys=default_camera_keys,
        )
    else:
        state_key = _state_key_from_features(input_features)
        if state_key not in input_features:
            modules = _lazy_import_pi05_modules()
            FeatureType = modules["FeatureType"]
            PolicyFeature = modules["PolicyFeature"]
            input_features[state_key] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        if not output_features:
            modules = _lazy_import_pi05_modules()
            FeatureType = modules["FeatureType"]
            PolicyFeature = modules["PolicyFeature"]
            output_features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            }

    config.input_features = input_features
    config.output_features = output_features
    return config


def _build_pi05_preprocessor(
    *,
    checkpoint_dir: Path | None,
    device: str,
    state_dim: int,
    action_dim: int,
    image_hw: int,
    local_tokenizer_path: str | None,
) -> tuple[Any, Any, dict[str, Any], str]:
    runtime = ensure_pi_runtime_compatibility(
        local_tokenizer_path=local_tokenizer_path,
        require_local_tokenizer=True,
    )
    if not runtime["ready"]:
        raise RuntimeError("; ".join(runtime["errors"]))

    resolved_tokenizer_path = runtime["local_tokenizer_path"]
    if resolved_tokenizer_path is None:
        raise RuntimeError("Offline tokenizer path resolution unexpectedly returned None.")

    if checkpoint_dir is not None:
        resolved_checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        try:
            preprocessor, details = load_policy_preprocessor_from_checkpoint(
                resolved_checkpoint_dir,
                device=device,
                local_tokenizer_path=resolved_tokenizer_path,
                require_local_tokenizer=True,
            )
            config = _maybe_load_checkpoint_config(resolved_checkpoint_dir, device)
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint assets exist but checkpoint-backed policy_preprocessor.json "
                "could not be loaded. Refusing to silently fall back to a constructed pipeline."
            ) from exc

        if config is None:
            raise RuntimeError(f"Failed to load PI05Config from checkpoint '{resolved_checkpoint_dir}'.")
        details["runtime_compatibility"] = runtime
        return preprocessor, config, details, "checkpoint_policy_preprocessor"

    config = _ensure_pi05_config(
        checkpoint_dir=None,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
        image_hw=image_hw,
    )
    dataset_stats = _build_dataset_stats(config.input_features, config.output_features)
    preprocessor = _build_constructed_pi05_preprocessor_pipeline(
        config=config,
        dataset_stats=dataset_stats,
        device=device,
        local_tokenizer_path=resolved_tokenizer_path,
    )

    details = {
        "checkpoint_dir": None,
        "config_filename": None,
        "device": device,
        "local_tokenizer_path": resolved_tokenizer_path,
        "runtime_compatibility": runtime,
        "overrides": {
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": resolved_tokenizer_path},
        },
        "steps": summarize_pipeline_steps(preprocessor),
    }
    return preprocessor, config, details, "constructed_pi05_preprocessor"


def _linspace_tensor(shape: tuple[int, ...], start: float, end: float) -> torch.Tensor:
    numel = 1
    for item in shape:
        numel *= item
    tensor = torch.linspace(start, end, steps=numel, dtype=torch.float32)
    return tensor.view(shape)


def _make_image(case_offset: float, height: int, width: int) -> torch.Tensor:
    image = _linspace_tensor((3, height, width), 0.0 + case_offset, 1.0 + case_offset)
    return torch.remainder(image, 1.0)


def _make_raw_case(
    *,
    case_name: str,
    selected_camera_keys: list[str],
    input_features: dict[str, Any],
    output_features: dict[str, Any],
    task_text: str,
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_key = _state_key_from_features(input_features)
    action_key = _action_key_from_features(output_features)
    state_shape = _feature_shape(input_features[state_key])
    action_shape = _feature_shape(output_features[action_key])
    state_dim = state_shape[-1] if state_shape else 14
    action_dim = action_shape[-1] if action_shape else 7
    height, width = _image_hw(selected_camera_keys, input_features, 224)

    raw_batch: dict[str, Any] = {
        state_key: _linspace_tensor((state_dim,), -0.75, 0.75),
        "task": task_text,
    }
    for index, camera_key in enumerate(selected_camera_keys):
        raw_batch[camera_key] = _make_image(index * 0.1, height, width)

    contract = {
        "case_name": case_name,
        "camera_keys": selected_camera_keys,
        "state_key": state_key,
        "action_key": action_key,
        "task_text": task_text,
        "height": height,
        "width": width,
        "chunk_size": int(config.chunk_size),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "raw_batch_is_batched": False,
    }
    return raw_batch, contract


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _summarize_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {key: _summarize_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_summarize_structure(item) for item in value]
    if isinstance(value, tuple):
        return [_summarize_structure(item) for item in value]
    return value


def _save_tensor_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_copy(payload), path)


def _prepare_pi05_cases(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = None
    if args.policy_path or args.checkpoint_dir:
        checkpoint_dir = resolve_checkpoint_dir(Path(args.policy_path or args.checkpoint_dir).expanduser())

    preprocessor, config, processor_details, processor_source = _build_pi05_preprocessor(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        image_hw=args.image_hw,
        local_tokenizer_path=args.local_tokenizer_path,
    )

    input_features = dict(config.input_features)
    output_features = dict(config.output_features)
    camera_keys = _camera_keys_from_features(input_features)
    if not camera_keys:
        raise RuntimeError("PI05 dummy batch preparation requires at least one camera feature.")

    case_specs = {
        "single_camera_minimal_case": {
            "camera_keys": [camera_keys[0]],
            "task_text": "Pick up the red block",
        },
        "multi_camera_nominal_case": {
            "camera_keys": list(camera_keys),
            "task_text": "Place the object into the tray",
        },
    }

    results: dict[str, Any] = {}
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_name in args.cases:
        spec = case_specs[case_name]
        raw_batch, contract = _make_raw_case(
            case_name=case_name,
            selected_camera_keys=spec["camera_keys"],
            input_features=input_features,
            output_features=output_features,
            task_text=spec["task_text"],
            config=config,
        )
        processed_batch = preprocessor(raw_batch)

        raw_path = output_dir / f"{case_name}.raw_input.pt"
        processed_path = output_dir / f"{case_name}.processed_batch.pt"
        _save_tensor_bundle(raw_path, raw_batch)
        _save_tensor_bundle(processed_path, processed_batch)

        case_record = {
            "status": "ok",
            "contract": contract,
            "notes": [],
            "raw_input_path": str(raw_path),
            "processed_batch_path": str(processed_path),
            "raw_input_summary": _summarize_structure(_cpu_copy(raw_batch)),
            "processed_batch_summary": _summarize_structure(_cpu_copy(processed_batch)),
        }
        if case_name == "multi_camera_nominal_case" and len(spec["camera_keys"]) < 2:
            case_record["notes"].append(
                "Checkpoint only exposes one camera key, so multi_camera_nominal_case is checkpoint-limited."
            )
        results[case_name] = case_record

    summary = {
        "schema_version": "pi_trt.dummy_batch.v2",
        "generated_at_utc": _utc_now(),
        "status": "ok",
        "variant": "pi05",
        "phase1_supported": True,
        "processor_source": processor_source,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "local_tokenizer_path": processor_details["local_tokenizer_path"],
        "runtime_compatibility": processor_details["runtime_compatibility"],
        "processor_overrides": processor_details["overrides"],
        "processor_steps": processor_details["steps"],
        "config_contract": {
            "chunk_size": int(config.chunk_size),
            "num_inference_steps": int(config.num_inference_steps),
            "tokenizer_max_length": int(config.tokenizer_max_length),
            "camera_keys": camera_keys,
            "device": str(config.device),
        },
        "cases": results,
    }
    summary_path = output_dir / "dummy_batch_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _prepare_pi0_stub(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "pi_trt.dummy_batch.v2",
        "generated_at_utc": _utc_now(),
        "status": "stub_only",
        "variant": "pi0",
        "phase1_supported": False,
        "checkpoint_dir": args.policy_path or args.checkpoint_dir,
        "limitations": [
            "Phase 1 only guarantees pi05 processor-driven dummy batches.",
            "pi0 keeps a clear CLI entrypoint here but does not emit controlled processed batches yet.",
        ],
    }
    summary_path = output_dir / "dummy_batch_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare processor-driven dummy PI batches with offline tokenizer support."
    )
    parser.add_argument(
        "--policy-path",
        help="Training run root, checkpoints/last path, or pretrained_model directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Direct checkpoint directory alias for --policy-path.",
    )
    parser.add_argument(
        "--variant",
        default="pi05",
        choices=("pi05", "pi0"),
        help="Phase 1 only fully supports pi05.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for emitted .pt and summary files.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("single_camera_minimal_case", "multi_camera_nominal_case"),
        choices=("single_camera_minimal_case", "multi_camera_nominal_case"),
        help="Controlled case set to emit.",
    )
    parser.add_argument("--device", default="cpu", help="Processor device override.")
    parser.add_argument(
        "--local-tokenizer-path",
        default=None,
        help="Optional explicit offline tokenizer directory override.",
    )
    parser.add_argument("--state-dim", type=int, default=14, help="Fallback state dimension when checkpoint is absent.")
    parser.add_argument("--action-dim", type=int, default=7, help="Fallback action dimension when checkpoint is absent.")
    parser.add_argument("--image-hw", type=int, default=224, help="Fallback image height/width when checkpoint is absent.")
    parser.add_argument("--print-json", action="store_true", help="Print summary JSON after writing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.variant == "pi0":
        summary = _prepare_pi0_stub(args)
    else:
        summary = _prepare_pi05_cases(args)

    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
