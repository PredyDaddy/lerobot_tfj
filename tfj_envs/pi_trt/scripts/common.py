#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
PI_TRT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PI_TRT_ROOT / "runs"

for candidate in (REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


DEFAULT_CONDA_ENV = "lerobot"
DEFAULT_PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
DEFAULT_POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"
LOCAL_TOKENIZER_ENV_KEYS = (
    "PI05_LOCAL_TOKENIZER_PATH",
    "PI_LOCAL_TOKENIZER_PATH",
    "PALIGEMMA_LOCAL_TOKENIZER_PATH",
)
DEFAULT_LOCAL_TOKENIZER_CANDIDATES = (
    Path("/home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224"),
    Path("/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base"),
)

KNOWN_VARIANTS = ("pi05", "pi0")
PHASE1_VARIANTS = ("pi05",)
REQUIRED_CHECKPOINT_ASSETS = (
    "config.json",
    "model.safetensors",
    DEFAULT_PREPROCESSOR_CONFIG_FILENAME,
    DEFAULT_POSTPROCESSOR_CONFIG_FILENAME,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_run_dir(run_dir: str | Path | None = None, *, prefix: str = "pi_trt") -> Path:
    if run_dir is None:
        candidate = RUNS_ROOT / f"{prefix}_{timestamp_slug()}"
    else:
        candidate = Path(run_dir).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
    return ensure_dir(candidate.resolve())


def prepare_run_layout(run_dir: str | Path | None = None, *, prefix: str = "pi_trt") -> dict[str, Path]:
    root = resolve_run_dir(run_dir, prefix=prefix)
    layout = {
        "run_dir": root,
        "logs_dir": root / "logs",
        "artifacts_dir": root / "artifacts",
        "onnx_dir": root / "artifacts" / "onnx",
        "engines_dir": root / "artifacts" / "engines",
        "reports_dir": root / "artifacts" / "reports",
    }
    for path in layout.values():
        ensure_dir(path)
    return layout


def stage_json_path(run_dir: str | Path, stage_name: str) -> Path:
    return ensure_dir(Path(run_dir).expanduser().resolve()) / f"{stage_name}.json"


def metadata_path(run_dir: str | Path) -> Path:
    return stage_json_path(run_dir, "pi_trt_metadata")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def normalize_variant(variant: str) -> str:
    return variant.strip().lower().replace("-", "")


def validate_variant(variant: str, *, phase1_only: bool = False) -> str:
    normalized = normalize_variant(variant)
    if normalized not in KNOWN_VARIANTS:
        allowed = ", ".join(KNOWN_VARIANTS)
        raise ValueError(f"Unsupported variant '{variant}'. Expected one of: {allowed}.")
    if phase1_only and normalized not in PHASE1_VARIANTS:
        allowed = ", ".join(PHASE1_VARIANTS)
        raise ValueError(f"Phase-1 only supports: {allowed}. Requested: {normalized}.")
    return normalized


def _iter_candidate_checkpoint_dirs(base_path: Path) -> Iterable[Path]:
    if base_path.is_file():
        yield base_path.parent
        return

    yield base_path
    yield base_path / "pretrained_model"
    yield base_path / "checkpoints" / "last"
    yield base_path / "checkpoints" / "last" / "pretrained_model"


def resolve_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    if checkpoint_path is None:
        raise FileNotFoundError("Checkpoint path was not provided.")

    base = Path(checkpoint_path).expanduser()
    if base.name == "config.json":
        base = base.parent

    resolved_base = base.resolve(strict=False)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for candidate in _iter_candidate_checkpoint_dirs(resolved_base):
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve checkpoint directory containing config.json. "
        f"Searched: {searched}"
    )


def checkpoint_asset_paths(checkpoint_dir: str | Path) -> dict[str, Path]:
    root = Path(checkpoint_dir).expanduser().resolve(strict=False)
    return {name: root / name for name in REQUIRED_CHECKPOINT_ASSETS}


def inspect_checkpoint_assets(checkpoint_dir: str | Path) -> dict[str, Any]:
    assets = checkpoint_asset_paths(checkpoint_dir)
    statuses = {name: path.is_file() for name, path in assets.items()}
    missing = [name for name, ok in statuses.items() if not ok]
    return {
        "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve(strict=False)),
        "assets": {name: str(path) for name, path in assets.items()},
        "asset_status": statuses,
        "missing_assets": missing,
        "all_required_present": not missing,
    }


def detect_conda_env() -> str | None:
    env_name = os.getenv("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    prefix = os.getenv("CONDA_PREFIX")
    if prefix:
        return Path(prefix).name
    return None


def _tokenizer_candidate_iter(explicit_path: str | Path | None = None) -> Iterable[Path]:
    if explicit_path is not None:
        yield Path(explicit_path).expanduser()
    for env_key in LOCAL_TOKENIZER_ENV_KEYS:
        value = os.getenv(env_key)
        if value:
            yield Path(value).expanduser()
    for candidate in DEFAULT_LOCAL_TOKENIZER_CANDIDATES:
        yield candidate


def is_local_tokenizer_dir(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        return False
    required_markers = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "tokenizer_config.json",
    )
    return any((candidate / marker).is_file() for marker in required_markers)


def discover_local_tokenizer_path(
    explicit_path: str | Path | None = None,
    *,
    require: bool = False,
) -> Path | None:
    seen: set[Path] = set()
    for candidate in _tokenizer_candidate_iter(explicit_path):
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_local_tokenizer_dir(resolved):
            return resolved

    if require:
        searched = ", ".join(str(path) for path in seen) or "<none>"
        raise FileNotFoundError(
            "Could not locate an offline tokenizer directory for PaliGemma. "
            f"Searched: {searched}"
        )
    return None


def install_siglip_check_shim() -> dict[str, Any]:
    summary = {
        "module_name": "transformers.models.siglip.check",
        "ready": False,
        "installed": False,
        "source": None,
        "error": None,
    }

    try:
        siglip_pkg = importlib.import_module("transformers.models.siglip")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    try:
        module = importlib.import_module(summary["module_name"])
    except ModuleNotFoundError as exc:
        if exc.name != summary["module_name"]:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            return summary
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary
    else:
        if not hasattr(module, "check_whether_transformers_replace_is_installed_correctly"):
            module.check_whether_transformers_replace_is_installed_correctly = lambda: True
            summary["installed"] = True
            summary["source"] = "patched_existing_module"
        else:
            summary["source"] = "native_module"
        sys.modules[summary["module_name"]] = module
        setattr(siglip_pkg, "check", module)
        summary["ready"] = True
        return summary

    shim = ModuleType(summary["module_name"])
    shim.__package__ = "transformers.models.siglip"
    shim.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules[summary["module_name"]] = shim
    setattr(siglip_pkg, "check", shim)

    summary["ready"] = True
    summary["installed"] = True
    summary["source"] = "runtime_shim"
    return summary


def ensure_pi_runtime_compatibility(
    *,
    local_tokenizer_path: str | Path | None = None,
    require_local_tokenizer: bool = True,
) -> dict[str, Any]:
    tokenizer_dir = discover_local_tokenizer_path(local_tokenizer_path, require=False)
    conda_env = detect_conda_env()
    shim_status = install_siglip_check_shim()

    errors: list[str] = []
    warnings: list[str] = []
    if not shim_status["ready"]:
        errors.append(
            "Runtime compatibility shim for transformers.models.siglip.check is not ready: "
            f"{shim_status['error']}"
        )
    if tokenizer_dir is None and require_local_tokenizer:
        errors.append(
            "Offline tokenizer assets were not found. "
            "Set PI05_LOCAL_TOKENIZER_PATH or place the tokenizer under the known local cache paths."
        )
    if conda_env is None:
        warnings.append(
            f"Could not detect an active conda environment. Expected '{DEFAULT_CONDA_ENV}' for PI TRT work."
        )
    elif conda_env != DEFAULT_CONDA_ENV:
        warnings.append(
            f"Active conda environment is '{conda_env}', but PI TRT scripts default to '{DEFAULT_CONDA_ENV}'."
        )

    return {
        "expected_conda_env": DEFAULT_CONDA_ENV,
        "detected_conda_env": conda_env,
        "conda_env_matches_default": conda_env == DEFAULT_CONDA_ENV if conda_env is not None else None,
        "siglip_check": shim_status,
        "local_tokenizer_path": str(tokenizer_dir) if tokenizer_dir is not None else None,
        "local_tokenizer_found": tokenizer_dir is not None,
        "errors": errors,
        "warnings": warnings,
        "ready": not errors,
    }


def build_processor_overrides(
    *,
    device: str = "cpu",
    local_tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "device_processor": {"device": device},
    }
    if local_tokenizer_path is not None:
        overrides["tokenizer_processor"] = {
            "tokenizer_name": str(Path(local_tokenizer_path).expanduser().resolve(strict=False))
        }
    return overrides


def summarize_pipeline_steps(pipeline: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, step in enumerate(getattr(pipeline, "steps", [])):
        record: dict[str, Any] = {
            "index": index,
            "class_name": type(step).__name__,
            "registry_name": getattr(type(step), "_registry_name", None),
        }
        for attr in (
            "device",
            "float_dtype",
            "max_length",
            "padding",
            "padding_side",
            "truncation",
            "task_key",
            "tokenizer_name",
            "max_state_dim",
        ):
            if hasattr(step, attr):
                value = getattr(step, attr)
                if isinstance(value, Path):
                    record[attr] = str(value)
                elif value is None or isinstance(value, (str, int, float, bool)):
                    record[attr] = value
                else:
                    record[attr] = str(value)
        summaries.append(record)
    return summaries


def _load_pi_processor_registry() -> None:
    install_siglip_check_shim()
    import lerobot.policies.pi0.processor_pi0 as _pi0_processor  # noqa: F401
    import lerobot.policies.pi05.processor_pi05 as _pi05_processor  # noqa: F401


def load_policy_preprocessor_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    local_tokenizer_path: str | Path | None = None,
    require_local_tokenizer: bool = True,
) -> tuple[Any, dict[str, Any]]:
    resolved_checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    runtime = ensure_pi_runtime_compatibility(
        local_tokenizer_path=local_tokenizer_path,
        require_local_tokenizer=require_local_tokenizer,
    )
    if not runtime["ready"]:
        raise RuntimeError("; ".join(runtime["errors"]))

    _load_pi_processor_registry()
    from lerobot.processor import PolicyProcessorPipeline

    overrides = build_processor_overrides(
        device=device,
        local_tokenizer_path=runtime["local_tokenizer_path"],
    )
    try:
        pipeline = PolicyProcessorPipeline.from_pretrained(
            resolved_checkpoint_dir,
            config_filename=DEFAULT_PREPROCESSOR_CONFIG_FILENAME,
            overrides=overrides,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load checkpoint policy preprocessor "
            f"from '{resolved_checkpoint_dir}'."
        ) from exc

    return pipeline, {
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "config_filename": DEFAULT_PREPROCESSOR_CONFIG_FILENAME,
        "device": device,
        "local_tokenizer_path": runtime["local_tokenizer_path"],
        "runtime_compatibility": runtime,
        "overrides": overrides,
        "steps": summarize_pipeline_steps(pipeline),
    }


def build_metadata_skeleton(
    *,
    run_dir: str | Path,
    variant: str,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    normalized_variant = validate_variant(variant)
    resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
    checkpoint_value = (
        str(Path(checkpoint_dir).expanduser().resolve(strict=False)) if checkpoint_dir is not None else None
    )
    return {
        "schema_version": 1,
        "phase": "phase-1",
        "created_at": iso_timestamp(),
        "variant": normalized_variant,
        "run_dir": str(resolved_run_dir),
        "checkpoint_dir": checkpoint_value,
        "contract": {
            "batch_size": 1,
            "fixed_variant_per_run": True,
            "phase1_variants": list(PHASE1_VARIANTS),
            "known_variants": list(KNOWN_VARIANTS),
            "frozen_boundaries": [
                "vision_encoder",
                "prefix_cache",
                "denoise_step",
            ],
        },
        "artifacts": {
            "stage0_env_check": str(stage_json_path(resolved_run_dir, "stage0_env_check")),
            "stage1_inspect_checkpoint": str(stage_json_path(resolved_run_dir, "stage1_inspect_checkpoint")),
            "stage2_export_onnx": str(stage_json_path(resolved_run_dir, "stage2_export_onnx")),
            "stage3_verify_onnx": str(stage_json_path(resolved_run_dir, "stage3_verify_onnx")),
            "stage4_build_engines": str(stage_json_path(resolved_run_dir, "stage4_build_engines")),
            "stage5_verify_trt": str(stage_json_path(resolved_run_dir, "stage5_verify_trt")),
            "compare_torch_onnx_trt": str(stage_json_path(resolved_run_dir, "compare_torch_onnx_trt")),
            "local_runtime_smoke": str(stage_json_path(resolved_run_dir, "local_runtime_smoke")),
            "pi_trt_metadata": str(metadata_path(resolved_run_dir)),
        },
    }


def module_probe(module_name: str, *, import_name: str | None = None) -> dict[str, Any]:
    target = import_name or module_name
    try:
        module = importlib.import_module(target)
    except Exception as exc:
        return {
            "name": module_name,
            "import_name": target,
            "available": False,
            "version": None,
            "module_path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    version = getattr(module, "__version__", None)
    if version is None and module_name == "tensorrt":
        version = getattr(module, "version", None)
    return {
        "name": module_name,
        "import_name": target,
        "available": True,
        "version": str(version) if version is not None else None,
        "module_path": getattr(module, "__file__", None),
        "error": None,
    }


def _run_optional_command(cmd: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {
            "available": False,
            "cmd": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": completed.returncode == 0,
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def probe_cuda() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": False,
        "torch_cuda_version": None,
        "device_count": 0,
        "devices": [],
        "cudnn_available": None,
        "cudnn_version": None,
        "nvidia_smi": _run_optional_command(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
    }

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        probe["torch_error"] = f"{type(exc).__name__}: {exc}"
        return probe

    cuda_available = bool(torch.cuda.is_available())
    probe["torch_cuda_available"] = cuda_available
    probe["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    probe["device_count"] = int(torch.cuda.device_count()) if cuda_available else 0

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        probe["cudnn_available"] = bool(cudnn.is_available())
        probe["cudnn_version"] = cudnn.version()

    if cuda_available:
        for index in range(probe["device_count"]):
            props = torch.cuda.get_device_properties(index)
            probe["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    return probe


def collect_env_probe(*, local_tokenizer_path: str | Path | None = None) -> dict[str, Any]:
    modules = {
        "python": {
            "executable": os.sys.executable,
            "version": os.sys.version,
        },
        "torch": module_probe("torch"),
        "onnx": module_probe("onnx"),
        "onnxruntime": module_probe("onnxruntime"),
        "tensorrt": module_probe("tensorrt"),
        "transformers": module_probe("transformers"),
    }
    return {
        "probed_at": iso_timestamp(),
        "modules": modules,
        "cuda": probe_cuda(),
        "runtime": ensure_pi_runtime_compatibility(
            local_tokenizer_path=local_tokenizer_path,
            require_local_tokenizer=False,
        ),
    }


def build_preflight_summary(
    *,
    variant: str,
    checkpoint_dir: str | Path | None,
    env_probe: dict[str, Any] | None = None,
    require_checkpoint: bool = True,
    require_local_tokenizer: bool = True,
) -> dict[str, Any]:
    normalized_variant = validate_variant(variant, phase1_only=False)
    probe = env_probe or collect_env_probe()
    runtime = probe.get("runtime") or ensure_pi_runtime_compatibility(
        require_local_tokenizer=require_local_tokenizer
    )

    errors: list[str] = []
    warnings: list[str] = list(runtime.get("warnings", []))

    for name in ("torch", "onnx", "onnxruntime", "tensorrt", "transformers"):
        if not probe["modules"][name]["available"]:
            errors.append(f"Missing required python module: {name}")

    if not probe["cuda"]["torch_cuda_available"]:
        errors.append("torch.cuda.is_available() returned False")

    if not runtime["siglip_check"]["ready"]:
        errors.append(
            "Runtime compatibility shim for PI05 is not ready: "
            f"{runtime['siglip_check']['error']}"
        )

    if require_local_tokenizer and not runtime["local_tokenizer_found"]:
        errors.append(
            "Offline tokenizer assets were not found; PI05 processor loading cannot be fully offline."
        )

    checkpoint_summary: dict[str, Any]
    if checkpoint_dir is None:
        checkpoint_summary = {
            "checkpoint_dir": None,
            "assets": {},
            "asset_status": {},
            "missing_assets": list(REQUIRED_CHECKPOINT_ASSETS) if require_checkpoint else [],
            "all_required_present": not require_checkpoint,
        }
        if require_checkpoint:
            errors.append("Checkpoint path was not provided.")
        else:
            warnings.append("Checkpoint path not provided; only environment checks were executed.")
    else:
        checkpoint_summary = inspect_checkpoint_assets(checkpoint_dir)
        if checkpoint_summary["missing_assets"]:
            errors.append(
                "Missing checkpoint assets: " + ", ".join(checkpoint_summary["missing_assets"])
            )

    return {
        "variant": normalized_variant,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "environment": probe,
        "runtime": runtime,
        "checkpoint": checkpoint_summary,
    }
