#!/usr/bin/env python

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

import onnx
import torch
from onnx import checker

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


DEFAULT_DEVICE = "cuda"
DEFAULT_CONDA_ENV = "lerobot_flex"
DEFAULT_VIDEO_VIEWS_1 = 1
DEFAULT_VIDEO_VIEWS_2 = 2
DEFAULT_SEQ_LEN_1 = 296
DEFAULT_SEQ_LEN_2 = 568
DEFAULT_MIN_SEQ_LEN = 80
DEFAULT_MAX_SEQ_LEN_1 = 300
DEFAULT_MAX_SEQ_LEN_2 = 600
DEFAULT_TENSORRT_PY_DIR = os.getenv("TENSORRT_PY_DIR", "").strip()
DEFAULT_TMPDIR = os.getenv("TMPDIR", "").strip()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "trt"
DEFAULT_STAGEWISE_PREFIX = "groot_stepwise"


def repo_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    py_parts = [SRC_ROOT.as_posix(), REPO_ROOT.as_posix()]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_parts)
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            resolved = str(value).strip()
            if not resolved:
                continue
            env[key] = resolved
    return env


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def default_stage_report_path(out_dir: Path, stage_name: str) -> Path:
    ensure_dir(out_dir)
    return out_dir / f"{stage_name}.json"


def default_run_dir(prefix: str = DEFAULT_STAGEWISE_PREFIX) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{prefix}_{now_ts()}"


def resolve_tensorrt_py_dir(path: str | None) -> str | None:
    candidate = (path or os.getenv("TENSORRT_PY_DIR", "")).strip()
    if not candidate:
        return None
    return Path(candidate).expanduser().resolve().as_posix()


def resolve_tmpdir(path: str | None, run_dir: Path | None = None) -> str:
    candidate = (path or os.getenv("TMPDIR", "")).strip()
    if candidate:
        return ensure_dir(Path(candidate).expanduser().resolve()).as_posix()
    base = run_dir if run_dir is not None else DEFAULT_OUTPUT_ROOT
    return ensure_dir(base / ".tmp").as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_conda_python_cmd(conda_env: str, script_path: Path, script_args: list[str]) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        script_path.as_posix(),
        *script_args,
    ]


def run_command(
    cmd: list[str],
    *,
    log_path: Path | None = None,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    env = repo_env(env_extra)
    workdir = cwd or REPO_ROOT
    started_at = time.time()
    proc = subprocess.run(
        cmd,
        cwd=workdir.as_posix(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    ended_at = time.time()
    output = proc.stdout
    if log_path is not None:
        ensure_dir(log_path.parent)
        log_path.write_text(output)
    result = {
        "cmd": cmd,
        "cwd": workdir.as_posix(),
        "returncode": proc.returncode,
        "seconds": ended_at - started_at,
        "log_path": log_path.as_posix() if log_path is not None else None,
        "output_tail": output[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def resolve_policy_dir(policy_path: str | Path) -> Path:
    path = Path(policy_path).expanduser().resolve()
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    searched = "\n".join(f"  - {cand}" for cand in candidates)
    raise FileNotFoundError(
        "Could not resolve a valid `pretrained_model/` directory.\n"
        f"Input: {path}\n"
        "Searched:\n"
        f"{searched}\n"
    )


def required_checkpoint_files(policy_dir: Path) -> dict[str, bool]:
    return {
        "config.json": (policy_dir / "config.json").is_file(),
        "model.safetensors": (policy_dir / "model.safetensors").is_file(),
        "policy_preprocessor.json": (policy_dir / "policy_preprocessor.json").is_file(),
        "policy_postprocessor.json": (policy_dir / "policy_postprocessor.json").is_file(),
    }


def load_policy(
    policy_dir: Path,
    device: str = DEFAULT_DEVICE,
    *,
    strict: bool = False,
) -> tuple[PreTrainedConfig, type[Any], Any]:
    cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    cfg.pretrained_path = policy_dir
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_dir), config=cfg, strict=bool(strict))
    policy.eval()
    policy.to(device)
    return cfg, policy_cls, policy


def load_pre_post_processors(policy_dir: Path) -> tuple[Any, Any]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def policy_summary(cfg: PreTrainedConfig, policy_cls: type[Any], policy: Any) -> dict[str, Any]:
    import lerobot

    backbone = policy._groot_model.backbone
    action_head = policy._groot_model.action_head
    if hasattr(backbone.eagle_model.vision_model, "vision_model"):
        num_patches = int(backbone.eagle_model.vision_model.vision_model.embeddings.num_patches)
    else:
        num_patches = int(backbone.eagle_model.vision_model.embeddings.num_patches)

    return {
        "lerobot_module": lerobot.__file__,
        "policy_type": cfg.type,
        "policy_class": policy_cls.__name__,
        "cfg_device": cfg.device,
        "cfg_pretrained_path": str(getattr(cfg, "pretrained_path", "")),
        "input_features": {
            key: {"shape": list(value.shape), "type": str(value.type)}
            for key, value in cfg.input_features.items()
        },
        "output_features": {
            key: {"shape": list(value.shape), "type": str(value.type)}
            for key, value in cfg.output_features.items()
        },
        "backbone": {
            "num_patches": num_patches,
            "select_layer": int(backbone.select_layer),
            "language_hidden_size": int(backbone.eagle_model.language_model.config.hidden_size),
            "use_pixel_shuffle": bool(getattr(backbone.eagle_model, "use_pixel_shuffle", False)),
        },
        "action_head": {
            "action_horizon": int(action_head.config.action_horizon),
            "action_dim": int(action_head.config.action_dim),
            "max_state_dim": int(action_head.config.max_state_dim),
            "num_target_vision_tokens": int(action_head.config.num_target_vision_tokens),
            "input_embedding_dim": int(action_head.config.input_embedding_dim),
            "hidden_size": int(action_head.config.hidden_size),
            "num_inference_timesteps": int(action_head.num_inference_timesteps),
        },
    }


def expected_onnx_contracts() -> dict[str, dict[str, dict[str, list[Any]]]]:
    return {
        "eagle2/vit_fp16.onnx": {
            "inputs": {"pixel_values": ["batch_size", 3, 224, 224], "position_ids": ["batch_size", 256]},
            "outputs": {"vit_embeds": ["batch_size", None, None]},
        },
        "eagle2/llm_fp16.onnx": {
            "inputs": {
                "inputs_embeds": ["batch_size", "sequence_length", 2048],
                "attention_mask": ["batch_size", "sequence_length"],
            },
            "outputs": {"embeddings": ["batch_size", "sequence_length", 2048]},
        },
        "action_head/vlln_vl_self_attention.onnx": {
            "inputs": {"backbone_features": ["batch_size", "sequence_length", 2048]},
            "outputs": {"output": ["batch_size", "sequence_length", 2048]},
        },
        "action_head/state_encoder.onnx": {
            "inputs": {"state": ["batch_size", 1, 64], "embodiment_id": ["batch_size"]},
            "outputs": {"output": ["batch_size", 1, 1536]},
        },
        "action_head/action_encoder.onnx": {
            "inputs": {
                "actions": ["batch_size", 16, 32],
                "timesteps_tensor": ["batch_size"],
                "embodiment_id": ["batch_size"],
            },
            "outputs": {"output": ["batch_size", 16, 1536]},
        },
        "action_head/DiT_fp16.onnx": {
            "inputs": {
                "sa_embs": ["batch_size", 49, 1536],
                "vl_embs": ["batch_size", "sequence_length", 2048],
                "timesteps_tensor": ["batch_size"],
            },
            "outputs": {"output": ["batch_size", 49, 1024]},
        },
        "action_head/action_decoder.onnx": {
            "inputs": {"model_output": ["batch_size", 49, 1024], "embodiment_id": ["batch_size"]},
            "outputs": {"output": ["batch_size", 49, 32]},
        },
    }


def expected_engine_files() -> list[str]:
    return [
        "vit_fp16.engine",
        "llm_fp16.engine",
        "vlln_vl_self_attention.engine",
        "state_encoder.engine",
        "action_encoder.engine",
        "DiT_fp16.engine",
        "action_decoder.engine",
    ]


def dim_to_obj(dim: Any) -> str | int | None:
    if dim.dim_param:
        return dim.dim_param
    if dim.dim_value:
        return int(dim.dim_value)
    return None


def tensor_shape(value_info: Any) -> list[str | int | None]:
    return [dim_to_obj(dim) for dim in value_info.type.tensor_type.shape.dim]


def validate_onnx_contracts(onnx_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for rel_path, contract in expected_onnx_contracts().items():
        path = onnx_dir / rel_path
        if not path.is_file():
            raise FileNotFoundError(path)
        model = onnx.load(path.as_posix())
        checker.check_model(model)
        inputs = {item.name: tensor_shape(item) for item in model.graph.input}
        outputs = {item.name: tensor_shape(item) for item in model.graph.output}
        report[rel_path] = {
            "checker_ok": True,
            "inputs": inputs,
            "outputs": outputs,
            "input_names_match": set(inputs) == set(contract["inputs"]),
            "output_names_match": set(outputs) == set(contract["outputs"]),
        }
        if not report[rel_path]["input_names_match"]:
            raise ValueError(f"Input contract mismatch for {rel_path}: {inputs}")
        if not report[rel_path]["output_names_match"]:
            raise ValueError(f"Output contract mismatch for {rel_path}: {outputs}")
    return report


def validate_engine_dir(engine_dir: Path) -> dict[str, Any]:
    missing = []
    present = []
    for name in expected_engine_files():
        path = engine_dir / name
        if path.is_file():
            present.append(
                {
                    "name": name,
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            "Missing required TensorRT engine files:\n" + "\n".join(f"  - {name}" for name in missing)
        )
    return {
        "engine_dir": engine_dir.as_posix(),
        "present": present,
        "missing": missing,
    }


def summarize_build_report(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    engines = data["engines"]
    return {
        "report_path": report_path.as_posix(),
        "tensorrt_version": data["tensorrt_version"],
        "num_engines": len(engines),
        "engine_names": [Path(item["engine"]).name for item in engines],
        "profile": data["build_profile"],
    }


def summarize_onnx_compare(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    results = data["results"]
    worst_key = min(results, key=lambda key: results[key]["cosine"])
    return {
        "report_path": report_path.as_posix(),
        "missing": data.get("missing"),
        "worst_cosine_key": worst_key,
        "worst_cosine": results[worst_key]["cosine"],
        "denoising_cosine": results["action_denoising_pipeline"]["cosine"],
        "llm_from_vit_cosine": results["llm_from_vit_pipeline"]["cosine"],
    }


def summarize_trt_compare(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    results = data["results"]
    worst_key = min(results, key=lambda key: results[key]["cosine"])
    return {
        "report_path": report_path.as_posix(),
        "tensorrt_version": data.get("tensorrt_version"),
        "worst_cosine_key": worst_key,
        "worst_cosine": results[worst_key]["cosine"],
        "denoising_cosine": results["action_denoising_pipeline"]["cosine"],
        "llm_from_vit_cosine": results["llm_from_vit_pipeline"]["cosine"],
    }


def summarize_mock_compare(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    return {
        "report_path": report_path.as_posix(),
        "overall_cosine": data["overall"]["cosine"],
        "overall_rmse": data["overall"]["rmse"],
        "overall_max_abs": data["overall"]["max_abs"],
        "worst_cosine_step": data["summary"]["worst_cosine_step"],
        "worst_cosine": data["summary"]["worst_cosine"],
    }
