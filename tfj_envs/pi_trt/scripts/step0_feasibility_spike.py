#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import json
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELING_FILE = REPO_ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"
DEFAULT_JSON_OUTPUT = (
    REPO_ROOT
    / "tfj_envs/pi_trt/tmp_pi_trt_execution_20260311_204130/implementation_reports/02_worker_feasibility.json"
)
DEFAULT_MARKDOWN_REPORT = (
    REPO_ROOT
    / "tfj_envs/pi_trt/tmp_pi_trt_execution_20260311_204130/implementation_reports/02_worker_feasibility.md"
)

VERDICT_EXIT_CODES = {
    "go": 0,
    "blocked": 1,
    "no-go": 2,
}


@dataclass
class CheckResult:
    check_id: str
    status: str
    reason: str
    line: int | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check_id": self.check_id,
            "status": self.status,
            "reason": self.reason,
        }
        if self.line is not None:
            payload["line"] = self.line
        if self.details:
            payload["details"] = self.details
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-0 feasibility gate for the PI05 prefix_cache / past_key_values TensorRT boundary. "
            "The default path runs a real tiny-Gemma cache roundtrip, ONNX export, ORT validation, "
            "and TensorRT parse/build probe."
        )
    )
    parser.add_argument(
        "--modeling-file",
        type=Path,
        default=DEFAULT_MODELING_FILE,
        help="Path to src/lerobot/policies/pi05/modeling_pi05.py",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Optional path to write the JSON feasibility report.",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
        help="Optional path to write the markdown implementation report.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for temporary ONNX artifacts. Defaults to a temporary directory outside the repo.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary ONNX artifacts even when --artifact-dir is not set.",
    )
    parser.add_argument(
        "--skip-package-probe",
        action="store_true",
        help="Skip importlib-based package availability probing in the report header.",
    )
    parser.add_argument(
        "--skip-runtime-probe",
        action="store_true",
        help="Skip the tiny runtime / ONNX / TensorRT feasibility probe and rely on static source analysis only.",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Skip ONNX export and the downstream ORT / TensorRT checks.",
    )
    parser.add_argument(
        "--skip-trt-build",
        action="store_true",
        help="Skip TensorRT parse/build checks after ONNX export.",
    )
    parser.add_argument(
        "--allow-static-go",
        action="store_true",
        help="Allow a `go` verdict from static source checks alone when the runtime probe is intentionally skipped.",
    )
    return parser.parse_args()


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_class_def(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def get_method_def(class_def: ast.ClassDef | None, method_name: str) -> ast.FunctionDef | None:
    if class_def is None:
        return None
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def has_all_snippets(text: str, snippets: list[str]) -> bool:
    return all(snippet in text for snippet in snippets)


def get_method_segment(source: str, method_def: ast.FunctionDef | None) -> str:
    if method_def is None:
        return ""
    return ast.get_source_segment(source, method_def) or ""


def get_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_probe(skip_probe: bool) -> dict[str, dict[str, Any]]:
    packages = ["torch", "onnx", "onnxruntime", "tensorrt", "transformers"]
    results: dict[str, dict[str, Any]] = {}
    for package_name in packages:
        if skip_probe:
            results[package_name] = {"available": None, "detail": "skipped", "version": None}
            continue
        spec = importlib.util.find_spec(package_name)
        results[package_name] = {
            "available": spec is not None,
            "detail": "found" if spec is not None else "missing",
            "version": get_package_version(package_name) if spec is not None else None,
        }
    return results


def analyze_source(modeling_file: Path) -> dict[str, Any]:
    source = read_source(modeling_file)
    tree = ast.parse(source)

    pi_class = get_class_def(tree, "PI05Pytorch")
    bridge_class = get_class_def(tree, "PaliGemmaWithExpertModel")

    embed_prefix = get_method_def(pi_class, "embed_prefix")
    sample_actions = get_method_def(pi_class, "sample_actions")
    denoise_step = get_method_def(pi_class, "denoise_step")
    bridge_forward = get_method_def(bridge_class, "forward")

    embed_prefix_text = get_method_segment(source, embed_prefix)
    sample_actions_text = get_method_segment(source, sample_actions)
    denoise_step_text = get_method_segment(source, denoise_step)
    bridge_forward_text = get_method_segment(source, bridge_forward)

    checks: list[CheckResult] = []

    if embed_prefix is None:
        checks.append(
            CheckResult(
                "embed_prefix_exists",
                "fail",
                "PI05Pytorch.embed_prefix() was not found, so the prefix-cache boundary cannot be inspected.",
            )
        )
    elif has_all_snippets(
        embed_prefix_text,
        [
            "embs = torch.cat(embs, dim=1)",
            "pad_masks = torch.cat(pad_masks, dim=1)",
            "return embs, pad_masks, att_masks",
        ],
    ):
        checks.append(
            CheckResult(
                "embed_prefix_contract",
                "pass",
                "embed_prefix() still exposes `(prefix_embs, prefix_pad_masks, prefix_att_masks)` as a standalone bundle.",
                embed_prefix.lineno,
            )
        )
    else:
        checks.append(
            CheckResult(
                "embed_prefix_contract",
                "fail",
                "embed_prefix() no longer clearly exposes the expected `(embs, pad_masks, att_masks)` prefix contract.",
                embed_prefix.lineno,
            )
        )

    if sample_actions is None:
        checks.append(
            CheckResult(
                "sample_actions_exists",
                "fail",
                "PI05Pytorch.sample_actions() was not found, so the cache-producing path cannot be inspected.",
            )
        )
    elif has_all_snippets(
        sample_actions_text,
        [
            "prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)",
            "prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)",
            "prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)",
            "_, past_key_values = self.paligemma_with_expert.forward(",
            "inputs_embeds=[prefix_embs, None]",
            "use_cache=True",
        ],
    ):
        checks.append(
            CheckResult(
                "prefix_cache_produced",
                "pass",
                "sample_actions() still produces `past_key_values` from a prefix-only forward with a precomputed 4D attention mask.",
                sample_actions.lineno,
            )
        )
    else:
        checks.append(
            CheckResult(
                "prefix_cache_produced",
                "fail",
                "sample_actions() no longer clearly shows the prefix-only cache-producing forward path expected by the hybrid runtime split.",
                sample_actions.lineno,
            )
        )

    if denoise_step is None:
        checks.append(
            CheckResult(
                "denoise_step_exists",
                "fail",
                "PI05Pytorch.denoise_step() was not found, so cache consumption cannot be inspected.",
            )
        )
    elif has_all_snippets(
        denoise_step_text,
        [
            "full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)",
            "full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)",
            "past_key_values=past_key_values",
            "inputs_embeds=[None, suffix_embs]",
            "use_cache=False",
            "return self.action_out_proj(suffix_out)",
        ],
    ):
        checks.append(
            CheckResult(
                "denoise_step_consumes_cache",
                "pass",
                "denoise_step() still consumes a precomputed cache on the suffix-only path with a fixed-shape 4D mask.",
                denoise_step.lineno,
            )
        )
    else:
        checks.append(
            CheckResult(
                "denoise_step_consumes_cache",
                "fail",
                "denoise_step() no longer clearly consumes `past_key_values` on the suffix-only path expected by the split runtime.",
                denoise_step.lineno,
            )
        )

    if bridge_forward is None:
        checks.append(
            CheckResult(
                "bridge_forward_exists",
                "fail",
                "PaliGemmaWithExpertModel.forward() was not found, so the producer/consumer bridge semantics are unknown.",
            )
        )
    elif has_all_snippets(
        bridge_forward_text,
        [
            "past_key_values: list[torch.FloatTensor] | None = None",
            "prefix_output = self.paligemma.language_model.forward(",
            "suffix_output = self.gemma_expert.model.forward(",
            "prefix_past_key_values = prefix_output.past_key_values",
            "return [prefix_output, suffix_output], prefix_past_key_values",
        ],
    ):
        checks.append(
            CheckResult(
                "bridge_forward_cache_contract",
                "pass",
                "The bridge forward path still returns Paligemma cache tensors for later suffix consumption by Gemma expert layers.",
                bridge_forward.lineno,
            )
        )
    else:
        checks.append(
            CheckResult(
                "bridge_forward_cache_contract",
                "warn",
                "The bridge forward path exists, but the producer/consumer cache contract is no longer explicit enough for a frozen export boundary.",
                bridge_forward.lineno,
            )
        )

    flatten_helper_present = "flatten" in source and "past_key_values" in source
    unflatten_helper_present = "unflatten" in source and "past_key_values" in source
    if flatten_helper_present and unflatten_helper_present:
        checks.append(
            CheckResult(
                "flatten_unflatten_helper_in_source",
                "pass",
                "A flatten/unflatten cache helper already exists in the PI05 source tree.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "flatten_unflatten_helper_in_source",
                "warn",
                "No dedicated flatten/unflatten helper is present in modeling_pi05.py, so the export script must freeze the cache tensor schema explicitly.",
            )
        )

    hard_fail = any(check.status == "fail" for check in checks)
    return {
        "checks": [check.as_dict() for check in checks],
        "hard_fail": hard_fail,
        "contract_summary": {
            "prefix_producer": "PI05Pytorch.sample_actions() -> paligemma.language_model.forward(..., use_cache=True)",
            "suffix_consumer": "PI05Pytorch.denoise_step() -> gemma_expert.model.forward(..., past_key_values=...)",
            "attention_mask_shape": "4D additive mask",
            "phase1_shape_policy": "fixed batch=1 / fixed prefix len / fixed suffix len during Stage-0 probe",
        },
    }


def status_rank(status: str) -> int:
    return {
        "pass": 0,
        "warn": 1,
        "skipped": 1,
        "blocked": 2,
        "fail": 3,
    }.get(status, 3)


def max_status(*statuses: str) -> str:
    return max(statuses, key=status_rank)


def tensor_name_for_layer(layer_idx: int, kind: str) -> str:
    return f"past_key_values.layer_{layer_idx:02d}.{kind}"


def build_prefix_causal_mask(torch_module: Any, prefix_len: int) -> Any:
    att_2d = torch_module.tril(torch_module.ones(prefix_len, prefix_len, dtype=torch_module.bool)).unsqueeze(0)
    return prepare_attention_mask_4d(torch_module, att_2d)


def build_suffix_attention_mask(torch_module: Any, batch_size: int, prefix_len: int, suffix_len: int) -> Any:
    prefix_pad = torch_module.ones(batch_size, prefix_len, dtype=torch_module.bool)
    suffix_causal = torch_module.tril(torch_module.ones(suffix_len, suffix_len, dtype=torch_module.bool)).unsqueeze(0)
    prefix_pad_2d = prefix_pad[:, None, :].expand(batch_size, suffix_len, prefix_len)
    full_att_2d = torch_module.cat([prefix_pad_2d, suffix_causal], dim=2)
    return prepare_attention_mask_4d(torch_module, full_att_2d)


def prepare_attention_mask_4d(torch_module: Any, att_2d: Any) -> Any:
    att_4d = att_2d[:, None, :, :]
    false_value = torch_module.tensor(torch_module.finfo(torch_module.float32).min)
    true_value = torch_module.tensor(0.0)
    return torch_module.where(att_4d, true_value, false_value)


def flatten_cache_tensors(cache: Any) -> tuple[list[Any], list[str], list[list[int]]]:
    flat_tensors: list[Any] = []
    flat_names: list[str] = []
    flat_shapes: list[list[int]] = []
    num_layers = len(cache)
    for layer_idx in range(num_layers):
        key_tensor, value_tensor = cache[layer_idx]
        for kind, tensor in (("key", key_tensor), ("value", value_tensor)):
            flat_tensors.append(tensor)
            flat_names.append(tensor_name_for_layer(layer_idx, kind))
            flat_shapes.append([int(dim) for dim in tensor.shape])
    return flat_tensors, flat_names, flat_shapes


def metric_summary_from_diff(diff: Any) -> dict[str, float]:
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def build_runtime_probe(args: argparse.Namespace) -> dict[str, Any]:
    required_packages = ["torch", "onnx", "onnxruntime", "transformers"]
    if not args.skip_trt_build:
        required_packages.append("tensorrt")

    missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
    if missing_packages:
        return {
            "status": "blocked",
            "mode": "tiny_gemma_cross_cache_probe",
            "reason": f"Required runtime packages are missing for the real probe: {', '.join(missing_packages)}.",
            "missing_packages": missing_packages,
        }

    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")
    transformers_cache_utils = importlib.import_module("transformers.cache_utils")
    transformers_gemma = importlib.import_module("transformers.models.gemma")

    DynamicCache = getattr(transformers_cache_utils, "DynamicCache")
    GemmaConfig = getattr(transformers_gemma, "GemmaConfig")
    GemmaForCausalLM = getattr(transformers_gemma, "GemmaForCausalLM")

    probe_config = {
        "batch_size": 1,
        "prefix_len": 5,
        "suffix_len": 3,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 128,
        "dtype": "float32",
        "mask_contract": "4D additive mask",
        "onnx_opset": 17,
    }

    @contextmanager
    def artifact_workspace() -> Iterator[tuple[Path, bool]]:
        if args.artifact_dir is not None:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            yield args.artifact_dir, True
            return
        if args.keep_artifacts:
            artifact_dir = Path(tempfile.mkdtemp(prefix="pi_stage0_feasibility_"))
            yield artifact_dir, True
            return
        with tempfile.TemporaryDirectory(prefix="pi_stage0_feasibility_") as temp_dir:
            yield Path(temp_dir), False

    torch.manual_seed(0)

    cfg = GemmaConfig(
        hidden_size=probe_config["hidden_size"],
        intermediate_size=probe_config["intermediate_size"],
        num_hidden_layers=probe_config["num_hidden_layers"],
        num_attention_heads=probe_config["num_attention_heads"],
        num_key_value_heads=probe_config["num_key_value_heads"],
        head_dim=probe_config["head_dim"],
        vocab_size=probe_config["vocab_size"],
    )

    producer = GemmaForCausalLM(cfg).model.eval().cpu()
    consumer = GemmaForCausalLM(cfg).model.eval().cpu()
    producer.config._attn_implementation = "eager"  # noqa: SLF001
    consumer.config._attn_implementation = "eager"  # noqa: SLF001

    batch_size = probe_config["batch_size"]
    prefix_len = probe_config["prefix_len"]
    suffix_len = probe_config["suffix_len"]
    hidden_size = probe_config["hidden_size"]

    prefix_embs = torch.randn(batch_size, prefix_len, hidden_size, dtype=torch.float32)
    prefix_mask_4d = build_prefix_causal_mask(torch, prefix_len).to(dtype=torch.float32)
    prefix_position_ids = torch.arange(prefix_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        prefix_out = producer(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_mask_4d,
            position_ids=prefix_position_ids,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = prefix_out.past_key_values
    flat_cache, flat_cache_names, flat_cache_shapes = flatten_cache_tensors(past_key_values)
    roundtrip_cache = DynamicCache.from_legacy_cache(
        tuple((flat_cache[idx], flat_cache[idx + 1]) for idx in range(0, len(flat_cache), 2))
    )

    suffix_embs = torch.randn(batch_size, suffix_len, hidden_size, dtype=torch.float32)
    suffix_mask_4d = build_suffix_attention_mask(torch, batch_size, prefix_len, suffix_len).to(dtype=torch.float32)
    suffix_position_ids = (prefix_len + torch.arange(suffix_len, dtype=torch.long)).unsqueeze(0)

    with torch.no_grad():
        original_consumer_out = consumer(
            inputs_embeds=suffix_embs,
            attention_mask=suffix_mask_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        roundtrip_consumer_out = consumer(
            inputs_embeds=suffix_embs,
            attention_mask=suffix_mask_4d,
            position_ids=suffix_position_ids,
            past_key_values=roundtrip_cache,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    cache_roundtrip_diff = (original_consumer_out - roundtrip_consumer_out).abs()
    cache_roundtrip_metrics = metric_summary_from_diff(cache_roundtrip_diff)
    cache_roundtrip_status = "pass" if cache_roundtrip_metrics["max_abs_diff"] <= 1e-7 else "fail"

    runtime_report: dict[str, Any] = {
        "status": cache_roundtrip_status,
        "mode": "tiny_gemma_cross_cache_probe",
        "reason": (
            "Tiny Gemma prefix->cache->suffix roundtrip completed."
            if cache_roundtrip_status == "pass"
            else "Flattened cache roundtrip changed the suffix output."
        ),
        "probe_config": probe_config,
        "cache_contract": {
            "cache_type": type(past_key_values).__name__,
            "num_layers": len(past_key_values),
            "flattened_tensor_names": flat_cache_names,
            "flattened_tensor_shapes": flat_cache_shapes,
        },
        "torch_cache_roundtrip": {
            "status": cache_roundtrip_status,
            **cache_roundtrip_metrics,
        },
    }

    if args.skip_onnx_export:
        runtime_report["status"] = max_status(runtime_report["status"], "blocked")
        runtime_report["onnx_prefix"] = {"status": "skipped", "reason": "ONNX export skipped by CLI flag."}
        runtime_report["onnx_consumer"] = {"status": "skipped", "reason": "ONNX export skipped by CLI flag."}
        runtime_report["trt_prefix"] = {"status": "skipped", "reason": "TensorRT probe depends on ONNX export."}
        runtime_report["trt_consumer"] = {"status": "skipped", "reason": "TensorRT probe depends on ONNX export."}
        return runtime_report

    class PrefixCacheWrapper(torch.nn.Module):
        def __init__(self, model: Any, num_layers: int):
            super().__init__()
            self.model = model
            self.num_layers = num_layers

        def forward(self, prefix_embs: Any, attention_mask: Any, position_ids: Any) -> tuple[Any, ...]:
            out = self.model(
                inputs_embeds=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            outputs: list[Any] = []
            for layer_idx in range(self.num_layers):
                key_tensor, value_tensor = out.past_key_values[layer_idx]
                outputs.extend([key_tensor, value_tensor])
            return tuple(outputs)

    class ConsumerWrapper(torch.nn.Module):
        def __init__(self, model: Any, dynamic_cache_cls: Any):
            super().__init__()
            self.model = model
            self.dynamic_cache_cls = dynamic_cache_cls

        def forward(self, suffix_embs: Any, attention_mask: Any, position_ids: Any, *flat_cache_inputs: Any) -> Any:
            legacy_cache = []
            for idx in range(0, len(flat_cache_inputs), 2):
                legacy_cache.append((flat_cache_inputs[idx], flat_cache_inputs[idx + 1]))
            cache = self.dynamic_cache_cls.from_legacy_cache(tuple(legacy_cache))
            out = self.model(
                inputs_embeds=suffix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
            )
            return out.last_hidden_state

    prefix_input_names = ["prefix_embs", "attention_mask", "position_ids"]
    consumer_input_names = ["suffix_embs", "attention_mask", "position_ids", *flat_cache_names]
    consumer_output_names = ["suffix_hidden_state"]

    prefix_feed = {
        "prefix_embs": prefix_embs.numpy(),
        "attention_mask": prefix_mask_4d.numpy(),
        "position_ids": prefix_position_ids.numpy(),
    }

    consumer_feed = {
        "suffix_embs": suffix_embs.numpy(),
        "attention_mask": suffix_mask_4d.numpy(),
        "position_ids": suffix_position_ids.numpy(),
    }

    prefix_onnx_status = "pass"
    consumer_onnx_status = "pass"
    trt_prefix_status = "skipped"
    trt_consumer_status = "skipped"

    with artifact_workspace() as (artifact_dir, artifacts_preserved):
        prefix_onnx_path = artifact_dir / "stage0_prefix_cache_probe.onnx"
        consumer_onnx_path = artifact_dir / "stage0_suffix_consumer_probe.onnx"

        torch.onnx.export(
            PrefixCacheWrapper(producer, len(past_key_values)).eval(),
            (prefix_embs, prefix_mask_4d, prefix_position_ids),
            prefix_onnx_path.as_posix(),
            input_names=prefix_input_names,
            output_names=flat_cache_names,
            opset_version=probe_config["onnx_opset"],
        )
        onnx.checker.check_model(prefix_onnx_path.as_posix())
        prefix_session = ort.InferenceSession(prefix_onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        prefix_ort_outputs = prefix_session.run(flat_cache_names, prefix_feed)
        prefix_ort_diffs = [
            float(np.max(np.abs(prefix_ort_outputs[idx] - flat_cache[idx].numpy()))) for idx in range(len(flat_cache))
        ]
        prefix_max_abs_diff = max(prefix_ort_diffs) if prefix_ort_diffs else 0.0
        prefix_onnx_status = "pass" if prefix_max_abs_diff <= 1e-5 else "fail"
        runtime_report["onnx_prefix"] = {
            "status": prefix_onnx_status,
            "path": str(prefix_onnx_path),
            "inputs": prefix_input_names,
            "outputs": flat_cache_names,
            "num_outputs": len(prefix_ort_outputs),
            "max_abs_diff": float(prefix_max_abs_diff),
            "max_abs_diff_per_output": prefix_ort_diffs,
            "artifacts_preserved": artifacts_preserved,
        }

        for name, value in zip(flat_cache_names, prefix_ort_outputs, strict=True):
            consumer_feed[name] = value

        torch.onnx.export(
            ConsumerWrapper(consumer, DynamicCache).eval(),
            (suffix_embs, suffix_mask_4d, suffix_position_ids, *flat_cache),
            consumer_onnx_path.as_posix(),
            input_names=consumer_input_names,
            output_names=consumer_output_names,
            opset_version=probe_config["onnx_opset"],
        )
        onnx.checker.check_model(consumer_onnx_path.as_posix())
        consumer_session = ort.InferenceSession(consumer_onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        consumer_ort_output = consumer_session.run(consumer_output_names, consumer_feed)[0]
        consumer_diff = np.abs(consumer_ort_output - original_consumer_out.numpy())
        consumer_onnx_status = "pass" if float(np.max(consumer_diff)) <= 1e-5 else "fail"
        runtime_report["onnx_consumer"] = {
            "status": consumer_onnx_status,
            "path": str(consumer_onnx_path),
            "inputs": consumer_input_names,
            "outputs": consumer_output_names,
            "max_abs_diff": float(np.max(consumer_diff)),
            "mean_abs_diff": float(np.mean(consumer_diff)),
            "artifacts_preserved": artifacts_preserved,
        }

        if args.skip_trt_build:
            runtime_report["trt_prefix"] = {"status": "skipped", "reason": "TensorRT probe skipped by CLI flag."}
            runtime_report["trt_consumer"] = {"status": "skipped", "reason": "TensorRT probe skipped by CLI flag."}
            runtime_report["status"] = max_status(
                runtime_report["status"],
                prefix_onnx_status,
                consumer_onnx_status,
                "blocked",
            )
            return runtime_report

        trt = importlib.import_module("tensorrt")

        def trt_probe(onnx_path: Path) -> dict[str, Any]:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            parse_ok = parser.parse_from_file(onnx_path.as_posix())
            errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
            result = {
                "status": "fail" if not parse_ok else "pass",
                "parse_success": bool(parse_ok),
                "num_errors": int(parser.num_errors),
                "errors": errors,
                "network_inputs": [network.get_input(idx).name for idx in range(network.num_inputs)],
                "network_outputs": [network.get_output(idx).name for idx in range(network.num_outputs)],
                "build_success": False,
                "fp16_enabled": False,
            }
            if not parse_ok:
                return result

            builder_config = builder.create_builder_config()
            fp16_enabled = bool(builder.platform_has_fast_fp16)
            if fp16_enabled:
                builder_config.set_flag(trt.BuilderFlag.FP16)
            engine_blob = builder.build_serialized_network(network, builder_config)
            build_success = engine_blob is not None
            result["status"] = "pass" if build_success else "fail"
            result["build_success"] = build_success
            result["fp16_enabled"] = fp16_enabled
            result["engine_blob_type"] = type(engine_blob).__name__ if engine_blob is not None else None
            return result

        runtime_report["trt_prefix"] = trt_probe(prefix_onnx_path)
        runtime_report["trt_consumer"] = trt_probe(consumer_onnx_path)
        trt_prefix_status = runtime_report["trt_prefix"]["status"]
        trt_consumer_status = runtime_report["trt_consumer"]["status"]

    runtime_report["status"] = max_status(
        runtime_report["status"],
        prefix_onnx_status,
        consumer_onnx_status,
        trt_prefix_status,
        trt_consumer_status,
    )
    return runtime_report


def decide_verdict(
    source_analysis: dict[str, Any],
    runtime_probe: dict[str, Any] | None,
    package_results: dict[str, dict[str, Any]],
    allow_static_go: bool,
    skip_runtime_probe: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    failing_source_checks = [check for check in source_analysis["checks"] if check["status"] == "fail"]
    if failing_source_checks:
        reasons.extend(check["reason"] for check in failing_source_checks)
        return "no-go", reasons

    package_missing = [name for name, result in package_results.items() if result["available"] is False]
    if package_missing and skip_runtime_probe:
        reasons.append(
            f"Static analysis passed, but package availability is incomplete for a real probe: {', '.join(package_missing)}."
        )
        return "blocked", reasons

    if runtime_probe is None:
        if allow_static_go and not package_missing:
            reasons.append("Static source checks passed and `--allow-static-go` was set.")
            return "go", reasons
        reasons.append("Static source checks passed, but the real runtime probe was skipped.")
        return "blocked", reasons

    reasons.append(runtime_probe.get("reason", "Runtime probe completed."))

    runtime_status = runtime_probe.get("status", "fail")
    if runtime_status == "pass":
        if any(check["status"] == "warn" for check in source_analysis["checks"]):
            reasons.append("Static source checks still report schema/documentation warnings that must be frozen in metadata.")
        reasons.append("Tiny Gemma roundtrip, ONNX export, ORT validation, and TensorRT parse/build all succeeded.")
        return "go", reasons

    if runtime_status == "blocked":
        reasons.append("The runtime probe could not complete because of environment or CLI gating constraints.")
        return "blocked", reasons

    reasons.append("The runtime probe exercised the contract directly and found a concrete failure.")
    return "no-go", reasons


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source_analysis = analyze_source(args.modeling_file)
    packages = package_probe(args.skip_package_probe)
    runtime_probe = None if args.skip_runtime_probe else build_runtime_probe(args)
    verdict, reasons = decide_verdict(
        source_analysis=source_analysis,
        runtime_probe=runtime_probe,
        package_results=packages,
        allow_static_go=args.allow_static_go,
        skip_runtime_probe=args.skip_runtime_probe,
    )

    phase1_risks = [
        "The probe uses a tiny synthetic Gemma configuration, so it proves cache contract viability rather than checkpoint-specific numerical fidelity.",
        "The ONNX and TensorRT checks use fixed shapes only; dynamic prefix length, tokenizer padding variance, and multi-camera prefixes are still out of scope.",
        "Stage-0 proves producer/consumer cache compatibility, but it does not export the real PI05 checkpoint or validate real processor-generated embeddings.",
    ]

    next_steps = [
        "Freeze the flattened cache schema in the export metadata using `past_key_values.layer_{layer_idx}.key/value` naming.",
        "Build the real PI05 prefix-cache export wrapper around the existing 4D attention-mask contract from modeling_pi05.py.",
        "Reuse the same cache names and shapes in the downstream denoise-step exporter and checkpoint inspection metadata.",
    ]

    return {
        "stage": "stage-0-feasibility-gate",
        "subject": "pi05 prefix_cache / past_key_values contract",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reason_count": len(reasons),
        "reasons": reasons,
        "source_modeling_file": str(args.modeling_file),
        "package_probe": packages,
        "source_analysis": source_analysis,
        "runtime_probe": runtime_probe,
        "phase1_risks": phase1_risks,
        "next_steps": next_steps,
    }


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any], json_output: Path | None) -> str:
    lines = [
        "# Worker 2 Feasibility Report",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Subject: `{report['subject']}`",
        f"- Modeling file: `{report['source_modeling_file']}`",
    ]
    if json_output is not None:
        lines.append(f"- JSON output: `{json_output}`")

    lines.extend(["", "## Why"])
    for reason in report["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(["", "## Source Checks"])
    for check in report["source_analysis"]["checks"]:
        line_suffix = f" (line {check['line']})" if "line" in check else ""
        lines.append(f"- `{check['status']}` `{check['check_id']}`{line_suffix}: {check['reason']}")

    runtime_probe = report.get("runtime_probe")
    lines.extend(["", "## Runtime Probe"])
    if runtime_probe is None:
        lines.append("- `skipped`: Runtime probe was not executed.")
    else:
        lines.append(f"- Status: `{runtime_probe['status']}`")
        lines.append(f"- Mode: `{runtime_probe.get('mode', 'n/a')}`")
        lines.append(f"- Reason: {runtime_probe.get('reason', 'n/a')}")
        if "cache_contract" in runtime_probe:
            lines.append(f"- Cache type: `{runtime_probe['cache_contract']['cache_type']}`")
            lines.append(f"- Flattened tensors: `{len(runtime_probe['cache_contract']['flattened_tensor_names'])}`")
        if "torch_cache_roundtrip" in runtime_probe:
            lines.append(
                "- Torch roundtrip: "
                f"`{runtime_probe['torch_cache_roundtrip']['status']}` "
                f"(max_abs_diff={runtime_probe['torch_cache_roundtrip']['max_abs_diff']:.3e}, "
                f"mean_abs_diff={runtime_probe['torch_cache_roundtrip']['mean_abs_diff']:.3e})"
            )
        if "onnx_prefix" in runtime_probe:
            lines.append(
                "- ONNX prefix export: "
                f"`{runtime_probe['onnx_prefix']['status']}` "
                f"(max_abs_diff={runtime_probe['onnx_prefix'].get('max_abs_diff', 0.0):.3e})"
            )
        if "onnx_consumer" in runtime_probe:
            lines.append(
                "- ONNX consumer export: "
                f"`{runtime_probe['onnx_consumer']['status']}` "
                f"(max_abs_diff={runtime_probe['onnx_consumer'].get('max_abs_diff', 0.0):.3e})"
            )
        if "trt_prefix" in runtime_probe:
            lines.append(f"- TensorRT prefix: `{runtime_probe['trt_prefix']['status']}`")
        if "trt_consumer" in runtime_probe:
            lines.append(f"- TensorRT consumer: `{runtime_probe['trt_consumer']['status']}`")

    lines.extend(["", "## Phase-1 Risks"])
    for risk in report["phase1_risks"]:
        lines.append(f"- {risk}")

    lines.extend(["", "## Next Steps"])
    for step in report["next_steps"]:
        lines.append(f"- {step}")

    if runtime_probe and "cache_contract" in runtime_probe:
        lines.extend(
            [
                "",
                "## Proposed Flattened Cache Names",
                "",
                "```json",
                json.dumps(runtime_probe["cache_contract"]["flattened_tensor_names"], indent=2, ensure_ascii=True),
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def write_markdown(path: Path | None, report: dict[str, Any], json_output: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report, json_output), encoding="utf-8")


def build_unexpected_error_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "stage": "stage-0-feasibility-gate",
        "subject": "pi05 prefix_cache / past_key_values contract",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "blocked",
        "reason_count": 1,
        "reasons": [f"Unexpected error while running the feasibility probe: {type(exc).__name__}: {exc}"],
        "source_modeling_file": str(args.modeling_file),
        "package_probe": package_probe(skip_probe=False),
        "source_analysis": {"checks": [], "hard_fail": False},
        "runtime_probe": {
            "status": "blocked",
            "mode": "unexpected_exception",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        },
        "phase1_risks": ["The probe failed unexpectedly before a stable go/no-go judgment could be completed."],
        "next_steps": ["Inspect the traceback and rerun the stage-0 probe once the immediate failure is resolved."],
    }


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except Exception as exc:  # noqa: BLE001
        report = build_unexpected_error_report(args, exc)
        write_json(args.json_output, report)
        write_markdown(args.report_markdown, report, args.json_output)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 3

    write_json(args.json_output, report)
    write_markdown(args.report_markdown, report, args.json_output)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return VERDICT_EXIT_CODES.get(report["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
