#!/usr/bin/env python3

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from build_pi_trt_engine import build_engine_from_onnx
from common import build_metadata_skeleton, metadata_path, prepare_run_layout, read_json, write_json


SUBGRAPH_FILES = {
    "vision_encoder": {
        "onnx": "pi_shared_vision_encoder.onnx",
        "engine": "pi_shared_vision_encoder.engine",
    },
    "prefix_cache": {
        "onnx": "pi_shared_prefix_cache.onnx",
        "engine": "pi_shared_prefix_cache.engine",
    },
    "denoise_step": {
        "onnx": "pi05_denoise_step.onnx",
        "engine": "pi05_denoise_step.engine",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4: build PI05 TensorRT engines.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument("--engine-dir", default=None)
    parser.add_argument(
        "--subgraphs",
        nargs="+",
        choices=sorted(SUBGRAPH_FILES),
        default=list(SUBGRAPH_FILES),
        help="Subset of PI05 subgraphs to build. Default: all three.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 math during fp32 fallback paths. Disabled by default to reduce drift.",
    )
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument(
        "--force-fp32-layer-types",
        nargs="*",
        default=None,
        help="Optional TensorRT layer types to force to fp32 while building lower-precision engines.",
    )
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-cache-dir", default=None)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after the first failed engine build instead of collecting partial results.",
    )
    return parser


def _resolve_artifact_paths(
    *,
    run_dir: Path,
    onnx_dir: Path,
    engine_dir: Path,
    onnx_dir_explicit: bool,
    engine_dir_explicit: bool,
) -> tuple[dict[str, Path], dict[str, Path]]:
    onnx_paths: dict[str, Path] = {}
    engine_paths: dict[str, Path] = {}
    metadata_file = metadata_path(run_dir)
    metadata = read_json(metadata_file) if metadata_file.is_file() else {}

    metadata_onnx = metadata.get("onnx_paths", {})
    metadata_engines = metadata.get("engine_paths", {})
    for subgraph, files in SUBGRAPH_FILES.items():
        if onnx_dir_explicit:
            onnx_candidate = (onnx_dir / files["onnx"]).resolve()
        else:
            onnx_candidate = Path(metadata_onnx.get(subgraph, onnx_dir / files["onnx"]))
        if engine_dir_explicit:
            engine_candidate = (engine_dir / files["engine"]).resolve()
        else:
            engine_candidate = Path(metadata_engines.get(subgraph, engine_dir / files["engine"]))
        if onnx_candidate.is_absolute():
            onnx_paths[subgraph] = onnx_candidate
        else:
            onnx_paths[subgraph] = (onnx_dir / onnx_candidate).resolve()
        if engine_candidate.is_absolute():
            engine_paths[subgraph] = engine_candidate
        else:
            engine_paths[subgraph] = (engine_dir / engine_candidate).resolve()

    return onnx_paths, engine_paths


def _overall_status(results: dict[str, dict[str, Any]]) -> str:
    statuses = [entry["status"] for entry in results.values()]
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _stage_report_path(run_dir: Path, metadata: dict[str, Any], stage_name: str) -> Path:
    gate_entry = metadata.get("validation_gates", {}).get(stage_name, {})
    explicit_report_path = gate_entry.get("report_path")
    if explicit_report_path:
        return Path(explicit_report_path).expanduser().resolve()
    return (run_dir / f"{stage_name}.json").resolve()


def _stage_gate_payload(stage_name: str, report_payload: dict[str, Any]) -> dict[str, Any] | None:
    if stage_name == "stage2_export_onnx":
        return report_payload.get("stage2_acceptance")
    if stage_name == "stage3_verify_onnx":
        return report_payload.get("stage3_acceptance")
    return None


def _load_stage_gate(stage_name: str, *, run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    report_path = _stage_report_path(run_dir, metadata, stage_name)
    report_payload = read_json(report_path) if report_path.is_file() else {}
    metadata_gate = metadata.get("validation_gates", {}).get(stage_name, {})
    report_gate = _stage_gate_payload(stage_name, report_payload) or {}
    metadata_stage_status = metadata.get("stage_status", {}).get(stage_name)

    if metadata_gate.get("status"):
        status = metadata_gate["status"]
        source = "metadata.validation_gates"
    elif report_gate.get("status"):
        status = report_gate["status"]
        source = "stage_report.acceptance"
    elif metadata_stage_status:
        status = metadata_stage_status
        source = "metadata.stage_status"
    elif report_payload.get("overall_status"):
        status = report_payload["overall_status"]
        source = "stage_report.overall_status"
    else:
        status = "missing"
        source = "unavailable"

    return {
        "stage": stage_name,
        "status": status,
        "source": source,
        "report_path": report_path.as_posix(),
        "report_found": report_path.is_file(),
        "metadata_gate": metadata_gate,
        "report_gate": report_gate,
    }


def _upstream_gate_check(*, run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    stages = {
        stage_name: _load_stage_gate(stage_name, run_dir=run_dir, metadata=metadata)
        for stage_name in ("stage2_export_onnx", "stage3_verify_onnx")
    }
    blocking_reasons = []
    for stage_name, gate in stages.items():
        if gate["status"] != "pass":
            blocking_reasons.append(
                f"{stage_name} gate status is {gate['status']} (source={gate['source']}, report={gate['report_path']})"
            )

    return {
        "status": "pass" if not blocking_reasons else "fail",
        "stages": stages,
        "blocking_reasons": blocking_reasons,
    }


def main() -> int:
    args = build_parser().parse_args()
    layout = prepare_run_layout(args.run_dir, prefix="pi05_trt")
    run_dir = layout["run_dir"]
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else layout["onnx_dir"]
    engine_dir = Path(args.engine_dir).expanduser().resolve() if args.engine_dir else layout["engines_dir"]
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (run_dir / "stage4_build_engines.json")
    )
    timing_cache_dir = (
        Path(args.timing_cache_dir).expanduser().resolve()
        if args.timing_cache_dir
        else (engine_dir / "timing_cache")
    )

    onnx_paths, engine_paths = _resolve_artifact_paths(
        run_dir=run_dir,
        onnx_dir=onnx_dir,
        engine_dir=engine_dir,
        onnx_dir_explicit=bool(args.onnx_dir),
        engine_dir_explicit=bool(args.engine_dir),
    )
    metadata_file = metadata_path(run_dir)
    if metadata_file.is_file():
        metadata = read_json(metadata_file)
    else:
        metadata = build_metadata_skeleton(run_dir=run_dir, variant="pi05")
    upstream_gate = _upstream_gate_check(run_dir=run_dir, metadata=metadata)
    results: dict[str, dict[str, Any]] = {}
    selected_subgraphs = list(dict.fromkeys(args.subgraphs))

    if upstream_gate["status"] != "pass":
        report = {
            "stage": "stage4_build_engines",
            "variant": metadata.get("variant"),
            "checkpoint_dir": metadata.get("checkpoint_dir"),
            "requested_precision": args.precision,
            "run_dir": run_dir.as_posix(),
            "onnx_dir": onnx_dir.as_posix(),
            "engine_dir": engine_dir.as_posix(),
            "requested_subgraphs": selected_subgraphs,
            "build_settings": {
                "precision": args.precision,
                "allow_tf32": bool(args.allow_tf32),
                "workspace_gb": float(args.workspace_gb),
                "opt_level": int(args.opt_level),
                "force_fp32_layer_types": [name.upper() for name in (args.force_fp32_layer_types or [])],
                "device": args.device,
                "timing_cache_dir": timing_cache_dir.as_posix(),
                "stop_on_error": bool(args.stop_on_error),
            },
            "artifact_paths": {
                subgraph: {
                    "onnx": onnx_paths[subgraph].as_posix(),
                    "engine": engine_paths[subgraph].as_posix(),
                }
                for subgraph in selected_subgraphs
            },
            "results": results,
            "upstream_gate": upstream_gate,
            "overall_status": "fail",
            "all_succeeded": False,
        }
        write_json(report_path, report)

        metadata.setdefault("stage_status", {})["stage4_build_engines"] = "fail"
        metadata.setdefault("artifacts", {})["stage4_build_engines"] = report_path.as_posix()
        metadata["requested_trt_precision"] = args.precision
        metadata.setdefault("validation_gates", {})["stage4_build_engines"] = {
            "status": "fail",
            "hard_fail": True,
            "failed_checks": list(upstream_gate["blocking_reasons"]),
            "report_path": report_path.as_posix(),
        }
        write_json(metadata_file, metadata)

        print(f"[FAIL] Stage 4 refused to build engines: {report_path}")
        return 1

    engine_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_dir.mkdir(parents=True, exist_ok=True)

    for subgraph in selected_subgraphs:
        onnx_path = onnx_paths[subgraph]
        engine_path = engine_paths[subgraph]
        per_engine_report_path = engine_dir / f"{subgraph}_build_report.json"
        timing_cache_path = timing_cache_dir / f"{subgraph}.timing.cache"
        try:
            build_report = build_engine_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision=args.precision,
                allow_tf32=bool(args.allow_tf32),
                workspace_gb=args.workspace_gb,
                opt_level=args.opt_level,
                device=args.device,
                timing_cache_path=timing_cache_path,
                force_fp32_layer_types=args.force_fp32_layer_types,
            )
            build_report["status"] = "pass"
            build_report["subgraph"] = subgraph
            write_json(per_engine_report_path, build_report)
            results[subgraph] = {
                "status": "pass",
                "onnx_path": onnx_path.as_posix(),
                "engine_path": engine_path.as_posix(),
                "timing_cache_path": timing_cache_path.as_posix(),
                "report_path": per_engine_report_path.as_posix(),
                "build_report": build_report,
            }
        except Exception as exc:
            error_payload = {
                "status": "error",
                "onnx_path": onnx_path.as_posix(),
                "engine_path": engine_path.as_posix(),
                "timing_cache_path": timing_cache_path.as_posix(),
                "report_path": per_engine_report_path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            write_json(per_engine_report_path, error_payload)
            results[subgraph] = error_payload
            if args.stop_on_error:
                break

    overall_status = _overall_status(results)
    report = {
        "stage": "stage4_build_engines",
        "variant": metadata.get("variant"),
        "checkpoint_dir": metadata.get("checkpoint_dir"),
        "requested_precision": args.precision,
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "requested_subgraphs": selected_subgraphs,
        "build_settings": {
            "precision": args.precision,
            "allow_tf32": bool(args.allow_tf32),
            "workspace_gb": float(args.workspace_gb),
            "opt_level": int(args.opt_level),
            "force_fp32_layer_types": [name.upper() for name in (args.force_fp32_layer_types or [])],
            "device": args.device,
            "timing_cache_dir": timing_cache_dir.as_posix(),
            "stop_on_error": bool(args.stop_on_error),
        },
        "artifact_paths": {
            subgraph: {
                "onnx": onnx_paths[subgraph].as_posix(),
                "engine": engine_paths[subgraph].as_posix(),
            }
            for subgraph in selected_subgraphs
        },
        "results": results,
        "upstream_gate": upstream_gate,
        "overall_status": overall_status,
        "all_succeeded": all(entry["status"] == "pass" for entry in results.values()),
    }
    write_json(report_path, report)

    metadata.setdefault("onnx_paths", {})
    metadata.setdefault("engine_paths", {})
    metadata.setdefault("artifacts", {})["stage4_build_engines"] = report_path.as_posix()
    for subgraph in selected_subgraphs:
        metadata["onnx_paths"][subgraph] = onnx_paths[subgraph].as_posix()
        if results.get(subgraph, {}).get("status") == "pass":
            metadata["engine_paths"][subgraph] = engine_paths[subgraph].as_posix()
    metadata["engine_build_settings"] = report["build_settings"]
    metadata["requested_trt_precision"] = args.precision
    metadata["trt_effective_precision_evidence"] = {
        subgraph: result.get("build_report", {}).get("effective_precision_evidence")
        for subgraph, result in results.items()
        if result.get("status") == "pass"
    }
    metadata.setdefault("stage_status", {})["stage4_build_engines"] = overall_status
    metadata.setdefault("validation_gates", {})["stage4_build_engines"] = {
        "status": overall_status,
        "hard_fail": overall_status in {"fail", "error"},
        "failed_checks": [
            subgraph
            for subgraph, result in results.items()
            if result.get("status") != "pass"
        ],
        "report_path": report_path.as_posix(),
    }
    if overall_status == "pass":
        metadata["last_completed_stage"] = "stage4_build_engines"
    write_json(metadata_file, metadata)

    print(f"[{overall_status.upper()}] Stage 4 report written to: {report_path}")
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
