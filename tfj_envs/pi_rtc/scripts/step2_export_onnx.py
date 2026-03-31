#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import build_metadata_skeleton, metadata_path, prepare_run_layout, resolve_checkpoint_dir, write_json
from export_subgraphs import export_all_subgraphs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: export PI05 ONNX subgraphs.")
    parser.add_argument("--policy-path", required=True, help="Checkpoint root or pretrained_model/ path.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory.")
    parser.add_argument("--onnx-dir", default=None, help="Explicit ONNX output directory.")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version.")
    parser.add_argument("--strict-load", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--report-path", default=None, help="Explicit JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = prepare_run_layout(args.run_dir, prefix="pi05_trt")
    run_dir = layout["run_dir"]
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else layout["onnx_dir"]
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (run_dir / "stage2_export_onnx.json")
    )
    policy_dir = resolve_checkpoint_dir(args.policy_path)

    export_summary = export_all_subgraphs(
        policy_path=policy_dir,
        onnx_dir=onnx_dir,
        opset=args.opset,
        strict=args.strict_load,
    )
    stage2_acceptance = export_summary["stage2_acceptance"]
    stage2_status = stage2_acceptance["status"]

    metadata_file = metadata_path(run_dir)
    if metadata_file.is_file():
        metadata = __import__("json").loads(metadata_file.read_text(encoding="utf-8"))
    else:
        metadata = build_metadata_skeleton(run_dir=run_dir, variant="pi05", checkpoint_dir=policy_dir)
    metadata["checkpoint_dir"] = policy_dir.as_posix()
    metadata["onnx_paths"] = export_summary["onnx_paths"]
    metadata["export_routes"] = export_summary.get("export_routes", {})
    metadata["local_tokenizer_path"] = export_summary["metadata_notes"]["local_tokenizer_path"]
    metadata["runtime_reference_contract"] = export_summary["runtime_reference_contract"]
    metadata["export_reference_contract"] = export_summary["export_reference_contract"]
    metadata["torch_reference_contract"] = export_summary["runtime_reference_contract"]
    metadata["onnx_contract"] = export_summary["onnx_contract"]
    metadata["contract_checks"] = export_summary["contract_checks"]
    metadata["compare_basis"] = export_summary["compare_basis"]
    metadata["analysis_notes"] = export_summary["analysis_notes"]
    metadata["onnx_file_sizes"] = export_summary["onnx_file_sizes"]
    metadata["stage2_immediate_onnx_compare"] = export_summary["immediate_onnx_compare"]
    metadata["stage2_acceptance"] = stage2_acceptance
    metadata.setdefault("stage_status", {})["stage2_export_onnx"] = stage2_status
    metadata.setdefault("validation_gates", {})["stage2_export_onnx"] = {
        "status": stage2_status,
        "hard_fail": bool(stage2_acceptance["hard_fail"]),
        "failed_checks": list(stage2_acceptance["failed_checks"]),
        "report_path": report_path.as_posix(),
    }
    if stage2_status == "pass":
        metadata["last_completed_stage"] = "stage2_export_onnx"

    report = {
        "stage": "stage2_export_onnx",
        "overall_status": stage2_status,
        "policy_dir": policy_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "onnx_dir": onnx_dir.as_posix(),
        "opset": args.opset,
        "strict_load": bool(args.strict_load),
        "onnx_paths": export_summary["onnx_paths"],
        "export_routes": export_summary.get("export_routes", {}),
        "onnx_file_sizes": export_summary["onnx_file_sizes"],
        "runtime_reference_contract": export_summary["runtime_reference_contract"],
        "export_reference_contract": export_summary["export_reference_contract"],
        "onnx_contract": export_summary["onnx_contract"],
        "contract_checks": export_summary["contract_checks"],
        "compare_basis": export_summary["compare_basis"],
        "analysis_notes": export_summary["analysis_notes"],
        "torch_reference": export_summary["torch_reference"],
        "immediate_onnx_compare": export_summary["immediate_onnx_compare"],
        "stage2_acceptance": stage2_acceptance,
        "metadata_notes": export_summary["metadata_notes"],
    }
    write_json(report_path, report)
    write_json(metadata_file, metadata)
    print(f"[{stage2_status.upper()}] Stage 2 report written to: {report_path}")
    return 0 if stage2_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
