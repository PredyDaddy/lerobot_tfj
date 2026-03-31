from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch


PI_TRT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PI_TRT_ROOT / "scripts"

scripts_dir_str = str(SCRIPTS_DIR)
if scripts_dir_str not in sys.path:
    sys.path.insert(0, scripts_dir_str)

import pi05_chunk_runtime as chunk_runtime
import run_pi05_onnx_infer_so101 as onnx_launcher


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_stub_file(path: Path, content: str = "stub\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_policy_dir(root: Path, name: str) -> Path:
    policy_dir = root / name
    policy_dir.mkdir(parents=True, exist_ok=True)
    _write_json(policy_dir / "config.json", {"type": "pi05"})
    return policy_dir


def _make_validate_args(
    *,
    policy_path: Path,
    calib_dir: Path,
    onnx_path: Path,
    stage2_report_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        policy_path=str(policy_path),
        robot_calibration_dir=str(calib_dir),
        onnx_path=str(onnx_path),
        onnx_stage2_report_path=str(stage2_report_path),
        local_tokenizer_path=None,
    )


def _assert_message_has_any(exc: BaseException, candidates: tuple[str, ...]) -> None:
    message = str(exc).lower()
    assert any(candidate in message for candidate in candidates), message


def _write_stage_reports(
    *,
    run_dir: Path,
    onnx_dir: Path,
    stage2_policy_dir: Path,
    onnx_paths: dict[str, Path],
    stage3_policy_dir: Path | None = None,
) -> tuple[Path, Path]:
    stage2_report = _write_json(
        run_dir / "stage2_export_onnx.json",
        {
            "stage": "stage2_export_onnx",
            "overall_status": "pass",
            "policy_dir": str(stage2_policy_dir),
            "run_dir": str(run_dir),
            "onnx_dir": str(onnx_dir),
            "onnx_paths": {name: str(path) for name, path in onnx_paths.items()},
        },
    )
    stage3_report = _write_json(
        run_dir / "stage3_verify_onnx.json",
        {
            "stage": "stage3_verify_onnx",
            "overall_status": "pass",
            "policy_dir": str(stage3_policy_dir or stage2_policy_dir),
            "run_dir": str(run_dir),
            "onnx_dir": str(onnx_dir),
            "artifact_paths": {name: str(path) for name, path in onnx_paths.items()},
            "stage2_context": {
                "stage2_report_path": str(stage2_report),
                "stage2_onnx_paths": {name: str(path) for name, path in onnx_paths.items()},
            },
        },
    )
    return stage2_report, stage3_report


def test_validate_paths_rejects_mixed_onnx_artifacts_from_different_runs(tmp_path: Path) -> None:
    policy_dir = _make_policy_dir(tmp_path, "policy")
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()

    primary_run = tmp_path / "run_primary"
    foreign_run = tmp_path / "run_foreign"

    vision = _write_stub_file(primary_run / "onnx" / "pi_shared_vision_encoder.onnx")
    denoise = _write_stub_file(primary_run / "onnx" / "pi05_denoise_step.onnx")
    prefix = _write_stub_file(foreign_run / "artifacts" / "onnx" / "pi_shared_prefix_cache.onnx")

    stage2_report, _ = _write_stage_reports(
        run_dir=primary_run,
        onnx_dir=primary_run / "onnx",
        stage2_policy_dir=policy_dir,
        onnx_paths={
            "vision_encoder": vision,
            "prefix_cache": prefix,
            "denoise_step": denoise,
        },
    )

    args = _make_validate_args(
        policy_path=policy_dir,
        calib_dir=calib_dir,
        onnx_path=primary_run / "onnx",
        stage2_report_path=stage2_report,
    )

    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)) as exc_info:
        onnx_launcher.validate_paths(args)

    _assert_message_has_any(exc_info.value, ("mixed", "provenance", "coherent", "directory", "run_dir"))


def test_validate_paths_rejects_stage2_policy_dir_mismatch(tmp_path: Path) -> None:
    policy_dir = _make_policy_dir(tmp_path, "policy_a")
    foreign_policy_dir = _make_policy_dir(tmp_path, "policy_b")
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()

    run_dir = tmp_path / "run"
    _write_stub_file(run_dir / "onnx" / "pi_shared_vision_encoder.onnx")
    _write_stub_file(run_dir / "onnx" / "pi_shared_prefix_cache.onnx")
    _write_stub_file(run_dir / "onnx" / "pi05_denoise_step.onnx")

    stage2_report, _ = _write_stage_reports(
        run_dir=run_dir,
        onnx_dir=run_dir / "onnx",
        stage2_policy_dir=foreign_policy_dir,
        stage3_policy_dir=foreign_policy_dir,
        onnx_paths={
            "vision_encoder": run_dir / "onnx" / "pi_shared_vision_encoder.onnx",
            "prefix_cache": run_dir / "onnx" / "pi_shared_prefix_cache.onnx",
            "denoise_step": run_dir / "onnx" / "pi05_denoise_step.onnx",
        },
    )

    args = _make_validate_args(
        policy_path=policy_dir,
        calib_dir=calib_dir,
        onnx_path=run_dir / "onnx",
        stage2_report_path=stage2_report,
    )

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        onnx_launcher.validate_paths(args)

    _assert_message_has_any(exc_info.value, ("policy", "checkpoint"))
    _assert_message_has_any(exc_info.value, ("stage2", "report", "mismatch"))


@pytest.mark.parametrize(
    "rtc_only_kwargs",
    [
        {"inference_delay": 0},
        {"execution_horizon": 1},
        {"prev_chunk_left_over": torch.zeros((1, 1), dtype=torch.float32)},
    ],
)
def test_build_chunk_predict_kwargs_fails_fast_for_explicit_rtc_inputs_when_disabled(
    rtc_only_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="RTC is not enabled"):
        chunk_runtime.build_chunk_predict_kwargs(rtc_enabled=False, **rtc_only_kwargs)


def test_merge_chunk_prediction_result_requires_real_delay_signal() -> None:
    class QueueSpy:
        def __init__(self) -> None:
            self.merge_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def merge(self, *args: object, **kwargs: object) -> None:
            self.merge_calls.append((args, kwargs))

    prediction = chunk_runtime.ChunkPredictionResult(
        original_actions=torch.zeros((2, 3), dtype=torch.float32),
        processed_actions=[
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        ],
        preprocess_time_s=0.0,
        inference_time_s=0.0,
        postprocess_time_s=0.0,
        processed_actions_tensor=torch.zeros((2, 3), dtype=torch.float32),
    )
    action_queue = QueueSpy()

    with pytest.raises(ValueError, match="real_delay"):
        chunk_runtime.merge_chunk_prediction_result(action_queue, prediction)

    assert action_queue.merge_calls == []


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0", 0),
        ("0", 0.0),
    ],
)
def test_parse_optional_zero_strings_preserve_numeric_zero(raw_value: str, expected: int | float) -> None:
    if isinstance(expected, int):
        assert onnx_launcher.parse_optional_int(raw_value) == expected
    else:
        assert onnx_launcher.parse_optional_float(raw_value) == expected
