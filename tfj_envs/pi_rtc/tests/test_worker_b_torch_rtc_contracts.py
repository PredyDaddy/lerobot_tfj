from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import torch


PI_TRT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PI_TRT_ROOT / "scripts"
TORCH_LAUNCHER_PATH = SCRIPTS_DIR / "run_pi05_torch_infer_so101.py"

scripts_dir_str = str(SCRIPTS_DIR)
if scripts_dir_str not in sys.path:
    sys.path.insert(0, scripts_dir_str)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name!r} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


chunk_runtime = _load_module("worker_b_pi05_chunk_runtime", SCRIPTS_DIR / "pi05_chunk_runtime.py")
torch_launcher = _load_module("worker_b_run_pi05_torch_infer_so101", TORCH_LAUNCHER_PATH)


def _torch_launcher_ast() -> ast.AST:
    return ast.parse(TORCH_LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(TORCH_LAUNCHER_PATH))


def test_torch_launcher_rtc_arguments_parse() -> None:
    parser = torch_launcher.build_parser()
    rtc_schedule = next(iter(torch_launcher.RTCAttentionSchedule)).value.lower()
    args = parser.parse_args(
        [
            "--rtc-enable",
            "--rtc-execution-horizon",
            "4",
            "--rtc-max-guidance-weight",
            "1.25",
            "--rtc-prefix-attention-schedule",
            rtc_schedule,
            "--rtc-debug",
            "--rtc-debug-maxlen",
            "32",
            "--dry-run",
        ]
    )

    assert args.rtc_enable is True
    assert args.rtc_execution_horizon == 4
    assert args.rtc_max_guidance_weight == 1.25
    assert args.rtc_prefix_attention_schedule == rtc_schedule
    assert args.rtc_debug is True
    assert args.rtc_debug_maxlen == 32
    assert args.dry_run is True


def test_torch_launcher_accepts_trt_compatibility_arguments() -> None:
    parser = torch_launcher.build_parser()
    args = parser.parse_args(
        [
            "--trt-path",
            "/tmp/fake_trt_run",
            "--trt-metadata-path",
            "/tmp/fake_trt_run/pi_trt_metadata.json",
            "--trt-device",
            "cuda:0",
            "--skip-trt-preflight",
            "--dry-run",
        ]
    )

    assert args.trt_path == "/tmp/fake_trt_run"
    assert args.trt_metadata_path == "/tmp/fake_trt_run/pi_trt_metadata.json"
    assert args.trt_device == "cuda:0"
    assert args.skip_trt_preflight is True
    assert args.dry_run is True


def test_torch_launcher_source_drops_legacy_predict_action_path() -> None:
    violations: list[str] = []
    tree = _torch_launcher_ast()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lerobot.utils.control_utils":
            for alias in node.names:
                if alias.name == "predict_action":
                    violations.append(f"legacy import on line {node.lineno}")

        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "predict_action":
            violations.append(f"legacy predict_action(...) call on line {node.lineno}")

    assert violations == [], (
        "Torch launcher should not depend on the legacy predict_action(...) path: "
        + ", ".join(violations)
    )


def test_build_chunk_predict_kwargs_covers_torch_launcher_rtc_off_and_on_paths() -> None:
    class RuntimeConfig:
        enabled = True
        execution_horizon = 5

    class RuntimeStub:
        def __init__(self) -> None:
            self.config = RuntimeConfig()

    class ActionQueueStub:
        def __init__(self, left_over: torch.Tensor) -> None:
            self._left_over = left_over

        def get_left_over(self) -> torch.Tensor:
            return self._left_over

    left_over = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    rtc_off_kwargs = chunk_runtime.build_chunk_predict_kwargs(
        rtc_enabled=False,
        action_queue=ActionQueueStub(left_over),
    )
    assert rtc_off_kwargs == {}

    rtc_on_kwargs = chunk_runtime.build_chunk_predict_kwargs(
        rtc_runtime=RuntimeStub(),
        action_queue=ActionQueueStub(left_over),
        predicted_delay_steps=2,
    )

    assert rtc_on_kwargs["inference_delay"] == 2
    assert rtc_on_kwargs["execution_horizon"] == 5
    assert torch.equal(rtc_on_kwargs["prev_chunk_left_over"], left_over)
    assert rtc_on_kwargs["prev_chunk_left_over"] is not left_over


def test_merge_chunk_prediction_result_passes_computed_real_delay_to_action_queue() -> None:
    class ActionQueueSpy:
        def __init__(self, action_index: int) -> None:
            self._action_index = action_index
            self.merge_calls: list[dict[str, object]] = []

        def get_action_index(self) -> int:
            return self._action_index

        def merge(
            self,
            original_actions: torch.Tensor,
            processed_actions: torch.Tensor,
            *,
            real_delay: int,
            action_index_before_inference: int | None,
        ) -> None:
            self.merge_calls.append(
                {
                    "original_actions": original_actions,
                    "processed_actions": processed_actions,
                    "real_delay": real_delay,
                    "action_index_before_inference": action_index_before_inference,
                }
            )

    processed_actions_tensor = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=torch.float32,
    )
    prediction = chunk_runtime.ChunkPredictionResult(
        original_actions=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        ),
        processed_actions=[row.clone() for row in processed_actions_tensor],
        preprocess_time_s=0.001,
        inference_time_s=0.002,
        postprocess_time_s=0.003,
        processed_actions_tensor=processed_actions_tensor,
        action_index_before_inference=7,
    )
    action_queue = ActionQueueSpy(action_index=10)

    resolved_real_delay = chunk_runtime.merge_chunk_prediction_result(action_queue, prediction)

    assert resolved_real_delay == 3
    assert len(action_queue.merge_calls) == 1
    merge_call = action_queue.merge_calls[0]
    assert merge_call["real_delay"] == 3
    assert merge_call["action_index_before_inference"] == 7
    assert torch.equal(merge_call["original_actions"], prediction.original_actions)
    assert torch.equal(merge_call["processed_actions"], processed_actions_tensor)
