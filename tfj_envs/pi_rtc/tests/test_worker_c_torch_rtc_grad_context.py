from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PI_TRT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PI_TRT_ROOT / "scripts"
TORCH_LAUNCHER_PATH = SCRIPTS_DIR / "run_pi05_torch_infer_so101.py"
REPO_ROOT = PI_TRT_ROOT.parents[1]
SRC_DIR = REPO_ROOT / "src"

for candidate in (SCRIPTS_DIR, REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name!r} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


torch_launcher = _load_module("worker_c_run_pi05_torch_infer_so101", TORCH_LAUNCHER_PATH)


class _GradProbePolicy:
    def __init__(self, *, rtc_enabled: bool) -> None:
        rtc_config = RTCConfig(
            enabled=rtc_enabled,
            execution_horizon=2,
            max_guidance_weight=1.0,
            debug=False,
            debug_maxlen=8,
        )
        self.config = SimpleNamespace(rtc_config=rtc_config)
        self.rtc_processor = RTCProcessor(rtc_config)
        self.events: list[dict[str, object]] = []

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.events.append(
            {
                "phase": "policy_entry",
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode_enabled": torch.is_inference_mode_enabled(),
                "kwarg_names": tuple(sorted(kwargs)),
            }
        )

        if not self.config.rtc_config.enabled:
            return torch.zeros((1, 2, 3), dtype=torch.float32)

        x_t = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            dtype=torch.float32,
        )

        def denoise_step_partial(input_x_t: torch.Tensor) -> torch.Tensor:
            self.events.append(
                {
                    "phase": "rtc_partial",
                    "grad_enabled": torch.is_grad_enabled(),
                    "inference_mode_enabled": torch.is_inference_mode_enabled(),
                }
            )
            return input_x_t * 0.5

        return self.rtc_processor.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=kwargs["prev_chunk_left_over"],
            inference_delay=kwargs["inference_delay"],
            time=torch.tensor(0.5),
            original_denoise_step_partial=denoise_step_partial,
            execution_horizon=kwargs["execution_horizon"],
        )


def test_torch_chunk_policy_runtime_keeps_rtc_off_path_in_inference_mode() -> None:
    policy = _GradProbePolicy(rtc_enabled=False)
    runtime = torch_launcher.TorchChunkPolicyRuntime(
        policy,
        device=torch.device("cpu"),
        use_amp=False,
    )

    result = runtime.predict_action_chunk({"dummy": torch.tensor(1.0)})

    assert result.shape == (1, 2, 3)
    assert policy.events == [
        {
            "phase": "policy_entry",
            "grad_enabled": False,
            "inference_mode_enabled": True,
            "kwarg_names": (),
        }
    ]


def test_torch_chunk_policy_runtime_keeps_rtc_on_path_out_of_inference_mode() -> None:
    policy = _GradProbePolicy(rtc_enabled=True)
    runtime = torch_launcher.TorchChunkPolicyRuntime(
        policy,
        device=torch.device("cpu"),
        use_amp=False,
    )

    result = runtime.predict_action_chunk(
        {"dummy": torch.tensor(1.0)},
        prev_chunk_left_over=torch.zeros((1, 2, 3), dtype=torch.float32),
        inference_delay=1,
        execution_horizon=2,
    )

    assert result.shape == (1, 2, 3)
    assert policy.events == [
        {
            "phase": "policy_entry",
            "grad_enabled": False,
            "inference_mode_enabled": False,
            "kwarg_names": ("execution_horizon", "inference_delay", "prev_chunk_left_over"),
        },
        {
            "phase": "rtc_partial",
            "grad_enabled": True,
            "inference_mode_enabled": False,
        },
    ]


def test_rtc_processor_grad_probe_still_fails_if_forced_under_inference_mode() -> None:
    rtc_processor = RTCProcessor(
        RTCConfig(
            enabled=True,
            execution_horizon=2,
            max_guidance_weight=1.0,
            debug=False,
            debug_maxlen=8,
        )
    )

    with pytest.raises(RuntimeError, match="does not require grad"):
        with torch.inference_mode():
            rtc_processor.denoise_step(
                x_t=torch.zeros((1, 2, 3), dtype=torch.float32),
                prev_chunk_left_over=torch.zeros((1, 2, 3), dtype=torch.float32),
                inference_delay=1,
                time=torch.tensor(0.5),
                original_denoise_step_partial=lambda input_x_t: input_x_t * 0.5,
                execution_horizon=2,
            )
