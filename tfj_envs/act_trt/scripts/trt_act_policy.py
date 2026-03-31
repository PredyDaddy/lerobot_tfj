from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act_trt_paths import REPO_ROOT, SRC_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from trt_runtime import TensorRTRunner


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("-", "_")


def feature_type_name(feature: Any) -> str | None:
    feature_type = getattr(feature, "type", None)
    if feature_type is None and isinstance(feature, dict):
        feature_type = feature.get("type")
    return getattr(feature_type, "name", feature_type)


def resolve_act_feature_keys(config: ACTConfig) -> tuple[str | None, list[str]]:
    visual_keys = [
        key
        for key, feature in config.input_features.items()
        if key.startswith("observation.images.") or feature_type_name(feature) == "VISUAL"
    ]
    state_key = "observation.state" if "observation.state" in config.input_features else None
    if len(visual_keys) != 2:
        raise ValueError(f"ACT TRT runner expected exactly 2 visual inputs, got {visual_keys}")
    return state_key, visual_keys


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrtIOMapping:
    state_input_name: str | None
    camera_input_names: dict[str, str]
    output_name: str


def resolve_trt_io_mapping(
    runner: TensorRTRunner,
    state_key: str | None,
    camera_keys: list[str],
    metadata: dict[str, Any] | None,
) -> TrtIOMapping:
    input_names = list(runner.input_names)
    output_names = list(runner.output_names)

    if metadata:
        camera_order = metadata.get("camera_order_visual_keys")
        if (
            state_key is None or "obs_state_norm" in input_names
        ) and all(name in input_names for name in ["img0_norm", "img1_norm"]) and "actions_norm" in output_names:
            mapping: dict[str, str] = {}
            if isinstance(camera_order, list) and len(camera_order) == 2 and set(camera_order) == set(camera_keys):
                mapping[camera_order[0]] = "img0_norm"
                mapping[camera_order[1]] = "img1_norm"
            else:
                mapping[camera_keys[0]] = "img0_norm"
                mapping[camera_keys[1]] = "img1_norm"
            return TrtIOMapping(
                state_input_name="obs_state_norm" if state_key is not None else None,
                camera_input_names=mapping,
                output_name="actions_norm",
            )

    sanitized_inputs = {key: sanitize_name(key) for key in ([state_key] if state_key else []) + camera_keys}
    if all(name in input_names for name in [sanitized_inputs[key] for key in camera_keys]):
        output_name = "action_chunk" if "action_chunk" in output_names else output_names[0]
        return TrtIOMapping(
            state_input_name=sanitized_inputs.get(state_key) if state_key else None,
            camera_input_names={key: sanitized_inputs[key] for key in camera_keys},
            output_name=output_name,
        )

    if len(camera_keys) == 2 and all(name in input_names for name in ["img0_norm", "img1_norm"]):
        output_name = "actions_norm" if "actions_norm" in output_names else output_names[0]
        return TrtIOMapping(
            state_input_name="obs_state_norm" if (state_key is not None and "obs_state_norm" in input_names) else None,
            camera_input_names={
                camera_keys[0]: "img0_norm",
                camera_keys[1]: "img1_norm",
            },
            output_name=output_name,
        )

    raise ValueError(
        "Unable to resolve TensorRT input/output names. "
        f"Engine inputs={input_names}, outputs={output_names}, camera_keys={camera_keys}, state_key={state_key}"
    )


class TrtActPolicyAdapter(PreTrainedPolicy):
    config_class = ACTConfig
    name = "act_trt"

    def __init__(
        self,
        config: ACTConfig,
        *,
        engine_path: str | Path,
        metadata_path: str | Path | None = None,
        trt_device: str = "cuda:0",
    ) -> None:
        super().__init__(config)
        self.config = config
        if not str(config.device).startswith("cuda"):
            raise ValueError("TensorRT real-robot runner expects `--policy-device=cuda`.")

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.metadata_path = Path(metadata_path).expanduser().resolve() if metadata_path else None
        self.metadata = load_json(self.metadata_path) if self.metadata_path and self.metadata_path.is_file() else None

        self.runner = TensorRTRunner(engine_path=self.engine_path, device=trt_device)
        self.state_key, self.camera_keys = resolve_act_feature_keys(config)
        self.io_mapping = resolve_trt_io_mapping(self.runner, self.state_key, self.camera_keys, self.metadata)
        self.output_device = self.runner.device

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        raise NotImplementedError("TrtActPolicyAdapter does not support training forward().")

    def reset(self) -> None:
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().to("cpu", dtype=torch.float32).numpy())

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        feed_dict: dict[str, np.ndarray] = {}
        if self.state_key is not None:
            if self.state_key not in batch:
                raise KeyError(f"Missing state input `{self.state_key}` in policy batch.")
            state_name = self.io_mapping.state_input_name
            if state_name is None:
                raise KeyError("TensorRT engine mapping did not expose a state input name.")
            feed_dict[state_name] = self._to_numpy(batch[self.state_key])

        for key in self.camera_keys:
            if key not in batch:
                raise KeyError(f"Missing visual input `{key}` in policy batch.")
            feed_dict[self.io_mapping.camera_input_names[key]] = self._to_numpy(batch[key])

        actions = self.runner.infer(feed_dict)[self.io_mapping.output_name]
        return actions.to(self.output_device, dtype=torch.float32)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()


def init_keyboard_listener():
    from lerobot.utils.control_utils import init_keyboard_listener as _init_keyboard_listener

    return _init_keyboard_listener()
