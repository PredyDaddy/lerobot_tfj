from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass(frozen=True)
class ActModelSpec:
    checkpoint: str
    visual_keys: list[str]
    state_dim: int
    image_height: int
    image_width: int
    action_dim: int
    chunk_size: int
    n_action_steps: int

    @property
    def obs_state_shape(self) -> tuple[int, int]:
        return (1, self.state_dim)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.image_height, self.image_width)

    @property
    def action_shape(self) -> tuple[int, int, int]:
        return (1, self.chunk_size, self.action_dim)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_visual_feature_keys_in_order(config_dict: dict[str, Any]) -> list[str]:
    input_features = config_dict.get("input_features", {})
    keys: list[str] = []
    for key, value in input_features.items():
        if not isinstance(value, dict):
            continue
        if value.get("type") == "VISUAL":
            keys.append(key)
    return keys


def load_model_spec(checkpoint: str | Path) -> ActModelSpec:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    config_dict = load_json(checkpoint_path / "config.json")
    visual_keys = get_visual_feature_keys_in_order(config_dict)
    if len(visual_keys) != 2:
        raise ValueError(f"Expected exactly 2 visual inputs, got {visual_keys}")

    state_shape = config_dict["input_features"]["observation.state"]["shape"]
    img_shape = config_dict["input_features"][visual_keys[0]]["shape"]
    action_shape = config_dict["output_features"]["action"]["shape"]

    if int(img_shape[0]) != 3:
        raise ValueError(f"Expected 3 image channels, got {img_shape[0]}")

    return ActModelSpec(
        checkpoint=str(checkpoint_path),
        visual_keys=visual_keys,
        state_dim=int(state_shape[0]),
        image_height=int(img_shape[1]),
        image_width=int(img_shape[2]),
        action_dim=int(action_shape[0]),
        chunk_size=int(config_dict["chunk_size"]),
        n_action_steps=int(config_dict["n_action_steps"]),
    )


def load_act_policy(checkpoint: str | Path, device: str = "cpu") -> ACTPolicy:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    config.device = device
    # Avoid any accidental external weight load paths. Checkpoint provides weights in safetensors.
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(str(checkpoint_path), config=config)
    policy.eval()
    return policy


@torch.no_grad()
def torch_core_forward(
    policy: ACTPolicy,
    obs_state_norm: torch.Tensor,
    img0_norm: torch.Tensor,
    img1_norm: torch.Tensor,
) -> torch.Tensor:
    batch = {
        OBS_STATE: obs_state_norm,
        OBS_IMAGES: [img0_norm, img1_norm],
    }
    actions, _ = policy.model(batch)
    return actions


@torch.no_grad()
def make_case_inputs(spec: ActModelSpec, seed: int | None = None, case: str = "random") -> tuple[torch.Tensor, ...]:
    if case not in {"random", "zeros", "ones", "linspace"}:
        raise ValueError(f"Unsupported case: {case}")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    if case == "random":
        obs_state = torch.randn(spec.obs_state_shape, dtype=torch.float32, generator=generator)
        img0 = torch.randn(spec.image_shape, dtype=torch.float32, generator=generator)
        img1 = torch.randn(spec.image_shape, dtype=torch.float32, generator=generator)
    elif case == "zeros":
        obs_state = torch.zeros(spec.obs_state_shape, dtype=torch.float32)
        img0 = torch.zeros(spec.image_shape, dtype=torch.float32)
        img1 = torch.zeros(spec.image_shape, dtype=torch.float32)
    elif case == "ones":
        obs_state = torch.ones(spec.obs_state_shape, dtype=torch.float32)
        img0 = torch.ones(spec.image_shape, dtype=torch.float32)
        img1 = torch.ones(spec.image_shape, dtype=torch.float32)
    else:
        obs_state = torch.linspace(-1.0, 1.0, steps=int(np.prod(spec.obs_state_shape)), dtype=torch.float32).reshape(
            spec.obs_state_shape
        )
        img0 = torch.linspace(-2.0, 2.0, steps=int(np.prod(spec.image_shape)), dtype=torch.float32).reshape(
            spec.image_shape
        )
        img1 = torch.linspace(2.0, -2.0, steps=int(np.prod(spec.image_shape)), dtype=torch.float32).reshape(
            spec.image_shape
        )

    return obs_state.contiguous(), img0.contiguous(), img1.contiguous()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().float().numpy(), dtype=np.float32)


def build_case_catalog(random_cases: int) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for seed in range(int(random_cases)):
        catalog.append({"name": f"random_seed_{seed}", "case": "random", "seed": seed})
    catalog.extend(
        [
            {"name": "zeros", "case": "zeros", "seed": None},
            {"name": "ones", "case": "ones", "seed": None},
            {"name": "linspace", "case": "linspace", "seed": None},
        ]
    )
    return catalog


def spec_to_dict(spec: ActModelSpec) -> dict[str, Any]:
    return asdict(spec)

