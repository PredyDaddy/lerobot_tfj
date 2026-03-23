#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import AlohaEnv, MetaworldEnv, PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.utils import get_safe_torch_device, init_logging


DEFAULT_RENAME_MAPS: dict[str, dict[str, str]] = {
    "aloha": {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    },
    "pusht": {
        "observation.image": "observation.images.camera1",
    },
    "metaworld": {
        "observation.image": "observation.images.camera1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA / RL+SmolVLA checkpoints on sim envs with controlled dim adaptation."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--env-type", required=True, choices=["aloha", "pusht", "metaworld"])
    parser.add_argument("--env-task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--max-episodes-rendered", type=int, default=10)
    parser.add_argument("--rename-map-json", default=None)
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def _default_task(env_type: str) -> str:
    if env_type == "aloha":
        return "AlohaInsertion-v0"
    if env_type == "pusht":
        return "PushT-v0"
    if env_type == "metaworld":
        return "push-v3"
    raise ValueError(f"Unsupported env_type: {env_type}")


def _default_dims(env_type: str) -> tuple[int, int]:
    if env_type == "aloha":
        return 6, 6
    if env_type == "pusht":
        return 2, 2
    if env_type == "metaworld":
        return 4, 4
    raise ValueError(f"Unsupported env_type: {env_type}")


def make_env_cfg(env_type: str, env_task: str, obs_type: str):
    if env_type == "aloha":
        return AlohaEnv(task=env_task, obs_type=obs_type)
    if env_type == "pusht":
        return PushtEnv(task=env_task, obs_type=obs_type)
    if env_type == "metaworld":
        return MetaworldEnv(task=env_task, obs_type=obs_type)
    raise ValueError(f"Unsupported env_type: {env_type}")


def load_stats_file(path: Path) -> dict[str, dict[str, Any]]:
    flat_state = load_file(str(path))
    nested: dict[str, dict[str, Any]] = {}
    for flat_key, tensor in flat_state.items():
        feature_key, stat_name = flat_key.rsplit(".", 1)
        value = tensor.detach().cpu()
        nested.setdefault(feature_key, {})[stat_name] = value.tolist()
    return nested


def slice_stats(
    stats: dict[str, dict[str, Any]],
    *,
    state_dim: int | None,
    action_dim: int | None,
) -> dict[str, dict[str, Any]]:
    sliced = json.loads(json.dumps(stats))
    dim_map = {
        "observation.state": state_dim,
        "action": action_dim,
    }
    for feature_key, dim in dim_map.items():
        if dim is None or feature_key not in sliced:
            continue
        for stat_name, value in sliced[feature_key].items():
            if stat_name == "count":
                continue
            if isinstance(value, list):
                sliced[feature_key][stat_name] = value[:dim]
    return sliced


def configure_policy(
    policy_path: Path,
    *,
    device: str,
    state_dim: int,
    action_dim: int,
    use_amp: bool,
):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_cfg.use_amp = use_amp

    if "observation.state" in policy_cfg.input_features:
        policy_cfg.input_features["observation.state"].shape = (state_dim,)
    if "action" in policy_cfg.output_features:
        policy_cfg.output_features["action"].shape = (action_dim,)
    return policy_cfg


def build_processors(
    policy_cfg,
    policy_path: Path,
    rename_map: dict[str, str],
    *,
    device: str,
    state_dim: int,
    action_dim: int,
):
    pre_stats_path = policy_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_stats_path = policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    pre_stats = slice_stats(
        load_stats_file(pre_stats_path),
        state_dim=state_dim,
        action_dim=action_dim,
    )
    post_stats = slice_stats(
        load_stats_file(post_stats_path),
        state_dim=state_dim,
        action_dim=action_dim,
    )

    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename_map},
        "normalizer_processor": {
            "stats": pre_stats,
            "features": {**policy_cfg.input_features, **policy_cfg.output_features},
            "norm_map": policy_cfg.normalization_mapping,
        },
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": post_stats,
            "features": policy_cfg.output_features,
            "norm_map": policy_cfg.normalization_mapping,
        }
    }

    return make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=output_dir / "cross_env_eval.log")

    env_task = args.env_task or _default_task(args.env_type)
    default_state_dim, default_action_dim = _default_dims(args.env_type)
    state_dim = args.state_dim or default_state_dim
    action_dim = args.action_dim or default_action_dim
    rename_map = (
        json.loads(args.rename_map_json)
        if args.rename_map_json
        else DEFAULT_RENAME_MAPS[args.env_type].copy()
    )

    policy_path = Path(args.policy_path)
    env_cfg = make_env_cfg(args.env_type, env_task, args.obs_type)
    device = get_safe_torch_device(args.device, log=True)
    policy_cfg = configure_policy(
        policy_path,
        device=str(device),
        state_dim=state_dim,
        action_dim=action_dim,
        use_amp=args.use_amp,
    )

    logging.info(
        "Cross-env eval config: policy=%s env=%s task=%s state_dim=%s action_dim=%s rename_map=%s",
        policy_path,
        args.env_type,
        env_task,
        state_dim,
        action_dim,
        rename_map,
    )

    envs = make_env(
        env_cfg,
        n_envs=args.batch_size,
        use_async_envs=False,
    )
    policy = make_policy(
        cfg=policy_cfg,
        env_cfg=env_cfg,
        rename_map=rename_map,
    )
    policy.eval()

    preprocessor, postprocessor = build_processors(
        policy_cfg,
        policy_path,
        rename_map,
        device=str(device),
        state_dim=state_dim,
        action_dim=action_dim,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg)

    try:
        autocast_ctx = torch.autocast(device_type=device.type) if args.use_amp else nullcontext()
        with torch.no_grad(), autocast_ctx:
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                max_episodes_rendered=min(args.max_episodes_rendered, args.n_episodes),
                videos_dir=output_dir / "videos",
                start_seed=args.seed,
                max_parallel_tasks=env_cfg.max_parallel_tasks,
            )
    finally:
        close_envs(envs)

    with (output_dir / "eval_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info["overall"], indent=2))


if __name__ == "__main__":
    main()
