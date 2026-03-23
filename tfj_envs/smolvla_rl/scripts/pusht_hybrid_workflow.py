#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train_smolvla_hybrid import (
    SmolVLAHybridCollectorConfig,
    SmolVLAHybridLossConfig,
    SmolVLAHybridReplayBufferConfig,
    TrainSmolVLAHybridConfig,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.smolvla_hybrid.trainer import train as train_smolvla_hybrid
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import get_safe_torch_device, init_logging


RENAME_MAP = {"observation.image": "observation.images.camera1"}
STATE_DIM = 2
ACTION_DIM = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def latest_snapshot(model_dir: Path) -> Path:
    snapshots_dir = model_dir / "snapshots"
    snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.is_dir() else []
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {snapshots_dir}")
    return snapshots[-1]


def default_smolvla_base_path() -> Path:
    return latest_snapshot(hf_cache_root() / "models--lerobot--smolvla_base")


def default_smolvlm2_path() -> Path:
    return latest_snapshot(
        hf_cache_root() / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    )


def default_teacher_policy_path() -> Path:
    candidates = [
        repo_root()
        / "outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model",
        repo_root()
        / "outputs/train/smolvla_hybrid_aloha_debug_block_in_bin_tune1/checkpoints/000200/pretrained_model",
        repo_root()
        / "outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find a default teacher policy. Pass --teacher-policy-path explicitly."
    )


def resolve_single_env(envs: dict[str, dict[int, Any]]) -> Any:
    flattened = [env for suite in envs.values() for env in suite.values()]
    if len(flattened) != 1:
        raise ValueError(f"Expected exactly one environment, found {len(flattened)}")
    return flattened[0]


def load_stats_file(path: Path) -> dict[str, dict[str, Any]]:
    flat_state = load_file(str(path))
    nested: dict[str, dict[str, Any]] = {}
    for flat_key, tensor in flat_state.items():
        feature_key, stat_name = flat_key.rsplit(".", 1)
        nested.setdefault(feature_key, {})[stat_name] = tensor.detach().cpu().tolist()
    return nested


def slice_stats(
    stats: dict[str, dict[str, Any]],
    *,
    state_dim: int,
    action_dim: int,
) -> dict[str, dict[str, Any]]:
    sliced = json.loads(json.dumps(stats))
    dim_map = {
        "observation.state": state_dim,
        "action": action_dim,
    }
    for feature_key, dim in dim_map.items():
        if feature_key not in sliced:
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
    vlm_path: Path,
    device: str,
    use_amp: bool,
) -> PreTrainedConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_cfg.use_amp = use_amp
    policy_cfg.push_to_hub = False
    if hasattr(policy_cfg, "vlm_model_name"):
        policy_cfg.vlm_model_name = str(vlm_path)

    if "observation.state" in policy_cfg.input_features:
        policy_cfg.input_features["observation.state"].shape = (STATE_DIM,)
    if "action" in policy_cfg.output_features:
        policy_cfg.output_features["action"].shape = (ACTION_DIM,)

    missing_image_slots = max(0, len(policy_cfg.image_features) - 1)
    if hasattr(policy_cfg, "empty_cameras"):
        policy_cfg.empty_cameras = max(int(policy_cfg.empty_cameras), missing_image_slots)

    return policy_cfg


def build_processors(
    policy_cfg: PreTrainedConfig,
    policy_path: Path,
    *,
    device: str,
) -> tuple[Any, Any]:
    pre_stats_path = policy_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_stats_path = policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    pre_stats = slice_stats(load_stats_file(pre_stats_path), state_dim=STATE_DIM, action_dim=ACTION_DIM)
    post_stats = slice_stats(load_stats_file(post_stats_path), state_dim=STATE_DIM, action_dim=ACTION_DIM)

    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": RENAME_MAP},
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


def pusht_dataset_features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    return {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "observation.image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
    }


def summarize_dataset(dataset_root: Path, repo_id: str) -> dict[str, Any]:
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    return {
        "repo_id": repo_id,
        "root": str(dataset_root),
        "num_episodes": meta.total_episodes,
        "num_frames": meta.total_frames,
        "fps": meta.fps,
        "features": {key: {"dtype": value["dtype"], "shape": value["shape"]} for key, value in meta.features.items()},
    }


def collect_dataset(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {dataset_root}. Use --overwrite to replace it."
            )
        import shutil

        shutil.rmtree(dataset_root)

    device = get_safe_torch_device(args.device, log=True)
    env_cfg = PushtEnv(task=args.task, obs_type="pixels_agent_pos")
    teacher_path = Path(args.teacher_policy_path).resolve()
    vlm_path = Path(args.vlm_path).resolve()

    policy_cfg = configure_policy(
        teacher_path,
        vlm_path=vlm_path,
        device=str(device),
        use_amp=args.use_amp,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, postprocessor = build_processors(policy_cfg, teacher_path, device=str(device))
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = resolve_single_env(envs)

    dataset: LeRobotDataset | None = None
    kept_rewards: list[float] = []
    kept_successes = 0

    try:
        raw_obs, _ = env.reset(seed=[args.seed])
        image_shape = tuple(int(dim) for dim in raw_obs["pixels"][0].shape)
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            fps=env_cfg.fps,
            features=pusht_dataset_features(image_shape),
            root=dataset_root,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=args.image_writer_threads,
        )
        task_text = args.task_text or args.task

        for attempt_idx in range(args.max_attempts):
            if len(kept_rewards) >= args.target_episodes:
                break

            policy.reset()
            raw_obs, _ = env.reset(seed=[args.seed + attempt_idx])
            episode_frames: list[dict[str, Any]] = []
            sum_reward = 0.0
            success = False

            for _step_idx in range(args.max_steps_per_episode):
                observation = preprocess_observation(raw_obs)
                observation = add_envs_task(env, observation)
                observation = env_preprocessor(observation)
                policy_input = preprocessor(observation)

                with torch.inference_mode():
                    policy_action = policy.select_action(policy_input)

                env_action = postprocessor(policy_action)
                env_action = env_postprocessor({ACTION: env_action})[ACTION]
                env_action_np = env_action[0].detach().cpu().numpy().astype(np.float32)

                episode_frames.append(
                    {
                        "observation.state": raw_obs["agent_pos"][0].astype(np.float32),
                        "observation.image": raw_obs["pixels"][0].astype(np.uint8),
                        "action": env_action_np,
                        "task": task_text,
                    }
                )

                next_obs, reward, terminated, truncated, info = env.step(env_action_np[None, :])
                sum_reward += float(reward[0])
                if "is_success" in info:
                    success = success or bool(info["is_success"][0])
                raw_obs = next_obs

                if bool(terminated[0] or truncated[0]):
                    break

            keep_episode = sum_reward >= args.min_sum_reward
            if keep_episode:
                for frame in episode_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                kept_rewards.append(sum_reward)
                kept_successes += int(success)

            if (attempt_idx + 1) % args.log_every == 0 or keep_episode:
                logging.info(
                    "collect attempt=%s kept=%s/%s reward=%.4f success=%s threshold=%.4f",
                    attempt_idx + 1,
                    len(kept_rewards),
                    args.target_episodes,
                    sum_reward,
                    success,
                    args.min_sum_reward,
                )
    finally:
        if dataset is not None:
            dataset.finalize()
        close_envs(envs)

    if not kept_rewards:
        raise RuntimeError("No PushT episodes passed the reward threshold; dataset is empty.")

    summary = {
        "teacher_policy_path": str(teacher_path),
        "vlm_path": str(vlm_path),
        "task": args.task,
        "task_text": args.task_text or args.task,
        "seed": args.seed,
        "target_episodes": args.target_episodes,
        "max_attempts": args.max_attempts,
        "min_sum_reward": args.min_sum_reward,
        "kept_episodes": len(kept_rewards),
        "kept_successes": kept_successes,
        "kept_success_rate": kept_successes / max(1, len(kept_rewards)),
        "reward_mean": float(np.mean(kept_rewards)),
        "reward_median": float(np.median(kept_rewards)),
        "reward_min": float(np.min(kept_rewards)),
        "reward_max": float(np.max(kept_rewards)),
        "dataset": summarize_dataset(dataset_root, args.dataset_repo_id),
    }
    (dataset_root / "collection_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Collection finished: %s", json.dumps(summary, indent=2))


def train_on_dataset(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")

    device = get_safe_torch_device(args.device, log=True)
    policy_path = Path(args.policy_path).resolve()
    vlm_path = Path(args.vlm_path).resolve()
    policy_cfg = configure_policy(
        policy_path,
        vlm_path=vlm_path,
        device=str(device),
        use_amp=args.use_amp,
    )
    policy_cfg.optimizer_lr = args.optimizer_lr
    policy_cfg.scheduler_warmup_steps = args.scheduler_warmup_steps
    policy_cfg.scheduler_decay_steps = args.scheduler_decay_steps
    policy_cfg.scheduler_decay_lr = args.scheduler_decay_lr

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(dataset_root),
        streaming=False,
    )
    env_cfg = PushtEnv(task=args.task, obs_type="pixels_agent_pos")
    cfg = TrainSmolVLAHybridConfig(
        dataset=dataset_cfg,
        env=env_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=args.job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        rename_map=RENAME_MAP,
    )
    cfg.collector = SmolVLAHybridCollectorConfig(
        n_envs=1,
        use_async_envs=False,
        chunks_per_step=args.chunks_per_step,
        warmup_chunks=args.warmup_chunks,
        max_steps_per_chunk=args.max_steps_per_chunk,
    )
    cfg.replay_buffer = SmolVLAHybridReplayBufferConfig(
        capacity=args.replay_capacity,
        online_batch_size=args.online_batch_size,
    )
    cfg.losses = SmolVLAHybridLossConfig(
        offline_loss_weight=args.offline_loss_weight,
        online_flow_loss_weight=args.online_flow_loss_weight,
        value_loss_weight=args.value_loss_weight,
        discount=args.discount,
        advantage_temperature=args.advantage_temperature,
        normalize_advantage=True,
        advantage_clip_min=-5.0,
        advantage_clip_max=5.0,
        max_advantage_weight=20.0,
    )
    cfg.wandb = WandBConfig(enable=False)

    logging.info(
        "Starting PushT hybrid training with config: %s",
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "dataset_repo_id": args.dataset_repo_id,
                "policy_path": str(policy_path),
                "vlm_path": str(vlm_path),
                "output_dir": str(output_dir),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "save_freq": args.save_freq,
                "log_freq": args.log_freq,
                "warmup_chunks": args.warmup_chunks,
                "chunks_per_step": args.chunks_per_step,
                "replay_capacity": args.replay_capacity,
                "online_batch_size": args.online_batch_size,
                "offline_loss_weight": args.offline_loss_weight,
                "online_flow_loss_weight": args.online_flow_loss_weight,
                "value_loss_weight": args.value_loss_weight,
            },
            indent=2,
        ),
    )
    train_smolvla_hybrid(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-friendly PushT collection and SmolVLA hybrid training workflow."
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect a local PushT LeRobot dataset.")
    collect.add_argument("--teacher-policy-path", default=str(default_teacher_policy_path()))
    collect.add_argument("--vlm-path", default=str(default_smolvlm2_path()))
    collect.add_argument("--dataset-root", required=True)
    collect.add_argument("--dataset-repo-id", default="local/pusht_teacher_rl_filtered")
    collect.add_argument("--task", default="PushT-v0")
    collect.add_argument("--task-text", default=None)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--target-episodes", type=int, default=128)
    collect.add_argument("--max-attempts", type=int, default=400)
    collect.add_argument("--min-sum-reward", type=float, default=5.0)
    collect.add_argument("--max-steps-per-episode", type=int, default=300)
    collect.add_argument("--image-writer-threads", type=int, default=4)
    collect.add_argument("--log-every", type=int, default=10)
    collect.add_argument("--use-amp", action="store_true")
    collect.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Train SmolVLA hybrid on a local PushT dataset.")
    train.add_argument("--policy-path", default=str(default_smolvla_base_path()))
    train.add_argument("--vlm-path", default=str(default_smolvlm2_path()))
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--dataset-repo-id", default="local/pusht_teacher_rl_filtered")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--job-name", default="smolvla_hybrid_pusht")
    train.add_argument("--task", default="PushT-v0")
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--save-freq", type=int, default=500)
    train.add_argument("--log-freq", type=int, default=50)
    train.add_argument("--warmup-chunks", type=int, default=32)
    train.add_argument("--chunks-per-step", type=int, default=1)
    train.add_argument("--max-steps-per-chunk", type=int, default=50)
    train.add_argument("--replay-capacity", type=int, default=4096)
    train.add_argument("--online-batch-size", type=int, default=16)
    train.add_argument("--offline-loss-weight", type=float, default=1.0)
    train.add_argument("--online-flow-loss-weight", type=float, default=0.3)
    train.add_argument("--value-loss-weight", type=float, default=1.0)
    train.add_argument("--discount", type=float, default=0.99)
    train.add_argument("--advantage-temperature", type=float, default=1.0)
    train.add_argument("--optimizer-lr", type=float, default=1e-4)
    train.add_argument("--scheduler-warmup-steps", type=int, default=100)
    train.add_argument("--scheduler-decay-steps", type=int, default=2000)
    train.add_argument("--scheduler-decay-lr", type=float, default=1e-5)
    train.add_argument("--use-amp", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_file = Path(args.log_file).resolve() if args.log_file else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=log_file)
    logging.getLogger("pymunk").setLevel(logging.WARNING)
    logging.getLogger("pygame").setLevel(logging.WARNING)
    logging.info("PushT hybrid workflow command=%s", args.command)

    if args.command == "collect":
        collect_dataset(args)
        return
    if args.command == "train":
        train_on_dataset(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
