#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from termcolor import colored

from lerobot.configs.train_smolvla_hybrid import TrainSmolVLAHybridConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rl.smolvla_hybrid.buffer import SmolVLAChunkReplayBuffer
from lerobot.rl.smolvla_hybrid.collector import SmolVLAChunkCollector, resolve_single_vector_env
from lerobot.rl.smolvla_hybrid.losses import compute_online_losses
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging


def _make_processors(cfg, dataset, policy):
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }

    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )


def _make_dataloader(cfg, dataset, device):
    sampler = None
    shuffle = True
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def _average_metrics(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / count for key, value in metric_sums.items()}


def train(cfg: TrainSmolVLAHybridConfig) -> None:
    cfg.validate()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=log_dir / "smolvla_hybrid_train.log")
    logging.info("Starting SmolVLA hybrid training")

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if not isinstance(policy, SmolVLAPolicy):
        raise TypeError(f"Expected SmolVLAPolicy, got {type(policy)}")

    preprocessor, postprocessor = _make_processors(cfg, dataset, policy)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(cfg.env)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    train_envs = make_env(
        cfg.env,
        n_envs=cfg.collector.n_envs,
        use_async_envs=cfg.collector.use_async_envs,
    )
    train_env = resolve_single_vector_env(train_envs)
    collector = SmolVLAChunkCollector(
        env=train_env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        discount=cfg.losses.discount,
        max_steps_per_chunk=cfg.collector.max_steps_per_chunk,
    )
    collector.reset(seed=cfg.seed)

    replay_buffer = SmolVLAChunkReplayBuffer(cfg.replay_buffer.capacity)
    if cfg.collector.warmup_chunks > 0:
        logging.info("Warmup collector with %s chunks", cfg.collector.warmup_chunks)
        replay_buffer.extend(collector.collect(cfg.collector.warmup_chunks))

    dataloader = _make_dataloader(cfg, dataset, device)
    dl_iter = cycle(dataloader)

    eval_envs = None
    if cfg.eval_freq > 0:
        eval_envs = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
        )

    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)

    num_learnable_params = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    num_total_params = sum(parameter.numel() for parameter in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    metric_sums: dict[str, float] = {}
    metric_count = 0
    policy.train()

    try:
        for _ in range(step, cfg.steps):
            start_collect = time.perf_counter()
            replay_buffer.extend(collector.collect(cfg.collector.chunks_per_step))
            collect_s = time.perf_counter() - start_collect

            start_data = time.perf_counter()
            offline_batch = preprocessor(next(dl_iter))
            dataloading_s = time.perf_counter() - start_data
            online_batch = replay_buffer.sample(cfg.replay_buffer.online_batch_size, device=device)

            optimizer.zero_grad()
            start_update = time.perf_counter()
            offline_loss, _ = policy.forward(offline_batch)
            online_policy_loss, value_loss, online_metrics = compute_online_losses(
                policy,
                online_batch,
                cfg.losses,
            )
            total_loss = (
                cfg.losses.offline_loss_weight * offline_loss
                + cfg.losses.online_flow_loss_weight * online_policy_loss
                + cfg.losses.value_loss_weight * value_loss
            )
            total_loss.backward()

            if cfg.optimizer.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    cfg.optimizer.grad_clip_norm,
                    error_if_nonfinite=False,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    float("inf"),
                    error_if_nonfinite=False,
                )

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            update_s = time.perf_counter() - start_update

            step += 1
            step_metrics = {
                "loss": total_loss.item(),
                "offline_loss": offline_loss.item(),
                "online_policy_loss": online_policy_loss.item(),
                "value_loss_total": value_loss.item(),
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "collect_s": collect_s,
                "data_s": dataloading_s,
                "update_s": update_s,
                "buffer_size": float(len(replay_buffer)),
                **online_metrics,
            }
            for key, value in step_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_count += 1

            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_save_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = eval_envs is not None and cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                averaged_metrics = _average_metrics(metric_sums, metric_count)
                logging.info("step=%s metrics=%s", step, averaged_metrics)
                if wandb_logger is not None:
                    wandb_logger.log_dict(averaged_metrics, step)
                metric_sums = {}
                metric_count = 0

            if cfg.save_checkpoint and is_save_step:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger is not None:
                    wandb_logger.log_policy(checkpoint_dir)

            if is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info("Evaluating policy at step %s", step)
                with torch.no_grad():
                    eval_info = eval_policy_all(
                        envs=eval_envs,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                eval_metrics = {
                    key: value
                    for key, value in eval_info["overall"].items()
                    if isinstance(value, (int, float))
                }
                logging.info("eval=%s", eval_metrics)
                if wandb_logger is not None:
                    wandb_logger.log_dict(eval_metrics, step, mode="eval")
                policy.train()
    finally:
        close_envs(train_envs)
        if eval_envs is not None:
            close_envs(eval_envs)

    logging.info("End of SmolVLA hybrid training")
