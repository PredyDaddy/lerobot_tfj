#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.distillation_utils import load_act_teacher_bundle
from lerobot.policies.act.modeling_act import _resolve_teacher_pretrained_path
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, reduce_metrics_dict
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_metadata import (
    KD_TEACHER_METADATA_FILENAME,
    KDTeacherMetadata,
    kd_teacher_metadata_to_dict,
    merge_kd_teacher_metadata,
)
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


ACT_KD_METADATA_FILENAME = KD_TEACHER_METADATA_FILENAME
POLICY_METRIC_SPECS = {
    "l1_loss": ("l1", ":.3f"),
    "kld_loss": ("kld", ":.3f"),
    "kd_l1_loss": ("kd_raw", ":.3f"),
    "kd_weighted_l1_loss": ("kd", ":.3f"),
    "kd_overlap_steps": ("kd_ov", ":.1f"),
    "kd_valid_ratio": ("kd_vr", ":.3f"),
    "kd_to_bc_ratio": ("kd/bc", ":.3f"),
    "kd_weighted_to_bc_ratio": ("kdw/bc", ":.3f"),
    "kd_prefix_l1_loss": ("kd_pre", ":.3f"),
    "kd_tail_l1_loss": ("kd_tail", ":.3f"),
    "decoder_kd_loss": ("dec_raw", ":.3f"),
    "decoder_kd_weighted_loss": ("dec", ":.3f"),
    "decoder_kd_prefix_loss": ("dec_pre", ":.3f"),
    "decoder_kd_tail_loss": ("dec_tail", ":.3f"),
    "decoder_kd_valid_ratio": ("dec_vr", ":.3f"),
    "decoder_kd_weighted_to_bc_ratio": ("dec/bc", ":.3f"),
    "decoder_kd_weighted_to_action_kd_ratio": ("dec/akd", ":.3f"),
    "decoder_prefix_to_tail_ratio": ("dec_p/t", ":.3f"),
    "noise_to_signal_ratio": ("noise/sig", ":.3f"),
    "decoder_grad_to_bc_grad_ratio": ("decg/bc", ":.3f"),
    "decoder_grad_to_action_kd_grad_ratio": ("decg/akd", ":.3f"),
    "decoder_grad_to_behavior_grad_ratio": ("decg/beh", ":.3f"),
    "student_teacher_decoder_cos": ("dec_cos", ":.3f"),
    "student_decoder_norm": ("s_dec_n", ":.3f"),
    "teacher_decoder_norm": ("t_dec_n", ":.3f"),
    "student_decoder_train_eval_gap": ("dec_gap", ":.3f"),
}


def _is_act_kd_training(cfg: TrainPipelineConfig) -> bool:
    return bool(cfg.policy and cfg.policy.type == "act" and getattr(cfg.policy, "kd", False))


def _make_policy_metric_meters(metric_names: list[str]) -> dict[str, AverageMeter]:
    meters = {}
    for metric_name in metric_names:
        display_name, fmt = POLICY_METRIC_SPECS.get(metric_name, (metric_name, ":.3f"))
        meters[metric_name] = AverageMeter(display_name, fmt)
    return meters


def _infer_teacher_source_kind(teacher_source: Path, teacher_pretrained_path: Path) -> str:
    if teacher_source.is_file():
        if teacher_source.name == "train_config.json":
            return "train_config_file"
        if teacher_source.name == "config.json":
            return "config_file"
        return "file"

    if teacher_source == teacher_pretrained_path:
        return "pretrained_dir"
    if teacher_source / "pretrained_model" == teacher_pretrained_path:
        return "checkpoint_dir"
    if (teacher_source / "checkpoints").is_dir():
        if (teacher_source / "checkpoints" / "last").exists():
            return "output_dir_last"
        return "output_dir_checkpoint_scan"
    return "directory"


def _extract_teacher_checkpoint_step(teacher_pretrained_path: Path) -> int | None:
    for candidate in (teacher_pretrained_path.parent, teacher_pretrained_path.parent.parent):
        if candidate.name.isdigit():
            return int(candidate.name)
    return None


def _prepare_act_kd_startup(
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    accelerator: Accelerator,
) -> KDTeacherMetadata | None:
    if not _is_act_kd_training(cfg):
        return None

    if not has_method(policy, "attach_teacher_bundle"):
        raise RuntimeError(
            "ACT KD startup requires the policy to expose `attach_teacher_bundle(...)`."
        )

    existing_metadata = cfg.get_kd_teacher_metadata()
    pinned_teacher_path = cfg.get_pinned_kd_teacher_pretrained_path()
    if pinned_teacher_path is not None:
        teacher_pretrained_path = Path(pinned_teacher_path)
        if not teacher_pretrained_path.exists():
            raise FileNotFoundError(
                "Pinned ACT KD teacher snapshot from run metadata does not exist anymore: "
                f"{teacher_pretrained_path}"
            )
    else:
        runtime_teacher_source = cfg.get_runtime_teacher_source_path()
        if runtime_teacher_source is None:
            raise RuntimeError("ACT KD is enabled but no teacher path is available at startup.")
        teacher_pretrained_path = _resolve_teacher_pretrained_path(runtime_teacher_source)

    teacher_bundle = load_act_teacher_bundle(
        student_policy=policy,
        student_preprocessor=preprocessor,
        teacher_pretrained_path=teacher_pretrained_path,
    )
    policy.attach_teacher_bundle(teacher_bundle)

    resolved_teacher_path = Path(
        teacher_bundle.resolved_policy_path or teacher_pretrained_path
    ).resolve()
    original_teacher_policy_path = getattr(cfg.policy, "teacher_policy_path", None)
    original_teacher_train_config = getattr(cfg.policy, "teacher_train_config", None)
    original_teacher_source = original_teacher_policy_path or original_teacher_train_config or resolved_teacher_path
    pinned_at = None
    if existing_metadata is not None:
        pinned_at = existing_metadata.extra.get("pinned_at")

    metadata = merge_kd_teacher_metadata(
        {
            "comparison_space": teacher_bundle.comparison_space,
            "processor_compatibility": (
                existing_metadata.processor_compatibility_mode
                if existing_metadata is not None and existing_metadata.processor_compatibility_mode is not None
                else "strict" if getattr(cfg.policy, "kd_strict_processor_compatibility", False) else "relaxed"
            ),
            "metric_aggregation_mode": (
                "mean_across_processes_before_logging"
                if accelerator.num_processes > 1
                else "single_process"
            ),
            "teacher": {
                "original_path": original_teacher_source,
                "pinned_pretrained_path": resolved_teacher_path,
                "checkpoint_step": (
                    existing_metadata.teacher_checkpoint_step
                    if existing_metadata is not None and existing_metadata.teacher_checkpoint_step is not None
                    else _extract_teacher_checkpoint_step(resolved_teacher_path)
                ),
                "source_kind": (
                    existing_metadata.teacher_source_kind
                    if existing_metadata is not None and existing_metadata.teacher_source_kind is not None
                    else _infer_teacher_source_kind(Path(original_teacher_source), resolved_teacher_path)
                ),
                "pinned_from_run_metadata": cfg.get_pinned_kd_teacher_pretrained_path() is not None,
            },
            "pinned_at": pinned_at or datetime.now(timezone.utc).isoformat(),
        },
        existing_metadata,
    )

    cfg.kd_teacher_metadata = kd_teacher_metadata_to_dict(metadata)

    if accelerator.is_main_process:
        metadata_path = cfg.pin_kd_teacher_metadata(metadata, cfg.output_dir)
        logging.info(
            "Pinned ACT KD teacher to %s (%s). Metadata written to %s",
            resolved_teacher_path,
            metadata.teacher_source_kind,
            metadata_path,
        )

    accelerator.wait_for_everyone()
    return metadata


def _update_train_tracker_from_output_dict(
    train_tracker: MetricsTracker,
    output_dict: dict[str, Any] | None,
    accelerator: Accelerator,
) -> dict[str, float]:
    reduced_output_dict = reduce_metrics_dict(
        output_dict,
        accelerator=accelerator,
        reduction="mean",
        device=accelerator.device,
    )
    if not reduced_output_dict:
        return {}

    if (
        "kd_to_bc_ratio" not in reduced_output_dict
        and "kd_weighted_l1_loss" in reduced_output_dict
        and "l1_loss" in reduced_output_dict
    ):
        l1_loss = reduced_output_dict["l1_loss"]
        if l1_loss != 0:
            reduced_output_dict["kd_weighted_to_bc_ratio"] = reduced_output_dict["kd_weighted_l1_loss"] / l1_loss

    missing_metric_names = [name for name in reduced_output_dict if name not in train_tracker.metrics]
    if missing_metric_names:
        train_tracker.register_metrics(_make_policy_metric_meters(missing_metric_names))

    train_tracker.update_from_dict(reduced_output_dict)
    return reduced_output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    _prepare_act_kd_startup(cfg, policy, preprocessor, accelerator)

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        _update_train_tracker_from_output_dict(train_tracker, output_dict, accelerator)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_logger.log_dict(train_tracker.to_dict(), step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
