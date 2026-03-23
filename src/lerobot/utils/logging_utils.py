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
from collections.abc import Callable, Mapping
from typing import Any

import torch

from lerobot.utils.utils import format_big_number


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(cfg, dataset, metrics, initial_step=step)

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "accelerator",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        accelerator: Callable | None = None,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        self.accelerator = accelerator

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def register_metrics(self, metrics: Mapping[str, AverageMeter], overwrite: bool = False) -> None:
        """Registers additional metrics after tracker construction."""
        for name, meter in metrics.items():
            if name in self.metrics and not overwrite:
                raise ValueError(f"Metric '{name}' is already registered.")
            self.metrics[name] = meter

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def update_from_dict(
        self,
        values: Mapping[str, Any],
        *,
        n: int = 1,
        ignore_missing: bool = False,
    ) -> None:
        """Updates registered meters from a dictionary of scalar values."""
        for key, value in values.items():
            if key not in self.metrics:
                if ignore_missing:
                    continue
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Metric '{key}' must be scalar, got tensor with shape {tuple(value.shape)}.")
                scalar = value.item()
            else:
                scalar = float(value)

            self.metrics[key].update(scalar, n=n)

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()


def reduce_metrics_dict(
    metrics: Mapping[str, Any] | None,
    accelerator: Any | None = None,
    *,
    reduction: str = "mean",
    device: torch.device | None = None,
) -> dict[str, float]:
    """Reduces a flat scalar metrics dictionary across processes."""
    if not metrics:
        return {}

    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            tensor = value.detach()
            target_device = device or tensor.device
            if tensor.device != target_device:
                tensor = tensor.to(target_device)
            tensor = tensor.reshape(())
        elif isinstance(value, bool):
            tensor = torch.tensor(float(value), device=device or getattr(accelerator, "device", "cpu"))
        elif isinstance(value, int | float):
            tensor = torch.tensor(float(value), device=device or getattr(accelerator, "device", "cpu"))
        else:
            continue

        if accelerator is not None and hasattr(accelerator, "reduce"):
            tensor = accelerator.reduce(tensor, reduction=reduction)

        reduced[key] = float(tensor.item() if isinstance(tensor, torch.Tensor) else tensor)

    return reduced
