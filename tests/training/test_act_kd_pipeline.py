#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
import socket
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.distillation_utils import ACTTeacherBundle, KDProcessorCompatibilityReport
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.scripts import lerobot_train
from lerobot.scripts.lerobot_train import (
    ACT_KD_METADATA_FILENAME,
    _prepare_act_kd_startup,
    _update_train_tracker_from_output_dict,
    train,
)
from lerobot.utils.logging_utils import MetricsTracker
from lerobot.utils.train_metadata import save_kd_teacher_metadata
from lerobot.utils.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint

INPUT_FEATURES = {
    "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(4,)),
}
OUTPUT_FEATURES = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
}
NORM_MAP = {
    FeatureType.ENV: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}
BASE_STATS = {
    "observation.environment_state": {
        "mean": torch.zeros(4, dtype=torch.float32),
        "std": torch.ones(4, dtype=torch.float32),
    },
    "action": {
        "mean": torch.zeros(2, dtype=torch.float32),
        "std": torch.ones(2, dtype=torch.float32),
    },
}


def _clone_stats(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {key: {name: value.clone() for name, value in values.items()} for key, values in stats.items()}


def _make_act_config(*, kd: bool, teacher_policy_path: Path | None = None) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        use_vae=False,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        normalization_mapping=NORM_MAP,
        kd=kd,
        teacher_policy_path=teacher_policy_path,
        chunk_size=2,
        n_action_steps=2,
    )


def _make_stage2_act_config(tmp_path: Path, *, teacher_policy_path: Path | None = None) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        use_vae=False,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        normalization_mapping=NORM_MAP,
        kd=True,
        teacher_policy_path=teacher_policy_path or (tmp_path / "teacher"),
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        decoder_kd={
            "enabled": True,
            "peak_weight": 0.2,
            "start_step": 0,
            "ramp_steps": 4,
            "anneal_start_step": 8,
            "end_step": 12,
            "enable_noise_gate": True,
            "enable_grad_gate": True,
            "log_grad_ratio": False,
        },
    )


def _make_preprocessor(
    *,
    rename_map: dict[str, str] | None = None,
    stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map=rename_map or {}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device="cpu"),
            NormalizerProcessorStep(
                features={**INPUT_FEATURES, **OUTPUT_FEATURES},
                norm_map=NORM_MAP,
                stats=_clone_stats(BASE_STATS) if stats is None else _clone_stats(stats),
                device="cpu",
            ),
        ],
        name="policy_preprocessor",
    )


def _make_postprocessor() -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(steps=[], name="policy_postprocessor")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.meta = SimpleNamespace(
            stats=_clone_stats(BASE_STATS),
            episodes={"dataset_from_index": [0, 1], "dataset_to_index": [1, 2]},
        )
        self.num_frames = 4
        self.num_episodes = 2
        self.episodes = [0, 1]

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation.environment_state": torch.full((4,), float(index), dtype=torch.float32),
            "action": torch.full((2,), 0.5 * (index + 1), dtype=torch.float32),
        }


class DummyPolicy(torch.nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.attached_teacher_bundles = []

    def forward(self, batch):
        prediction = torch.ones_like(batch["action"]) * self.weight
        l1_loss = torch.nn.functional.l1_loss(prediction, batch["action"])
        output_dict = {"l1_loss": float(l1_loss.detach())}
        if self.config.kd:
            output_dict["kd_l1_loss"] = float((l1_loss.detach() * 0.5).item())
            output_dict["kd_weighted_l1_loss"] = float((l1_loss.detach() * 0.75).item())
            output_dict["kd_overlap_steps"] = 2.0
        return l1_loss, output_dict

    def predict_action_chunk(self, batch):
        action = batch["action"]
        return torch.zeros(action.shape[0], 1, action.shape[-1], dtype=action.dtype, device=action.device)

    def select_action(self, batch):
        return batch["action"]

    def attach_teacher_bundle(self, teacher_bundle):
        self.attached_teacher_bundles.append(teacher_bundle)
        self.__dict__["_teacher_bundle"] = teacher_bundle


class DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.num_processes = 1

    def wait_for_everyone(self):
        return None

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def prepare(self, *args):
        return args

    def unwrap_model(self, model, keep_fp32_wrapper=False):
        return model

    def reduce(self, tensor, reduction="sum", scale=1.0):
        return tensor

    def end_training(self):
        return None


class DistributedTestAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_processes = dist.get_world_size()
        self.is_main_process = dist.get_rank() == 0

    def reduce(self, tensor, reduction="sum", scale=1.0):
        reduced = tensor.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            reduced /= dist.get_world_size()
        elif reduction not in {"sum", "none"}:
            raise ValueError(reduction)
        return reduced


def _make_train_cfg(tmp_path: Path, policy_cfg: ACTConfig) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/repo"),
        policy=policy_cfg,
        output_dir=tmp_path / "run",
        steps=1,
        eval_freq=-1,
        log_freq=1,
        num_workers=0,
        batch_size=2,
        save_checkpoint=False,
    )
    cfg.validate = lambda: None
    cfg.optimizer = AdamWConfig(lr=0.1, grad_clip_norm=0.0)
    return cfg


def _patch_training_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    student_policy: DummyPolicy,
    teacher_policy: DummyPolicy,
    student_preprocessor: PolicyProcessorPipeline,
    teacher_preprocessor: PolicyProcessorPipeline,
    resolved_teacher_path: Path,
    bundle_error: Exception | None = None,
):
    dummy_dataset = DummyDataset()

    def fake_load_act_teacher_bundle(*, student_policy, student_preprocessor, teacher_pretrained_path):
        assert Path(teacher_pretrained_path) == resolved_teacher_path
        if bundle_error is not None:
            raise bundle_error
        return ACTTeacherBundle(
            policy=teacher_policy,
            preprocessor=teacher_preprocessor,
            processor_compatibility=KDProcessorCompatibilityReport(compatible=True),
            resolved_policy_path=resolved_teacher_path,
        )

    monkeypatch.setattr(lerobot_train, "init_logging", lambda accelerator=None: None)
    monkeypatch.setattr(lerobot_train, "make_dataset", lambda cfg: dummy_dataset)
    monkeypatch.setattr(lerobot_train, "make_policy", lambda cfg, ds_meta, rename_map: student_policy)
    monkeypatch.setattr(
        lerobot_train,
        "make_pre_post_processors",
        lambda policy_cfg, pretrained_path=None, **kwargs: (student_preprocessor, _make_postprocessor()),
    )
    monkeypatch.setattr(
        lerobot_train,
        "make_optimizer_and_scheduler",
        lambda cfg, policy: (torch.optim.SGD(policy.parameters(), lr=0.1), None),
    )
    monkeypatch.setattr(lerobot_train, "_resolve_teacher_pretrained_path", lambda path: resolved_teacher_path)
    monkeypatch.setattr(lerobot_train, "load_act_teacher_bundle", fake_load_act_teacher_bundle)


def test_train_pins_teacher_and_writes_run_metadata(tmp_path, monkeypatch):
    teacher_source = tmp_path / "teacher_run"
    resolved_teacher_path = teacher_source / "checkpoints" / "000123" / "pretrained_model"
    resolved_teacher_path.mkdir(parents=True)
    cfg = _make_train_cfg(tmp_path, _make_act_config(kd=True, teacher_policy_path=teacher_source))
    student_policy = DummyPolicy(cfg.policy)
    teacher_policy = DummyPolicy(_make_act_config(kd=False))
    student_preprocessor = _make_preprocessor()
    teacher_preprocessor = _make_preprocessor()

    _patch_training_harness(
        monkeypatch,
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        student_preprocessor=student_preprocessor,
        teacher_preprocessor=teacher_preprocessor,
        resolved_teacher_path=resolved_teacher_path,
    )

    train(cfg, accelerator=DummyAccelerator())

    metadata_path = cfg.output_dir / ACT_KD_METADATA_FILENAME
    assert metadata_path.is_file()
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["comparison_space"] == "normalized_action_space"
    assert metadata["processor_compatibility"] == "strict"
    assert metadata["metric_aggregation_mode"] == "single_process"
    assert metadata["teacher"]["original_path"] == str(teacher_source)
    assert metadata["teacher"]["pinned_pretrained_path"] == str(resolved_teacher_path.resolve())
    assert metadata["teacher"]["checkpoint_step"] == 123
    assert metadata["teacher"]["source_kind"] == "output_dir_checkpoint_scan"
    assert metadata["teacher"]["pinned_from_run_metadata"] is False
    assert "pinned_at" in metadata
    assert len(student_policy.attached_teacher_bundles) == 1
    assert isinstance(student_policy.attached_teacher_bundles[0], ACTTeacherBundle)
    assert student_policy.attached_teacher_bundles[0].policy is teacher_policy
    assert student_policy.attached_teacher_bundles[0].preprocessor is teacher_preprocessor


def test_train_fails_fast_on_incompatible_teacher_processor(tmp_path, monkeypatch):
    teacher_source = tmp_path / "teacher_run"
    resolved_teacher_path = teacher_source / "checkpoints" / "000123" / "pretrained_model"
    resolved_teacher_path.mkdir(parents=True)
    cfg = _make_train_cfg(tmp_path, _make_act_config(kd=True, teacher_policy_path=teacher_source))
    student_policy = DummyPolicy(cfg.policy)
    teacher_policy = DummyPolicy(_make_act_config(kd=False))
    student_preprocessor = _make_preprocessor()
    teacher_preprocessor = _make_preprocessor(
        rename_map={"observation.environment_state_raw": "observation.environment_state"}
    )

    _patch_training_harness(
        monkeypatch,
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        student_preprocessor=student_preprocessor,
        teacher_preprocessor=teacher_preprocessor,
        resolved_teacher_path=resolved_teacher_path,
        bundle_error=ValueError("Student and teacher rename behavior must match."),
    )

    with pytest.raises(ValueError, match="rename behavior must match"):
        train(cfg, accelerator=DummyAccelerator())


def test_prepare_act_kd_startup_keeps_validate_pinned_checkpoint_metadata(tmp_path, monkeypatch):
    moving_teacher_source = tmp_path / "moving_teacher"
    checkpoint_pinned_teacher_path = tmp_path / "teacher" / "checkpoints" / "000007" / "pretrained_model"
    checkpoint_pinned_teacher_path.mkdir(parents=True)
    stale_run_level_teacher_path = tmp_path / "teacher" / "checkpoints" / "000003" / "pretrained_model"
    stale_run_level_teacher_path.mkdir(parents=True)

    cfg = _make_train_cfg(tmp_path, _make_act_config(kd=True, teacher_policy_path=moving_teacher_source))
    cfg.resume = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.kd_teacher_metadata = {
        "comparison_space": "normalized_action_space",
        "processor_compatibility": "strict",
        "metric_aggregation_mode": "mean_across_processes_before_logging",
        "teacher": {
            "original_path": str(moving_teacher_source),
            "pinned_pretrained_path": str(checkpoint_pinned_teacher_path),
            "source_kind": "checkpoint_dir",
            "checkpoint_step": 7,
            "pinned_from_run_metadata": True,
        },
    }
    save_kd_teacher_metadata(
        {
            "comparison_space": "normalized_action_space",
            "processor_compatibility": "strict",
            "teacher": {
                "original_path": str(moving_teacher_source),
                "pinned_pretrained_path": str(stale_run_level_teacher_path),
                "source_kind": "output_dir_last",
                "checkpoint_step": 3,
                "pinned_from_run_metadata": True,
            },
        },
        cfg.output_dir,
    )

    student_policy = DummyPolicy(cfg.policy)
    teacher_policy = DummyPolicy(_make_act_config(kd=False))
    student_preprocessor = _make_preprocessor()
    teacher_preprocessor = _make_preprocessor()
    loaded_paths = []

    monkeypatch.setattr(
        lerobot_train,
        "_resolve_teacher_pretrained_path",
        lambda path: (_ for _ in ()).throw(AssertionError("moving target should not be re-resolved")),
    )
    monkeypatch.setattr(
        lerobot_train,
        "load_act_teacher_bundle",
        lambda *, student_policy, student_preprocessor, teacher_pretrained_path: loaded_paths.append(
            Path(teacher_pretrained_path)
        )
        or ACTTeacherBundle(
            policy=teacher_policy,
            preprocessor=teacher_preprocessor,
            processor_compatibility=KDProcessorCompatibilityReport(compatible=True),
            resolved_policy_path=Path(teacher_pretrained_path),
        ),
    )

    metadata = _prepare_act_kd_startup(cfg, student_policy, student_preprocessor, DummyAccelerator())

    assert loaded_paths == [checkpoint_pinned_teacher_path]
    assert metadata is not None
    assert metadata.resolved_teacher_pretrained_path == checkpoint_pinned_teacher_path.resolve()
    assert metadata.teacher_source_kind == "checkpoint_dir"
    assert metadata.teacher_checkpoint_step == 7


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_logging_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        tracker = MetricsTracker(
            batch_size=2,
            num_frames=8,
            num_episodes=2,
            metrics={},
            accelerator=None,
        )
        accelerator = DistributedTestAccelerator()
        _update_train_tracker_from_output_dict(
            tracker,
            {
                "l1_loss": 2.0 + rank,
                "kd_l1_loss": 1.0 + rank,
                "kd_weighted_l1_loss": 1.0 + rank,
                "kd_overlap_steps": 4.0 + rank,
            },
            accelerator,
        )
        with open(Path(output_dir) / f"rank_{rank}.json", "w", encoding="utf-8") as f:
            json.dump(tracker.to_dict(), f)
    finally:
        dist.destroy_process_group()


def test_update_train_tracker_from_output_dict_two_process_smoke(tmp_path):
    world_size = 2
    port = _get_free_port()
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_distributed_logging_worker, args=(rank, world_size, port, str(tmp_path)))
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    results = []
    for rank in range(world_size):
        with open(tmp_path / f"rank_{rank}.json", encoding="utf-8") as f:
            results.append(json.load(f))

    for result in results:
        assert result["l1_loss"] == pytest.approx(2.5)
        assert result["kd_l1_loss"] == pytest.approx(1.5)
        assert result["kd_weighted_l1_loss"] == pytest.approx(1.5)
        assert result["kd_overlap_steps"] == pytest.approx(4.5)
        assert result["kd_weighted_to_bc_ratio"] == pytest.approx(0.6)
        assert "kd_to_bc_ratio" not in result


def test_update_train_tracker_preserves_model_side_kd_to_bc_ratio():
    tracker = MetricsTracker(batch_size=2, num_frames=8, num_episodes=2, metrics={})

    reduced = _update_train_tracker_from_output_dict(
        tracker,
        {
            "l1_loss": 2.0,
            "kd_l1_loss": 1.0,
            "kd_weighted_l1_loss": 1.5,
            "kd_to_bc_ratio": 0.75,
        },
        DummyAccelerator(),
    )

    assert reduced["kd_to_bc_ratio"] == pytest.approx(0.75)
    assert "kd_weighted_to_bc_ratio" not in reduced
    assert tracker.to_dict()["kd_to_bc_ratio"] == pytest.approx(0.75)


def test_update_train_tracker_accepts_decoder_kd_metrics_without_phase1_pollution():
    tracker = MetricsTracker(batch_size=2, num_frames=8, num_episodes=2, metrics={})

    reduced = _update_train_tracker_from_output_dict(
        tracker,
        {
            "l1_loss": 2.0,
            "decoder_kd_loss": 1.2,
            "decoder_kd_weighted_loss": 0.6,
            "decoder_kd_valid_ratio": 0.75,
            "decoder_kd_weighted_to_bc_ratio": 0.3,
            "noise_to_signal_ratio": 0.4,
        },
        DummyAccelerator(),
    )

    tracker_dict = tracker.to_dict()
    assert reduced["decoder_kd_loss"] == pytest.approx(1.2)
    assert reduced["decoder_kd_weighted_loss"] == pytest.approx(0.6)
    assert reduced["decoder_kd_weighted_to_bc_ratio"] == pytest.approx(0.3)
    assert tracker_dict["decoder_kd_loss"] == pytest.approx(1.2)
    assert tracker_dict["decoder_kd_weighted_loss"] == pytest.approx(0.6)
    assert tracker_dict["decoder_kd_valid_ratio"] == pytest.approx(0.75)
    assert tracker_dict["noise_to_signal_ratio"] == pytest.approx(0.4)
    assert "kd_l1_loss" not in tracker_dict
    assert "kd_weighted_l1_loss" not in tracker_dict
    assert tracker.metrics["decoder_kd_loss"].name == "dec_raw"
    assert tracker.metrics["decoder_kd_weighted_to_bc_ratio"].name == "dec/bc"


def test_update_train_tracker_phase1_payload_does_not_register_decoder_metrics():
    tracker = MetricsTracker(batch_size=2, num_frames=8, num_episodes=2, metrics={})

    _update_train_tracker_from_output_dict(
        tracker,
        {
            "l1_loss": 2.0,
            "kd_l1_loss": 1.0,
            "kd_weighted_l1_loss": 1.5,
            "kd_overlap_steps": 2.0,
        },
        DummyAccelerator(),
    )

    tracker_dict = tracker.to_dict()
    assert tracker_dict["kd_l1_loss"] == pytest.approx(1.0)
    assert tracker_dict["kd_weighted_l1_loss"] == pytest.approx(1.5)
    assert tracker_dict["kd_weighted_to_bc_ratio"] == pytest.approx(0.75)
    assert "decoder_kd_loss" not in tracker_dict
    assert "decoder_kd_weighted_loss" not in tracker_dict
    assert "decoder_kd_weighted_to_bc_ratio" not in tracker_dict


def test_stage2_checkpoint_save_and_resume_restore_decoder_kd_state(tmp_path):
    cfg = _make_train_cfg(tmp_path, _make_stage2_act_config(tmp_path))
    cfg.save_checkpoint = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy(cfg.policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    preprocessor = _make_preprocessor()
    postprocessor = _make_postprocessor()

    policy._decoder_kd_step_buffer.fill_(3)
    expected_scheduler_weight = policy._get_decoder_kd_scheduler_weight(policy._get_decoder_kd_step())
    checkpoint_step = 2
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, checkpoint_step)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=checkpoint_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        scheduler=None,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    pretrained_dir = checkpoint_dir / "pretrained_model"
    restored_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
    restored_policy = ACTPolicy.from_pretrained(pretrained_dir)
    restored_optimizer = torch.optim.SGD(restored_policy.parameters(), lr=0.1)
    restored_step, _, _ = load_training_state(checkpoint_dir, restored_optimizer, None)

    assert restored_cfg.policy.decoder_kd.enabled is True
    assert restored_cfg.policy.decoder_kd.peak_weight == pytest.approx(0.2)
    assert restored_cfg.policy.decoder_kd.require_action_kd is True
    assert restored_policy._get_decoder_kd_step() == 3
    assert restored_policy._get_decoder_kd_scheduler_weight(restored_policy._get_decoder_kd_step()) == pytest.approx(
        expected_scheduler_weight
    )
    assert restored_step == checkpoint_step
