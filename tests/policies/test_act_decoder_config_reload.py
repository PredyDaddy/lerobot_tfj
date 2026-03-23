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

from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_act_config(tmp_path: Path) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        kd=True,
        teacher_policy_path=tmp_path / "teacher",
        decoder_kd={
            "enabled": True,
            "peak_weight": 0.25,
            "loss_type": "smooth_l1",
            "smooth_l1_beta": 0.5,
            "latent_mode": "zero",
            "overlap_steps": 8,
            "temporal_decay": 0.1,
            "prefix_weight": 1.5,
            "tail_weight": 0.75,
            "start_step": 10,
            "ramp_steps": 20,
            "anneal_start_step": 40,
            "end_step": 80,
            "projection": {
                "enabled": True,
                "kind": "linear",
                "placement": "student_only",
                "output_dim": 256,
                "bias": False,
            },
        },
    )


def test_act_config_from_pretrained_round_trip_preserves_decoder_kd(tmp_path):
    config = _make_act_config(tmp_path)
    save_dir = tmp_path / "act_config"

    config.save_pretrained(save_dir)
    loaded = ACTConfig.from_pretrained(save_dir)

    assert isinstance(loaded, ACTConfig)
    assert loaded.kd is True
    assert loaded.decoder_kd.enabled is True
    assert loaded.decoder_kd.peak_weight == 0.25
    assert loaded.decoder_kd.loss_type == "smooth_l1"
    assert loaded.decoder_kd.smooth_l1_beta == 0.5
    assert loaded.decoder_kd.latent_mode == "zero"
    assert loaded.decoder_kd.overlap_steps == 8
    assert loaded.decoder_kd.temporal_decay == 0.1
    assert loaded.decoder_kd.prefix_weight == 1.5
    assert loaded.decoder_kd.tail_weight == 0.75
    assert loaded.decoder_kd.start_step == 10
    assert loaded.decoder_kd.ramp_steps == 20
    assert loaded.decoder_kd.anneal_start_step == 40
    assert loaded.decoder_kd.end_step == 80
    assert loaded.decoder_kd.projection is not None
    assert loaded.decoder_kd.projection.enabled is True
    assert loaded.decoder_kd.projection.kind == "linear"
    assert loaded.decoder_kd.projection.placement == "student_only"
    assert loaded.decoder_kd.projection.output_dim == 256
    assert loaded.decoder_out_dim == 1024
    assert loaded.dim_model == 512


def test_train_pipeline_config_from_pretrained_round_trip_preserves_decoder_kd(tmp_path):
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/test"),
        policy=_make_act_config(tmp_path),
        output_dir=tmp_path / "outputs",
    )
    save_dir = tmp_path / "train_config_dir"

    train_cfg.save_pretrained(save_dir)
    loaded = TrainPipelineConfig.from_pretrained(save_dir)

    assert isinstance(loaded.policy, ACTConfig)
    assert loaded.policy.decoder_kd.enabled is True
    assert loaded.policy.decoder_kd.loss_type == "smooth_l1"
    assert loaded.policy.decoder_kd.latent_mode == "zero"
    assert loaded.policy.decoder_kd.projection is not None
    assert loaded.policy.decoder_kd.projection.output_dim == 256
    assert loaded.policy.decoder_out_dim == 1024
