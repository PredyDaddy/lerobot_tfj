from pathlib import Path
from unittest.mock import patch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_infer import InferConfig, _resolve_pretrained_policy_path, infer
from tests.mocks.mock_robot import MockRobotConfig


def test_resolve_pretrained_policy_path_accepts_parent_directory(tmp_path):
    policy_root = tmp_path / "pi_model"
    pretrained_dir = policy_root / "pretrained_model"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "config.json").write_text("{}", encoding="utf-8")

    assert _resolve_pretrained_policy_path(policy_root) == pretrained_dir


def test_infer_uses_record_loop_without_dataset():
    policy_cfg = PI05Config(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        pretrained_path=Path("pi_model"),
    )
    cfg = InferConfig(
        robot=MockRobotConfig(),
        policy=policy_cfg,
        fps=15,
        task="test task",
        run_time_s=0.2,
    )

    with (
        patch("lerobot.scripts.lerobot_infer.load_policy_runtime", return_value=("policy", "pre", "post")),
        patch(
            "lerobot.scripts.lerobot_infer.init_keyboard_listener",
            return_value=(None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}),
        ),
        patch("lerobot.scripts.lerobot_infer.is_headless", return_value=True),
        patch("lerobot.scripts.lerobot_infer.record_loop") as mock_record_loop,
    ):
        infer(cfg)

    kwargs = mock_record_loop.call_args.kwargs
    assert kwargs["dataset"] is None
    assert kwargs["teleop"] is None
    assert kwargs["policy"] == "policy"
    assert kwargs["preprocessor"] == "pre"
    assert kwargs["postprocessor"] == "post"
    assert kwargs["fps"] == 15
    assert kwargs["single_task"] == "test task"
    assert kwargs["control_time_s"] == 0.2
