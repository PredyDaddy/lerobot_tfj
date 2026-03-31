import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
REPO_SRC_STR = str(REPO_SRC)
if REPO_SRC_STR not in sys.path:
    sys.path.insert(0, REPO_SRC_STR)

from lerobot.runtime.so101_pickplace.schemas import ActionCommand, GuardResult
from lerobot.scripts import lerobot_record as record_script
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


class _FakeClock:
    def __init__(self, step: float = 0.01):
        self.current = 0.0
        self.step = step

    def perf_counter(self) -> float:
        value = self.current
        self.current += self.step
        return value


class _StaticBridge:
    def __init__(self, payload):
        self.payload = payload

    def decide(self, **kwargs):
        return self.payload


class _StaticGuard:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def validate(self, **kwargs):
        self.calls += 1
        return self.payload


class _CaptureLogger:
    def __init__(self):
        self.steps = []
        self.guard_rejects = []

    def log_step(self, **payload):
        self.steps.append(payload)

    def log_guard_reject(self, **payload):
        self.guard_rejects.append(payload)


class _CaptureDataset:
    def __init__(self, fps: int):
        self.fps = fps
        self.frames = []
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            },
            "action": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            },
        }

    def add_frame(self, frame):
        self.frames.append(frame)


class _ResettableStub:
    def __init__(self):
        self.reset_calls = 0
        self.config = type("Cfg", (), {"device": "cpu", "use_amp": False})()

    def reset(self):
        self.reset_calls += 1


def _make_robot_and_teleop():
    robot = MockRobot(
        MockRobotConfig(
            random_values=False,
            static_values=[0.0, 0.0, 0.0],
        )
    )
    teleop = MockTeleop(
        MockTeleopConfig(
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    robot.connect()
    teleop.connect()
    return robot, teleop


@pytest.fixture
def identity_processors():
    return {
        "teleop_action_processor": lambda pair: pair[0],
        "robot_action_processor": lambda pair: pair[0],
        "robot_observation_processor": lambda obs: obs,
    }


def _policy_feature_dict():
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
        }
    }, {
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
        }
    }


def test_record_loop_guard_reject_skips_send_and_still_sleeps(monkeypatch, identity_processors):
    robot, teleop = _make_robot_and_teleop()
    dataset = _CaptureDataset(fps=10)
    sleep_calls = []
    send_calls = []
    logger = _CaptureLogger()
    guard = _StaticGuard({"accept": False, "reason": "reject_once"})
    clock = _FakeClock(step=0.01)

    monkeypatch.setattr(record_script, "precise_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(record_script.time, "perf_counter", clock.perf_counter)
    original_send_action = robot.send_action
    robot.send_action = lambda action: send_calls.append(action) or original_send_action(action)

    try:
        record_script.record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False},
            fps=10,
            teleop=teleop,
            dataset=dataset,
            control_time_s=0.025,
            single_task="guard reject",
            safety_guard=guard,
            step_event_logger=logger,
            runtime_state={},
            **identity_processors,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert guard.calls == 1
    assert send_calls == []
    assert dataset.frames == []
    assert sleep_calls == [pytest.approx(0.09)]
    assert logger.steps == []
    assert len(logger.guard_rejects) == 1
    assert logger.guard_rejects[0]["guard_result"]["reason"] == "reject_once"


def test_record_loop_bridge_delta_is_applied_before_send(monkeypatch, identity_processors):
    robot, teleop = _make_robot_and_teleop()
    dataset = _CaptureDataset(fps=10)
    sleep_calls = []
    sent_actions = []
    logger = _CaptureLogger()
    bridge = _StaticBridge(
        {
            "action_delta": {"motor_1.pos": 0.5},
            "task_override": "bridge override",
        }
    )
    guard = _StaticGuard({"accept": True})
    clock = _FakeClock(step=0.01)

    monkeypatch.setattr(record_script, "precise_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(record_script.time, "perf_counter", clock.perf_counter)
    original_send_action = robot.send_action
    robot.send_action = lambda action: sent_actions.append(action.copy()) or original_send_action(action)

    try:
        record_script.record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False},
            fps=10,
            teleop=teleop,
            dataset=dataset,
            control_time_s=0.025,
            single_task="original task",
            perception_bridge=bridge,
            safety_guard=guard,
            step_event_logger=logger,
            runtime_state={},
            **identity_processors,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert sleep_calls == [pytest.approx(0.09)]
    assert sent_actions == [{"motor_1.pos": 1.5, "motor_2.pos": 2.0, "motor_3.pos": 3.0}]
    assert len(dataset.frames) == 1
    assert dataset.frames[0]["action"].tolist() == pytest.approx([1.5, 2.0, 3.0])
    assert len(logger.steps) == 1
    assert logger.steps[0]["task"] == "bridge override"
    assert logger.steps[0]["sent_action"]["motor_1.pos"] == pytest.approx(1.5)


def test_record_loop_runtime_guard_reject_sends_fail_safe_and_persists_sent_action(monkeypatch, identity_processors):
    robot, teleop = _make_robot_and_teleop()
    dataset = _CaptureDataset(fps=10)
    sleep_calls = []
    send_calls = []
    logger = _CaptureLogger()
    guard = _StaticGuard(
        GuardResult.reject(
            error_code="reject_once",
            reason="reject_once",
            fail_safe_action=ActionCommand(joint_positions=(9.0, 8.0, 7.0), label="hold_position"),
        )
    )
    clock = _FakeClock(step=0.01)

    monkeypatch.setattr(record_script, "precise_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(record_script.time, "perf_counter", clock.perf_counter)
    original_send_action = robot.send_action
    robot.send_action = lambda action: send_calls.append(action.copy()) or original_send_action(action)

    try:
        record_script.record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False},
            fps=10,
            teleop=teleop,
            dataset=dataset,
            control_time_s=0.025,
            single_task="runtime reject",
            safety_guard=guard,
            step_event_logger=logger,
            runtime_state={},
            **identity_processors,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert guard.calls == 1
    assert send_calls == [{"motor_1.pos": 9.0, "motor_2.pos": 8.0, "motor_3.pos": 7.0}]
    assert sleep_calls == [pytest.approx(0.09)]
    assert len(dataset.frames) == 1
    assert dataset.frames[0]["action"].tolist() == pytest.approx([9.0, 8.0, 7.0])
    assert len(logger.steps) == 1
    assert len(logger.guard_rejects) == 1
    assert logger.guard_rejects[0]["guard_result"]["decision"] == "reject"
    assert logger.steps[0]["sent_action"]["motor_1.pos"] == pytest.approx(9.0)


def test_record_loop_runtime_guard_halt_sets_stop_recording_and_preserves_sleep(monkeypatch, identity_processors):
    robot, teleop = _make_robot_and_teleop()
    sleep_calls = []
    send_calls = []
    logger = _CaptureLogger()
    guard = _StaticGuard(
        GuardResult.halt(
            error_code="fatal_guard",
            reason="fatal_guard",
            fail_safe_action=ActionCommand(joint_positions=(0.0, 0.0, 0.0), label="hold_position"),
        )
    )
    clock = _FakeClock(step=0.01)
    events = {"exit_early": False, "stop_recording": False}

    monkeypatch.setattr(record_script, "precise_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(record_script.time, "perf_counter", clock.perf_counter)
    original_send_action = robot.send_action
    robot.send_action = lambda action: send_calls.append(action.copy()) or original_send_action(action)

    try:
        record_script.record_loop(
            robot=robot,
            events=events,
            fps=10,
            teleop=teleop,
            dataset=None,
            control_time_s=1.0,
            single_task="halt",
            safety_guard=guard,
            step_event_logger=logger,
            runtime_state={},
            **identity_processors,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert guard.calls == 1
    assert send_calls == [{"motor_1.pos": 0.0, "motor_2.pos": 0.0, "motor_3.pos": 0.0}]
    assert events["stop_recording"] is True
    assert sleep_calls == [pytest.approx(0.09)]
    assert len(logger.steps) == 1
    assert len(logger.guard_rejects) == 1
    assert logger.guard_rejects[0]["halt"] is True
    assert logger.guard_rejects[0]["guard_result"]["decision"] == "halt"


def test_record_loop_policy_supports_dataset_none_with_explicit_action_features(monkeypatch, identity_processors):
    robot, teleop = _make_robot_and_teleop()
    sleep_calls = []
    sent_actions = []
    logger = _CaptureLogger()
    guard = _StaticGuard({"accept": True})
    clock = _FakeClock(step=0.01)
    policy = _ResettableStub()
    preprocessor = _ResettableStub()
    postprocessor = _ResettableStub()
    policy_observation_features, policy_action_features = _policy_feature_dict()

    monkeypatch.setattr(record_script, "precise_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(record_script.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(
        record_script,
        "predict_action",
        lambda **kwargs: torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32),
    )
    original_send_action = robot.send_action
    robot.send_action = lambda action: sent_actions.append(action.copy()) or original_send_action(action)

    try:
        record_script.record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False},
            fps=10,
            teleop=None,
            dataset=None,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_observation_features=policy_observation_features,
            policy_action_features=policy_action_features,
            control_time_s=0.025,
            single_task="policy direct",
            safety_guard=guard,
            step_event_logger=logger,
            runtime_state={},
            **identity_processors,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert policy.reset_calls == 1
    assert preprocessor.reset_calls == 1
    assert postprocessor.reset_calls == 1
    assert guard.calls == 1
    assert sent_actions == [{"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0}]
    assert sleep_calls == [pytest.approx(0.09)]
    assert len(logger.steps) == 1
    assert logger.steps[0]["task"] == "policy direct"
    assert logger.steps[0]["sent_action"]["motor_2.pos"] == pytest.approx(5.0)
