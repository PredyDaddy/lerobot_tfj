import math

from lerobot.runtime.so101_pickplace.runtime_state import RuntimeState
from lerobot.runtime.so101_pickplace.safety_guard import SafetyGuard
from lerobot.runtime.so101_pickplace.schemas import (
    ActionCommand,
    GuardDecision,
    JointLimit,
    PerceptionFrame,
    SafetyProfile,
    SceneQuality,
    WorkspaceAABB,
)


def make_guard(*, allow_clamp: bool = True, halt_on_stale_perception: bool = True) -> SafetyGuard:
    return SafetyGuard(
        SafetyProfile(
            joint_limits=(JointLimit(-1.0, 1.0), JointLimit(-1.0, 1.0)),
            max_joint_delta=0.2,
            workspace_aabb=WorkspaceAABB(min_xyz=(-0.3, -0.3, 0.0), max_xyz=(0.3, 0.3, 0.4)),
            allow_clamp=allow_clamp,
            halt_on_stale_perception=halt_on_stale_perception,
        )
    )


def test_guard_accepts_safe_joint_position_command() -> None:
    guard = make_guard()

    result = guard.validate(
        ActionCommand(joint_positions=(0.1, -0.1), label="move"),
        current_joint_positions=(0.0, 0.0),
    )

    assert result.decision is GuardDecision.ACCEPT
    assert result.action is not None
    assert result.action.joint_positions == (0.1, -0.1)


def test_guard_clamps_large_joint_step_when_allowed() -> None:
    guard = make_guard()

    result = guard.validate(
        ActionCommand(joint_positions=(0.8, 0.0), label="move"),
        current_joint_positions=(0.0, 0.0),
    )

    assert result.decision is GuardDecision.CLAMP_AND_ACCEPT
    assert result.action is not None
    assert result.action.joint_positions == (0.2, 0.0)
    assert result.error_code == "joint_step_limit_exceeded"


def test_guard_rejects_workspace_violation_when_clamp_disabled() -> None:
    guard = make_guard(allow_clamp=False)

    result = guard.validate(ActionCommand(cartesian_target=(0.5, 0.0, 0.1), label="align"))

    assert result.decision is GuardDecision.REJECT
    assert result.error_code == "workspace_violation"
    assert result.requires_fail_safe is True


def test_guard_halts_on_non_finite_joint_values() -> None:
    guard = make_guard()

    result = guard.validate(ActionCommand(joint_positions=(math.nan, 0.0), label="bad"), current_joint_positions=(0.0, 0.0))

    assert result.decision is GuardDecision.HALT
    assert result.error_code == "invalid_joint_values"


def test_guard_halts_on_stale_perception_when_profile_requires_it() -> None:
    guard = make_guard(halt_on_stale_perception=True)
    perception = PerceptionFrame(scene_quality=SceneQuality(stale_ms=900.0))

    result = guard.validate(
        ActionCommand(joint_positions=(0.0, 0.0), label="hold"),
        current_joint_positions=(0.0, 0.0),
        perception=perception,
    )

    assert result.decision is GuardDecision.HALT
    assert result.error_code == "perception_stale"


def test_guard_halts_when_failure_budget_is_exhausted() -> None:
    guard = make_guard()
    runtime_state = RuntimeState(consecutive_failures=3)

    result = guard.validate(
        ActionCommand(joint_positions=(0.0, 0.0), label="hold"),
        current_joint_positions=(0.0, 0.0),
        runtime_state=runtime_state,
    )

    assert result.decision is GuardDecision.HALT
    assert result.error_code == "failure_budget_exhausted"
