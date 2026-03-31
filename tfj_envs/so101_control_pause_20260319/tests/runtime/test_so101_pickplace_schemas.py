from lerobot.runtime.so101_pickplace.schemas import (
    ActionCommand,
    GuardDecision,
    GuardResult,
    JointLimit,
    SafetyProfile,
    TaskIntent,
    TaskSlots,
    WorkspaceAABB,
)


def test_task_intent_exposes_slot_shortcuts() -> None:
    intent = TaskIntent(
        raw_text="pick the red block",
        slots=TaskSlots(target_object="red block", target_container="left bin"),
    )

    assert intent.target_object == "red block"
    assert intent.target_container == "left bin"


def test_workspace_clamp_and_contains() -> None:
    workspace = WorkspaceAABB(min_xyz=(-0.1, -0.2, 0.0), max_xyz=(0.3, 0.2, 0.5))

    assert workspace.contains((0.0, 0.0, 0.2))
    assert workspace.clamp((0.5, -0.5, 0.6)) == (0.3, -0.2, 0.5)


def test_safety_profile_validates_retry_budget() -> None:
    profile = SafetyProfile(
        joint_limits=(JointLimit(-1.0, 1.0), JointLimit(-0.5, 0.5)),
        max_joint_delta=0.2,
        max_retries=3,
    )

    assert profile.max_retries == 3
    assert len(profile.joint_limits) == 2


def test_guard_result_semantics_are_explicit() -> None:
    action = ActionCommand(joint_positions=(0.1, 0.2), label="move")
    accepted = GuardResult.accept(action)
    rejected = GuardResult.reject(error_code="workspace_violation", reason="outside workspace")

    assert accepted.decision is GuardDecision.ACCEPT
    assert accepted.should_send_action is True
    assert accepted.requires_fail_safe is False

    assert rejected.decision is GuardDecision.REJECT
    assert rejected.should_send_action is False
    assert rejected.requires_fail_safe is True
