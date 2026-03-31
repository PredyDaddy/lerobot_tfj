from lerobot.runtime.so101_pickplace.intent_parser import parse_task_intent
from lerobot.runtime.so101_pickplace.perception_action_bridge import PerceptionActionBridge
from lerobot.runtime.so101_pickplace.post_check import PostCheck
from lerobot.runtime.so101_pickplace.retry_manager import RetryManager
from lerobot.runtime.so101_pickplace.runtime_state import RuntimeState
from lerobot.runtime.so101_pickplace.safety_guard import SafetyGuard
from lerobot.runtime.so101_pickplace.schemas import (
    DetectedObject,
    GuardDecision,
    PerceptionFrame,
    PlannerDisposition,
    PostCheckStatus,
    RuntimePhase,
    SafetyProfile,
    SceneQuality,
    WorkspaceAABB,
)
from lerobot.runtime.so101_pickplace.task_planner import TaskPlanner



def test_bridge_guard_and_planner_form_minimal_closed_loop() -> None:
    intent = parse_task_intent("Pick the red block and place it into the left bin")
    runtime_state = RuntimeState(phase=RuntimePhase.OBSERVE)
    bridge = PerceptionActionBridge()
    guard = SafetyGuard(
        SafetyProfile(
            workspace_aabb=WorkspaceAABB(min_xyz=(-0.2, -0.2, 0.0), max_xyz=(0.2, 0.2, 0.3)),
        )
    )
    planner = TaskPlanner()
    post_check = PostCheck()
    retry_manager = RetryManager()

    perception = PerceptionFrame(
        objects=(
            DetectedObject(
                object_id="obj-red-1",
                label="red block",
                score=0.95,
                center_xyz=(0.25, 0.0, 0.1),
                container_label="left bin",
            ),
        ),
        scene_quality=SceneQuality(stale_ms=20.0),
    )

    bridge_decision = bridge.decide(intent, perception, runtime_state)
    assert bridge_decision.accepted is True
    assert bridge_decision.action is not None

    guard_result = guard.validate(bridge_decision.action)
    assert guard_result.decision is GuardDecision.CLAMP_AND_ACCEPT
    assert guard_result.action is not None
    assert guard_result.action.cartesian_target == (0.2, 0.0, 0.1)

    planner_decision = planner.plan_next(intent, perception, runtime_state, bridge_decision=bridge_decision)
    assert planner_decision.next_phase is RuntimePhase.PREGRASP
    assert planner_decision.retry_or_stop is PlannerDisposition.CONTINUE

    runtime_state.advance(RuntimePhase.VERIFY)
    verify_result = post_check.evaluate(intent, perception, runtime_state)
    assert verify_result.status is PostCheckStatus.PASSED
    assert verify_result.success is True

    retry_directive = retry_manager.decide(verify_result, runtime_state, guard.profile)
    assert retry_directive.action.value == "complete"
    assert retry_directive.next_phase is RuntimePhase.DONE


def test_bridge_rejection_leads_planner_to_retry() -> None:
    intent = parse_task_intent("Pick the blue block and place it into the left bin")
    runtime_state = RuntimeState(phase=RuntimePhase.OBSERVE)
    bridge = PerceptionActionBridge(max_scene_stale_ms=100.0)
    planner = TaskPlanner()
    perception = PerceptionFrame(scene_quality=SceneQuality(stale_ms=250.0))

    bridge_decision = bridge.decide(intent, perception, runtime_state)
    planner_decision = planner.plan_next(intent, perception, runtime_state, bridge_decision=bridge_decision)

    assert bridge_decision.rejection_reason == "perception_stale"
    assert planner_decision.retry_or_stop is PlannerDisposition.RETRY
    assert planner_decision.next_phase is RuntimePhase.OBSERVE
