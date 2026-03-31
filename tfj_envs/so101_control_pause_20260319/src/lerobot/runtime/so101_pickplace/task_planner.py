from __future__ import annotations

from dataclasses import dataclass

from .runtime_state import RuntimeState
from .schemas import (
    BridgeDecision,
    PerceptionFrame,
    PlannerDecision,
    PlannerDisposition,
    PostCheckResult,
    PostCheckStatus,
    RuntimePhase,
    TaskIntent,
)


@dataclass
class TaskPlanner:
    def plan_next(
        self,
        intent: TaskIntent,
        perception: PerceptionFrame,
        runtime_state: RuntimeState,
        *,
        bridge_decision: BridgeDecision | None = None,
        post_check: PostCheckResult | None = None,
    ) -> PlannerDecision:
        del intent  # Intent is reserved for future object-specific branching.
        if runtime_state.phase is RuntimePhase.HALTED:
            return PlannerDecision(next_phase=RuntimePhase.HALTED, retry_or_stop=PlannerDisposition.HALT, reason="runtime_halted")

        if post_check is not None and runtime_state.phase is RuntimePhase.VERIFY:
            if post_check.status is PostCheckStatus.PASSED:
                return PlannerDecision(next_phase=RuntimePhase.DONE, action_goal="task_complete")
            if post_check.should_retry:
                return PlannerDecision(
                    next_phase=RuntimePhase.RECOVER,
                    retry_or_stop=PlannerDisposition.RETRY,
                    action_goal="reobserve_scene",
                    reason=post_check.reason,
                )
            return PlannerDecision(
                next_phase=RuntimePhase.HALTED,
                retry_or_stop=PlannerDisposition.STOP,
                action_goal="await_operator",
                reason=post_check.reason,
            )

        if bridge_decision is not None:
            if bridge_decision.rejection_reason:
                return PlannerDecision(
                    next_phase=RuntimePhase.OBSERVE,
                    retry_or_stop=PlannerDisposition.RETRY,
                    action_goal="reobserve_scene",
                    reason=bridge_decision.rejection_reason,
                )
            if bridge_decision.phase_hint is not None:
                return PlannerDecision(
                    next_phase=bridge_decision.phase_hint,
                    action_goal=(bridge_decision.action.label if bridge_decision.action else None),
                )

        if not perception.objects:
            return PlannerDecision(
                next_phase=RuntimePhase.OBSERVE,
                retry_or_stop=PlannerDisposition.RETRY,
                action_goal="wait_for_target",
                reason="no_objects_detected",
            )

        progression = {
            RuntimePhase.OBSERVE: RuntimePhase.SELECT,
            RuntimePhase.SELECT: RuntimePhase.PREGRASP,
            RuntimePhase.PREGRASP: RuntimePhase.GRASP,
            RuntimePhase.GRASP: RuntimePhase.LIFT,
            RuntimePhase.LIFT: RuntimePhase.PLACE,
            RuntimePhase.PLACE: RuntimePhase.VERIFY,
            RuntimePhase.RECOVER: RuntimePhase.OBSERVE,
        }
        return PlannerDecision(next_phase=progression.get(runtime_state.phase, runtime_state.phase))
