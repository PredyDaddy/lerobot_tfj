from __future__ import annotations

from dataclasses import dataclass

from .runtime_state import RuntimeState
from .schemas import PerceptionFrame, PostCheckResult, PostCheckStatus, RuntimePhase, TaskIntent


@dataclass
class PostCheck:
    score_threshold: float = 0.5

    def evaluate(
        self,
        intent: TaskIntent,
        perception: PerceptionFrame,
        runtime_state: RuntimeState,
    ) -> PostCheckResult:
        if runtime_state.phase is not RuntimePhase.VERIFY:
            return PostCheckResult(
                status=PostCheckStatus.UNVERIFIED,
                success=False,
                should_retry=False,
                reason="phase_not_verify",
            )

        target_matches = [
            obj for obj in perception.objects if _matches_target(obj.label, intent.target_object) and obj.score >= self.score_threshold
        ]
        if intent.target_object and not target_matches:
            return PostCheckResult(
                status=PostCheckStatus.FAILED,
                success=False,
                should_retry=True,
                reason="target_not_found_during_verify",
            )

        if intent.target_container:
            placed = [obj for obj in target_matches if _matches_target(obj.container_label, intent.target_container)]
            if placed:
                return PostCheckResult(
                    status=PostCheckStatus.PASSED,
                    success=True,
                    should_retry=False,
                    matched_object_id=placed[0].object_id,
                )
            return PostCheckResult(
                status=PostCheckStatus.FAILED,
                success=False,
                should_retry=True,
                reason="target_not_in_container",
            )

        if target_matches:
            return PostCheckResult(
                status=PostCheckStatus.PASSED,
                success=True,
                should_retry=False,
                matched_object_id=target_matches[0].object_id,
            )

        if perception.objects:
            return PostCheckResult(status=PostCheckStatus.PASSED, success=True, should_retry=False)

        return PostCheckResult(
            status=PostCheckStatus.FAILED,
            success=False,
            should_retry=True,
            reason="empty_scene_on_verify",
        )



def _matches_target(candidate: str | None, target: str | None) -> bool:
    if target is None:
        return True
    if candidate is None:
        return False
    normalized_candidate = " ".join(candidate.lower().split())
    normalized_target = " ".join(target.lower().split())
    return normalized_target in normalized_candidate or normalized_candidate in normalized_target
