from __future__ import annotations

from dataclasses import dataclass

from .runtime_state import RuntimeState
from .schemas import (
    ActionCommand,
    BridgeDecision,
    DetectedObject,
    PerceptionFrame,
    RuntimePhase,
    TaskIntent,
)


@dataclass
class PerceptionActionBridge:
    max_scene_stale_ms: float = 500.0
    image_size_px: tuple[int, int] = (640, 480)
    pixel_to_joint_gain: float = 0.0015

    def decide(
        self,
        intent: TaskIntent,
        perception: PerceptionFrame,
        runtime_state: RuntimeState,
    ) -> BridgeDecision:
        if perception.scene_quality.stale_ms > self.max_scene_stale_ms:
            return BridgeDecision(
                rejection_reason="perception_stale",
                should_retry=True,
                metadata={"stale_ms": perception.scene_quality.stale_ms},
            )

        candidate = self._select_candidate(intent, perception)
        if candidate is None:
            return BridgeDecision(rejection_reason="target_not_visible", should_retry=True)
        if not candidate.graspable:
            return BridgeDecision(
                target_object_id=candidate.object_id,
                rejection_reason="target_not_graspable",
                should_retry=True,
            )

        runtime_state.remember_target(candidate, timestamp_ms=perception.timestamp_ms)
        action = self._build_action(candidate)
        if action is None:
            return BridgeDecision(
                target_object_id=candidate.object_id,
                rejection_reason="target_pose_unavailable",
                should_retry=True,
            )

        return BridgeDecision(
            target_object_id=candidate.object_id,
            target_container=intent.target_container,
            action=action,
            phase_hint=self._next_phase(runtime_state.phase),
            confidence=candidate.score,
            metadata={"object_label": candidate.label},
        )

    def _select_candidate(self, intent: TaskIntent, perception: PerceptionFrame) -> DetectedObject | None:
        if intent.target_object:
            target_label = _normalize_label(intent.target_object)
            matches = [
                obj
                for obj in perception.objects
                if target_label in _normalize_label(obj.label) or _normalize_label(obj.label) in target_label
            ]
            if matches:
                return max(matches, key=lambda item: item.score)

        if perception.best_candidate_id:
            best_candidate = perception.get_object(perception.best_candidate_id)
            if best_candidate is not None:
                return best_candidate

        if perception.objects:
            return max(perception.objects, key=lambda item: item.score)
        return None

    def _build_action(self, candidate: DetectedObject) -> ActionCommand | None:
        if candidate.center_xyz is not None:
            return ActionCommand(
                cartesian_target=candidate.center_xyz,
                reference_frame="base",
                label="align_to_target",
                metadata={"object_id": candidate.object_id},
            )

        if candidate.center_px is None:
            return None

        image_center = (self.image_size_px[0] / 2.0, self.image_size_px[1] / 2.0)
        dx = (candidate.center_px[0] - image_center[0]) * self.pixel_to_joint_gain
        dy = (image_center[1] - candidate.center_px[1]) * self.pixel_to_joint_gain
        return ActionCommand(
            joint_deltas=(dx, dy),
            label="pixel_align",
            metadata={"object_id": candidate.object_id},
        )

    def _next_phase(self, phase: RuntimePhase) -> RuntimePhase:
        if phase in {RuntimePhase.OBSERVE, RuntimePhase.SELECT}:
            return RuntimePhase.PREGRASP
        if phase is RuntimePhase.PREGRASP:
            return RuntimePhase.GRASP
        if phase is RuntimePhase.GRASP:
            return RuntimePhase.LIFT
        if phase is RuntimePhase.LIFT:
            return RuntimePhase.PLACE
        if phase is RuntimePhase.PLACE:
            return RuntimePhase.VERIFY
        return phase



def _normalize_label(text: str) -> str:
    return " ".join(text.lower().strip().replace("_", " ").split())
