from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from .runtime_state import RuntimeState
from .schemas import ActionCommand, GuardResult, PerceptionFrame, SafetyProfile


@dataclass
class SafetyGuard:
    profile: SafetyProfile

    def validate(
        self,
        action: ActionCommand | None,
        *,
        current_joint_positions: Sequence[float] | None = None,
        perception: PerceptionFrame | None = None,
        runtime_state: RuntimeState | None = None,
    ) -> GuardResult:
        hold_action = self._hold_position_action(current_joint_positions)

        if runtime_state is not None and runtime_state.consecutive_failures >= self.profile.max_consecutive_failures:
            return GuardResult.halt(
                error_code="failure_budget_exhausted",
                reason="Consecutive failure budget exhausted",
                fail_safe_action=hold_action,
                details={"consecutive_failures": runtime_state.consecutive_failures},
            )

        if perception is not None and perception.scene_quality.stale_ms > self.profile.max_perception_stale_ms:
            reason = "Perception frame is stale"
            if self.profile.halt_on_stale_perception:
                return GuardResult.halt(
                    error_code="perception_stale",
                    reason=reason,
                    fail_safe_action=hold_action,
                    details={"stale_ms": perception.scene_quality.stale_ms},
                )
            return GuardResult.reject(
                error_code="perception_stale",
                reason=reason,
                fail_safe_action=hold_action,
                details={"stale_ms": perception.scene_quality.stale_ms},
            )

        if action is None:
            return GuardResult.reject(
                error_code="empty_action",
                reason="No action provided",
                fail_safe_action=hold_action,
            )

        if self._contains_non_finite(action.joint_positions) or self._contains_non_finite(action.joint_deltas):
            return GuardResult.halt(
                error_code="invalid_joint_values",
                reason="Action contains NaN or Inf joint values",
                fail_safe_action=hold_action,
            )

        if self._contains_non_finite(action.cartesian_target):
            return GuardResult.halt(
                error_code="invalid_cartesian_target",
                reason="Action contains NaN or Inf Cartesian values",
                fail_safe_action=hold_action,
            )

        clamped_action = action
        clamp_reasons: list[str] = []
        clamp_details: dict[str, float | list[float]] = {}

        if action.joint_positions is not None:
            guard_result = self._validate_joint_positions(
                action,
                action.joint_positions,
                current_joint_positions=current_joint_positions,
                hold_action=hold_action,
            )
            if guard_result is not None:
                return guard_result
            clamp_result = self._clamp_joint_positions(
                clamped_action,
                current_joint_positions=current_joint_positions,
                hold_action=hold_action,
            )
            if isinstance(clamp_result, GuardResult):
                return clamp_result
            clamped_action, clamp_reasons, clamp_details = clamp_result

        if action.joint_deltas is not None:
            guard_result = self._validate_joint_deltas(
                action.joint_deltas,
                current_joint_positions=current_joint_positions,
                hold_action=hold_action,
            )
            if guard_result is not None:
                return guard_result
            clamp_result = self._clamp_joint_deltas(
                clamped_action,
                current_joint_positions=current_joint_positions,
                hold_action=hold_action,
            )
            if isinstance(clamp_result, GuardResult):
                return clamp_result
            clamped_action, extra_reasons, extra_details = clamp_result
            clamp_reasons.extend(extra_reasons)
            clamp_details.update(extra_details)

        if action.cartesian_target is not None and self.profile.workspace_aabb is not None:
            clamped_target = self.profile.workspace_aabb.clamp(action.cartesian_target)
            if clamped_target != action.cartesian_target:
                if not self.profile.allow_clamp:
                    return GuardResult.reject(
                        error_code="workspace_violation",
                        reason="Cartesian target lies outside the workspace",
                        fail_safe_action=hold_action,
                        details={"cartesian_target": list(action.cartesian_target)},
                    )
                clamped_action = replace(clamped_action, cartesian_target=clamped_target)
                clamp_reasons.append("workspace_violation")
                clamp_details["cartesian_target"] = list(clamped_target)

        if clamp_reasons:
            return GuardResult.clamp_and_accept(
                clamped_action,
                error_code=clamp_reasons[0],
                reason="; ".join(clamp_reasons),
                details=clamp_details,
            )
        return GuardResult.accept(clamped_action)

    def _validate_joint_positions(
        self,
        action: ActionCommand,
        joint_positions: tuple[float, ...],
        *,
        current_joint_positions: Sequence[float] | None,
        hold_action: ActionCommand | None,
    ) -> GuardResult | None:
        if self.profile.joint_limits and len(joint_positions) != len(self.profile.joint_limits):
            return GuardResult.halt(
                error_code="joint_dimension_mismatch",
                reason="Joint position dimensionality does not match configured limits",
                fail_safe_action=hold_action,
                details={"joint_count": len(joint_positions), "limit_count": len(self.profile.joint_limits)},
            )
        if current_joint_positions is not None and len(current_joint_positions) != len(joint_positions):
            return GuardResult.halt(
                error_code="joint_state_dimension_mismatch",
                reason="Current joint state dimensionality does not match requested joint positions",
                fail_safe_action=hold_action,
                details={"joint_count": len(joint_positions), "state_count": len(current_joint_positions)},
            )
        return None

    def _validate_joint_deltas(
        self,
        joint_deltas: tuple[float, ...],
        *,
        current_joint_positions: Sequence[float] | None,
        hold_action: ActionCommand | None,
    ) -> GuardResult | None:
        if current_joint_positions is None:
            return GuardResult.reject(
                error_code="missing_joint_state",
                reason="Joint delta commands require the current joint positions",
                fail_safe_action=hold_action,
            )
        if len(current_joint_positions) != len(joint_deltas):
            return GuardResult.halt(
                error_code="joint_state_dimension_mismatch",
                reason="Current joint state dimensionality does not match requested joint deltas",
                fail_safe_action=hold_action,
                details={"delta_count": len(joint_deltas), "state_count": len(current_joint_positions)},
            )
        return None

    def _clamp_joint_positions(
        self,
        action: ActionCommand,
        *,
        current_joint_positions: Sequence[float] | None,
        hold_action: ActionCommand | None,
    ) -> GuardResult | tuple[ActionCommand, list[str], dict[str, float | list[float]]]:
        assert action.joint_positions is not None
        proposed = list(action.joint_positions)
        reasons: list[str] = []
        details: dict[str, float | list[float]] = {}

        if current_joint_positions is not None:
            clamped_step = []
            step_clamped = False
            for current, requested in zip(current_joint_positions, proposed, strict=True):
                delta = requested - float(current)
                clamped_delta = min(self.profile.max_joint_delta, max(-self.profile.max_joint_delta, delta))
                if not math.isclose(delta, clamped_delta):
                    step_clamped = True
                clamped_step.append(float(current) + clamped_delta)
            if step_clamped:
                if not self.profile.allow_clamp:
                    return GuardResult.reject(
                        error_code="joint_step_limit_exceeded",
                        reason="Requested joint position step exceeds max_joint_delta",
                        fail_safe_action=hold_action,
                        details={"max_joint_delta": self.profile.max_joint_delta},
                    )
                proposed = clamped_step
                reasons.append("joint_step_limit_exceeded")
                details["max_joint_delta"] = self.profile.max_joint_delta

        if self.profile.joint_limits:
            clamped_limits = []
            limit_clamped = False
            for limit, requested in zip(self.profile.joint_limits, proposed, strict=True):
                bounded = limit.clamp(requested)
                if not math.isclose(requested, bounded):
                    limit_clamped = True
                clamped_limits.append(bounded)
            if limit_clamped:
                if not self.profile.allow_clamp:
                    return GuardResult.reject(
                        error_code="joint_limit_exceeded",
                        reason="Requested joint position lies outside configured joint limits",
                        fail_safe_action=hold_action,
                        details={"joint_positions": proposed},
                    )
                proposed = clamped_limits
                reasons.append("joint_limit_exceeded")
                details["joint_positions"] = list(proposed)

        return replace(action, joint_positions=tuple(proposed)), reasons, details

    def _clamp_joint_deltas(
        self,
        action: ActionCommand,
        *,
        current_joint_positions: Sequence[float] | None,
        hold_action: ActionCommand | None,
    ) -> GuardResult | tuple[ActionCommand, list[str], dict[str, float | list[float]]]:
        assert action.joint_deltas is not None
        assert current_joint_positions is not None
        requested_deltas = list(action.joint_deltas)
        reasons: list[str] = []
        details: dict[str, float | list[float]] = {}

        clamped_deltas = []
        delta_clamped = False
        for delta in requested_deltas:
            clamped_delta = min(self.profile.max_joint_delta, max(-self.profile.max_joint_delta, delta))
            if not math.isclose(delta, clamped_delta):
                delta_clamped = True
            clamped_deltas.append(clamped_delta)
        if delta_clamped:
            if not self.profile.allow_clamp:
                return GuardResult.reject(
                    error_code="joint_step_limit_exceeded",
                    reason="Requested joint delta exceeds max_joint_delta",
                    fail_safe_action=hold_action,
                    details={"max_joint_delta": self.profile.max_joint_delta},
                )
            reasons.append("joint_step_limit_exceeded")
            details["joint_deltas"] = list(clamped_deltas)

        target_positions = [
            float(current) + clamped_delta
            for current, clamped_delta in zip(current_joint_positions, clamped_deltas, strict=True)
        ]
        if self.profile.joint_limits:
            clamped_positions = []
            limit_clamped = False
            for limit, target in zip(self.profile.joint_limits, target_positions, strict=True):
                bounded = limit.clamp(target)
                if not math.isclose(target, bounded):
                    limit_clamped = True
                clamped_positions.append(bounded)
            if limit_clamped:
                if not self.profile.allow_clamp:
                    return GuardResult.reject(
                        error_code="joint_limit_exceeded",
                        reason="Requested joint delta reaches outside configured joint limits",
                        fail_safe_action=hold_action,
                        details={"joint_positions": target_positions},
                    )
                clamped_deltas = [
                    bounded - float(current)
                    for current, bounded in zip(current_joint_positions, clamped_positions, strict=True)
                ]
                reasons.append("joint_limit_exceeded")
                details["joint_positions"] = list(clamped_positions)

        return replace(action, joint_deltas=tuple(clamped_deltas)), reasons, details

    def _hold_position_action(self, current_joint_positions: Sequence[float] | None) -> ActionCommand | None:
        if current_joint_positions is None:
            return None
        return ActionCommand(joint_positions=tuple(float(position) for position in current_joint_positions), label="hold_position")

    def _contains_non_finite(self, values: Sequence[float] | None) -> bool:
        if values is None:
            return False
        return any(not math.isfinite(float(value)) for value in values)
