from __future__ import annotations

from dataclasses import dataclass

from .runtime_state import RuntimeState
from .schemas import PostCheckResult, RetryAction, RetryDirective, RuntimePhase, SafetyProfile


@dataclass
class RetryManager:
    def decide(self, post_check: PostCheckResult, runtime_state: RuntimeState, profile: SafetyProfile) -> RetryDirective:
        if post_check.success:
            return RetryDirective(action=RetryAction.COMPLETE, next_phase=RuntimePhase.DONE)

        if runtime_state.consecutive_failures >= profile.max_consecutive_failures:
            return RetryDirective(
                action=RetryAction.HALT,
                next_phase=RuntimePhase.HALTED,
                reason="failure_budget_exhausted",
            )

        if post_check.should_retry and runtime_state.retry_count < profile.max_retries:
            return RetryDirective(
                action=RetryAction.RETRY,
                next_phase=RuntimePhase.OBSERVE,
                reason=post_check.reason,
                metadata={"retry_index": runtime_state.retry_count + 1},
            )

        return RetryDirective(
            action=RetryAction.STOP,
            next_phase=RuntimePhase.HALTED,
            reason=post_check.reason or "retry_budget_exhausted",
        )
