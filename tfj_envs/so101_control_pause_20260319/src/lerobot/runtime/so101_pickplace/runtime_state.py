from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import DetectedObject, RuntimePhase, Vector3


@dataclass
class RuntimeState:
    phase: RuntimePhase = RuntimePhase.OBSERVE
    retry_count: int = 0
    consecutive_failures: int = 0
    last_seen_target: str | None = None
    last_target_position: Vector3 | None = None
    last_failure_reason: str | None = None
    last_event_ts_ms: int | None = None
    phase_history: list[RuntimePhase] = field(default_factory=list)

    def advance(self, next_phase: RuntimePhase, *, timestamp_ms: int | None = None) -> None:
        if self.phase != next_phase:
            self.phase_history.append(self.phase)
        self.phase = next_phase
        self.last_event_ts_ms = timestamp_ms

    def remember_target(self, target: DetectedObject, *, timestamp_ms: int | None = None) -> None:
        self.last_seen_target = target.object_id
        self.last_target_position = target.center_xyz
        self.last_event_ts_ms = timestamp_ms

    def register_failure(self, reason: str, *, timestamp_ms: int | None = None) -> None:
        self.consecutive_failures += 1
        self.last_failure_reason = reason.strip() or None
        self.last_event_ts_ms = timestamp_ms

    def register_success(self, *, timestamp_ms: int | None = None) -> None:
        self.consecutive_failures = 0
        self.last_failure_reason = None
        self.last_event_ts_ms = timestamp_ms

    def schedule_retry(self, *, next_phase: RuntimePhase = RuntimePhase.OBSERVE, timestamp_ms: int | None = None) -> None:
        self.retry_count += 1
        self.advance(next_phase, timestamp_ms=timestamp_ms)
