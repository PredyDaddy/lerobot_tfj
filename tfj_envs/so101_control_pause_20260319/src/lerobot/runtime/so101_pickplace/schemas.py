from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]


class RuntimePhase(str, Enum):
    OBSERVE = "observe"
    SELECT = "select"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    VERIFY = "verify"
    RECOVER = "recover"
    DONE = "done"
    HALTED = "halted"


class GuardDecision(str, Enum):
    ACCEPT = "accept"
    CLAMP_AND_ACCEPT = "clamp_and_accept"
    REJECT = "reject"
    HALT = "halt"


class PlannerDisposition(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    STOP = "stop"
    HALT = "halt"


class PostCheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class RetryAction(str, Enum):
    RETRY = "retry"
    COMPLETE = "complete"
    STOP = "stop"
    HALT = "halt"



def _coerce_float_tuple(
    value: Sequence[float] | None,
    *,
    name: str,
    expected_len: int | None = None,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in value)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(result)}")
    return result


@dataclass(frozen=True)
class TaskSlots:
    target_object: str | None = None
    target_container: str | None = None
    source_container: str | None = None
    modifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_object", _clean_text(self.target_object))
        object.__setattr__(self, "target_container", _clean_text(self.target_container))
        object.__setattr__(self, "source_container", _clean_text(self.source_container))
        object.__setattr__(self, "modifiers", tuple(str(item).strip() for item in self.modifiers if str(item).strip()))


@dataclass(frozen=True)
class TaskIntent:
    raw_text: str
    verb: str = "pick_place"
    slots: TaskSlots = field(default_factory=TaskSlots)
    constraints: dict[str, JsonValue] = field(default_factory=dict)
    language: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_text", self.raw_text.strip())
        object.__setattr__(self, "verb", self.verb.strip() or "pick_place")
        object.__setattr__(self, "language", (self.language or "unknown").strip() or "unknown")

    @property
    def target_object(self) -> str | None:
        return self.slots.target_object

    @property
    def target_container(self) -> str | None:
        return self.slots.target_container


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if minimum > maximum:
            raise ValueError(f"Invalid joint limit range: {minimum} > {maximum}")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, value))

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class WorkspaceAABB:
    min_xyz: Vector3
    max_xyz: Vector3

    def __post_init__(self) -> None:
        min_xyz = _coerce_float_tuple(self.min_xyz, name="min_xyz", expected_len=3)
        max_xyz = _coerce_float_tuple(self.max_xyz, name="max_xyz", expected_len=3)
        assert min_xyz is not None
        assert max_xyz is not None
        if any(min_v > max_v for min_v, max_v in zip(min_xyz, max_xyz, strict=True)):
            raise ValueError("Workspace min_xyz must be <= max_xyz on every axis")
        object.__setattr__(self, "min_xyz", min_xyz)
        object.__setattr__(self, "max_xyz", max_xyz)

    def contains(self, point: Sequence[float]) -> bool:
        xyz = _coerce_float_tuple(point, name="point", expected_len=3)
        assert xyz is not None
        return all(
            min_v <= value <= max_v
            for value, min_v, max_v in zip(xyz, self.min_xyz, self.max_xyz, strict=True)
        )

    def clamp(self, point: Sequence[float]) -> Vector3:
        xyz = _coerce_float_tuple(point, name="point", expected_len=3)
        assert xyz is not None
        return tuple(
            min(max(value, min_v), max_v)
            for value, min_v, max_v in zip(xyz, self.min_xyz, self.max_xyz, strict=True)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class ActionCommand:
    joint_positions: tuple[float, ...] | None = None
    joint_deltas: tuple[float, ...] | None = None
    cartesian_target: Vector3 | None = None
    reference_frame: str = "base"
    gripper_closed: bool | None = None
    label: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        joint_positions = _coerce_float_tuple(self.joint_positions, name="joint_positions")
        joint_deltas = _coerce_float_tuple(self.joint_deltas, name="joint_deltas")
        cartesian_target = _coerce_float_tuple(self.cartesian_target, name="cartesian_target", expected_len=3)
        if joint_positions is not None and joint_deltas is not None:
            raise ValueError("Only one of joint_positions or joint_deltas may be set")
        if cartesian_target is not None and (joint_positions is not None or joint_deltas is not None):
            raise ValueError("Cartesian targets are mutually exclusive with joint position or delta commands")
        if joint_positions is None and joint_deltas is None and cartesian_target is None and self.gripper_closed is None:
            raise ValueError("ActionCommand must define at least one action target")
        object.__setattr__(self, "joint_positions", joint_positions)
        object.__setattr__(self, "joint_deltas", joint_deltas)
        object.__setattr__(self, "cartesian_target", cartesian_target)
        object.__setattr__(self, "reference_frame", self.reference_frame.strip() or "base")
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True)
class DetectedObject:
    object_id: str
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    center_px: Vector2 | None = None
    center_xyz: Vector3 | None = None
    container_label: str | None = None
    graspable: bool = True
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "object_id", self.object_id.strip())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "bbox_xyxy", _coerce_float_tuple(self.bbox_xyxy, name="bbox_xyxy", expected_len=4))
        object.__setattr__(self, "center_px", _coerce_float_tuple(self.center_px, name="center_px", expected_len=2))
        object.__setattr__(self, "center_xyz", _coerce_float_tuple(self.center_xyz, name="center_xyz", expected_len=3))
        object.__setattr__(self, "container_label", _clean_text(self.container_label))


@dataclass(frozen=True)
class SceneQuality:
    blur: float = 0.0
    exposure: float = 0.0
    stale_ms: float = 0.0

    def __post_init__(self) -> None:
        blur = float(self.blur)
        exposure = float(self.exposure)
        stale_ms = float(self.stale_ms)
        if stale_ms < 0:
            raise ValueError("stale_ms must be non-negative")
        object.__setattr__(self, "blur", blur)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "stale_ms", stale_ms)


@dataclass(frozen=True)
class PerceptionFrame:
    objects: tuple[DetectedObject, ...] = ()
    best_candidate_id: str | None = None
    scene_quality: SceneQuality = field(default_factory=SceneQuality)
    frame_id: str = ""
    timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "best_candidate_id", _clean_text(self.best_candidate_id))
        object.__setattr__(self, "frame_id", self.frame_id.strip())

    def get_object(self, object_id: str) -> DetectedObject | None:
        return next((obj for obj in self.objects if obj.object_id == object_id), None)


@dataclass(frozen=True)
class SafetyProfile:
    joint_limits: tuple[JointLimit, ...] = ()
    max_joint_delta: float = 0.25
    workspace_aabb: WorkspaceAABB | None = None
    max_perception_stale_ms: float = 500.0
    max_consecutive_failures: int = 3
    max_retries: int = 2
    allow_clamp: bool = True
    halt_on_stale_perception: bool = True

    def __post_init__(self) -> None:
        joint_limits = tuple(self.joint_limits)
        max_joint_delta = float(self.max_joint_delta)
        max_perception_stale_ms = float(self.max_perception_stale_ms)
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")
        if max_perception_stale_ms < 0:
            raise ValueError("max_perception_stale_ms must be non-negative")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        object.__setattr__(self, "joint_limits", joint_limits)
        object.__setattr__(self, "max_joint_delta", max_joint_delta)
        object.__setattr__(self, "max_perception_stale_ms", max_perception_stale_ms)


@dataclass(frozen=True)
class BridgeDecision:
    target_object_id: str | None = None
    target_container: str | None = None
    action: ActionCommand | None = None
    phase_hint: RuntimePhase | None = None
    confidence: float = 0.0
    rejection_reason: str | None = None
    should_retry: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        object.__setattr__(self, "target_object_id", _clean_text(self.target_object_id))
        object.__setattr__(self, "target_container", _clean_text(self.target_container))
        object.__setattr__(self, "confidence", confidence)

    @property
    def accepted(self) -> bool:
        return self.action is not None and self.rejection_reason is None


@dataclass(frozen=True)
class GuardResult:
    decision: GuardDecision
    action: ActionCommand | None = None
    error_code: str | None = None
    reason: str | None = None
    fail_safe_action: ActionCommand | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def should_send_action(self) -> bool:
        return self.decision in {GuardDecision.ACCEPT, GuardDecision.CLAMP_AND_ACCEPT} and self.action is not None

    @property
    def requires_fail_safe(self) -> bool:
        return self.decision in {GuardDecision.REJECT, GuardDecision.HALT}

    @property
    def requires_halt(self) -> bool:
        return self.decision is GuardDecision.HALT

    @classmethod
    def accept(cls, action: ActionCommand, *, details: dict[str, JsonValue] | None = None) -> "GuardResult":
        return cls(decision=GuardDecision.ACCEPT, action=action, details=details or {})

    @classmethod
    def clamp_and_accept(
        cls,
        action: ActionCommand,
        *,
        reason: str,
        error_code: str,
        details: dict[str, JsonValue] | None = None,
    ) -> "GuardResult":
        return cls(
            decision=GuardDecision.CLAMP_AND_ACCEPT,
            action=action,
            reason=reason,
            error_code=error_code,
            details=details or {},
        )

    @classmethod
    def reject(
        cls,
        *,
        error_code: str,
        reason: str,
        fail_safe_action: ActionCommand | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> "GuardResult":
        return cls(
            decision=GuardDecision.REJECT,
            error_code=error_code,
            reason=reason,
            fail_safe_action=fail_safe_action,
            details=details or {},
        )

    @classmethod
    def halt(
        cls,
        *,
        error_code: str,
        reason: str,
        fail_safe_action: ActionCommand | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> "GuardResult":
        return cls(
            decision=GuardDecision.HALT,
            error_code=error_code,
            reason=reason,
            fail_safe_action=fail_safe_action,
            details=details or {},
        )


@dataclass(frozen=True)
class PlannerDecision:
    next_phase: RuntimePhase
    retry_or_stop: PlannerDisposition = PlannerDisposition.CONTINUE
    action_goal: str | None = None
    reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PostCheckResult:
    status: PostCheckStatus
    success: bool
    should_retry: bool
    reason: str | None = None
    matched_object_id: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class RetryDirective:
    action: RetryAction
    next_phase: RuntimePhase | None = None
    reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class StepEvent:
    event_type: str
    run_id: str
    episode_id: str
    step_id: int
    phase: RuntimePhase
    timestamp_ms: int
    task_slots: TaskSlots = field(default_factory=TaskSlots)
    latency_ms: float | None = None
    guard_decision: GuardDecision | None = None
    action_summary: str | None = None
    success: bool | None = None
    fail_reason: str | None = None
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self.event_type.strip())
        object.__setattr__(self, "run_id", self.run_id.strip())
        object.__setattr__(self, "episode_id", self.episode_id.strip())
        object.__setattr__(self, "action_summary", _clean_text(self.action_summary))
        object.__setattr__(self, "fail_reason", _clean_text(self.fail_reason))
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")



def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None
