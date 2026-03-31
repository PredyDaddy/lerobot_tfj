from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .schemas import GuardResult, StepEvent


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: StepEvent | Mapping[str, Any]) -> None:
        payload = self._serialize(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def log_run_start(self, *, run_id: str, episode_id: str, payload: Mapping[str, Any] | None = None) -> None:
        self.log(
            {
                "event_type": "run_start",
                "run_id": run_id,
                "episode_id": episode_id,
                "timestamp": self._utc_iso_timestamp(),
                "payload": dict(payload or {}),
            }
        )

    def log_guard_result(self, *, run_id: str, episode_id: str, step_id: int, guard_result: GuardResult) -> None:
        self.log(
            {
                "event_type": "guard_result",
                "run_id": run_id,
                "episode_id": episode_id,
                "step_id": step_id,
                "timestamp": self._utc_iso_timestamp(),
                "guard_result": guard_result,
            }
        )

    def log_run_end(self, *, run_id: str, episode_id: str, success: bool, payload: Mapping[str, Any] | None = None) -> None:
        self.log(
            {
                "event_type": "run_end",
                "run_id": run_id,
                "episode_id": episode_id,
                "timestamp": self._utc_iso_timestamp(),
                "success": bool(success),
                "payload": dict(payload or {}),
            }
        )

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return self._serialize(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): self._serialize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._serialize(item) for item in value]
        return value

    def _utc_iso_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
