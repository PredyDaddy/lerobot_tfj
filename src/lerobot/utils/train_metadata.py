# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lerobot.datasets.utils import load_json, write_json

KD_TEACHER_METADATA_FILENAME = "kd_teacher_metadata.json"
KD_TEACHER_METADATA_SCHEMA_VERSION = 1

_ROOT_STRING_FIELDS = {"comparison_space", "processor_compatibility", "metric_aggregation_mode"}
_ROOT_INT_FIELDS = {"schema_version"}
_ROOT_RESERVED_KEYS = {
    *_ROOT_STRING_FIELDS,
    *_ROOT_INT_FIELDS,
    "teacher",
    "processor_compatibility_mode",
    "teacher_policy_path",
    "teacher_train_config",
    "resolved_teacher_pretrained_path",
    "teacher_source_kind",
    "teacher_checkpoint_step",
    "pinned_pretrained_path",
    "original_path",
    "checkpoint_step",
    "source_kind",
    "pinned_from_run_metadata",
}

_TEACHER_PATH_FIELDS = {"original_path", "pinned_pretrained_path"}
_TEACHER_INT_FIELDS = {"checkpoint_step"}
_TEACHER_BOOL_FIELDS = {"pinned_from_run_metadata"}
_TEACHER_STRING_FIELDS = {"source_kind"}
_TEACHER_RESERVED_KEYS = {
    *_TEACHER_PATH_FIELDS,
    *_TEACHER_INT_FIELDS,
    *_TEACHER_BOOL_FIELDS,
    *_TEACHER_STRING_FIELDS,
}


def _normalize_path(value: Path | str | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _normalize_int(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return value


def _normalize_bool(value: bool | str | int | None) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"true", "1", "yes"}:
            return True
        if normalized_value in {"false", "0", "no"}:
            return False
        raise ValueError(f"Unsupported boolean value: {value}")
    return bool(value)


def _pick(primary: Any, fallback: Any) -> Any:
    return primary if primary is not None else fallback


@dataclass
class KDTeacherSnapshotMetadata:
    original_path: Path | None = None
    pinned_pretrained_path: Path | None = None
    checkpoint_step: int | None = None
    source_kind: str | None = None
    pinned_from_run_metadata: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            value is not None
            for value in (
                self.original_path,
                self.pinned_pretrained_path,
                self.checkpoint_step,
                self.source_kind,
                self.pinned_from_run_metadata,
            )
        ) and not self.extra

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.original_path is not None:
            payload["original_path"] = str(self.original_path)
        if self.pinned_pretrained_path is not None:
            payload["pinned_pretrained_path"] = str(self.pinned_pretrained_path)
        if self.checkpoint_step is not None:
            payload["checkpoint_step"] = self.checkpoint_step
        if self.source_kind is not None:
            payload["source_kind"] = self.source_kind
        if self.pinned_from_run_metadata is not None:
            payload["pinned_from_run_metadata"] = self.pinned_from_run_metadata
        payload.update(self.extra)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KDTeacherSnapshotMetadata | None":
        if not data:
            return None

        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key, value in data.items():
            if key in _TEACHER_PATH_FIELDS:
                kwargs[key] = _normalize_path(value)
            elif key in _TEACHER_INT_FIELDS:
                kwargs[key] = _normalize_int(value)
            elif key in _TEACHER_BOOL_FIELDS:
                kwargs[key] = _normalize_bool(value)
            elif key in _TEACHER_STRING_FIELDS:
                kwargs[key] = value
            else:
                extra[key] = value

        teacher_metadata = cls(**kwargs, extra=extra)
        if teacher_metadata.is_empty():
            return None
        return teacher_metadata


def _legacy_teacher_metadata_from_mapping(
    data: Mapping[str, Any] | None,
) -> KDTeacherSnapshotMetadata | None:
    if not data:
        return None

    original_path = data.get("teacher_policy_path")
    if original_path is None:
        original_path = data.get("teacher_train_config")
    if original_path is None:
        original_path = data.get("original_path")

    teacher_metadata = KDTeacherSnapshotMetadata(
        original_path=_normalize_path(original_path),
        pinned_pretrained_path=_normalize_path(
            data.get("resolved_teacher_pretrained_path", data.get("pinned_pretrained_path"))
        ),
        checkpoint_step=_normalize_int(data.get("teacher_checkpoint_step", data.get("checkpoint_step"))),
        source_kind=data.get("teacher_source_kind", data.get("source_kind")),
        pinned_from_run_metadata=_normalize_bool(data.get("pinned_from_run_metadata")),
    )
    if teacher_metadata.is_empty():
        return None
    return teacher_metadata


def as_kd_teacher_snapshot_metadata(
    metadata: KDTeacherSnapshotMetadata | Mapping[str, Any] | None,
) -> KDTeacherSnapshotMetadata | None:
    if metadata is None:
        return None
    if isinstance(metadata, KDTeacherSnapshotMetadata):
        if metadata.is_empty():
            return None
        return metadata
    return KDTeacherSnapshotMetadata.from_mapping(metadata)


def merge_kd_teacher_snapshot_metadata(
    primary: KDTeacherSnapshotMetadata | Mapping[str, Any] | None,
    fallback: KDTeacherSnapshotMetadata | Mapping[str, Any] | None,
) -> KDTeacherSnapshotMetadata | None:
    primary_metadata = as_kd_teacher_snapshot_metadata(primary)
    fallback_metadata = as_kd_teacher_snapshot_metadata(fallback)

    if primary_metadata is None:
        return fallback_metadata
    if fallback_metadata is None:
        return primary_metadata

    return KDTeacherSnapshotMetadata(
        original_path=_pick(primary_metadata.original_path, fallback_metadata.original_path),
        pinned_pretrained_path=_pick(
            primary_metadata.pinned_pretrained_path, fallback_metadata.pinned_pretrained_path
        ),
        checkpoint_step=_pick(primary_metadata.checkpoint_step, fallback_metadata.checkpoint_step),
        source_kind=_pick(primary_metadata.source_kind, fallback_metadata.source_kind),
        pinned_from_run_metadata=_pick(
            primary_metadata.pinned_from_run_metadata, fallback_metadata.pinned_from_run_metadata
        ),
        extra={**fallback_metadata.extra, **primary_metadata.extra},
    )


@dataclass
class KDTeacherMetadata:
    schema_version: int = KD_TEACHER_METADATA_SCHEMA_VERSION
    comparison_space: str | None = None
    processor_compatibility: str | None = None
    metric_aggregation_mode: str | None = None
    teacher: KDTeacherSnapshotMetadata = field(default_factory=KDTeacherSnapshotMetadata)
    extra: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            self.comparison_space is None
            and self.processor_compatibility is None
            and self.metric_aggregation_mode is None
            and self.teacher.is_empty()
            and not self.extra
        )

    @property
    def teacher_policy_path(self) -> Path | None:
        return self.teacher.original_path

    @property
    def teacher_train_config(self) -> Path | None:
        teacher_source_path = self.teacher.original_path
        if teacher_source_path is not None and teacher_source_path.name == "train_config.json":
            return teacher_source_path
        return None

    @property
    def resolved_teacher_pretrained_path(self) -> Path | None:
        return self.teacher.pinned_pretrained_path

    @property
    def teacher_source_kind(self) -> str | None:
        return self.teacher.source_kind

    @property
    def teacher_checkpoint_step(self) -> int | None:
        return self.teacher.checkpoint_step

    @property
    def processor_compatibility_mode(self) -> str | None:
        return self.processor_compatibility

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": self.schema_version}
        if self.comparison_space is not None:
            payload["comparison_space"] = self.comparison_space
        if self.processor_compatibility is not None:
            payload["processor_compatibility"] = self.processor_compatibility
        if self.metric_aggregation_mode is not None:
            payload["metric_aggregation_mode"] = self.metric_aggregation_mode

        teacher_payload = self.teacher.to_dict()
        if teacher_payload:
            payload["teacher"] = teacher_payload

        payload.update(self.extra)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KDTeacherMetadata | None":
        if not data:
            return None

        teacher_metadata = None
        teacher_mapping = data.get("teacher")
        if isinstance(teacher_mapping, Mapping):
            teacher_metadata = KDTeacherSnapshotMetadata.from_mapping(teacher_mapping)

        teacher_metadata = merge_kd_teacher_snapshot_metadata(
            teacher_metadata, _legacy_teacher_metadata_from_mapping(data)
        ) or KDTeacherSnapshotMetadata()

        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key, value in data.items():
            if key == "teacher":
                continue
            if key in _ROOT_INT_FIELDS:
                kwargs[key] = _normalize_int(value)
            elif key in _ROOT_STRING_FIELDS:
                kwargs[key] = value
            elif key == "processor_compatibility_mode":
                kwargs["processor_compatibility"] = value
            elif key not in _ROOT_RESERVED_KEYS:
                extra[key] = value

        metadata = cls(
            schema_version=kwargs.get("schema_version") or KD_TEACHER_METADATA_SCHEMA_VERSION,
            comparison_space=kwargs.get("comparison_space"),
            processor_compatibility=kwargs.get("processor_compatibility"),
            metric_aggregation_mode=kwargs.get("metric_aggregation_mode"),
            teacher=teacher_metadata,
            extra=extra,
        )
        if metadata.is_empty():
            return None
        return metadata


def create_kd_teacher_metadata(
    *,
    teacher_source_path: Path | str | None = None,
    teacher_pretrained_path: Path | str | None = None,
    teacher_checkpoint_step: int | None = None,
    teacher_source_kind: str | None = None,
    pinned_from_run_metadata: bool | None = None,
    comparison_space: str | None = None,
    processor_compatibility: str | None = None,
    metric_aggregation_mode: str | None = None,
    schema_version: int = KD_TEACHER_METADATA_SCHEMA_VERSION,
    extra: Mapping[str, Any] | None = None,
    teacher_extra: Mapping[str, Any] | None = None,
) -> KDTeacherMetadata:
    return KDTeacherMetadata(
        schema_version=schema_version,
        comparison_space=comparison_space,
        processor_compatibility=processor_compatibility,
        metric_aggregation_mode=metric_aggregation_mode,
        teacher=KDTeacherSnapshotMetadata(
            original_path=_normalize_path(teacher_source_path),
            pinned_pretrained_path=_normalize_path(teacher_pretrained_path),
            checkpoint_step=teacher_checkpoint_step,
            source_kind=teacher_source_kind,
            pinned_from_run_metadata=pinned_from_run_metadata,
            extra=dict(teacher_extra or {}),
        ),
        extra=dict(extra or {}),
    )


def as_kd_teacher_metadata(
    metadata: KDTeacherMetadata | Mapping[str, Any] | None,
) -> KDTeacherMetadata | None:
    if metadata is None:
        return None
    if isinstance(metadata, KDTeacherMetadata):
        if metadata.is_empty():
            return None
        return metadata
    return KDTeacherMetadata.from_mapping(metadata)


def kd_teacher_metadata_to_dict(
    metadata: KDTeacherMetadata | Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_metadata = as_kd_teacher_metadata(metadata)
    if normalized_metadata is None:
        return {}
    return normalized_metadata.to_dict()


def merge_kd_teacher_metadata(
    primary: KDTeacherMetadata | Mapping[str, Any] | None,
    fallback: KDTeacherMetadata | Mapping[str, Any] | None,
) -> KDTeacherMetadata | None:
    primary_metadata = as_kd_teacher_metadata(primary)
    fallback_metadata = as_kd_teacher_metadata(fallback)

    if primary_metadata is None:
        return fallback_metadata
    if fallback_metadata is None:
        return primary_metadata

    return KDTeacherMetadata(
        schema_version=primary_metadata.schema_version or fallback_metadata.schema_version,
        comparison_space=_pick(primary_metadata.comparison_space, fallback_metadata.comparison_space),
        processor_compatibility=_pick(
            primary_metadata.processor_compatibility, fallback_metadata.processor_compatibility
        ),
        metric_aggregation_mode=_pick(
            primary_metadata.metric_aggregation_mode, fallback_metadata.metric_aggregation_mode
        ),
        teacher=merge_kd_teacher_snapshot_metadata(
            primary_metadata.teacher, fallback_metadata.teacher
        )
        or KDTeacherSnapshotMetadata(),
        extra={**fallback_metadata.extra, **primary_metadata.extra},
    )


def get_kd_teacher_metadata_path(root_dir: Path) -> Path:
    return Path(root_dir) / KD_TEACHER_METADATA_FILENAME


def load_kd_teacher_metadata(root_dir: Path | None) -> KDTeacherMetadata | None:
    if root_dir is None:
        return None
    metadata_path = get_kd_teacher_metadata_path(root_dir)
    if not metadata_path.is_file():
        return None
    return KDTeacherMetadata.from_mapping(load_json(metadata_path))


def save_kd_teacher_metadata(
    metadata: KDTeacherMetadata | Mapping[str, Any],
    root_dir: Path,
) -> Path:
    normalized_metadata = as_kd_teacher_metadata(metadata)
    if normalized_metadata is None:
        raise ValueError("KD teacher metadata is empty and cannot be saved.")

    metadata_path = get_kd_teacher_metadata_path(root_dir)
    write_json(normalized_metadata.to_dict(), metadata_path)
    return metadata_path


def resolve_kd_teacher_metadata_for_resume(
    *,
    checkpoint_dir: Path | None = None,
    run_dir: Path | None = None,
    embedded_metadata: KDTeacherMetadata | Mapping[str, Any] | None = None,
) -> KDTeacherMetadata | None:
    merged_metadata = load_kd_teacher_metadata(checkpoint_dir)
    merged_metadata = merge_kd_teacher_metadata(merged_metadata, load_kd_teacher_metadata(run_dir))
    return merge_kd_teacher_metadata(merged_metadata, embedded_metadata)
