from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import TaskIntent, TaskSlots

_ENGLISH_OBJECT_CONTAINER_PATTERNS = (
    re.compile(
        r"(?:pick(?:\s+up)?|grab|take|move|put|place)\s+"
        r"(?P<object>.+?)\s+(?:and\s+)?(?:place|put|move)?\s*(?:it\s+)?"
        r"(?:into|in|to|onto|on)\s+(?P<container>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:put|place|move)\s+(?P<object>.+?)\s+(?:into|in|to|onto|on)\s+(?P<container>.+)$",
        re.IGNORECASE,
    ),
)
_ENGLISH_OBJECT_ONLY_PATTERN = re.compile(
    r"(?:pick(?:\s+up)?|grab|take|move|put|place)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
_CHINESE_PATTERNS = (
    re.compile(r"(?:请)?(?:把)?(?P<object>.+?)(?:放到|放进|放在|放入|拿到)(?P<container>.+)$"),
    re.compile(r"(?:抓起|拾起|拿起|移动)(?P<object>.+?)(?:放到|放进|放在|放入)(?P<container>.+)$"),
)
_RETRY_PATTERNS = (
    re.compile(r"(?:with\s+)?(?P<count>\d+)\s+(?:retry|retries|times)", re.IGNORECASE),
    re.compile(r"(?:retry|retries)\s*(?P<count>\d+)", re.IGNORECASE),
    re.compile(r"(?:最多)?重试\s*(?P<count>\d+)\s*次"),
)
_SPEED_PATTERNS = (
    (re.compile(r"\b(slowly|slow|carefully|gentle|gently)\b", re.IGNORECASE), "slow"),
    (re.compile(r"\b(quickly|fast|rapidly)\b", re.IGNORECASE), "fast"),
    (re.compile(r"(慢一点|慢速|谨慎)"), "slow"),
    (re.compile(r"(快速|迅速)"), "fast"),
)
_SAFETY_PATTERNS = (
    (re.compile(r"\b(?:in\s+)?(safe\s+mode|safety\s+mode|conservative)\b", re.IGNORECASE), "conservative"),
    (re.compile(r"(安全模式|谨慎模式)"), "conservative"),
)
_POLITE_PREFIXES = ("please ", "please, ", "请", "请把")
_TRAILING_FILLERS = re.compile(
    r"(?:\b(?:please|now|thanks)\b|[，,。.!！]+)$",
    re.IGNORECASE,
)
_ENGLISH_DETERMINERS = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


@dataclass
class IntentParser:
    default_verb: str = "pick_place"

    def parse(self, raw_task_text: str) -> TaskIntent:
        text = (raw_task_text or "").strip()
        language = "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
        cleaned_text, constraints = self._extract_constraints(text)
        slots = self._parse_chinese(cleaned_text) if language == "zh" else self._parse_english(cleaned_text)
        return TaskIntent(
            raw_text=text,
            verb=self.default_verb,
            slots=slots,
            constraints=constraints,
            language=language,
        )

    def _extract_constraints(self, text: str) -> tuple[str, dict[str, str | int]]:
        working = text
        constraints: dict[str, str | int] = {}

        for pattern in _RETRY_PATTERNS:
            match = pattern.search(working)
            if match:
                constraints["max_retries"] = int(match.group("count"))
                working = pattern.sub(" ", working, count=1)
                break

        for pattern, speed in _SPEED_PATTERNS:
            if pattern.search(working):
                constraints["speed"] = speed
                working = pattern.sub(" ", working)
                break

        for pattern, mode in _SAFETY_PATTERNS:
            if pattern.search(working):
                constraints["safety_mode"] = mode
                working = pattern.sub(" ", working)
                break

        return self._normalize_whitespace(working), constraints

    def _parse_english(self, text: str) -> TaskSlots:
        working = self._strip_polite_prefix(text.lower())
        for pattern in _ENGLISH_OBJECT_CONTAINER_PATTERNS:
            match = pattern.search(working)
            if match:
                return TaskSlots(
                    target_object=self._clean_slot(match.group("object")),
                    target_container=self._clean_slot(match.group("container")),
                )

        match = _ENGLISH_OBJECT_ONLY_PATTERN.search(working)
        if match:
            return TaskSlots(target_object=self._clean_slot(match.group("object")))
        return TaskSlots()

    def _parse_chinese(self, text: str) -> TaskSlots:
        working = self._strip_polite_prefix(text)
        for pattern in _CHINESE_PATTERNS:
            match = pattern.search(working)
            if match:
                return TaskSlots(
                    target_object=self._clean_slot(match.group("object")),
                    target_container=self._clean_slot(match.group("container")),
                )

        fallback = re.search(r"(?:抓起|拾起|拿起|移动|把)?(?P<object>.+)$", working)
        if fallback:
            return TaskSlots(target_object=self._clean_slot(fallback.group("object")))
        return TaskSlots()

    def _strip_polite_prefix(self, text: str) -> str:
        working = text.strip()
        for prefix in _POLITE_PREFIXES:
            if working.startswith(prefix):
                working = working[len(prefix) :].strip()
                break
        return self._normalize_whitespace(working)

    def _clean_slot(self, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = self._normalize_whitespace(value)
        cleaned = _TRAILING_FILLERS.sub("", cleaned).strip()
        cleaned = _ENGLISH_DETERMINERS.sub("", cleaned).strip()
        if cleaned.endswith("里"):
            cleaned = cleaned[:-1].strip()
        return cleaned or None

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" ，,。.!！")


_default_parser = IntentParser()


def parse_task_intent(raw_task_text: str) -> TaskIntent:
    return _default_parser.parse(raw_task_text)
