#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SCRIPT_DIR = Path(__file__).resolve().parent
PI_TRT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lerobot.policies.utils import prepare_observation_for_inference


_UNSET = object()


@dataclass(frozen=True)
class ChunkPredictionResult:
    original_actions: Tensor
    processed_actions: list[Any]
    preprocess_time_s: float
    inference_time_s: float
    postprocess_time_s: float
    processed_actions_tensor: Tensor | None = None
    action_index_before_inference: int | None = None
    submit_time_s: float | None = None
    ready_time_s: float | None = None
    real_delay: int | None = None

    @property
    def total_time_s(self) -> float:
        return self.preprocess_time_s + self.inference_time_s + self.postprocess_time_s

    @property
    def submit_to_ready_time_s(self) -> float | None:
        if self.submit_time_s is None or self.ready_time_s is None:
            return None
        return max(0.0, self.ready_time_s - self.submit_time_s)

    @property
    def actions(self) -> list[Any]:
        return self.processed_actions

    @property
    def num_actions(self) -> int:
        if self.processed_actions_tensor is not None:
            return int(self.processed_actions_tensor.shape[0])
        if self.original_actions.ndim == 2:
            return int(self.original_actions.shape[0])
        if self.original_actions.ndim == 3 and int(self.original_actions.shape[0]) == 1:
            return int(self.original_actions.shape[1])
        return len(self.processed_actions)

    def with_real_delay(self, real_delay: int | None) -> "ChunkPredictionResult":
        normalized_delay = None if real_delay is None else max(0, int(real_delay))
        return replace(self, real_delay=normalized_delay)

    def require_original_actions_matrix(self) -> Tensor:
        return _ensure_action_sequence_matrix(
            self.original_actions,
            tensor_name="ChunkPredictionResult.original_actions",
        )

    def require_processed_actions_tensor(self) -> Tensor:
        if self.processed_actions_tensor is None:
            raise TypeError(
                "Processed actions are not tensor-convertible. "
                "This chunk result can be used for list-based rollout but not ActionQueue.merge(...)."
            )
        return self.processed_actions_tensor

    def action_queue_payload(self) -> tuple[Tensor, Tensor]:
        return (
            self.require_original_actions_matrix(),
            self.require_processed_actions_tensor(),
        )


def _resolve_n_action_steps(policy: Any, n_action_steps: int | None) -> int:
    if n_action_steps is not None:
        resolved = int(n_action_steps)
    else:
        policy_config = getattr(policy, "config", None)
        resolved = getattr(policy_config, "n_action_steps", None)
        if resolved is None:
            raise ValueError(
                "`n_action_steps` must be provided when policy.config.n_action_steps is unavailable."
            )
        resolved = int(resolved)

    if resolved <= 0:
        raise ValueError(f"`n_action_steps` must be positive, got {resolved}")
    return resolved


def _ensure_action_chunk_tensor(action_chunk: Any) -> Tensor:
    if not isinstance(action_chunk, torch.Tensor):
        raise TypeError(
            "`policy.predict_action_chunk(...)` must return a torch.Tensor, "
            f"got {type(action_chunk)}"
        )
    if action_chunk.ndim == 2:
        return action_chunk.unsqueeze(0)
    if action_chunk.ndim != 3:
        raise ValueError(
            "`policy.predict_action_chunk(...)` must return a tensor with shape (T, A) or (B, T, A), "
            f"got {tuple(action_chunk.shape)}"
        )
    return action_chunk


def _ensure_action_sequence_matrix(action_sequence: Tensor, *, tensor_name: str) -> Tensor:
    if action_sequence.ndim == 2:
        return action_sequence.detach().clone().contiguous()
    if action_sequence.ndim == 3 and int(action_sequence.shape[0]) == 1:
        return action_sequence.squeeze(0).detach().clone().contiguous()
    raise ValueError(
        f"{tensor_name} must have shape (T, A) or (1, T, A), got {tuple(action_sequence.shape)}"
    )


def _coerce_processed_action_tensor(
    processed_action: Any,
    *,
    action_index: int,
) -> Tensor:
    if isinstance(processed_action, torch.Tensor):
        action_tensor = processed_action.detach().cpu()
    else:
        action_tensor = torch.as_tensor(processed_action)

    if action_tensor.ndim == 2 and int(action_tensor.shape[0]) == 1:
        action_tensor = action_tensor.squeeze(0)
    if action_tensor.ndim != 1:
        raise ValueError(
            "Each processed action must be convertible to a 1D tensor for queue-based rollout, "
            f"but step {action_index} has shape {tuple(action_tensor.shape)}"
        )
    return action_tensor.to(dtype=torch.float32).contiguous()


def _maybe_stack_processed_actions(
    processed_actions: Sequence[Any],
    *,
    empty_action_dim: int | None,
) -> Tensor | None:
    if not processed_actions:
        if empty_action_dim is None:
            return None
        return torch.empty((0, int(empty_action_dim)), dtype=torch.float32)

    processed_tensors: list[Tensor] = []
    try:
        for index, processed_action in enumerate(processed_actions):
            processed_tensors.append(
                _coerce_processed_action_tensor(processed_action, action_index=index)
            )
    except (TypeError, ValueError, RuntimeError):
        return None

    try:
        return torch.stack(processed_tensors, dim=0).contiguous()
    except RuntimeError:
        return None


def _normalize_predict_action_chunk_kwargs(
    *,
    predict_kwargs: Mapping[str, Any] | None,
    extra_predict_action_chunk_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_kwargs: dict[str, Any] = {}
    if predict_kwargs is not None:
        normalized_kwargs.update(dict(predict_kwargs))

    for key, value in extra_predict_action_chunk_kwargs.items():
        if key in normalized_kwargs:
            raise TypeError(
                f"Duplicate predict_action_chunk kwarg {key!r} passed via both "
                "`predict_kwargs` and direct keyword arguments."
            )
        normalized_kwargs[key] = value
    return normalized_kwargs


def _read_object_field(source: Any, field_name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _iter_rtc_sources(
    *,
    rtc_runtime: Any | None,
    action_queue: Any | None,
) -> Any:
    pending = [rtc_runtime, action_queue]
    seen: set[int] = set()

    while pending:
        source = pending.pop(0)
        if source is None:
            continue

        source_id = id(source)
        if source_id in seen:
            continue
        seen.add(source_id)
        yield source

        for nested_field_name in ("config", "rtc_config", "cfg"):
            nested_source = _read_object_field(source, nested_field_name)
            if nested_source is not None:
                pending.append(nested_source)


def _resolve_rtc_enabled(
    *,
    rtc_enabled: bool | None,
    rtc_runtime: Any | None,
    action_queue: Any | None,
) -> bool:
    if rtc_enabled is not None:
        return bool(rtc_enabled)

    for source in _iter_rtc_sources(rtc_runtime=rtc_runtime, action_queue=action_queue):
        enabled = _read_object_field(source, "enabled")
        if enabled is not None:
            return bool(enabled)
    return False


def _normalize_optional_non_negative_int(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"`{field_name}` must be non-negative, got {normalized}")
    return normalized


def _normalize_optional_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"`{field_name}` must be positive, got {normalized}")
    return normalized


def _resolve_execution_horizon(
    *,
    execution_horizon: int | None,
    rtc_runtime: Any | None,
    action_queue: Any | None,
) -> int | None:
    resolved_value = execution_horizon
    if resolved_value is None:
        for source in _iter_rtc_sources(rtc_runtime=rtc_runtime, action_queue=action_queue):
            candidate = _read_object_field(source, "execution_horizon")
            if candidate is not None:
                resolved_value = candidate
                break

    return _normalize_optional_positive_int(
        resolved_value,
        field_name="execution_horizon",
    )


def _resolve_left_over_value(
    provider: Any,
) -> Any:
    if provider is None:
        return None
    if isinstance(provider, torch.Tensor):
        return provider
    if callable(provider):
        return provider()
    get_left_over = _read_object_field(provider, "get_left_over")
    if callable(get_left_over):
        return get_left_over()
    return provider


def _normalize_prev_chunk_left_over(
    prev_chunk_left_over: Any,
    *,
    clone_left_over: bool,
    drop_empty_left_over: bool,
) -> Tensor | None:
    if prev_chunk_left_over is None:
        return None

    if isinstance(prev_chunk_left_over, torch.Tensor):
        normalized = prev_chunk_left_over.detach()
        if clone_left_over:
            normalized = normalized.clone()
    else:
        normalized = torch.as_tensor(prev_chunk_left_over)
        if clone_left_over:
            normalized = normalized.clone()

    normalized = normalized.contiguous()
    if drop_empty_left_over and normalized.numel() == 0:
        return None
    return normalized


def _collect_explicit_rtc_predict_fields(
    *,
    prev_chunk_left_over: Any,
    inference_delay: int | None,
    execution_horizon: int | None,
) -> tuple[str, ...]:
    explicit_fields: list[str] = []
    if prev_chunk_left_over is not _UNSET:
        explicit_fields.append("prev_chunk_left_over")
    if inference_delay is not None:
        explicit_fields.append("inference_delay")
    if execution_horizon is not None:
        explicit_fields.append("execution_horizon")
    return tuple(explicit_fields)


def build_chunk_predict_kwargs(
    *,
    rtc_enabled: bool | None = None,
    rtc_runtime: Any | None = None,
    action_queue: Any | None = None,
    left_over_provider: Any | None = None,
    prev_chunk_left_over: Any = _UNSET,
    predicted_delay_steps: int | None = None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
    clone_left_over: bool = True,
    drop_empty_left_over: bool = True,
) -> dict[str, Any]:
    """Build RTC predict kwargs without binding to launcher-private runtime types."""
    rtc_is_enabled = _resolve_rtc_enabled(
        rtc_enabled=rtc_enabled,
        rtc_runtime=rtc_runtime,
        action_queue=action_queue,
    )
    explicit_rtc_fields = _collect_explicit_rtc_predict_fields(
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
    )

    if not rtc_is_enabled:
        if explicit_rtc_fields:
            formatted_fields = ", ".join(f"`{field_name}`" for field_name in explicit_rtc_fields)
            raise ValueError(
                f"RTC predict kwargs {formatted_fields} were provided, but RTC is not enabled. "
                "Pass `rtc_enabled=True` (or an enabled `rtc_runtime` / `action_queue`) when "
                "supplying RTC-only inputs, or omit those inputs for non-RTC inference."
            )
        return {}

    if prev_chunk_left_over is _UNSET:
        left_over_source = left_over_provider if left_over_provider is not None else action_queue
        raw_prev_chunk_left_over = _resolve_left_over_value(left_over_source)
    else:
        raw_prev_chunk_left_over = prev_chunk_left_over

    normalized_prev_chunk_left_over = _normalize_prev_chunk_left_over(
        raw_prev_chunk_left_over,
        clone_left_over=clone_left_over,
        drop_empty_left_over=drop_empty_left_over,
    )
    normalized_inference_delay = _normalize_optional_non_negative_int(
        inference_delay if inference_delay is not None else predicted_delay_steps,
        field_name="inference_delay",
    )
    if normalized_prev_chunk_left_over is not None and normalized_inference_delay is None:
        raise ValueError("`inference_delay` is required when `prev_chunk_left_over` is provided.")

    return {
        "inference_delay": normalized_inference_delay,
        "prev_chunk_left_over": normalized_prev_chunk_left_over,
        "execution_horizon": _resolve_execution_horizon(
            execution_horizon=execution_horizon,
            rtc_runtime=rtc_runtime,
            action_queue=action_queue,
        ),
    }


def prepare_policy_observation(
    observation_frame: Mapping[str, Any],
    *,
    device: Any,
    preprocessor: Any,
    task: str,
    robot_type: str,
) -> dict[str, Any]:
    observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task,
        robot_type,
    )
    return preprocessor(observation)


def postprocess_action_chunk(
    action_chunk: Tensor,
    *,
    postprocessor: Any,
) -> list[Any]:
    normalized_chunk = _ensure_action_chunk_tensor(action_chunk)
    processed_actions: list[Any] = []
    for index in range(int(normalized_chunk.shape[1])):
        processed_action = postprocessor(normalized_chunk[:, index, :])
        if isinstance(processed_action, torch.Tensor):
            processed_action = processed_action.squeeze(0).detach().cpu()
        processed_actions.append(processed_action)
    return processed_actions


def compute_real_delay(
    *,
    action_index_before_inference: int | None,
    action_index_after_inference: int | None,
) -> int | None:
    if action_index_before_inference is None or action_index_after_inference is None:
        return None
    return max(0, int(action_index_after_inference) - int(action_index_before_inference))


def _resolve_merge_real_delay(
    *,
    action_queue: Any,
    prediction: ChunkPredictionResult,
    action_index_before_inference: int | None,
    action_index_after_inference: int | None,
    real_delay: int | None,
) -> int:
    normalized_real_delay = _normalize_optional_non_negative_int(
        real_delay,
        field_name="real_delay",
    )
    if normalized_real_delay is not None:
        return normalized_real_delay

    resolved_action_index_after_inference = action_index_after_inference
    get_action_index = getattr(action_queue, "get_action_index", None)
    action_index_source = "argument"
    if resolved_action_index_after_inference is None and callable(get_action_index):
        resolved_action_index_after_inference = get_action_index()
        action_index_source = "action_queue.get_action_index()"

    computed_real_delay = compute_real_delay(
        action_index_before_inference=action_index_before_inference,
        action_index_after_inference=resolved_action_index_after_inference,
    )
    if computed_real_delay is not None:
        return computed_real_delay

    normalized_prediction_real_delay = _normalize_optional_non_negative_int(
        prediction.real_delay,
        field_name="prediction.real_delay",
    )
    if normalized_prediction_real_delay is not None:
        return normalized_prediction_real_delay

    failure_details: list[str] = []
    if action_index_before_inference is None:
        failure_details.append(
            "`action_index_before_inference` is missing (pass it explicitly or keep it on the prediction result)"
        )
    if resolved_action_index_after_inference is None:
        if callable(get_action_index):
            failure_details.append(f"`{action_index_source}` returned None")
        else:
            failure_details.append(
                "`action_index_after_inference` is missing and action_queue does not expose `get_action_index()`"
            )

    detail_suffix = ""
    if failure_details:
        detail_suffix = " Details: " + "; ".join(failure_details) + "."
    raise ValueError(
        "Cannot merge chunk prediction without a reliable `real_delay`. Provide explicit "
        "`real_delay`, or provide both action indexes so the helper can compute it, or set "
        "`prediction.real_delay` before merging."
        + detail_suffix
    )


def merge_chunk_prediction_result(
    action_queue: Any,
    prediction: ChunkPredictionResult,
    *,
    action_index_after_inference: int | None = None,
    action_index_before_inference: int | None = None,
    real_delay: int | None = None,
) -> int:
    """Merge a predicted chunk into an action queue with explicit real-delay semantics."""
    resolved_action_index_before_inference = (
        prediction.action_index_before_inference
        if action_index_before_inference is None
        else int(action_index_before_inference)
    )
    resolved_real_delay = _resolve_merge_real_delay(
        action_queue=action_queue,
        prediction=prediction,
        action_index_before_inference=resolved_action_index_before_inference,
        action_index_after_inference=action_index_after_inference,
        real_delay=real_delay,
    )

    original_actions, processed_actions = prediction.action_queue_payload()
    action_queue.merge(
        original_actions,
        processed_actions,
        real_delay=int(resolved_real_delay),
        action_index_before_inference=resolved_action_index_before_inference,
    )
    return int(resolved_real_delay)


def predict_processed_action_chunk(
    *,
    policy: Any,
    observation_frame: Mapping[str, Any],
    device: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: str,
    robot_type: str,
    n_action_steps: int | None = None,
    action_index_before_inference: int | None = None,
    submit_time_s: float | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
    **predict_action_chunk_kwargs: Any,
) -> ChunkPredictionResult:
    effective_submit_time_s = time.perf_counter() if submit_time_s is None else float(submit_time_s)
    resolved_n_action_steps = _resolve_n_action_steps(policy, n_action_steps)
    resolved_predict_kwargs = _normalize_predict_action_chunk_kwargs(
        predict_kwargs=predict_kwargs,
        extra_predict_action_chunk_kwargs=predict_action_chunk_kwargs,
    )

    preprocess_start_s = time.perf_counter()
    policy_observation = prepare_policy_observation(
        observation_frame,
        device=device,
        preprocessor=preprocessor,
        task=task,
        robot_type=robot_type,
    )
    preprocess_time_s = time.perf_counter() - preprocess_start_s

    inference_start_s = time.perf_counter()
    raw_action_chunk = policy.predict_action_chunk(policy_observation, **resolved_predict_kwargs)
    batched_action_chunk = _ensure_action_chunk_tensor(raw_action_chunk)
    truncated_action_chunk = batched_action_chunk[:, :resolved_n_action_steps, :].detach()
    inference_time_s = time.perf_counter() - inference_start_s

    postprocess_start_s = time.perf_counter()
    processed_actions = postprocess_action_chunk(
        truncated_action_chunk,
        postprocessor=postprocessor,
    )
    postprocess_time_s = time.perf_counter() - postprocess_start_s
    processed_actions_tensor = _maybe_stack_processed_actions(
        processed_actions,
        empty_action_dim=int(truncated_action_chunk.shape[-1]),
    )

    # The live control path uses batch size 1, so store a 2D tensor to remain
    # directly compatible with ActionQueue.merge(...). Keep the batched form for
    # multi-sample calls.
    if truncated_action_chunk.shape[0] == 1:
        original_actions = truncated_action_chunk.squeeze(0).clone()
    else:
        original_actions = truncated_action_chunk.clone()

    ready_time_s = time.perf_counter()
    return ChunkPredictionResult(
        original_actions=original_actions,
        processed_actions=processed_actions,
        preprocess_time_s=preprocess_time_s,
        inference_time_s=inference_time_s,
        postprocess_time_s=postprocess_time_s,
        processed_actions_tensor=processed_actions_tensor,
        action_index_before_inference=action_index_before_inference,
        submit_time_s=effective_submit_time_s,
        ready_time_s=ready_time_s,
    )


def estimate_prefetch_threshold(
    *,
    configured_threshold: int | None,
    n_action_steps: int,
    chunk_latency_s: float | None,
    step_time_s: float | None,
    fallback_fps: float,
) -> int:
    if configured_threshold is not None:
        return max(0, int(configured_threshold))

    effective_step_time_s = step_time_s
    if effective_step_time_s is None:
        effective_step_time_s = 1.0 / max(float(fallback_fps), 1e-6)

    if chunk_latency_s is None or effective_step_time_s <= 0:
        return max(1, int(n_action_steps) // 2)

    refill_steps = max(1, math.ceil(float(chunk_latency_s) / effective_step_time_s))
    return max(1, min(int(n_action_steps), refill_steps + 1))


class AsyncChunkPrefetcher:
    def __init__(
        self,
        *,
        policy: Any,
        device: Any,
        preprocessor: Any,
        postprocessor: Any,
        task: str,
        robot_type: str,
        n_action_steps: int | None = None,
        thread_name_prefix: str = "pi05_chunk_prefetch",
    ) -> None:
        self.policy = policy
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task = task
        self.robot_type = robot_type
        self.n_action_steps = n_action_steps
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self.future: Future[ChunkPredictionResult] | None = None
        self.submitted_at_s: float | None = None

    def _submit_impl(
        self,
        observation_frame: dict[str, Any],
        *,
        action_index_before_inference: int | None,
        submit_time_s: float | None,
        predict_kwargs: Mapping[str, Any] | None,
        extra_predict_action_chunk_kwargs: dict[str, Any],
    ) -> ChunkPredictionResult:
        return predict_processed_action_chunk(
            policy=self.policy,
            observation_frame=observation_frame,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            task=self.task,
            robot_type=self.robot_type,
            n_action_steps=self.n_action_steps,
            action_index_before_inference=action_index_before_inference,
            submit_time_s=submit_time_s,
            predict_kwargs=predict_kwargs,
            **extra_predict_action_chunk_kwargs,
        )

    def predict_sync(
        self,
        observation_frame: Mapping[str, Any],
        *,
        action_index_before_inference: int | None = None,
        submit_time_s: float | None = None,
        predict_kwargs: Mapping[str, Any] | None = None,
        **predict_action_chunk_kwargs: Any,
    ) -> ChunkPredictionResult:
        return self._submit_impl(
            dict(observation_frame),
            action_index_before_inference=action_index_before_inference,
            submit_time_s=submit_time_s,
            predict_kwargs=predict_kwargs,
            extra_predict_action_chunk_kwargs=dict(predict_action_chunk_kwargs),
        )

    def has_pending(self) -> bool:
        return self.future is not None and not self.future.done()

    def has_future(self) -> bool:
        return self.future is not None

    def maybe_submit(
        self,
        observation_frame: Mapping[str, Any],
        *,
        action_index_before_inference: int | None = None,
        submit_time_s: float | None = None,
        predict_kwargs: Mapping[str, Any] | None = None,
        **predict_action_chunk_kwargs: Any,
    ) -> bool:
        if self.future is not None:
            return False

        effective_submit_time_s = time.perf_counter() if submit_time_s is None else float(submit_time_s)
        self.future = self.executor.submit(
            self._submit_impl,
            dict(observation_frame),
            action_index_before_inference=action_index_before_inference,
            submit_time_s=effective_submit_time_s,
            predict_kwargs=predict_kwargs,
            extra_predict_action_chunk_kwargs=dict(predict_action_chunk_kwargs),
        )
        self.submitted_at_s = effective_submit_time_s
        return True

    def maybe_collect(self) -> ChunkPredictionResult | None:
        if self.future is None or not self.future.done():
            return None
        try:
            return self.future.result()
        finally:
            self.future = None
            self.submitted_at_s = None

    def wait_for_result(self, timeout_s: float) -> ChunkPredictionResult | None:
        if self.future is None:
            return None
        try:
            result = self.future.result(timeout=timeout_s)
        except TimeoutError:
            return None

        self.future = None
        self.submitted_at_s = None
        return result

    def close(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=True)


__all__ = [
    "AsyncChunkPrefetcher",
    "ChunkPredictionResult",
    "build_chunk_predict_kwargs",
    "compute_real_delay",
    "estimate_prefetch_threshold",
    "merge_chunk_prediction_result",
    "postprocess_action_chunk",
    "predict_processed_action_chunk",
    "prepare_policy_observation",
]
