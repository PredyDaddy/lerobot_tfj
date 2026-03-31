# Worker A Runtime Helper Report

## Scope

- Task: implement the RTC shared runtime helper described in `RTC_ACCEL_IMPLEMENTATION_PLAN.md`
- Code file to change: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py`
- Explicit non-goal: do not place RTC queue merge logic in this helper

## Required API Surface

- `ChunkPredictionResult`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `estimate_prefetch_threshold(...)`
- `AsyncChunkPrefetcher`

## Minimum Schema

- `original_actions`
- `processed_actions`
- `preprocess_time_s`
- `inference_time_s`
- `postprocess_time_s`
- `action_index_before_inference`
- `submit_time_s`
- `ready_time_s`

## Change Summary

- Added `ChunkPredictionResult` with the required RTC-facing timing and submission fields, plus optional `real_delay` and convenience latency properties.
- Added `prepare_policy_observation(...)` and `postprocess_action_chunk(...)` by lifting the ONNX launcher's chunk preprocess/postprocess flow into a backend-agnostic helper.
- Added `predict_processed_action_chunk(...)` that:
  - prepares the policy observation
  - calls `policy.predict_action_chunk(...)`
  - truncates to `n_action_steps`
  - preserves `original_actions`
  - returns `processed_actions`
  - forwards RTC kwargs such as `prev_chunk_left_over`, `inference_delay`, and `execution_horizon`
- Added `estimate_prefetch_threshold(...)` using the existing latency-aware threshold heuristic.
- Added `AsyncChunkPrefetcher` with `predict_sync(...)`, `maybe_submit(...)`, `maybe_collect(...)`, `wait_for_result(...)`, and `close(...)`.
- For batch size 1, `original_actions` is stored as a 2D tensor so the future launcher-side `ActionQueue.merge(...)` path can consume it directly without another squeeze.

## Self-Check

- Command: `python -m py_compile scripts/pi05_chunk_runtime.py`
- Result: pass, exit code `0`

## Remaining Risks

- This worker only added the shared helper file. ONNX/TRT launchers still need to import and use it before behavior can be validated end-to-end.
- `AsyncChunkPrefetcher` intentionally mirrors the current ONNX single-worker-thread pattern. It does not by itself prove TensorRT adapter/thread ownership safety; that must still be handled when Worker B/C wire TRT async execution.
- `ChunkPredictionResult.original_actions` is queue-friendly for the live batch-1 control path, but multi-sample callers will still receive a batched 3D tensor and must handle that explicitly.
