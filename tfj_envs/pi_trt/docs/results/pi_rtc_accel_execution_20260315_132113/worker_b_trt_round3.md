# Worker B TRT Launcher Round 3

## Change Summary

- Modified only `scripts/run_pi05_trt_infer_so101.py`.
- Removed the launcher-local RTC glue helpers `build_chunk_predict_kwargs(...)` and `merge_prediction_chunk(...)`.
- Switched the TRT launcher main path to the shared helper APIs from `scripts/pi05_chunk_runtime.py`:
  - `build_chunk_predict_kwargs(...)`
  - `merge_chunk_prediction_result(...)`
- Kept RTC default-off behavior and existing CLI compatibility intact.
- Expanded runtime observability so periodic logs carry enough context to interpret `real_delay` and refill behavior.

## Self-Check Commands

Commands run:

```bash
python -m py_compile scripts/run_pi05_trt_infer_so101.py
python scripts/run_pi05_trt_infer_so101.py --help
python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help
python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help
```

Results:

- `py_compile` passed.
- `--help` passed.
- `--rtc-enable --help` passed.
- `--rtc-enabled --help` passed.
- No hardware, robot, camera, or TRT engine execution was performed in this round.

## Shared Helper Wiring

- The launcher now imports `build_chunk_predict_kwargs(...)` and `merge_chunk_prediction_result(...)` directly from `scripts/pi05_chunk_runtime.py`.
- Shared `build_chunk_predict_kwargs(...)` is now used for:
  - initial synchronous warmup chunk
  - async prefetch submission
  - synchronous refill chunk generation
- Shared `merge_chunk_prediction_result(...)` is now used for:
  - async collect
  - async wait success path
  - initial synchronous warmup merge
  - synchronous refill merge
- After the shared merge helper returns `real_delay`, the launcher copies that value back into the local `ChunkPredictionResult` via `with_real_delay(...)` only so the existing logging and latency bookkeeping can continue to read `prediction.real_delay`.
- RTC default-off behavior is unchanged because the launcher still resolves runtime enablement through the existing `ResolvedRTCRuntimeConfig` path, and the shared helper respects that resolved runtime state.

## Logging Semantics Changes

- Periodic logs now include `rtc_enabled=...`, so `real_delay=...` is no longer shown without runtime context.
- Periodic logs continue to include `refill_mode=...`.
- Added `sync_refill_count=...` to periodic logs.
- `queue_underrun_count` and `hold_step_count` remain hold-path counters only.
- `sync_refill_count` now makes blocking refill events visible so those hold counters are not misread as "all queue drain events".
- The synchronous refill warning is now more explicit and prints:
  - `refill_mode=sync_refill`
  - `reason=async_wait_timeout` or `reason=no_inflight_async_chunk`
  - `sync_refill_count`
  - `rtc_enabled`
  - `prefetch_pending`
  - `predicted_delay_steps`
- The sync refill warning also states that `real_delay=0` on this path reflects blocking refill semantics rather than healthy async overlap.

## Remaining Risks

- This round did not modify the shared helper itself. If worker A changes the helper contract again, the TRT launcher should be rechecked against the new API.
- No real robot or TRT runtime smoke run was executed, so this report only covers static wiring and CLI-level validation.
- The shared merge helper still falls back to `0` if no usable delay signal is available. In the current launcher path that should not regress behavior, but this remains a semantic risk for future callers.
