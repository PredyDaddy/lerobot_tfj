# Worker C ONNX Launcher Round 2

## 改动摘要

- 在 `scripts/run_pi05_onnx_infer_so101.py` 补齐 RTC CLI 参数：
  `--rtc-enable`、`--rtc-enabled`、`--rtc-execution-horizon`、`--rtc-max-guidance-weight`、
  `--rtc-prefix-attention-schedule`、`--rtc-debug`、`--rtc-debug-maxlen`。
- 新增 `ResolvedRTCRuntimeConfig`，读取 checkpoint 的 `rtc_config`，应用 CLI override，并保持默认 RTC 关闭；
  只有显式传入 `--rtc-*` 才会在 launcher runtime 打开 RTC。resolved config 会在构造 ONNX adapter 前写回 `policy_cfg.rtc_config`。
- 将 launcher 的 chunk 队列从 `deque` 切换到 `ActionQueue` 语义。
  预测时会向 adapter 透传 `prev_chunk_left_over`、`inference_delay`、`execution_horizon`；
  chunk 完成后基于 `action_index_before_inference` 和当前 `ActionQueue` 位置计算 `real_delay`，再执行 `ActionQueue.merge(...)`。
- 尽量复用 `scripts/pi05_chunk_runtime.py`，删除 launcher 内部重复的 chunk runtime 实现，改为直接复用：
  `AsyncChunkPrefetcher`、`ChunkPredictionResult`、`compute_real_delay`、`estimate_prefetch_threshold`。
- 启动摘要新增 resolved RTC config 输出；当 checkpoint 自带 RTC enabled 但 launcher runtime 未显式开启时，会打印 warning。
- 周期日志补齐 `rtc_enabled` 与 `real_delay`，并把 `queue_size / queue_underrun_count / hold_step_count / chunk_latency_s / prefetch_threshold`
  统一到更接近 TRT launcher 的口径。
- 修正 `sync_refill_timeout_s` 的等待条件：
  现在只有在 `action_queue.empty()` 且 `prefetcher.has_future()` 时才调用 `wait_for_result(...)`，
  能真正等待 pending future，而不是旧逻辑那种在没有 pending future 时才进入等待。

## 自检命令

- `python -m py_compile scripts/run_pi05_onnx_infer_so101.py`
  结果：通过。
- `python scripts/run_pi05_onnx_infer_so101.py --help`
  结果：通过，CLI 可正常展开。
- `python scripts/run_pi05_onnx_infer_so101.py --help | rg -- '--rtc|sync-refill|prefetch-threshold'`
  结果：通过，确认 `--rtc-enable, --rtc-enabled` 以及其余 RTC 选项、`--sync-refill-timeout-s`、`--prefetch-threshold` 已暴露。

说明：本轮未做硬件联调，也未运行真实 robot loop。

## TRT对齐情况

- RTC 参数层：已对齐到 TRT launcher 的最小闭环，并额外补了 `--rtc-enabled` 兼容别名。
- runtime config 层：已对齐 TRT 的 checkpoint 读取、CLI override、默认关闭策略、以及 adapter 构造前写回 `policy_cfg.rtc_config`。
- queue merge 层：已对齐 TRT 的 `ActionQueue`、`prev_chunk_left_over`、`inference_delay`、`execution_horizon`、`real_delay` merge 语义。
- 指标层：已对齐 TRT 的 resolved RTC summary / checkpoint-warning / 周期性 runtime 指标口径。
- 等待语义层：本次 ONNX 侧已修成“队列空且存在 future 时等待”，这点比当前 TRT 文件里的旧条件更符合 RTC async refill 预期。
- shared helper 复用：ONNX 侧已切到 `pi05_chunk_runtime.py`，避免继续保留本地重复实现。

## 剩余差异

- ONNX launcher 仍保留 ONNX 特有的 provider / stage2 report / artifact 路径摘要；TRT launcher 的 metadata safety、precision、build report 相关摘要不在本轮 ONNX 范围内。
- TRT launcher 里还有 TRT 特有的 `assert_finite_robot_action(...)`、camera fourcc、artifact safety 逻辑；本轮未迁入 ONNX launcher，因为不属于 ONNX launcher 的 RTC 闭环最小整改范围。
- 同步回填分支里 `real_delay` 仍固定为 `0`，这是与 TRT 一致的设计：进入同步分支时队列已经空了，阻塞式生成期间不会继续消费旧 chunk。

## 剩余风险

- 没有做硬件在环验证，`ActionQueue.merge(...)`、RTC kwargs 透传、以及 pending-future 等待逻辑只做了静态和 CLI 级自检。
- 共享 helper / adapter / TRT launcher 由其他 worker 并行修改；我当前按 `pi05_chunk_runtime.py` 和现有 `onnx_pi_adapter.py` API 对齐，若共享接口后续再变，需要重新跑一轮 smoke test。
- 当前自检没有覆盖真实 checkpoint 中 `rtc_config` 为 enabled 的实跑场景，因此 warning、override 和 non-RTC baseline 的运行表现仍需联调确认。
