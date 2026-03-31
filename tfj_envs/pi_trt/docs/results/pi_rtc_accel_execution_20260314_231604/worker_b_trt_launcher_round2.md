# Worker B TRT Launcher Round 2

## 改动摘要

- 仅修改了 `scripts/run_pi05_trt_infer_so101.py`。
- TRT launcher 已改为优先复用 `scripts/pi05_chunk_runtime.py` 中的共享能力：
  - `ChunkPredictionResult`
  - `AsyncChunkPrefetcher`
  - `compute_real_delay`
  - `estimate_prefetch_threshold`
- 删除了 launcher 内本地重复的 chunk 预测、postprocess、prefetcher 与 threshold 估算实现，避免继续漂移。
- `ActionQueue.merge(...)` 现在通过 shared helper 的 `action_queue_payload()` 取回 `(original_actions, processed_actions_tensor)`，保持原有 queue merge 所需数据不丢失。
- 保留了原有 TRT runtime 关键指标字段：
  - `queue_underrun_count`
  - `hold_step_count`
  - `real_delay`
- 为减少误读，周期日志新增了 `refill_mode`，用于区分 `initial_sync`、`async_collect`、`async_wait`、`hold_pending_async`、`sync_refill`。

## 自检命令

执行命令：

```bash
python -m py_compile scripts/run_pi05_trt_infer_so101.py
python scripts/run_pi05_trt_infer_so101.py --help
python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help
python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help
```

结果：

- `py_compile` 通过。
- `--help` 通过，并确认 CLI 同时暴露 `--rtc-enable, --rtc-enabled`。
- `--rtc-enabled --help` 与 `--rtc-enable --help` 均可成功解析，确认新旧命名都可用。

## CLI 兼容性说明

- 兼容设计文档中的 `--rtc-enabled`。
- 保留既有 `--rtc-enable` 用法不变。
- 两者共享同一 `dest=rtc_enable`，不会改变现有 override 判断逻辑。
- RTC 默认仍然是 off；只有显式传入 `--rtc-enable/--rtc-enabled` 或其他 `--rtc-*` override 时才会打开，不改变安全基线。

## 等待语义修正说明

旧行为问题：

- queue 已空时，代码只在 `not prefetcher.has_pending()` 的条件下才尝试 `wait_for_result(...)`。
- 这会跳过“已有 future 但仍在跑”的关键等待路径，导致还没给异步 chunk 一个 grace wait 就直接走 hold。

新行为：

1. 当 `action_queue.empty()` 且存在 future 时，先执行 `wait_for_result(sync_refill_timeout_s)`。
2. 如果在 timeout 内拿到 chunk，立刻 merge，并按正常路径更新 `real_delay` / latency / chunk_count。
3. 如果 timeout 后 future 仍在，进入 hold，继续维持“不并发发起第二个同步推理”的安全行为。
4. 只有 queue 为空且当前没有 future 可用时，才进入同步 refill。

补充说明：

- 同步 refill 仍然会得到 `real_delay=0`，因为这是阻塞式补货下的事实结果，不应伪造为其他值。
- 为避免把这个 `0` 误读成“RTC overlap 健康”，日志额外输出 `refill_mode=sync_refill`，并在触发同步 refill 时给出更明确的 warning 文案。

## 剩余风险

- 本轮没有修改 `scripts/pi05_chunk_runtime.py`。如果别的 worker 后续继续扩展 shared helper API，TRT launcher 目前是按现有公开方法 `with_real_delay()`、`action_queue_payload()`、`wait_for_result()` 适配的；若 helper 做破坏性改动，仍需重新回归验证。
- 本轮未做真机或 TRT engine 实跑，只做了无硬件最小自检；因此无法在本报告中证明真实 camera/robot/TRT 环境下的时序表现。
- `real_delay` 字段本身保持兼容，没有改名；日志层面已补充 `refill_mode` 来降低误读，但消费日志的外部脚本如果只看 `real_delay` 仍可能忽略 refill 路径差异。
