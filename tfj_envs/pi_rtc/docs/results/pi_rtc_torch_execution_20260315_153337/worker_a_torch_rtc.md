# Worker A Torch RTC Report

## 改动摘要

- 仅修改 `scripts/run_pi05_torch_infer_so101.py`，未触碰 shared helper、ONNX/TRT launcher、adapter、`src/` 策略实现。
- 为 torch launcher 补齐 RTC CLI/runtime config：
  - 新增 `--rtc-enable/--rtc-enabled`、`--rtc-execution-horizon`、`--rtc-max-guidance-weight`、`--rtc-prefix-attention-schedule`、`--rtc-debug`、`--rtc-debug-maxlen`。
  - 新增 `--prefetch-threshold`、`--sync-refill-timeout-s`。
  - 通过本地 `ResolvedRTCRuntimeConfig` 将 resolved config 写回 `policy_cfg.rtc_config`，默认 RTC 仍保持 off，只有显式 `--rtc-*` override 时才开启。
- 纯 PyTorch 实时主路径不再调用 `predict_action(...)` / `policy.select_action(...)` 的旧 wrapper 队列逻辑，改为显式 chunk runtime：
  - 使用 `AsyncChunkPrefetcher` 驱动后台 chunk 预测。
  - 使用 `ActionQueue` 统一消费动作。
  - 直接走 `policy.predict_action_chunk(...)`，并通过 `build_chunk_predict_kwargs(...)` / `merge_chunk_prediction_result(...)` 接 RTC glue。
- 保留现有控制保护逻辑，并补充 finite check：
  - 保留 `smooth_robot_action(...)`。
  - 保留 `clamp_robot_action_delta(...)`。
  - 发送前新增 `assert_finite_robot_action(...)`，拒绝非有限值。
- 日志补齐 RTC 关键字段：
  - 周期日志显式输出 `rtc_enabled`、`real_delay`、`refill_mode`、`sync_refill_count`。
  - 同步 refill warning 也显式输出 `refill_mode=sync_refill`、`sync_refill_count`、`rtc_enabled`，避免把 `real_delay=0` 误读为健康异步 overlap。

## RTC 接线路径

1. CLI / config：
   - parser 收集 `--rtc-*`、`--prefetch-threshold`、`--sync-refill-timeout-s`
   - `apply_pi_runtime_overrides(...)` 调用 `resolve_rtc_runtime_config(...)`
   - resolved config 写回 `policy_cfg.rtc_config`

2. 推理 runtime：
   - `TorchChunkPolicyRuntime` 仅包一层 AMP/inference_mode
   - shared `AsyncChunkPrefetcher` 内部调用 shared `predict_processed_action_chunk(...)`
   - shared helper 最终直接调用 `policy.predict_action_chunk(...)`

3. RTC glue / queue：
   - startup initial chunk、async submit、sync refill 都通过 shared `build_chunk_predict_kwargs(...)` 组装 `predict_action_chunk(...)` 的 RTC kwargs
   - `ActionQueue` 负责外部 rollout，不再依赖 policy 私有 `_action_queue`
   - async collect / async wait 使用 shared `merge_chunk_prediction_result(...)` 合并 chunk，并从 queue index 推导真实 `real_delay`
   - initial sync / sync refill 仍显式传 `real_delay=0`，保留阻塞式补货语义

4. 发送路径：
   - `ActionQueue.get()` -> `make_robot_action(...)` -> `robot_action_processor(...)`
   - 然后继续走 smoothing、delta clamp、finite check、`robot.send_action(...)`

## 自检命令

已执行：

```bash
python -m py_compile scripts/run_pi05_torch_infer_so101.py
python scripts/run_pi05_torch_infer_so101.py --help
python scripts/run_pi05_torch_infer_so101.py --rtc-enable --help
python - <<'PY'
from scripts.run_pi05_torch_infer_so101 import build_parser
args = build_parser().parse_args(['--rtc-enable'])
print({'rtc_enable': args.rtc_enable, 'sync_refill_timeout_s': args.sync_refill_timeout_s, 'prefetch_threshold': args.prefetch_threshold})
PY
```

结果：

- `py_compile` 通过。
- 两条 `--help` 都正常返回，RTC 参数成功出现在帮助输出中。
- 轻量 import/parse smoke 返回：

```python
{'rtc_enable': True, 'sync_refill_timeout_s': 1.0, 'prefetch_threshold': None}
```

## 仍未覆盖的风险

- 未做真机验证；`real_delay`、`refill_mode`、`sync_refill_count` 在真实相机/串口节拍下是否与现场体感完全一致，还需要设备侧回归。
- 纯 PyTorch 路径现在也使用后台 `AsyncChunkPrefetcher`；虽然静态检查和 parser smoke 正常，但 CUDA 上的线程化 chunk 预取仍缺少无硬件的状态机级自动化测试。
- `TorchChunkPolicyRuntime` 为了保留 AMP，在 shared helper 前加了一层本地 runtime wrapper；核心推理仍直接落到 `policy.predict_action_chunk(...)`，但这一层本地包装的 AMP 语义还没有经过长时间运行验证。
