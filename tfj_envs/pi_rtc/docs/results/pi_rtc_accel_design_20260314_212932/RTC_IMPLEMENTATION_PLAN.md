# PI0.5 RTC 加速实施计划

## 1. 目标

本计划对应的实现目标是：

- 不改 TRT engine 边界
- 把 RTC 重新接回 PI0.5 的 TRT runtime
- 把 TRT launcher 从同步 `select_action()` 路径升级成 RTC-aware async chunk runtime

## 2. 计划拆分

### Step A：在 adapter 中接回 RTC 数学逻辑

修改文件：

- [trt_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py)

具体改动：

- 引入 `RTCProcessor`
- 在 adapter 中持有 `rtc_processor`
- 新增 `_rtc_enabled()`
- 扩展 `predict_action_chunk()` 支持：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- 在 denoise loop 中定义 `denoise_step_partial(...)`
- RTC 开启时通过 `rtc_processor.denoise_step(...)` 计算 `v_t`
- RTC 关闭时保持现有路径不变

验收标准：

- 非 RTC 模式行为不变
- RTC 模式下 `predict_action_chunk()` 能正常跑通
- 输出 shape / dtype / finite 检查通过

### Step B：抽共享 chunk runtime helper

新增文件：

- [pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py)

建议新增内容：

- `ChunkPredictionResult`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `estimate_prefetch_threshold(...)`
- `AsyncChunkPrefetcher`
- `RtcRuntimeStats`

关键设计：

- `ChunkPredictionResult` 里同时保存：
  - `original_actions`
  - `processed_actions`
  - `preprocess_time_s`
  - `inference_time_s`
  - `postprocess_time_s`
  - `action_index_before_inference`
  - `submitted_at_s`
  - `completed_at_s`

验收标准：

- helper 不依赖具体 backend，只依赖 policy 对象提供 `predict_action_chunk`
- ONNX / TRT launcher 都能导入该 helper

### Step C：把 TRT launcher 切到 RTC-aware chunk loop

修改文件：

- [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py)

具体改动：

- 新增 CLI：
  - `--rtc-enabled`
  - `--rtc-execution-horizon`
  - `--rtc-prefix-attention-schedule`
  - `--rtc-max-guidance-weight`
  - `--rtc-debug`
  - `--prefetch-threshold`
  - `--sync-refill-timeout-s`
- 新增 runtime override helper，把 CLI 写回 `policy_cfg.rtc_config`
- RTC 关闭时保留现有同步路径
- RTC 开启时：
  - 不再走 `control_utils.predict_action()`
  - 改用显式 chunk runtime loop
  - 使用 `ActionQueue`
  - 使用 `AsyncChunkPrefetcher`
  - low-watermark 触发后台 chunk
  - 完成后执行 `queue.merge(...)`

验收标准：

- RTC 关闭时与当前 launcher 行为一致
- RTC 开启时能完整跑 loop，不出现 queue/index 异常
- 日志能打印：
  - `queue_size`
  - `real_delay`
  - `chunk_latency`
  - `underrun_count`
  - `hold_step_count`

### Step D：把 ONNX launcher 迁移到共享 helper

修改文件：

- [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py)

具体改动：

- 把已有 async prefetch 逻辑迁移到共享 helper
- 尽量使用同一套结果结构和统计字段
- 为后续 TRT / ONNX 行为对比统一口径

验收标准：

- ONNX launcher 行为保持现有功能
- shared helper 没有引入功能回退

### Step E：补 RTC 专用 benchmark 与报告

建议新增文件：

- [benchmark_pi_rtc_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/benchmark_pi_rtc_runtime.py)

如果第一轮不想新建脚本，也至少要在 launcher 中输出结构化统计。

建议统计：

- `mean_loop_ms`
- `p95_loop_ms`
- `over_budget_rate`
- `queue_underrun_count`
- `hold_step_count`
- `sync_refill_count`
- `chunk_latency_mean_ms`
- `chunk_latency_p95_ms`
- `real_delay_mean`
- `real_delay_p95`
- `smoothing_event_count`
- `delta_clip_event_count`

验收标准：

- 能明确区分“模型推理快”和“控制循环不饿死”

## 3. 不建议修改的文件

第一轮不建议改：

- [build_pi_trt_engine.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/build_pi_trt_engine.py)
- [step2_export_onnx.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/step2_export_onnx.py)
- [step4_build_engines.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/step4_build_engines.py)
- [step5_verify_trt.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/step5_verify_trt.py)

原因：

- 本次 RTC 方案不应该打断当前可部署工件链路
- 这是 runtime orchestration 任务，不是 export/build 任务

## 4. 验证顺序

建议按这个顺序验证：

1. `python -m py_compile` 验证修改文件
2. `--help` 检查新增 CLI
3. `--dry-run` 验证路径和配置
4. `--preflight-only` 验证 adapter + runtime 初始化
5. 离线 mock control loop benchmark
6. 真机短时运行

## 5. 真正的成功判据

这次 RTC 实施是否成功，不看一句“平均推理变快了”，而看以下组合：

- 同样 fps 下 queue underrun 是否下降
- hold step 是否下降
- 真实 control loop over-budget 比例是否下降
- smoothing / delta clamp 是否没有明显恶化
- 非 RTC 路径是否完全不回退

## 6. 风险控制

### 6.1 强制保留双路径

必须保留：

- RTC off
- RTC on

两条路径，且默认保持 RTC off。

### 6.2 先在 TRT 路径落地，再考虑共用

虽然最终建议抽共享 helper，但实施顺序上先确保 TRT 路径跑通最重要。

### 6.3 日志要足够强

如果没有足够日志，RTC 出问题时会非常难排。

建议日志必须直接打印：

- queue low-watermark
- pending future 状态
- action_index_before_inference
- real_delay
- merge 后 queue 长度

## 7. 实施优先级总结

推荐执行顺序：

1. `trt_pi_adapter.py`
2. `pi05_chunk_runtime.py`
3. `run_pi05_trt_infer_so101.py`
4. `run_pi05_onnx_infer_so101.py`
5. RTC benchmark / report

这套顺序最适合先把 TRT 上机收益做出来，再收拢共享抽象。
