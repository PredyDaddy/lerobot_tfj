# PI05 RTC 加速实施 Plan

## 1. 范围

本 plan 对应的目标是：

- 在不修改 TRT engine 边界的前提下
- 给 PI05 的 TRT/ONNX runtime 接入 RTC 语义
- 并把 TRT launcher 从同步 `select_action()` 模式升级为 async chunk runtime

本 plan 只描述代码改造与验证顺序，不在这一份文档里承诺具体性能结果。

## 1.5 Step 0: 先固定 runtime schema

在进入 Commit A 之前，先固定一份最小 runtime schema，至少覆盖：

- `ChunkPredictionResult`
- queue/runtime stats 字段
- launcher 日志字段

最小字段建议包括：

- `original_actions`
- `processed_actions`
- `preprocess_time_s`
- `inference_time_s`
- `postprocess_time_s`
- `action_index_before_inference`
- `submit_time_s`
- `ready_time_s`
- `real_delay`

原因：

- `ActionQueue.merge(...)`、async prefetch、benchmark、日志都会消费这份 schema
- 如果这一步不先定下来，后面 `TRT/ONNX launcher` 和 `benchmark` 很容易各写各的，最后无法比较

## 2. 改动分解

### Commit A: 抽共享 chunk runtime helper

新增文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py`

主要内容：

- `ChunkPredictionResult`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `estimate_prefetch_threshold(...)`
- `AsyncChunkPrefetcher`

要求：

- 不区分 ONNX/TRT 后端
- 只依赖 policy adapter 提供 `predict_action_chunk(...)`
- 结果里同时保留 `original_actions` 与 `processed_actions`

不做：

- 不引入 RTC merge
- 不改 launcher 主循环

验收：

- 新 helper 可被 ONNX launcher 替换使用
- 现有 ONNX async 行为不回退

### Commit B: 给 ONNX/TRT adapter 接 RTCProcessor

修改文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/onnx_pi_adapter.py`

主要内容：

- 新增 `rtc_config` / `rtc_processor` 持有逻辑
- 新增 `_rtc_enabled()`
- `predict_action_chunk(...)` 接受 RTC kwargs：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- 在 denoise loop 中用 `RTCProcessor.denoise_step(...)` 包住当前的 denoise closure

要求：

- 不改 engine 输入输出契约
- 不改 preflight 基础逻辑
- RTC disabled 时行为必须与当前一致

不做：

- 不把 `ActionQueue` 塞进 adapter
- 不修改 `select_action()` 成为 RTC-aware

验收：

- adapter 在 RTC off 时与当前行为一致
- adapter 在 RTC on 时能接受 kwargs 并完成一次 chunk 生成

### Commit C: TRT launcher 切到 chunk runtime

修改文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py`

主要内容：

- 不再使用 `control_utils.predict_action()`
- 改走：
  - observation preprocess
  - `policy.predict_action_chunk(...)`
  - chunk postprocess
  - queue consume
- 引入 `AsyncChunkPrefetcher`
- 引入 `ActionQueue`
- 支持 hold fallback

要求：

- TRT launcher 在 RTC disabled 时，仍能稳定跑通
- 保留现有 smoothing / clamp / finite check / robot send_action 逻辑

不做：

- 不修改真机安全策略
- 不改已有 TRT provenance 校验

验收：

- TRT launcher 在 dry-run / preflight-only / live loop 三种模式下都能工作
- 日志能打印 queue size、chunk latency、underrun 相关指标

### Commit D: ONNX launcher 共享同一 runtime helper

修改文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py`

主要内容：

- 将现有 async prefetch 逻辑迁移到共享 helper
- 与 TRT launcher 对齐 chunk result / queue / logging 结构
- 可选接入 `ActionQueue`

要求：

- 不破坏现有 ONNX async 体验
- 统一与 TRT 的 runtime 概念

验收：

- ONNX launcher 行为与当前一致或更清晰
- TRT/ONNX 共享大部分 runtime helper

### Commit E: CLI 与 RTCConfig 贯通

修改文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py`

必要时可补一处共享解析辅助：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/common.py`

主要内容：

- 新增 CLI：
  - `--rtc-enabled`
  - `--rtc-execution-horizon`
  - `--rtc-max-guidance-weight`
  - `--rtc-prefix-attention-schedule`
  - `--rtc-debug`
- 将 CLI 注入 `policy_cfg.rtc_config`

要求：

- 不写回 checkpoint
- 仅 runtime override

验收：

- summary 中能打印 resolved RTC config
- RTC off / on 都能正确运行

### Commit F: 验证与 benchmark 补强

修改文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/benchmark_pi_inference.py`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/benchmark_pi_select_action.py`

可选新增：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/benchmark_pi_control_loop.py`

主要内容：

- 补 runtime-oriented 指标输出
- 至少写出：
  - chunk latency
  - queue underrun
  - hold steps
  - real delay
  - queue depth

要求：

- 明确区分：
  - pure inference benchmark
  - control-loop benchmark

验收：

- 文档与报告不会把 RTC 误写成 engine latency 加速

## 3. 推荐实现顺序

推荐顺序固定如下：

1. `Commit A`
2. `Commit B`
3. `Commit C`
4. `Commit D`
5. `Commit E`
6. `Commit F`

原因：

- 先抽 helper，减少 TRT/ONNX 漂移
- 再接 adapter 语义
- 再改 launcher 主循环
- 最后才加 CLI 和 benchmark

## 4. 关键代码触点

### 4.1 `trt_pi_adapter.py`

预计修改：

- `__init__`
- `reset`
- `runtime_summary`
- `predict_action_chunk`

新增：

- `_rtc_enabled`
- `rtc_processor` 初始化

### 4.2 `onnx_pi_adapter.py`

预计修改：

- `__init__`
- `reset`
- `runtime_summary`
- `predict_action_chunk`

新增：

- `_rtc_enabled`
- `rtc_processor` 初始化

### 4.3 `run_pi05_trt_infer_so101.py`

预计修改较大：

- CLI parser
- `print_summary`
- inference loop
- 异步 prefetch / queue / fallback

### 4.4 `run_pi05_onnx_infer_so101.py`

预计中等修改：

- 提炼现有 async prefetch 为共享 helper
- 对齐 queue / metrics 语义

## 5. 风险控制点

### 风险 1：引入 RTC 后仍然走 `select_action()`

规避：

- launcher 必须显式改为 chunk runtime

### 风险 2：leftover 来源错误

规避：

- queue 中必须同时保留 original / processed 两份动作

### 风险 3：real_delay 口径混乱

规避：

- 优先用 action index 差值
- wall-clock 只做辅助日志

### 风险 4：TRT/ONNX 两套 runtime 漂移

规避：

- 尽量共享 `pi05_chunk_runtime.py`

### 风险 5：用 benchmark 证明错问题

规避：

- 区分 pure inference 与 control loop benchmark

## 6. 实施后的最小验收清单

### 静态检查

- `python -m py_compile` 通过：
  - `scripts/pi05_chunk_runtime.py`
  - `scripts/trt_pi_adapter.py`
  - `scripts/onnx_pi_adapter.py`
  - `scripts/run_pi05_trt_infer_so101.py`
  - `scripts/run_pi05_onnx_infer_so101.py`

### CLI 检查

- `python scripts/run_pi05_trt_infer_so101.py --help`
- `python scripts/run_pi05_onnx_infer_so101.py --help`

### 纯推理检查

- TRT adapter RTC off 能跑
- TRT adapter RTC on 能跑
- ONNX adapter RTC off 能跑
- ONNX adapter RTC on 能跑

### runtime 检查

- async prefetch 能提前提交
- queue merge 不报错
- queue empty 时 fallback 正常

### 指标检查

- 日志中能看到：
  - queue size
  - chunk latency
  - underrun count
  - hold count
  - real delay

## 7. 交付结果

本 plan 对应的最终代码交付应该至少包含：

- 一个 shared chunk runtime helper
- TRT/ONNX adapter 的 RTC 接口
- TRT launcher 的 RTC-aware async runtime
- ONNX launcher 的共享化重构
- 一套能证明 RTC 真实 runtime 收益的指标输出

## 8. 最终判断

这套实现最值得优先推进的不是“让模型更快算”，而是：

- 让 `PI05 TRT` 从“同步 chunk 刷新 + deque”进化成
- “RTC-aware chunk runtime + ActionQueue + async prefetch”

如果这条线做成，哪怕 raw engine latency 没变，真实上机体验也有机会明显改善。
