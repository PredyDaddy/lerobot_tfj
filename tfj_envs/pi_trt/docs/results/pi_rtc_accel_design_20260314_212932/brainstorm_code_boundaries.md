# RTC 代码边界脑暴报告

## 1. 最小可改代码面

如果目标是“用 RTC 策略给 PI05 TRT 加速”，我认为最小可改代码面只有 4 块：

1. `scripts/trt_pi_adapter.py`
2. `scripts/run_pi05_trt_infer_so101.py`
3. `scripts/run_pi05_onnx_infer_so101.py`
4. 一个新的共享 runtime helper 文件

最推荐的新文件名是：

- `scripts/pi05_chunk_runtime.py`

原因是当前代码里已经存在两套相邻但不一致的 runtime：

- TRT：同步 `select_action()` 驱动
- ONNX：显式 async prefetch 驱动

如果不抽 helper，后面 RTC 逻辑会在两个 launcher 里复制两遍。

## 2. 哪些逻辑可以直接复用

### 2.1 可以直接复用的 RTC 数学逻辑

可直接复用：

- `src/lerobot/policies/rtc/modeling_rtc.py`
  - `RTCProcessor`
  - `denoise_step(...)`
  - `get_prefix_weights(...)`

这块不应该重写。

原因：

- 它已经把 RTC 的核心数学和 prefix guidance 封装好了
- 它的输入形式正好适合 adapter 里的单步 `denoise_step` closure

### 2.2 可以直接复用的 queue 语义

可直接复用：

- `src/lerobot/policies/rtc/action_queue.py`

它已经定义了：

- `get()`
- `qsize()`
- `get_left_over()`
- `merge(...)`
- RTC replace / non-RTC append 两种模式

这块最值得直接拿来用，不要再造一个新的 deque 语义。

### 2.3 可以直接复用的 ONNX launcher 思路

可直接复用的设计，而不是原样复制代码：

- `estimate_prefetch_threshold(...)`
- `AsyncChunkPrefetcher`
- `predict_processed_action_chunk(...)`
- hold action / sync refill / queue low-watermark 逻辑

这些在 `scripts/run_pi05_onnx_infer_so101.py` 里已经有不错雏形。

## 3. 哪些地方必须新增 adapter/runtime state

### 3.1 `TrtPi05PolicyAdapter` 需要新增

建议新增：

- `self.rtc_processor`
- `self.rtc_enabled`
- `self._rtc_enabled()` helper

并且让 `predict_action_chunk()` 支持：

- `prev_chunk_left_over`
- `inference_delay`
- `execution_horizon`

这些 kwargs 目前虽然已有 `**kwargs` 入口，但完全没消费。

### 3.2 需要一个独立的 chunk prediction result

现在 ONNX prefetch 只保存：

- processed actions
- preprocess/infer/postprocess 时间

RTC 要求额外保存：

- `original_actions`
- `processed_actions`
- `action_index_before_inference`
- `submitted_at_s`
- `completed_at_s`
- 可能还要保存 `observation_frame_id` 或 step id

否则 `ActionQueue.merge(...)` 没法正确工作。

### 3.3 launcher 需要状态机，而不是纯 while 循环

当前 TRT launcher 基本是：

- 读 observation
- `predict_action()`
- 发 action

而 RTC runtime 至少要维护：

- `ActionQueue`
- pending future
- `action_index_before_inference`
- `chunk_latency_ema`
- `step_time_ema`
- `queue_underrun_count`
- `hold_step_count`

这些不适合继续塞在当前“单层 while 循环 + generic predict_action”里。

## 4. 配置和 CLI 怎么贯通

### 4.1 配置层其实已经够了

`PI05Config` 已经有：

- `rtc_config: RTCConfig | None`

`RTCConfig` 已经有：

- `enabled`
- `prefix_attention_schedule`
- `max_guidance_weight`
- `execution_horizon`
- `debug`

所以不需要再改模型配置 dataclass。

### 4.2 真正缺的是 launcher CLI

当前 `run_pi05_trt_infer_so101.py` 和 `run_pi05_onnx_infer_so101.py` 没有把 RTC 暴露成运行时参数。

建议新增：

- `--rtc-enabled`
- `--rtc-execution-horizon`
- `--rtc-prefix-attention-schedule`
- `--rtc-max-guidance-weight`
- `--rtc-debug`

另外保留/统一：

- `--prefetch-threshold`
- `--sync-refill-timeout-s`

### 4.3 需要一个统一的 runtime override helper

建议新增一个 helper，例如：

- `apply_pi_rtc_runtime_overrides(args, policy_cfg)`

职责：

- 根据 CLI 创建/更新 `policy_cfg.rtc_config`
- 做参数合法性检查
- 在 launcher 启动摘要中打印最终 RTC 配置

## 5. 不建议做的事

### 5.1 不建议把 RTC 下沉进 TensorRT engine

原因：

- RTC 需要 `prev_chunk_left_over`
- 需要 queue state
- 需要 inference delay
- 需要 runtime merge

这本质上是运行时控制逻辑，不是稳定的 engine 边界。

### 5.2 不建议改 `control_utils.predict_action()`

`control_utils.predict_action()` 的设计天然是：

- preprocessor
- `policy.select_action()`
- postprocessor

但 PI05 本身已经说明：

- RTC 不支持 `select_action()`

所以如果强行改这层，只会把 generic utility 搞脏，还会让其他 policy 跟着背锅。

更好的方式是：

- TRT launcher 为 RTC 模式走显式 chunk runtime
- 非 RTC 模式继续保留现有 `predict_action()` 快路径

### 5.3 不建议第一版就碰 PyTorch policy 主干

PI05 PyTorch 主干已经定义了 RTC 的数学语义，当前更缺的是 runtime integration。

所以第一版不要去改：

- `src/lerobot/policies/pi05/modeling_pi05.py`

除非确实发现 adapter 复用时缺一个小 helper 才回头抽函数。

## 6. 我建议的代码结构

### 6.1 `scripts/trt_pi_adapter.py`

新增：

- `RTCProcessor` 引入
- `_rtc_enabled()`
- RTC-aware `predict_action_chunk(...)`
- 可选的 `predict_action_chunk_with_metadata(...)`

### 6.2 `scripts/pi05_chunk_runtime.py`

新增共享 helper：

- `ChunkPredictionResult`
- `AsyncChunkPrefetcher`
- `estimate_prefetch_threshold(...)`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `RtcChunkRuntimeLoop` 或同等 helper

### 6.3 `scripts/run_pi05_trt_infer_so101.py`

改造为：

- 非 RTC：保留现有同步路径
- RTC：走共享 `RtcChunkRuntimeLoop`

### 6.4 `scripts/run_pi05_onnx_infer_so101.py`

改造成共享 helper 的消费者，避免两套 async runtime 长期漂移。

## 7. 最务实的收敛路线

我建议分三步走：

1. `trt_pi_adapter.py` 接 RTC 数学逻辑
2. `run_pi05_trt_infer_so101.py` 接 `ActionQueue + async prefetch`
3. 抽 `pi05_chunk_runtime.py`，再把 ONNX launcher 一起迁移

这样可以先把 TRT 的真实部署收益做出来，再收拾共享抽象。

## 8. 最终建议

RTC 方案的关键不是“改很多代码”，而是“把改动压在 runtime orchestration 层”。

最小可行边界就是：

- engine 不动
- adapter 接 RTC denoise
- launcher 接 async prefetch + ActionQueue
- 配置和 CLI 打通

只要这 4 件事成立，这条路就是能落地的。
