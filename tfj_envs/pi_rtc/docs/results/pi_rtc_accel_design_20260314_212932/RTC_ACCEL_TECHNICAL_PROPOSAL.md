# PI05 基于 RTC 的运行时加速技术方案

## 1. 目标

在 **不修改 TensorRT engine 边界** 的前提下，为 `PI05` 的 `TRT/ONNX runtime` 接入 `RTC` 策略，以降低真实控制回路里的 chunk 刷新阻塞、queue drain 和 hold fallback 频率。

本方案的核心目标不是让单个 engine 更快，而是让整条实时执行链路更稳。

## 2. 非目标

本方案明确不做下面这些事：

- 不重新设计 `vision_encoder/prefix_cache/denoise_step` 的 engine 边界
- 不把 `RTC` 下沉进 ONNX 或 TensorRT engine
- 不修改 `Stage 2 -> Stage 5` 导出/构建/验证链
- 不把 `control_utils.predict_action()` 改造成 RTC-aware 通用入口
- 不在这一轮里宣称“RTC 一定降低单次 chunk 推理时间”

## 3. 现状问题

### 3.1 PyTorch 原生 PI05 已经有 RTC 语义

在 [modeling_pi05.py](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py) 中：

- `PI05Config` 已经有 `rtc_config`
- `PI05Pytorch.sample_actions()` 已经支持：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- `RTCProcessor.denoise_step(...)` 已经实现 prefix guidance

同时，`PI05Policy.select_action()` 明确写了：

- `RTC is not supported for select_action, use it with predict_action_chunk`

这说明：

- RTC 的正确接入点是 `predict_action_chunk/sample_actions`
- 不是 `select_action`

### 3.2 TRT/ONNX adapter 目前没有 RTC

当前 [trt_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py) 与 [onnx_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/onnx_pi_adapter.py) 都是：

- 同步生成整段 chunk
- 用简单 `deque` 做 action queue
- 没有 `RTCProcessor`
- 没有 `ActionQueue`
- 没有 leftover merge

### 3.3 TRT launcher 入口也不适合 RTC

当前 [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py) 仍走 `predict_action()`，而 `predict_action()` 内部会调用 `policy.select_action()`。

这条入口天然拿不到：

- `prev_chunk_left_over`
- `inference_delay`
- `execution_horizon`

所以它无法承载 RTC。

### 3.4 ONNX launcher 已经有半套对的 runtime

当前 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py) 已经有：

- `AsyncChunkPrefetcher`
- `prefetch_threshold`
- queue drain fallback
- sync refill timeout

这说明当前最有价值的演进方向不是碰 engine，而是把 ONNX 的 async runtime 抽象出来，并让 TRT runtime 也走同一套调度骨架。

## 4. 方案核心判断

### 4.1 RTC 是运行时加速，不是 engine 加速

本方案把 “RTC 加速” 明确定义为：

- 降低实时 loop 对 chunk latency 的敏感度
- 通过 leftover guidance 和 queue merge 隐藏部分推理延迟
- 降低 queue drain / hold step / stall 概率

本方案不把 “RTC 加速” 定义为：

- 降低 `vision_encoder/prefix_cache/denoise_step` 的单次执行时间

### 4.2 不改 engine 边界，优先改 3 层

本方案只动三层：

1. adapter 层
2. shared runtime helper 层
3. launcher 层

## 5. 最终架构

### 5.1 adapter 层：在 denoise loop 里接 RTCProcessor

文件：

- [trt_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py)
- [onnx_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/onnx_pi_adapter.py)

改造原则：

- 不改 engine I/O
- 不改 prefix cache tensor schema
- 不改 denoise engine 输入契约

改造方式：

- 在 adapter 内创建 `rtc_processor`
- 新增 `_rtc_enabled()`
- 保持现有 `predict_action_chunk()` 结构不变，只在 denoise loop 中插入 RTC wrapper

逻辑从：

```python
for timestep in timestep_values:
    v_t = denoise_runner.infer(...)
    x_t = x_t + dt * v_t
```

变成：

```python
for timestep in timestep_values:
    def denoise_partial(input_x_t):
        return denoise_runner.infer(..., x_t=input_x_t)["v_t"]

    if rtc_enabled:
        v_t = rtc_processor.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            inference_delay=kwargs.get("inference_delay"),
            time=timestep_scalar,
            original_denoise_step_partial=denoise_partial,
            execution_horizon=kwargs.get("execution_horizon"),
        )
    else:
        v_t = denoise_partial(x_t)

    x_t = x_t + dt * v_t
```

这样做的价值：

- 完整复用已有 `RTCProcessor`
- 不需要重新导出 engine
- 语义上对齐 PyTorch `sample_actions()`

### 5.2 runtime helper 层：抽共享 chunk runtime

建议新增文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py`

职责：

- `ChunkPredictionResult`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `estimate_prefetch_threshold(...)`
- `AsyncChunkPrefetcher`

新增数据结构建议：

```python
@dataclass
class ChunkPredictionResult:
    original_actions: torch.Tensor
    processed_actions: list[Any]
    preprocess_time_s: float
    inference_time_s: float
    postprocess_time_s: float
    action_index_before_inference: int | None
    submit_time_s: float | None
    ready_time_s: float | None
```

这里最关键的是同时保留：

- `original_actions`
- `processed_actions`

因为 RTC merge 需要前者，而机器人执行需要后者。

### 5.3 launcher 层：引入 ActionQueue + async prefetch

文件：

- [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py)
- [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py)

关键变化：

1. 不再走 `control_utils.predict_action() -> policy.select_action()`
2. 改成显式：
   - preprocess observation
   - `policy.predict_action_chunk(...)`
   - postprocess chunk
   - `ActionQueue.merge(...)`

3. 当 queue 长度低于阈值时异步提交下一次 chunk
4. 提交时把：
   - `prev_chunk_left_over`
   - `inference_delay`
   - `execution_horizon`
   一起传进 `predict_action_chunk(...)`

5. 当 future 完成时，根据 `real_delay` merge queue
6. 当 queue 已空但 future 未完成时，执行 hold fallback

### 5.4 queue 层：直接复用 RTC ActionQueue

不建议在 `pi_trt` 下重造 queue。

直接复用：

- [action_queue.py](/data/tfj/lerobot_tfj/src/lerobot/policies/rtc/action_queue.py)

原因：

- 它已经区分 `original_queue` 和 `processed_queue`
- 已经支持 RTC enabled / disabled 两种模式
- 已经有 `get_left_over()` 和 `merge(...)`

## 6. 配置与 CLI 设计

### 6.1 配置注入方式

优先使用 runtime override，而不是改 checkpoint 配置文件。

流程：

1. 读取 `PI05Config`
2. 解析 CLI
3. 组装 `RTCConfig`
4. 注入 `policy_cfg.rtc_config`
5. 用这个 config 创建 adapter

### 6.2 建议新增 CLI

对 TRT 和 ONNX launcher 都加：

- `--rtc-enabled`
- `--rtc-execution-horizon`
- `--rtc-max-guidance-weight`
- `--rtc-prefix-attention-schedule`
- `--rtc-debug`

建议保留现有 async 参数，并统一语义：

- `--prefetch-threshold`
- `--sync-refill-timeout-s`

如果要更稳，可补：

- `--rtc-delay-source {queue_index, wall_clock}`

### 6.3 默认运行模式

第一阶段默认仍保持：

- `RTC off`
- 原有安全策略不变
- 只有在显式传入 RTC CLI 参数时才进入 RTC-aware runtime

这样做的目的不是保守，而是保留一条可直接对照的基线：

- 同一套 engine
- 同一套 launcher
- 只切换 runtime orchestration / RTC 策略

否则后续 benchmark 很难判断收益到底来自：

- async prefetch
- RTC leftover guidance
- 还是别的伴随改动

## 7. 关键实现细节

### 7.1 `select_action()` 不作为 RTC 入口

保留 adapter 内 `select_action()`，但 RTC 模式下 launcher 不再使用它。

原因：

- `select_action()` 的调用契约太窄
- 无法传递 RTC kwargs

### 7.2 `real_delay` 的来源

建议优先使用 queue index 差值作为主来源，wall-clock 推导作为辅助指标。

原因：

- queue index 更贴近“控制回路实际消耗了多少 action”
- wall-clock 更容易受 sleep、camera jitter 影响

### 7.3 smoothing / clamp 继续留在 launcher

不把：

- smoothing
- delta clamp
- finite check

挪到 queue merge 或 adapter。

原因：

- 这是机器人安全逻辑，不是模型语义

### 7.4 leftover 必须来自原始动作

RTC prefix guidance 使用的 `prev_chunk_left_over` 必须来自：

- original model output chunk

不能来自：

- postprocess 后动作
- smoothing 后动作
- clamp 后动作

## 8. 风险控制

### 8.1 最大风险

最大风险不是“实现太复杂”，而是“把 RTC 的收益写错”。

必须明确：

- 这套方案优化的是 control-loop 稳定性和 latency hiding
- 不是承诺 raw chunk inference 变快

### 8.2 验收指标

必须记录：

- `queue_underrun_count`
- `hold_step_count`
- `real_delay`
- `chunk_latency_s`
- `step_time_ema_s`
- `chunk boundary jump`
- `smoothing_event_count`
- `delta_clip_event_count`

### 8.3 失败 fallback

若 async chunk 尚未返回且 queue 已空：

- 默认 hold 当前位姿
- 不自动并发启动第二个 chunk

否则会让 GPU runtime 更混乱，问题更难定位。

## 9. 推荐落地顺序

### 阶段 A

抽 shared chunk runtime helper，但不打开 RTC。

目标：

- 让 TRT 和 ONNX 共享 async chunk runtime
- 先把 `TRT sync select_action loop` 替换掉

### 阶段 B

adapter 内接 RTCProcessor。

目标：

- 在 `predict_action_chunk()` 里支持 RTC kwargs

### 阶段 C

launcher 接 `ActionQueue` + `prev_chunk_left_over` + `real_delay`。

目标：

- 真正让 RTC 参与 chunk merge

### 阶段 D

补 benchmark / logging / safety summary。

目标：

- 证明这不是“纸面上的加速”

## 10. 最终结论

对于“用 RTC 策略给 PI05 加速”，我给出的最终技术判断是：

- 最应该动的是 `runtime scheduling`
- 最不应该动的是 `engine boundary`
- 最关键的代码复用点是：
  - `RTCProcessor`
  - `ActionQueue`
  - `PI05 sample_actions()` 的 RTC 语义
- 最关键的工程动作是：
  - TRT launcher 放弃 `select_action()` 入口
  - TRT/ONNX 共用 async chunk runtime
  - 在 adapter 的 denoise loop 内挂 RTC wrapper

这条路线的价值在于：

- 它不要求重新导出 engine
- 它能最大化复用现有 PyTorch RTC 语义
- 它直接作用于你最关心的真实上机 wall-clock 行为
