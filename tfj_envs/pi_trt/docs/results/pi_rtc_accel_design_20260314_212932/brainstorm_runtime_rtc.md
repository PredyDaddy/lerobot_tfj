# RTC 运行时调度脑暴报告

## 1. 我理解的 RTC 加速点

这里的 `RTC` 不是让单次模型前向本身变快，也不是让 TensorRT kernel 更快。

它真正能带来的加速是运行时层面的：

- 通过 `prev_chunk_left_over` 把上一段 chunk 的未执行前缀拿来做当前 chunk 的引导，降低 chunk 之间的割裂感。
- 通过显式建模 `inference_delay`，允许系统在“上一段动作还没吃完”的时候就启动下一段推理。
- 通过 queue 替换而不是简单追加，让新的 chunk 在考虑时延后覆盖旧 queue，从而减少 stale action 的持续时间。
- 通过 prefetch + delay hiding，把原本阻塞 control loop 的整段 chunk 计算尽量挪到后台。

所以 RTC 策略的核心收益不是：

- `predict_action_chunk()` 单次耗时下降

而是：

- 更少的 queue underrun
- 更低的 hold-step 次数
- 更平滑的 chunk 过渡
- 在同样 chunk latency 下，真实控制循环更不容易饿死

## 2. 当前 TRT runtime 缺什么

当前仓库里，PyTorch 的 PI05 运行时已经有 RTC 逻辑，但 TRT runtime 没接上。

### 2.1 PyTorch 已经具备的 RTC 关键点

在 `src/lerobot/policies/pi05/modeling_pi05.py` 里：

- `PI05Pytorch.sample_actions()` 会在 denoise loop 中根据 `self._rtc_enabled()` 决定是否走 `self.rtc_processor.denoise_step(...)`
- RTC 包装的是单步 denoiser closure，而不是整个大图
- 输入包括：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`

这说明 RTC 本来就是“运行时 denoise 包装层”，不是导出边界的一部分。

### 2.2 仓库里已经有现成的 queue 语义

在 `src/lerobot/policies/rtc/action_queue.py` 里已经有：

- `get()`
- `get_left_over()`
- `merge(original_actions, processed_actions, real_delay, action_index_before_inference)`

而且 queue 语义已经明确区分：

- RTC enabled: replace queue
- RTC disabled: append queue

这意味着仓库里不缺“RTC queue 模型”，缺的是把它接到 TRT runtime。

### 2.3 TRT runtime 当前还是最原始的 chunk queue

`scripts/trt_pi_adapter.py` 当前只有：

- `_action_queue = deque(...)`
- `select_action()` 里队列空了就同步 `predict_action_chunk()`

它没有：

- `RTCProcessor`
- `ActionQueue`
- `prev_chunk_left_over`
- `inference_delay`
- 异步 chunk prefetch
- queue merge / replace 语义

这意味着现在 TRT 路径本质上还是：

- queue 见底
- 同步重算整个 chunk
- 期间 control loop 被大 chunk latency 卡住

### 2.4 ONNX runtime 反而已经有可复用的异步壳

`scripts/run_pi05_onnx_infer_so101.py` 已经有：

- `AsyncChunkPrefetcher`
- `estimate_prefetch_threshold(...)`
- queue low-watermark 触发的后台 chunk 生成
- queue 见底时的同步回填和 hold action 策略

但它也还没有真正接 RTC 的 queue merge 语义，仍然只是“预取 chunk + append actions”。

## 3. 最值得做的设计方案

我认为最值得做的是一条明确的“两层式 RTC runtime 加速”方案：

### 3.1 第一层：在 adapter 内把 RTC 接回 denoise loop

目标：

- 不改 TRT engine
- 不改 Stage 2-5 的导出边界
- 直接复用 PyTorch PI05 的 RTC 数学逻辑

做法：

- 在 `TrtPi05PolicyAdapter` 中实例化 `RTCProcessor`
- 在 `predict_action_chunk()` 里像 PyTorch `sample_actions()` 一样，定义单步 `denoise_step_partial(x_t)` closure
- 如果 `rtc_config.enabled`，就让每一步 `v_t` 来自：
  - `rtc_processor.denoise_step(...)`
- 否则保持当前直接调用 TRT `denoise_runner`

这一步的价值是：

- RTC 的数学语义与 PyTorch 保持一致
- 不碰 engine graph
- 可直接在 TRT / ONNX runtime 共用

### 3.2 第二层：在 launcher 层加 RTC-aware async chunk runtime

目标：

- 真正把“推理时延隐藏”做出来
- 让 queue 刷新从“同步 refill”变成“后台推理 + RTC merge”

做法：

- 不再让 TRT launcher 走 `control_utils.predict_action()` 这条 `select_action()` 直连路径
- 仿照 ONNX launcher，使用显式的 chunk runtime loop
- 引入 `ActionQueue`
- 在后台 future 启动 chunk inference 时，记录：
  - `action_index_before_inference`
  - 提交时 observation snapshot
- future 完成后：
  - 获得 `original_actions`
  - 对其做 postprocess 得到 `processed_actions`
  - 根据当前 queue 消耗量求 `real_delay`
  - 调用 `ActionQueue.merge(...)`

### 3.3 为什么优先做 TRT launcher 而不是先做 ONNX

因为你当前真实目标是 PI05 上机加速，而 TRT 是实际部署后端。

但我建议实现上把公共逻辑抽出来，这样：

- TRT 先吃到收益
- ONNX 可以作为 debug/reference runtime 跟着共享逻辑
- PyTorch / ONNX / TRT 三条 runtime 的控制逻辑更一致，benchmark 更可比

## 4. 推荐的模块划分

最合理的拆法不是把所有 RTC 代码硬塞进 `trt_pi_adapter.py`，而是拆成两层：

### 4.1 adapter 层

职责：

- 子图推理
- prefix cache 构建
- denoise loop
- RTC-guided denoise

建议保留在：

- `scripts/trt_pi_adapter.py`
- `scripts/onnx_pi_adapter.py`

### 4.2 runtime orchestration 层

职责：

- async prefetch
- queue low-watermark 判断
- observation snapshot
- `prev_chunk_left_over`
- `real_delay`
- hold action
- queue merge / replace

建议抽新文件，例如：

- `scripts/pi05_chunk_runtime.py`

这样：

- `run_pi05_trt_infer_so101.py`
- `run_pi05_onnx_infer_so101.py`

都能复用同一套 runtime orchestration。

## 5. 推荐的伪代码

### 5.1 adapter 内的 RTC denoise loop

```python
def predict_action_chunk(self, batch, **kwargs):
    runtime_inputs = self._extract_runtime_inputs(batch)
    prefix_outputs = self._run_prefix(runtime_inputs)

    x_t = self._init_noise(...)
    timestep_values = ...

    for timestep_value in timestep_values:
        current_timestep = timestep_value.expand(batch_size)

        def denoise_step_partial(input_x_t, current_timestep=current_timestep):
            denoise_feed = {
                "x_t": input_x_t,
                "timestep": current_timestep,
                "prefix_pad_masks": prefix_outputs["prefix_pad_masks"],
                **cache_tensors,
            }
            return self.denoise_runner.infer(denoise_feed)["v_t"].to(torch.float32)

        if self._rtc_enabled():
            v_t = self.rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                inference_delay=kwargs.get("inference_delay"),
                time=timestep_value,
                original_denoise_step_partial=denoise_step_partial,
                execution_horizon=kwargs.get("execution_horizon"),
            )
        else:
            v_t = denoise_step_partial(x_t)

        x_t = x_t + dt * v_t

    return x_t[:, :, :self.original_action_dim]
```

### 5.2 launcher 内的 RTC async chunk runtime

```python
queue = ActionQueue(rtc_cfg)
prefetcher = AsyncChunkPrefetcher(...)

while control_loop_running:
    obs = robot.get_observation()
    obs_frame = build_dataset_frame(...)

    completed = prefetcher.maybe_collect()
    if completed is not None:
        real_delay = queue.get_action_index() - completed.action_index_before_inference
        queue.merge(
            original_actions=completed.original_actions,
            processed_actions=completed.processed_actions,
            real_delay=real_delay,
            action_index_before_inference=completed.action_index_before_inference,
        )

    if queue.qsize() <= current_threshold:
        prefetcher.maybe_submit(
            observation_frame=obs_frame,
            prev_chunk_left_over=queue.get_left_over(),
            inference_delay=queue.qsize(),
            execution_horizon=rtc_cfg.execution_horizon,
            action_index_before_inference=queue.get_action_index(),
        )

    action = queue.get()
    if action is None:
        action = hold_action(...)

    robot.send_action(action)
```

## 6. 我建议的第一优先级实现

如果只能做一轮，我建议按这个顺序：

1. 在 `trt_pi_adapter.py` 中接入 `RTCProcessor`
2. 在 `run_pi05_trt_infer_so101.py` 中废掉当前同步 `predict_action()` 路径，改成显式 chunk runtime loop
3. 在 launcher 中接入 `ActionQueue`
4. 再把 ONNX launcher 提炼成共享 helper

不要一开始就：

- 改 engine
- 改 ONNX export
- 改 verification gate

## 7. 风险

### 7.1 最大风险不是数值，而是控制逻辑复杂化

RTC 方案引入后，最容易出问题的是：

- action queue 替换时机
- `real_delay` 计算错误
- leftover 使用错 observation 对齐关系

这些问题不会像数值 diff 一样显眼，但会直接搞坏真实控制循环。

### 7.2 当前 `select_action()` 体系不能直接承载 RTC

PI05 PyTorch 本身就写死了：

- `RTC is not supported for select_action, use it with predict_action_chunk`

因此 TRT runtime 若继续沿用 `control_utils.predict_action()`，就等于从入口设计上把 RTC 封死。

### 7.3 不能把“RTC 加速”误写成“模型更快”

RTC 最终能优化的是：

- 控制循环的 wall-clock 有效吞吐
- queue 不饿死
- chunk 刷新体验

不是：

- 单次 `pipeline_chunk` benchmark 必然下降

如果 benchmark 口径写错，后续很容易误判。

## 8. 最终建议

我的主张是：

- 不改 TRT engine 边界
- 把 RTC 重新接回 adapter 的 denoise loop
- 把 TRT launcher 从“同步 select_action 驱动”改成“RTC-aware async chunk runtime”
- 复用现有 `RTCProcessor` 和 `ActionQueue`
- 再用 ONNX runtime 作为共享 orchestration 的参考实现

这条路线最贴近当前仓库已有代码，也最像真正能把 PI05 真机控制链路加速起来的方案。
