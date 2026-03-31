# PI0.5 RTC 加速技术方案

## 1. 目标与边界

本方案的目标不是继续优化 TensorRT engine 内核，而是利用 RTC 的运行时策略，改善 PI0.5 在真实控制循环中的有效吞吐和时延隐藏能力。

一句话定义：

- 让 PI0.5 的 chunk 生成从“同步 refill”升级成“RTC-aware async refill”

本方案要优化的是：

- queue 不被吃空
- chunk 切换更平滑
- inference latency 被后台预取尽量掩盖
- 在不改 engine 边界的前提下，提高真实控制循环稳定性

本方案不做：

- 不重导出 ONNX
- 不重建 TRT engine
- 不修改 Stage 2-5 验证边界
- 不把 RTC 下沉进 TensorRT engine
- 不把 RTC 伪装成单次 kernel 加速

## 2. 当前现状

### 2.1 PI05 PyTorch 已经支持 RTC

在 [modeling_pi05.py](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py) 中：

- `sample_actions()` 在 denoise loop 中支持 `RTCProcessor`
- RTC 使用：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- `select_action()` 明确不支持 RTC

这说明：

- RTC 的正确接入点是 `predict_action_chunk/sample_actions`
- 不是 `select_action`

### 2.2 现有 TRT runtime 完全没接 RTC

在 [trt_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py) 中：

- `predict_action_chunk()` 只是固定：
  - vision
  - prefix
  - denoise loop
- `select_action()` 仍然是普通 deque queue
- 没有 `RTCProcessor`
- 没有 `ActionQueue`
- 没有 `prev_chunk_left_over`
- 没有 `inference_delay`

### 2.3 ONNX launcher 已经有 async prefetch 雏形

在 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py) 中，已经存在：

- `AsyncChunkPrefetcher`
- `estimate_prefetch_threshold(...)`
- low-watermark 提前提交后台 chunk
- hold action / sync refill 逻辑

但它还不是 RTC-aware queue，只是“异步预取 + append”。

### 2.4 当前 TRT launcher 仍是同步 select_action 路径

在 [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py) 中：

- 主循环仍调用 `control_utils.predict_action()`
- 内部最终还是 `policy.select_action()`

这条路径天然不适合 RTC。

## 3. 设计原则

本方案遵循 4 条原则：

1. RTC 只放在 Python runtime，不放进 engine
2. TRT / ONNX runtime 尽量共享同一套 orchestration
3. adapter 负责数值与 denoise loop，launcher 负责 queue 与调度
4. 验证目标是控制循环 wall-clock，不是单纯 kernel benchmark

## 4. 总体架构

方案拆成两层。

### 4.1 Adapter 层：RTC-aware denoise

在 `TrtPi05PolicyAdapter.predict_action_chunk()` 中：

- 保持现有 vision / prefix / denoise engine 边界不变
- 对每个 denoise timestep 定义 `denoise_step_partial(x_t)` closure
- 如果 RTC 开启，就让 `RTCProcessor.denoise_step(...)` 包装这个 closure
- 否则保持当前直接调用 denoise engine 的路径

这一步让 TRT runtime 的 denoise 数学路径与 PyTorch PI05 的 RTC 语义对齐。

### 4.2 Runtime 层：RTC-aware async chunk orchestration

在 launcher / 共享 helper 中：

- 使用 `ActionQueue` 代替普通 deque
- 使用 `AsyncChunkPrefetcher` 在 queue 低水位时后台启动下一段 chunk
- 后台任务提交时记录：
  - `action_index_before_inference`
  - `prev_chunk_left_over`
  - `inference_delay`
- 后台任务完成后使用：
  - `ActionQueue.merge(original_actions, processed_actions, real_delay, action_index_before_inference)`

这一步让系统真正具备：

- delay hiding
- queue replace
- leftover-aware chunk 过渡

## 5. 建议的代码结构

### 5.1 修改 [trt_pi_adapter.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py)

新增能力：

- `RTCProcessor` 成员
- `_rtc_enabled()`
- `predict_action_chunk(..., prev_chunk_left_over=None, inference_delay=None, execution_horizon=None)`
- 可选的 `predict_action_chunk_with_metadata(...)`

不改：

- engine 读法
- engine contract
- Stage 5 safety 逻辑

### 5.2 新增共享 helper，例如：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py`

建议包含：

- `ChunkPredictionResult`
- `prepare_policy_observation(...)`
- `postprocess_action_chunk(...)`
- `predict_processed_action_chunk(...)`
- `AsyncChunkPrefetcher`
- `estimate_prefetch_threshold(...)`
- `RtcChunkRuntimeState`
- `RtcChunkRuntimeLoop`

### 5.3 修改 [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py)

分成两条路径：

- 非 RTC：保留现有同步 `predict_action()` 路径
- RTC：使用显式 chunk runtime loop

### 5.4 修改 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py)

目标不是先改功能，而是迁移到共享 helper，避免 TRT / ONNX runtime orchestration 分叉。

## 6. 关键运行时数据流

### 6.1 首次启动

1. 读取 observation
2. 同步生成首个 chunk
3. 将：
   - `original_actions`
   - `processed_actions`
   写入 `ActionQueue`
4. 开始消费 queue

### 6.2 正常运行

1. 每个 control step 从 `ActionQueue.get()` 拿一个 action
2. 当 `qsize() <= threshold` 时，后台提交新的 chunk 推理
3. 提交时把：
   - `prev_chunk_left_over = queue.get_left_over()`
   - `inference_delay = queue.qsize()`
   - `action_index_before_inference = queue.get_action_index()`
   一起传给后台任务
4. 后台任务完成后，按实际消耗的 action 数计算 `real_delay`
5. 调用 `queue.merge(...)`

### 6.3 queue 饿死时

策略保持保守：

- 优先 hold action
- 必要时 sync refill

RTC 的目标是减少这种情况，而不是允许系统在 queue 空了之后乱补。

## 7. 为什么这条路比继续碰 engine 更合适

### 7.1 它直接复用现有 PI05 RTC 语义

当前 PyTorch PI05 已经把 RTC 放在：

- denoise loop 外层

TRT adapter 的现状刚好也是：

- denoise loop 在 Python

所以这是天然可复用的边界。

### 7.2 它对当前 TRT 工件零侵入

现在你已经有可部署的 TRT FP32 工件。

RTC 方案如果不碰 engine，就能：

- 降低工程风险
- 不打断现有 Stage 2-5 体系
- 先把 runtime scheduling 做起来

### 7.3 它更接近真实“上机加速”

你现在要的不是论文意义上的 kernel 最优，而是：

- 真机 loop 更稳
- queue 不空
- 动作连续

RTC runtime 正好打的是这块。

## 8. 关键风险

### 8.1 最大风险：把 RTC 误写成模型推理加速

要明确：

- RTC 优化的是 runtime wall-clock 结构
- 不是单次 `pipeline_chunk` 必然变快

### 8.2 最大工程风险：queue / delay 错位

如果这几项算错：

- `prev_chunk_left_over`
- `real_delay`
- `action_index_before_inference`

系统会在不显眼的地方引入控制错位。

### 8.3 共享 helper 抽象不稳

如果 TRT 和 ONNX 各写一套，很快就会漂。

所以共享 helper 虽然多一步抽象，但长期是必须的。

## 9. 成功判据

本方案的成功判据必须是 runtime / control 口径，而不是只看模型 benchmark。

必须至少满足：

- queue underrun 明显减少
- hold step 次数下降
- over-budget control steps 比例下降
- smoothing / delta clamp 不明显恶化
- 动作边界跳变可控

可以接受但不作为唯一指标：

- chunk latency 轻微上升
- 单次 `predict_action_chunk()` 平均耗时不变

## 10. 实施优先级

推荐顺序：

1. TRT adapter 接 RTC denoise
2. TRT launcher 接 RTC-aware async runtime
3. 抽共享 runtime helper
4. ONNX launcher 迁移到共享 helper
5. 补 RTC 专用 benchmark / report

## 11. 最终建议

当前最务实的 RTC 加速路线不是继续折腾 engine，而是：

- 用已有的 `RTCProcessor`
- 用已有的 `ActionQueue`
- 在 TRT adapter 把 RTC 接回 denoise loop
- 在 TRT launcher 引入 async chunk runtime

这条路线对现有部署工件最温和，也最像能在真实控制循环里拿到收益的方案。
