# PI05 基于 RTC 的 Runtime Scheduling 加速脑暴报告

## 0. 阅读范围与当前链路理解

本报告只基于本地仓库以下实现做分析，没有用 skills，也没有上网：

- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/rtc/modeling_rtc.py`
- `src/lerobot/policies/rtc/action_queue.py`
- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`
- `scripts/trt_pi_adapter.py`

我对现状的抽象如下：

1. PI05 原生 PyTorch 路径已经把 RTC 放在“chunk 级 runtime 层”而不是模型结构层。
   - `PI05.sample_actions()` 在每个 denoise step 里，如果启用 RTC，会把 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 传进 `RTCProcessor.denoise_step()`，也就是 RTC 语义是在采样循环外层做的，不需要改 token/prefix 网络结构。
   - 关键位置：`modeling_pi05.py:791-833`。

2. RTC 的核心不是“单次 chunk 算得更快”，而是“在 inference 存在延迟时，仍能让 chunk 更实时、更少浪费”。
   - `RTCProcessor.denoise_step()` 用上一 chunk 未执行完的 leftover 作为 prefix guidance。
   - `ActionQueue.merge()` 在 RTC 模式下不是 append，而是按 `real_delay` 丢弃新 chunk 前缀后整体替换队列。
   - 关键位置：`modeling_rtc.py:116-220`，`action_queue.py:128-174`。

3. ONNX real-robot runtime 已经实现了一个比较完整的 runtime scheduling 雏形。
   - 有 `AsyncChunkPrefetcher`、低水位 `prefetch_threshold`、按 chunk latency 自动估计阈值、queue underrun 时 hold pose、同步兜底 refill。
   - 关键位置：`run_pi05_onnx_infer_so101.py:631-791`，`run_pi05_onnx_infer_so101.py:896-1010`。

4. TRT real-robot runtime 目前还是同步控制环。
   - 主循环每步都调用 `predict_action()`，在这个调用里做 observation prepare、preprocess、policy.select_action、postprocess。
   - `TrtPi05PolicyAdapter` 虽然内部有 `_action_queue`，但只是“队空时同步生成整块，再逐步出队”，没有异步 prefetch，也没有 RTC 的 leftover/delay-aware merge。
   - 关键位置：`run_pi05_trt_infer_so101.py:1376-1431`，`trt_pi_adapter.py:471-522`。

5. TRT 的结构切分其实很适合做 scheduling。
   - 已经拆成 `vision_runner`、`prefix_runner`、`denoise_runner` 三段，说明机会主要在 runtime orchestration，而不是 engine kernel。
   - 关键位置：`trt_pi_adapter.py:126-215`，`trt_pi_adapter.py:471-514`。

---

## 1. 我理解的 RTC 加速点

### 1.1 RTC 真正加速的是“延迟隐藏”和“动作新鲜度”

RTC 在这里不是让单次 denoise kernel 更快，而是让系统在 chunk latency 不可避免时，尽量做到：

- 当前 chunk 还在执行时，下一 chunk 已经在后台生成。
- 新 chunk 完成时，不是盲目 append，而是按真实消耗进度裁掉已经过期的前缀。
- 下一 chunk 的采样可以感知上一 chunk 还没执行完的 leftover，从而减少 chunk 边界的跳变。

也就是说，RTC 的收益来自三件事叠加：

- `queue` 和 `compute` 解耦
- `refresh` 和 `execute` 解耦
- `delay` 被显式建模，而不是隐式吞掉

### 1.2 RTC 的语义核心是“双队列 + 双延迟”

从 `ActionQueue` 和 `RTCProcessor` 看，RTC 依赖两个量：

1. `prev_chunk_left_over`
   - 上一 chunk 在 policy action space 里尚未执行的尾巴。
   - 它不是 robot command space 的动作，而是原始 policy 输出。

2. `inference_delay` / `real_delay`
   - `RTCProcessor.denoise_step()` 需要一个 delay 来构造 prefix weights。
   - `ActionQueue.merge()` 则需要真实 delay 来裁掉新 chunk 前面已经过时的动作。

这意味着要把 RTC 真正搬到 TRT runtime，不能只保留“处理后的动作队列”，必须同时保留：

- 原始 chunk：给下一轮 RTC guidance 用
- 处理后 chunk：给机器人 rollout 用

这一点与当前 ONNX runtime 的 `deque(processed_actions)` 和 TRT adapter 的简单 `_action_queue` 有本质差别。

### 1.3 在 PI05 上，RTC 特别适合 chunk 式扩散采样

PI05 的 chunk 生成有天然延迟：

- 先 vision
- 再 prefix cache
- 再多步 denoise Euler rollout

而机器人控制是每一步都要按固定频率发 action。  
这就天然适合 RTC 的“当前在播，后台刷新下一段”的模式。

### 1.4 TRT 现有三段式子图，为 runtime scheduling 留了很大空间

从 `TrtPi05PolicyAdapter.predict_action_chunk()` 看，当前顺序是：

1. 图像预处理 `_extract_runtime_inputs()`
2. `vision_runner` top / wrist
3. `prefix_runner`
4. 多次 `denoise_runner`
5. 返回整块 action chunk

这说明可以拆出至少三类 scheduling：

- 整块异步 prefetch
- prefix cache 提前刷新
- denoise rollout 与当前 queue 消费并行

所以这里的最佳加速方向，确实不是“再挤 engine kernel 百分之几”，而是把整条 runtime pipeline 变成有队列、有低水位、有双缓冲的流式系统。

---

## 2. TRT runtime 当前缺什么

### 2.1 缺少异步 chunk prefetch

ONNX runtime 已经有：

- `AsyncChunkPrefetcher`
- `prefetch_threshold`
- `chunk_latency_ema`
- `step_time_ema`
- queue underrun 的 hold/sync fallback

TRT runtime 主循环则仍是每步同步：

- 取 observation
- preprocess
- `predict_action()`
- postprocess
- send_action

虽然 policy 内部有 `_action_queue`，但它只是减少了“每步都跑完整 chunk”的概率，没有把 chunk 生成从控制回路里拿出去。

结果是：

- 第一次生成 chunk 的整段延迟仍然阻塞控制环
- 每次 queue 耗尽时会出现同步 refill 尖峰
- 无法在当前 chunk 消费期间隐藏下一 chunk 的准备时间

### 2.2 缺少 RTC 风格的 queue ownership

当前 TRT path 使用的是简单 `deque`，不具备以下 RTC 必需语义：

- `original_queue` 和 `processed_queue` 分离
- `get_left_over()`
- `action_index_before_inference`
- `merge(..., real_delay=...)`
- replace 而不是 append

这会直接导致两个问题：

1. 没法把上一 chunk 剩余的原始动作拿来做下一 chunk guidance。
2. 即使后台算出了更新 chunk，也没法按“实际已经消耗了多少步”来裁掉 stale 前缀。

### 2.3 缺少 delay-aware refresh 策略

当前 TRT runtime 没有显式的：

- chunk latency 测量
- queue low watermark
- 预测 delay 步数
- refresh 触发点
- staleness age 上限

而 ONNX runtime 已经证明，这套最少控制面是必要的。

对 RTC 来说更进一步，至少要区分两个 delay：

- `predicted_delay_steps`
  - inference 启动前估计值，用来给 RTC guidance 构权重
- `real_delay_steps`
  - inference 结束后实际值，用来 merge 时裁 stale prefix

如果这两个量没有被 runtime 显式管理，RTC 只能停留在“理论上支持”，不能在真实机器人回路里可靠落地。

### 2.4 缺少 prefix refresh / cache double-buffer

TRT adapter 已经把 prefix cache 单独导出成子图，但 runtime 仍然把它和 denoise 紧耦合在一个同步 `predict_action_chunk()` 里。

因此当前缺少：

- `active_prefix_cache`
- `staged_prefix_cache`
- prefix refresh cadence
- prefix 结果的年龄/版本号
- 何时只刷新 prefix、何时连同 denoise 一起刷新

这其实浪费了现有子图切分带来的 scheduling 空间。

### 2.5 缺少 stage-level latency telemetry

目前 TRT runtime 看不到下面这些时间：

- preprocess_time
- vision_time
- prefix_time
- denoise_rollout_time
- postprocess_time

没有这些指标，就无法回答几个关键工程问题：

- 低水位应该设几步？
- prefix double-buffer 是否值得做？
- 是图像准备占大头，还是 denoise rollout 占大头？
- 哪个环节最适合并行化？

### 2.6 缺少 observation snapshot / staleness policy

当前 TRT 同步路径默认每次拿到的是“最新 observation”，但一旦要做异步 prefetch，就必须引入：

- 提交 chunk 时使用的 observation snapshot id
- chunk 完成时与当前时间的 age
- 是否允许老 observation 对应的 chunk 上线
- 是否因为视觉变化过大而丢弃已算好的 chunk

这部分 ONNX runtime 也还没做到严格版本化，但 TRT 若要上 RTC，最好一开始就设计进去。

---

## 3. 最值得做的设计方案

我的结论不是“直接把所有 RTC 功能一次性塞进 TRT”，而是分三层推进：

### 3.1 Phase 1：先把 ONNX 的异步 chunk prefetch 平移到 TRT

这是性价比最高、风险最低的第一步。

目标：

- 不再使用通用的 `predict_action()` 单步同步调用路径。
- 改成像 ONNX 一样的 chunk scheduler：
  - 低水位触发后台 chunk 生成
  - 完成后一次性补充到 queue
  - queue 空但 future 未完成时，hold pose
  - queue 空且无 future 时，同步兜底 refill

为什么这一层值得先做：

- 基本不依赖 RTC 算法细节
- 可以直接复用 ONNX runtime 的控制面经验
- 很可能单靠这一步就能吃到最大头的“延迟隐藏”收益

这一层的本质是：  
先把 TRT 从“同步 chunk refill”升级成“异步 chunk refill”。

### 3.2 Phase 2：把 simple deque 升级成 RTC ActionQueue

这是从“有 prefetch”走向“有 RTC 语义”的关键一步。

目标：

- queue 内同时保存：
  - `original_actions`
  - `processed_actions`
- worker 提交时记录：
  - `action_index_before_inference`
  - `prev_chunk_left_over`
  - `predicted_delay_steps`
- worker 完成时计算：
  - `real_delay_steps`
- 用 `ActionQueue.merge()` 做 replace，而不是简单 append

这样一来，新的 chunk 就不是“算出来就往后接”，而是：

- 根据真实消费进度丢掉前缀
- 用更实时的 chunk 覆盖旧队列尾部

这是 RTC 在系统层真正有效的地方。

### 3.3 Phase 3：利用 TRT 三子图做 prefix double-buffer

如果 Phase 1/2 打通后，profiling 显示 `vision + prefix` 占比显著，那么最值得继续做的是：

- 把 prefix refresh 从完整 chunk 生成里拆出来
- 做双缓冲 prefix cache

推荐逻辑：

1. 当前 chunk 还在执行时，后台先拿较新的 observation snapshot 跑：
   - preprocess
   - vision
   - prefix
2. 得到 `staged_prefix_cache`
3. 当 queue 低于更低的阈值时，再基于这个 staged cache 跑 denoise rollout
4. rollout 完成后做 RTC merge

这样做的价值是：

- 把 prefix 这段 latency 再往前推
- 把 chunk refresh 变成两段式流水
- 为后续“只在需要时刷新 prefix”创造空间

### 3.4 我认为最应该先做的版本

如果只能选一个最值得做的设计，我会选：

**“TRT 专用 RTC Chunk Scheduler：异步整块 prefetch + RTC ActionQueue merge + delay-aware refresh”**

原因：

- 它已经能覆盖 80% 的 runtime scheduling 收益。
- 不依赖重新导出 engine。
- 直接对齐当前代码库中已经存在的两套成熟语义：
  - ONNX runtime 的 async prefetch
  - RTC 的 leftover/delay-aware queue replace

而 prefix double-buffer 我会作为第二阶段优化，不会第一天就硬上。

---

## 4. 建议的模块划分

### 4.1 `RtcChunkScheduler`

职责：

- 维护主控制循环里的 action buffer
- 决定何时提交 refresh
- 决定何时 merge 新 chunk
- 处理 underrun / hold / sync fallback

核心状态建议：

- `action_queue: ActionQueue`
- `prefetch_future`
- `predicted_delay_steps_ema`
- `chunk_latency_ema_s`
- `step_time_ema_s`
- `last_completed_snapshot_id`
- `last_refresh_step`

### 4.2 `TrtChunkWorker`

职责：

- 独占 TRT runners 和 pre/postprocessor
- 接收 observation snapshot
- 生成一个 `ChunkArtifact`

建议不要让主线程和 worker 线程共享同一组 `TensorRTRunner` 做并发推理。  
更稳妥的方式是：

- worker 线程拥有自己的 runner 实例
- 主线程只消费结果，不直接碰 runner

这样能规避潜在的 CUDA context / runner thread-safety 问题。

### 4.3 `ChunkArtifact`

建议至少包含：

```python
@dataclass
class ChunkArtifact:
    snapshot_id: int
    submitted_at_s: float
    completed_at_s: float
    action_index_before_inference: int
    predicted_delay_steps: int
    original_actions: torch.Tensor
    processed_actions: list[Any]
    preprocess_time_s: float
    vision_time_s: float
    prefix_time_s: float
    denoise_time_s: float
    postprocess_time_s: float
```

注意 `original_actions` 必须保留。  
没有这个字段，就没法做真正的 RTC leftover guidance。

### 4.4 `PrefixCacheHandle`（Phase 3 用）

职责：

- 持有 prefix outputs
- 记录 snapshot id / age
- 支持 active / staged 双缓冲切换

建议字段：

- `snapshot_id`
- `prefix_pad_masks`
- `cache_tensors`
- `created_at_s`
- `visual_signature` 或 embedding hash

### 4.5 `RefreshPolicy`

职责：

- 根据队列长度、延迟 EMA、chunk age、视觉变化决定是否 refresh

输入：

- `queue.qsize()`
- `chunk_latency_ema_s`
- `step_time_ema_s`
- `last_refresh_age_steps`
- `observation_delta_score`

输出：

- `should_submit`
- `predicted_delay_steps`
- `refresh_mode`:
  - `full_chunk`
  - `prefix_only`
  - `skip`

---

## 5. 可能的伪代码

### 5.1 推荐的最小可落地版本

```python
queue = ActionQueue(rtc_cfg)
scheduler = RtcChunkScheduler(...)
worker = TrtChunkWorker(...)

while True:
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    observation_frame = build_dataset_frame(..., obs_processed, ...)

    completed = worker.maybe_collect()
    if completed is not None:
        real_delay = queue.get_action_index() - completed.action_index_before_inference
        queue.merge(
            original_actions=completed.original_actions,
            processed_actions=tensorize_processed(completed.processed_actions),
            real_delay=max(real_delay, 0),
            action_index_before_inference=completed.action_index_before_inference,
        )
        scheduler.update_latency_ema(completed)

    predicted_delay_steps = scheduler.estimate_delay_steps()
    should_submit = scheduler.should_submit_refresh(queue.qsize())
    if should_submit and not worker.has_pending():
        worker.submit(
            observation_frame=observation_frame,
            prev_chunk_left_over=queue.get_left_over(),
            predicted_delay_steps=predicted_delay_steps,
            action_index_before_inference=queue.get_action_index(),
        )

    action = queue.get()
    if action is None:
        if worker.has_pending():
            robot_action = hold_current_pose(obs)
        else:
            sync_chunk = worker.predict_sync(
                observation_frame=observation_frame,
                prev_chunk_left_over=queue.get_left_over(),
                predicted_delay_steps=predicted_delay_steps,
                action_index_before_inference=queue.get_action_index(),
            )
            queue.merge(
                original_actions=sync_chunk.original_actions,
                processed_actions=tensorize_processed(sync_chunk.processed_actions),
                real_delay=0,
                action_index_before_inference=queue.get_action_index(),
            )
            action = queue.get()
            robot_action = to_robot_action(action, obs)
    else:
        robot_action = to_robot_action(action, obs)

    robot.send_action(robot_action)
```

### 5.2 `TrtChunkWorker` 内部更合理的调用链

建议不要继续走 `control_utils.predict_action()`，因为那个 API 是单步思维，会把 queue / chunk 语义压扁。  
TRT worker 应该走显式 chunk API：

```python
def predict_chunk(...):
    policy_observation = prepare_policy_observation(...)
    original_actions = trt_policy.predict_action_chunk(
        policy_observation,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=predicted_delay_steps,
    )
    processed_actions = postprocess_action_chunk(original_actions[:, :n_action_steps, :])
    return ChunkArtifact(...)
```

这里最关键的是：

- `predict_action_chunk()` 必须显式接收 RTC 所需参数
- 返回值必须同时保留 raw chunk 和 processed chunk

### 5.3 Phase 3 的 prefix double-buffer 伪代码

```python
if refresh_policy.need_prefix_refresh() and not prefix_worker.has_pending():
    prefix_worker.submit(observation_snapshot)

prefix_ready = prefix_worker.maybe_collect()
if prefix_ready is not None:
    staged_prefix = prefix_ready

if queue.qsize() <= denoise_threshold and staged_prefix is not None and not denoise_worker.has_pending():
    denoise_worker.submit(
        prefix_cache=staged_prefix,
        prev_chunk_left_over=queue.get_left_over(),
        predicted_delay_steps=estimate_delay_steps(),
    )
```

这个版本的前提是：

- prefix 占比足够高，值得拆
- worker 生命周期和缓存一致性已经被设计好

---

## 6. 工程上最可能有效的几个优化点

### 6.1 低水位异步 refill

这是最直接的收益点。  
当前 TRT 只要 queue 空了才会同步 refill，而 ONNX 已经证明“在 queue 还没空时就启动下一 chunk”是最有价值的调度策略。

### 6.2 raw/processed 双轨队列

这是让 RTC 真正成立的基础设施，而不是锦上添花。  
否则只能做到“异步生成 chunk”，做不到“RTC merge”。

### 6.3 delay 预测与实际 delay 分离

推荐：

- `predicted_delay_steps = ceil(chunk_latency_ema / step_time_ema)`
- `real_delay_steps = queue.get_action_index() - action_index_before_inference`

前者给 guidance，后者给 merge。  
两者不一致时记录 telemetry，不要静默吞掉。

### 6.4 prefix cache 双缓冲

这是最像“runtime scheduling”而不是“普通异步推理”的优化。  
因为它真正利用了 TRT 已拆好的子图边界。

### 6.5 静态输入预烘焙

这不是主优化，但应该顺手做：

- task 文本通常整个 episode 不变
- `tokens` / `token_attention_mask` 可以预先准备
- 可以减少每次 chunk refresh 的 CPU preprocess 噪音

### 6.6 观测快照版本化

异步 refresh 一旦引入，必须给 chunk 打标签：

- `snapshot_id`
- `submitted_step`
- `completed_step`
- `age_steps`

否则后面很难解释“为什么这个 chunk 明明算出来了却看起来更旧”。

---

## 7. 风险与注意事项

### 7.1 `TensorRTRunner` 的并发/线程安全风险

当前代码没有证明 runner 可以被多线程安全共享。  
如果主线程和 worker 线程同时碰同一个 runner，可能出现：

- CUDA stream/context 冲突
- runner 内部 buffer 复用冲突
- 不可重复的时序 bug

所以更推荐“worker 独占 runner”。

### 7.2 preprocessor / postprocessor 的可重入风险

ONNX runtime 现在把 pre/postprocessor 放进 worker 线程里做，是因为只有一个 worker。  
TRT 若做多 worker 或 prefix/denoise 分拆，就要确认：

- processor 是否带内部状态
- 是否线程安全
- reset 语义是否只应发生在主线程

### 7.3 RTC leftover 必须在 policy action space 里维护

如果 leftover 来自：

- robot_action_processor 之后
- smoothing 之后
- delta clamp 之后

那么它就不再对应模型采样空间，会让 RTC guidance 语义错位。  
所以 leftover 要从 raw policy action chunk 里取，而不是从最终发给机器人的命令里取。

### 7.4 prefix 缓存复用会引入“视觉过期”风险

如果 Phase 3 做成“prefix 不是每个 chunk 都刷新”，必须定义边界：

- 最多允许复用几步
- 视觉变化超过什么阈值必须强制刷新
- 机械臂快速接近目标时是否禁用 cache reuse

否则很容易出现“算得更快，但动作盲了”的情况。

### 7.5 hold pose 策略会影响操作手感

ONNX runtime 当前在 queue underrun 时选择 hold pose。  
这很安全，但会：

- 降低动作连续性
- 在抓取末端造成 hesitation

TRT 若要上线，也许可以比较：

- hold current pose
- replay last action
- replay last low-pass-smoothed action

三者的实际手感差异。

### 7.6 fixed noise / stochastic noise 会影响 chunk 边界稳定性

当前脚本已经有 `--policy-fixed-noise`。  
对于 RTC 来说，固定噪声通常更利于 chunk 间可重复和 merge 稳定；但也可能牺牲部分探索性。  
这至少应进入实验矩阵，而不是忽略。

### 7.7 不要把所有优化一次性叠加

推荐验证顺序：

1. 先做 TRT 异步整块 prefetch
2. 再做 raw/processed 双轨队列
3. 再接入 RTC leftover + delay-aware merge
4. 最后再评估 prefix double-buffer / refresh skipping

这样每一步的收益和副作用都更容易归因。

---

## 8. 最终建议

如果目标是“用 RTC 策略给 PI05 TRT runtime 加速”，我认为最正确的方向不是改 engine，而是把 TRT runtime 从当前的同步单步调用，升级成一个显式的 **chunk scheduler**：

- 以 queue low watermark 触发 refresh
- 以后台 worker 隐藏 preprocess + vision + prefix + denoise 的大部分延迟
- 以 `ActionQueue` 维护原始动作和处理后动作的双轨状态
- 以 `predicted_delay_steps` 做 guidance，以 `real_delay_steps` 做 merge
- 在第二阶段再把 `prefix_cache` 单独双缓冲化

一句话概括：

**最值得做的不是“让 TRT 再快一点”，而是“让 chunk 的生成、刷新、替换在时间轴上更聪明”。**

