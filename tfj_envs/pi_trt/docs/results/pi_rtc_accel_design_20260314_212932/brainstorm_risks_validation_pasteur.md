# PI05 RTC 加速的挑刺与验证头脑风暴

## 0. 先说结论

基于当前本地仓库，我认为“用 RTC 策略加速 PI05”这件事最容易自欺的地方，不是把一个 kernel benchmark 看错，而是把“隐藏 chunk 刷新阻塞”误写成“模型本身更快”或者“真实控制环已经稳定无 underrun”。

在当前仓库语义下，`RTC` 不是一个已经落地成统一名词的模块。我只能按现有代码反推它最接近的实现含义：

1. `run_pi05_onnx_infer_so101.py` 里已经存在一个接近 RTC 的策略原型：
   - `AsyncChunkPrefetcher`
   - `prefetch_threshold`
   - `sync_refill_timeout_s`
   - `queue_underrun_count`
   - `hold_step_count`
2. 当前 `run_pi05_trt_infer_so101.py` 还没有这层异步预取，只是同步调用 `predict_action(...)`，而 `TrtPi05PolicyAdapter.select_action()` 内部只在本地 queue 空时同步生成新 chunk。

所以，如果现在讨论“RTC 加速 PI05”，更准确的说法应该是：

- 它是一个 runtime scheduling / overlap 策略
- 目标是把 `chunk refresh` 的 wall-clock 从控制线程关键路径里移出去
- 它不是直接降低 `predict_action_chunk()` 自身计算量的算法

这个前提不说清楚，后面所有 benchmark 结论都容易写歪。

## 1. RTC 加速到底是在降哪种 wall-clock 成本

## 1.1 不是在降什么

先说最容易写错的：

- RTC 不是直接降低 `vision_encoder` / `prefix_cache` / `denoise_step` 的裸计算耗时
- RTC 不是直接让 `pipeline_chunk` 的离线模型时延变短
- RTC 也不是直接让单次 chunk 刷新从 120ms magically 变成 20ms

如果某个设计声称“加了 RTC 以后 `pipeline_chunk` benchmark 直接变快了”，那大概率不是纯 RTC，而是混入了别的变量：

1. 改了 `num_inference_steps`
2. 改了 `n_action_steps`
3. 改了 pre/post 逻辑
4. 改了 precision
5. 改了缓存或线程模型

因为当前仓库里 `benchmark_pi_inference.py` 测的是模型推理链本身，而不是异步调度策略。

## 1.2 真正降低的 wall-clock 成本

RTC 真正打到的是“控制线程上、会导致当前 step 卡住的 chunk 刷新成本”。

在当前实现里，同步模式的 refresh-step 成本大致是：

1. 当前 observation 采集完成
2. `prepare_observation_for_inference(...)`
3. `preprocessor(...)`
4. `policy.predict_action_chunk(...)`
   - 2 次 vision
   - 1 次 prefix
   - `num_inference_steps` 次 denoise
5. `postprocessor(...)`
6. 再把这一个 step 的 action 发给机器人

对当前 `TRT launcher` 来说，这一整段都在控制线程里。

证据：

- `run_pi05_trt_infer_so101.py` 每步都调用 `predict_action(...)`
- `src/lerobot/utils/control_utils.py` 里的 `predict_action(...)` 明确包含：
  - `prepare_observation_for_inference(...)`
  - `preprocessor(...)`
  - `policy.select_action(...)`
  - `postprocessor(...)`
- `TrtPi05PolicyAdapter.select_action()` 只有在 `_action_queue` 为空时才同步生成整个 chunk

所以 RTC 如果要真有价值，它降低的是：

1. refresh-step 对当前 control step 的阻塞 wall-clock
2. queue 被刷空后的同步 refill stall
3. 因等待新 chunk 而发生的 hold / slow-step / missed-period

换句话说，RTC 不是降“chunk 计算量”，而是降“chunk 计算对当前 step 的显式阻塞”。

## 1.3 当前 ONNX launcher 已经给了一个很好的原型

`run_pi05_onnx_infer_so101.py` 现有的 async 设计本质上做了三件事：

1. 在 queue 还没空的时候，提前异步生成下一块 chunk
2. 把 `prepare_observation_for_inference + preprocessor + predict_action_chunk + postprocess_action_chunk` 都搬到 worker 线程
3. 当 queue 真空了但 future 还没好时，允许短暂 `hold current pose`

所以这条策略真正减少的是：

- 主线程“必须同步等 chunk 算完”这件事的发生频率

它打的是 control-thread critical path，不是 raw model path。

## 1.4 当前 benchmark 数字已经能说明这个差别

本地现成 benchmark 明确区分了两类口径：

1. `pipeline_chunk`
2. `1000-step pure select_action`

现有实测里已经写得很明白：

- `pipeline_chunk` 是“完整一次 action chunk 生成”
- `amortized_per_action_step` 只是 `chunk / n_action_steps`
- `1000-step pure select_action` 反映的是 queue 刷新和复用后的均摊纯推理吞吐

当前本地数值很适合拿来打脸自欺式表述：

安全 `TRT FP32`：

- `pipeline_chunk = 123.501 ms`
- `1000-step select_action = 2.491 ms/step`

诊断 `unsafe TRT FP16`：

- `pipeline_chunk = 50.665 ms`
- `1000-step select_action = 1.015 ms/step`

如果控制频率是当前 launcher 默认的 `30 FPS`，单步 wall-clock budget 只有大约：

- `33.3 ms`

那这组数字直接告诉我们：

1. 同步 chunk 刷新时，`123.5 ms` 一定超预算
2. 即使是 `50.7 ms`，同步 chunk 刷新也仍然超预算
3. 但平均 `select_action` 又看起来很快

这说明平均步耗时小，不代表 refresh-step 不会卡住控制环。RTC 的意义就是去隐藏这种 refresh-step stall。

## 2. 哪些指标必须实测，不然方案可能是假的

如果只是做“RTC acceleration proposal”，我认为下面这些指标必须实测。少一个，方案都可能是假的。

## 2.1 模型离线指标：证明你没有偷改模型口径

这些用来证明你没有把 scheduling 优化伪装成模型加速：

1. `pipeline_chunk` 的离线实测
   - 继续用 `scripts/benchmark_pi_inference.py`
   - 目的：确认模型自身 chunk 成本没有被神秘改写
2. 子阶段耗时：
   - `vision_encoder_pair`
   - `prefix_cache`
   - `denoise_step`
3. 固定：
   - `policy_num_inference_steps`
   - `policy_n_action_steps`
   - precision
   - provider / engine set

如果 RTC 方案上线后，离线 `pipeline_chunk` 也“变快了”，那要么不是纯 RTC，要么实验变量没控住。

## 2.2 运行时 chunk 指标：证明你真的把 stall 隐藏了

这是 RTC 最关键的实测层。

当前 ONNX async 原型已经有一部分字段了，TRT 如果要做 RTC，必须镜像补齐：

1. 每个 chunk 的总耗时
   - `chunk_total_time_s`
2. 每个 chunk 的拆分耗时
   - `chunk_preprocess_time_s`
   - `chunk_inference_time_s`
   - `chunk_postprocess_time_s`
3. chunk 提交时 queue 剩余长度
   - `queue_size_at_submit`
4. chunk 收到结果时 queue 剩余长度
   - `queue_size_at_collect`
5. chunk 是异步成功隐藏，还是最终退化成同步 refill
   - `async_collect_count`
   - `sync_refill_count`

原因很直接：

- RTC 的核心不是“算没算出来”
- 而是“是否在 queue 用尽前算出来”

## 2.3 真实 control loop 指标：证明你不是只让平均值更好看

这是最不能省的一层。

必须实测每一步的真实 loop wall-clock，而不是只看平均 `select_action`：

1. `loop_dt_ms`
   - 每一步从 `loop_t = time.perf_counter()` 到 `send_action()` 结束的真实耗时
2. `loop_over_budget_count`
   - `dt_s > 1 / camera_fps` 的次数
3. `loop_over_budget_ratio`
4. `loop_dt_p50 / p95 / p99 / max`
5. `sleep_budget_ms`
   - `1/fps - dt_s`
6. `sleep_zero_count`
   - 进入 `precise_sleep(max(..., 0.0))` 时被截成 `0` 的次数

当前两个 launcher 都有一个容易误导人的共同点：

- 它们最后都调用 `precise_sleep(max(1 / args.camera_fps - dt_s, 0.0))`

这意味着：

- 一旦某一步已经超预算，脚本不会报错
- 它只会不 sleep
- loop 就会静默跑慢

如果你不额外统计 `dt_s > period`，就很容易出现“日志没报错，所以实时性没问题”的假象。

## 2.4 queue 指标：证明你没有拿 hold action 掩盖失败

尤其是参考 ONNX async 方案时，这组指标必须实测：

1. `queue_underrun_count`
2. `hold_step_count`
3. `generated_new_chunk_count`
4. `prefetch_pending_ratio`
5. `queue_size` 的分布而不是单点日志

原因：

- 在 `run_pi05_onnx_infer_so101.py` 里，如果 queue 空了但 future 还没好，系统会直接发 hold action

这很容易制造一种错觉：

- loop 还在稳稳跑
- 机器人也没停机

但实际上：

- policy 已经没跟上
- 只是用 hold action 把 stall 藏起来了

所以如果 RTC 方案只汇报：

- `mean_step_ms` 下降了
- `loop 没崩`

但不汇报：

- `hold_step_count`
- `queue_underrun_count`

那这个方案很可能是假的。

## 2.5 stale-observation 指标：证明你没有靠“更旧的 observation”换稳定

这是我认为最容易被忽略、但最容易让方案“表面实时、实质退化”的指标。

一旦用预取，就会在 queue 还没空的时候提前用当前 observation 生成下一 chunk。

这意味着 action 的观测条件会过时。

必须实测：

1. `prefetch_threshold`
2. `observation_age_steps_at_first_use`
3. `observation_age_ms_at_first_use`
4. `observation_age_ms_at_last_use`
5. `stale_chunk_ratio`

一个很实用的推导量：

- `staleness_horizon_ms ≈ prefetch_threshold * control_period_ms`

如果 `camera_fps = 30`，每步约 `33.3 ms`：

- `prefetch_threshold = 3` 约等于提前 `100 ms`
- `prefetch_threshold = 10` 约等于提前 `333 ms`

这个数字非常大，而且很可能比模型加速本身更影响控制质量。

## 2.6 行为正确性指标：证明 RTC 没把行为语义搞变

当前 benchmark 里一个大洞是：

- 它们大多比较耗时，不比较“RTC 后实际动作和同步基线差了多少”

如果 RTC 设计改变了：

1. prefetch 触发时机
2. 使用的 observation 时刻
3. `n_action_steps`
4. `num_inference_steps`
5. `fixed_noise`

那就必须做 side-by-side action diff：

1. 同一 observation 序列
2. 同步基线控制器
3. RTC 控制器
4. 比较每一步最终发送给机器人的 action

至少记录：

1. `action_max_abs_diff`
2. `action_mean_abs_diff`
3. `action_p95_abs_diff`
4. 超过 `joint_delta_limit` / `gripper_delta_limit` 前后差异

不做这层，RTC 可能只是“更早地产生了旧动作”。

## 3. 如何区分“平均步耗时下降”和“真实 control loop 不 underrun”

这是整个 RTC 设计里最关键的辨析点。

## 3.1 平均步耗时下降，是什么意思

当前 `scripts/benchmark_pi_select_action.py` 已经写得很清楚：

- 它测的是 `select_action()` 的均摊纯推理吞吐
- 计时包含 queue 的刷新与复用
- 不是单次 chunk 刷新时延

所以：

- `mean_per_step_ms` 下降
- 只代表平均起来每一步更便宜
- 不代表任意一个 refresh-step 没有长尾 stall

## 3.2 真实 control loop 不 underrun，是什么意思

我建议把“不 underrun”定义成至少同时满足下面三条：

1. `loop_dt_p99 <= control_period_ms`
2. `loop_over_budget_count = 0` 或极低且有明确阈值
3. `hold_step_count = 0`

只有这样，才能说：

- 控制线程没因为 chunk 刷新而卡住
- 也没靠 hold action 把 stall 藏掉

## 3.3 一个很容易误导人的反例

本地现成数字就是最好的反例。

以 `30 FPS` 为例：

- 单步 budget ≈ `33.3 ms`

当前本地 `TRT FP32`：

- `pipeline_chunk = 123.501 ms`
- `select_action mean = 2.491 ms/step`

当前本地 `unsafe TRT FP16`：

- `pipeline_chunk = 50.665 ms`
- `select_action mean = 1.015 ms/step`

如果只看平均值，会以为：

- 这已经非常实时了

但如果没有 RTC overlap：

- 每次 queue 空时都要同步刷新 chunk
- 那一步还是会直接超 33.3ms budget

所以“平均很小”与“不会 underrun”是两回事。

## 3.4 正确的判定逻辑应该是什么

RTC 方案应该按下面这组关系判断，而不是看平均数：

1. `control_period_s = 1 / camera_fps`
2. `available_overlap_s = queue_remaining_at_submit * control_period_s`
3. `chunk_total_latency_p95_s`
4. `chunk_total_latency_max_s`

如果：

- `chunk_total_latency_p95_s < available_overlap_s - safety_margin`

并且：

- `hold_step_count = 0`
- `sync_refill_count = 0`
- `loop_over_budget_count = 0`

那才可以说：

- RTC 真正把 refresh stall 隐藏住了

否则只能说：

- 平均吞吐不错
- 但 control loop 是否真实稳定，证据不足

## 4. 设计里最可能踩的坑

## 4.1 把 scheduling 优化写成 model 加速

这是第一大坑。

RTC 的本质是 overlap，不是 kernel-level optimization。

如果文档里把它写成：

- “RTC 让 PI05 推理更快了”

就不够严谨。

更准确的写法应该是：

- “RTC 降低了控制线程的可见 refresh stall wall-clock，使 chunk 生成更容易被动作队列掩蔽”

## 4.2 用 `benchmark_pi_select_action.py` 证明 RTC 成功

这也是大坑。

原因：

1. 当前 `benchmark_pi_select_action.py` 不接机器人
2. 它不测 `robot.get_observation()`
3. 不测 `send_action()`
4. 不测 smoothing / delta clamp
5. 不测 hold steps
6. 不测 loop lateness

它最多只能证明：

- 当前 `policy.select_action()` 的均摊纯推理吞吐如何

它不能证明：

- RTC 真正解决了 live loop underrun

## 4.3 用 `pipeline_chunk / n_action_steps` 当成真实每步时延

仓库文档已经在很多地方提醒这一点了，但这是最容易被再次写错的坑。

`amortized_per_action_step_ms` 只是：

- `pipeline_chunk / n_action_steps`

它不是：

- 每个 control step 的最坏时延
- refresh-step 的时延
- live control loop 的 wall-clock

## 4.4 为了隐藏 stall 把 `n_action_steps` 拉大

这是非常危险的“看起来成功”的路径。

因为 launcher 允许：

- `--policy-n-action-steps`

把 `n_action_steps` 拉大，确实会：

1. 降低 refresh 频率
2. 增大 overlap 窗口
3. 让 average 看起来更漂亮

但同时也会：

1. 拉大 open-loop 执行动作长度
2. 提高 observation staleness
3. 降低控制的反馈性

所以 RTC 方案如果通过：

- 增大 `n_action_steps`

取得成功，必须把这件事单独写成 tradeoff，而不是当成“纯加速”。

## 4.5 为了变快偷偷改 `num_inference_steps`

同理，launcher 允许：

- `--policy-num-inference-steps`

如果 RTC 方案一边讲 scheduling，一边偷偷把 `num_inference_steps` 从 10 改到更小，那结论就混了。

这种情况下：

- 加速不再是纯 RTC
- 而是 runtime scheduling + 模型精度/采样质量 tradeoff

这个变量必须钉死，或者单列 ablation。

## 4.6 用 `policy_fixed_noise` 让 jitter 变小，然后误当成实时更稳

当前 ONNX / TRT launcher 都支持：

- `--policy-fixed-noise`
- `--policy-noise-seed`

固定噪声可能让 chunk 表现看起来更平滑、抖动更少，但这不等于 RTC 方案更好。

它可能只是：

1. 降低了随机性
2. 提高了 action 可重复性
3. 让 benchmark 更稳定

如果没把 noise 策略固定住，RTC 对比很容易变假。

## 4.7 用 hold action 掩盖 queue drained

这个坑在 ONNX async 原型里已经真实存在。

当 queue 空了但 future 还没 ready：

- 代码会发送 `hold current pose`

这在工程上是合理保守的，但在评估上非常容易自欺。

因为你会看到：

1. 机器人没炸
2. control loop 还在跑
3. 甚至 `dt_s` 也不大

但实际上：

- policy 已经没及时给出新动作

如果 RTC 报告里不显式写：

1. `queue_underrun_count`
2. `hold_step_count`

那这个设计可能只是“优雅地失败”。

## 4.8 过早 prefetch 导致 observation 过时

当前 `estimate_prefetch_threshold(...)` 是按：

- `chunk_latency_s`
- `step_time_s`
- `fallback_fps`

动态估阈值的。

但代码里的 `step_time_ema_s` 是：

- 每步 work time
- 不包含最终 `precise_sleep(...)`

这会让阈值估算偏保守，也就是：

- 可能比真正需要的更大
- 更早开始 prefetch
- 更早使用旧 observation

这对“避免 queue 空”是好事，但对“动作是否仍然够新”是坏事。

所以 RTC 方案必须同时看：

1. underrun 风险
2. staleness 风险

只优化前者，会把系统推成一个“稳定执行旧动作”的控制器。

## 4.9 直接把 ONNX async 策略搬到 TRT，忽略线程/上下文风险

这也是我会重点挑刺的点。

`ONNX` 当前 async 方案用的是：

- `ThreadPoolExecutor(max_workers=1)`

但 `TRT` 这边的 `TensorRTRunner` 每个 runner 都持有：

1. execution context
2. CUDA stream
3. device state

当前仓库没有任何地方证明：

- 直接把同一套 `TrtPi05PolicyAdapter` 放到 worker thread 里做后台 prefetch，同时主线程还安全使用它，是 thread-safe 的

所以如果要做 TRT 版 RTC，我认为更安全的做法不是“直接照抄 ONNX 版”，而是：

1. 单独的 worker adapter / 独立 runner 资源
2. 明确的 ownership
3. 不在两个线程并发触同一个 TensorRT execution context

不先把这件事讲清楚，设计很容易在 demo 时看起来能跑，长期跑却出偶发错。

## 4.10 只看 queue 指标，不看真实 loop deadline

即使 `queue_underrun_count = 0`，也不代表 control loop 真没问题。

因为下面这些仍然可能让 loop 超预算：

1. `robot.get_observation()`
2. camera decode / capture 抖动
3. smoothing / clamp
4. `send_action()`
5. Python 本身调度 jitter

所以 RTC 方案必须同时看两类东西：

1. queue 层：有没有来得及准备好下一 chunk
2. loop 层：本步总 wall-clock 有没有超目标周期

只看 queue 层，会错把“模型不堵了”当成“控制环没堵”。

## 4.11 只在 deterministic baseline batch 上验证

`benchmark_pi_inference.py` 当前明确使用：

- `build_runtime_context()` 生成的 deterministic baseline batch

这很适合做固定口径对比，但不代表真实任务分布。

RTC 特别依赖 observation 的时间性，所以只在单一 deterministic batch 上看吞吐，风险很大：

1. 看不见真实图像 jitter
2. 看不见真实 observation 分布
3. 看不见预取带来的 stale observation 影响

所以 RTC 的最终验证一定要有 live loop 或至少 recorded observation replay。

## 5. 我建议的验证框架

如果后面真要推进 RTC，我建议按下面四层做验证，少一层都不要写“成功”。

## 5.1 层一：模型离线不变性

目标：

- 证明你没有改模型本体，只改 runtime scheduling

必须保留：

1. `benchmark_pi_inference.py`
2. 相同 `num_inference_steps`
3. 相同 `n_action_steps`
4. 相同 engine / provider

预期：

- `pipeline_chunk` 不应该因为 RTC 本身而变化

## 5.2 层二：纯 select_action 均摊吞吐

目标：

- 看均摊吞吐是否改善

工具：

- `benchmark_pi_select_action.py`

但只能作为辅助指标，不能当最终 deploy 证据。

## 5.3 层三：实时 launcher 级验证

目标：

- 真正验证 control loop

必须新增或补齐的 runtime 指标：

1. `loop_dt_ms`
2. `loop_over_budget_count`
3. `sleep_zero_count`
4. `queue_underrun_count`
5. `hold_step_count`
6. `sync_refill_count`
7. `queue_size_at_submit`
8. `observation_age_ms_at_chunk_use`
9. `chunk_pre / infer / post / total`

## 5.4 层四：行为退化验证

目标：

- 防止“实时更稳，但动作更旧”这类伪成功

必须做：

1. 同 observation replay 的同步基线 vs RTC 对比
2. action drift 分布
3. stale chunk 使用比例
4. clamp / smoothing 触发频率变化

## 6. 我会如何写最终的挑刺句子

如果要对 RTC 方案做一句最难被反驳的挑刺，我会写：

- “在当前仓库里，RTC 最多只能被宣称为一种隐藏 `predict_action_chunk` refresh stall 的 runtime scheduling 策略；除非同时拿出 live loop 的 `over_budget_count=0`、`hold_step_count=0`、`queue_underrun_count=0`、以及 observation staleness 与 action drift 的实测证据，否则‘平均 step 更快’并不能证明它真的让 PI05 在真实 control loop 下更实时。” 

## 7. 最终判断

一句话总结：

- RTC 真正要验证的不是“平均值有没有更漂亮”，而是“在不靠 hold action、不靠更旧 observation、不靠偷偷改 `n_action_steps/num_inference_steps` 的前提下，chunk refresh 是否真的从控制线程关键路径里消失了”。

当前本地仓库已经给了一个很好的提醒：

1. `benchmark_pi_inference.py` 告诉你 chunk 本体多贵
2. `benchmark_pi_select_action.py` 告诉你均摊后多便宜
3. `run_pi05_onnx_infer_so101.py` 告诉你真正的实时策略需要 queue / prefetch / hold 监控
4. `run_pi05_trt_infer_so101.py` 告诉你当前 TRT live loop 还没有这层保护和可观测性

所以 RTC 方案最容易自欺的地方就是：

- 把“均摊吞吐提升”写成“真实控制环已经稳了”

这两者在当前仓库里，不是一回事。
