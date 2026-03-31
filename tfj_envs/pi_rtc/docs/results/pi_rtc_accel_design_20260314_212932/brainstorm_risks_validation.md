# RTC 风险与验证脑暴报告

## 1. RTC 加速到底在降什么成本

RTC 在这里降低的不是“单次模型推理时间”，而是“真实控制循环里因为 chunk 生成过慢导致的有效停顿成本”。

更具体地说，它试图优化的是：

- queue 被吃空前，后台 chunk 已经准备好
- 即使 chunk 不是完全无缝，也能用 leftover guidance 让切换更平滑
- 在已有 chunk latency 不变甚至略升的情况下，control loop 依然不 underrun

所以 RTC 的目标函数不是：

- `pipeline_chunk mean_ms` 最小

而是：

- `queue_underrun_count` 最小
- `hold_step_count` 最小
- `control_loop_over_budget_rate` 最小
- `real_delay` 可控
- `动作切换抖动` 可控

## 2. 哪些指标必须实测

如果这些指标不测，RTC 方案很容易看起来“更快”，实际却没有改善真机表现。

### 2.1 control loop 级指标

必须测：

- 平均 loop wall-clock
- p95 / p99 loop wall-clock
- 超过 `1 / fps` 预算的比例
- queue underrun 次数
- hold action 次数
- sync refill 次数

### 2.2 chunk runtime 级指标

必须测：

- chunk preprocess / inference / postprocess 分项耗时
- chunk latency EMA
- prefetch 提交到完成的 wall-clock
- `prefetch_threshold` 实际是否足够早
- `real_delay` 的真实分布

### 2.3 行为质量指标

必须测：

- action chunk 与非 RTC 基线的差异
- 前几个 action 的跳变幅度
- chunk 边界处 action discontinuity
- smoothing / delta clamp 触发次数

如果只看吞吐、不看动作边界，RTC 很容易把“算得更晚但拼接更激进”的问题藏起来。

## 3. 如何区分“平均步耗时下降”和“真实 control loop 不 underrun”

这两件事必须分开。

### 3.1 平均步耗时下降

这通常是 benchmark 口径，例如：

- `1000-step pure inference`
- `pipeline_chunk mean_ms`

这类数字只能说明：

- 模型/后端本身的平均推理吞吐

不能说明：

- 真机控制循环是否连续

### 3.2 真实 control loop 不 underrun

必须看：

- action queue 是否持续非空
- prefetch 是否总能在 queue 吃空前完成
- 当推理超时后，系统是否频繁进入 hold action
- 实际机器人发送动作的 wall-clock 是否稳定

所以 RTC 方案最重要的新 benchmark 不是：

- 再跑一组纯 `select_action()` 1000 步

而是：

- 一个离线的 runtime loop simulation benchmark
- 一个真机或 mock control loop benchmark

## 4. 最容易自欺的地方

### 4.1 把 RTC 说成“模型加速”

这是最容易写错的。

RTC 更像：

- 调度优化
- queue 语义优化
- 时延隐藏

不是：

- 模型本体 FLOPs 下降
- TRT engine kernel 直接更快

### 4.2 用 `select_action()` benchmark 给 RTC 站台

PI05 主模型已经明确：

- RTC 不支持 `select_action()`
- 要配合 `predict_action_chunk()`

所以如果最后还是用 `benchmark_pi_select_action.py` 来论证 RTC 方案，只能得到误导性结论。

### 4.3 只看平均值，不看尾延迟

RTC 是否有效，尾延迟比均值更重要。

因为控制循环真正出问题的时刻，往往是：

- 某一两个 chunk latency 突然抖高
- queue 没撑住
- 机器人进入 hold 或 sync refill

所以只看：

- mean ms

几乎一定会过于乐观。

### 4.4 把 smoother/delta clamp 当成 RTC 成功证据

如果 RTC 导致 chunk 边界更跳，但后面的：

- smoothing
- delta clamp

把动作“磨平”了，表面上也许看不出炸，但本质上是后处理在兜底，不是 RTC 真的好。

所以必须记录：

- smoothing_event_count
- delta_clip_event_count

而且要对比 RTC 前后变化。

## 5. 当前代码里最可能踩的坑

### 5.1 TRT launcher 入口不对

当前 `run_pi05_trt_infer_so101.py` 还是走：

- `control_utils.predict_action()`
- `policy.select_action()`

而 PI05 已经明确写了：

- RTC 不支持 `select_action`

如果不改这条入口，RTC 从设计上就接不进去。

### 5.2 `real_delay` 计算错位

`ActionQueue.merge(...)` 依赖：

- `real_delay`
- `action_index_before_inference`

如果这两个值错位，会出现：

- queue 替换错位置
- leftover 错前缀
- action chunk 时间对齐错误

这类 bug 在数值上不一定炸，但控制上会很怪。

### 5.3 observation snapshot 与 chunk 对齐错误

后台 prefetch 使用的是“提交时 observation”，而 merge 发生在“完成时”。

如果系统没有清楚记录：

- 这是哪一帧 observation 发起的推理
- 当前 queue 已经消耗了多少步

RTC leftover 就可能和错误的 observation 语境混在一起。

### 5.4 把 ONNX async 逻辑和 TRT 逻辑分叉两套

当前 ONNX launcher 已经有 async prefetch 雏形，而 TRT 没有。

如果 TRT 为 RTC 再写一套新 orchestration，后面很容易出现：

- threshold 算法不一致
- queue 行为不一致
- benchmark 不可比

所以共享 helper 很重要。

## 6. 我建议新增的验证工件

除了现有 benchmark，至少再补 3 类：

### 6.1 `rtc_runtime_benchmark.json/md`

内容：

- prefetch submit/collect 时间
- queue low-watermark
- chunk latency EMA
- `real_delay`
- queue underrun / hold step / sync refill 统计

### 6.2 `rtc_action_diff_report.json/md`

内容：

- RTC vs non-RTC 的 action chunk 差异
- chunk 边界 jump
- smoothing / delta clamp 触发对比

### 6.3 `rtc_mock_control_loop_report.json/md`

内容：

- 固定 fps 下的离线 control loop 仿真
- over-budget rate
- queue empty rate
- hold rate

## 7. 最终建议

如果要让 RTC 方案站得住，验证口径必须从“模型推理 benchmark”升级成“runtime control benchmark”。

最硬的成功判据不该只是：

- `mean chunk latency` 下降

而应该是：

- 同样 fps 下，queue underrun 减少
- hold step 减少
- over-budget rate 降低
- smoothing / delta clamp 没明显恶化

只有这样，RTC 才能被说成对 PI05 真正有效的加速策略。
