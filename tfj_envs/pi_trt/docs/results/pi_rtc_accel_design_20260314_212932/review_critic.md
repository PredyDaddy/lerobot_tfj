# RTC 方案挑刺审查

## 结论

这个 RTC 方案最大的风险不是实现不了，而是做完之后拿错指标、自我感觉“加速成功”。

## 我最担心的三个误判

### 1. 把 `predict_action_chunk()` 耗时当成 RTC 成功证据

RTC 可能让单次 chunk 逻辑更复杂，甚至平均 chunk 耗时略升。

只要：

- queue 不饿死
- control loop 更稳

它依然可能是成功的。

所以如果最后只汇报：

- `mean inference ms`

这个方案很可能被误判。

### 2. 只看均值，不看尾延迟

RTC 的价值体现在：

- p95 / p99 loop 预算
- queue empty rate
- hold rate

如果这些不看，只看均值，方案容易显得“没提升”或者“提升很大”，两种都可能是错觉。

### 3. smoothing / clamp 把问题藏掉

如果 RTC 带来更大的边界跳变，而：

- smoothing
- delta clamp

把这些问题吃掉了，表面上机器人可能仍然动得过去，但这不代表 RTC 真好。

所以必须把：

- `smoothing_event_count`
- `delta_clip_event_count`

作为一等指标。

## 我建议必须补的证据

- RTC on/off 的 mock control loop 对比报告
- queue underrun / hold step / sync refill 统计
- action boundary diff 报告
- real_delay 分布

## 我不建议的说法

不要写：

- “RTC 让 PI05 TensorRT 更快了”

更准确的说法应该是：

- “RTC 有望改善 PI05 TensorRT runtime 的控制循环时延隐藏和 chunk 续接能力”

## 最终意见

方案可以做，但只有在 runtime control 指标也一起落盘的前提下，这件事才有工程意义。
