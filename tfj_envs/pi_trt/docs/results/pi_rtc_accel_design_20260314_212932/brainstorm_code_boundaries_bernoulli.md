# PI05 TRT/ONNX 引入 RTC 的代码边界头脑风暴

日期：2026-03-14

## 目标与约束

目标是：**在不改动现有 TRT engine / ONNX subgraph 边界的前提下**，把 RTC 逻辑引入 PI05 的 TRT/ONNX runtime，并尽量复用现有 PI05 PyTorch/RTC 代码。

这里的“不改动 engine 边界”我理解为：

- 不改 `vision_encoder / prefix_cache / denoise_step` 三段 TRT engine / ONNX 的输入输出契约；
- 不新增 engine 输入，例如 `prev_chunk_left_over`、`guidance_weight`、`inference_delay`；
- RTC 只能作为 **host 侧 runtime orchestration**，包在现有 `prefix_cache + denoise_step` 推理循环之外。

本报告主要基于以下本地文件：

- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/rtc/configuration_rtc.py`
- `src/lerobot/policies/rtc/modeling_rtc.py`
- `scripts/trt_pi_adapter.py`
- `scripts/onnx_pi_adapter.py`

补充参考了本地 launcher / runtime 边界代码：

- `src/lerobot/policies/rtc/action_queue.py`
- `src/lerobot/utils/control_utils.py`
- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`

## 一句话结论

如果坚持不改 engine 边界，那么 RTC 最合适的落点不是 exporter、不是 engine、也不是 `predict_action()` 这种单步通用入口，而是：

1. 在 `trt_pi_adapter.py` / `onnx_pi_adapter.py` 的 **denoise loop** 内，复用 `RTCProcessor` 包住单步 denoise；
2. 在 real-run launcher 层维护 **leftover / inference_delay / action queue** 这类 runtime state；
3. 配置通过 `PI05Config.rtc_config` 贯通，CLI 只做 override，不另起一套平行配置体系。

换句话说，**RTC 应该是“adapter 负责数值修正，launcher 负责时序状态”**。

---

## 关键观察

## 1. PyTorch 版 RTC 已经是“包裹单步 denoise”的结构

`PI05Pytorch.sample_actions()` 的核心结构很适合直接迁移到 TRT/ONNX adapter：

- 先跑一次 prefix，拿到 `past_key_values`
- 再做 denoise loop
- 每个 step 里构造 `denoise_step_partial_call(x_t)`
- 如果 RTC 打开，则把这个 partial 交给 `self.rtc_processor.denoise_step(...)`
- 否则走普通 denoise

这一段在 `src/lerobot/policies/pi05/modeling_pi05.py` 非常清楚：

- `PI05Pytorch.sample_actions(...)` 在 `749-834`
- RTC 接入点在 `809-821`
- `ActionSelectKwargs` 已经定义了：
  - `inference_delay`
  - `prev_chunk_left_over`
  - `execution_horizon`

这意味着 **PI05 的 RTC 契约本来就是 chunk-level、denoise-loop-level 的，不是 select_action-level 的**。

## 2. PI05Policy 明确说了 RTC 不支持 `select_action`

`PI05Policy.select_action()` 里有显式断言：

- `assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"`

位置：

- `src/lerobot/policies/pi05/modeling_pi05.py:1215-1219`

这点非常重要。它直接说明：

- 不应把 RTC 强行塞进通用单步 `predict_action()` 调用路径；
- 更合理的做法是以 `predict_action_chunk(...)` 为原子单位，把 launcher 也改成 chunk 驱动，而不是一拍一拍地隐式 refill。

## 3. TRT/ONNX adapter 当前已经有现成的最小插桩点

两个 adapter 当前结构都非常像 PyTorch 的 chunk sampling，只是把 PyTorch 的 denoise 替换成了外部 runtime：

### ONNX adapter

`scripts/onnx_pi_adapter.py`

- prefix 计算：`279-291`
- denoise loop：`309-318`
- 每步当前就是：
  - 组装 `denoise_feed`
  - `v_t = self.denoise_runner.infer(... )["v_t"]`
  - `x_t = x_t + dt * v_t`

### TRT adapter

`scripts/trt_pi_adapter.py`

- prefix 计算：`474-486`
- denoise loop：`504-512`
- 结构和 ONNX 基本一致

所以从代码边界看，RTC 的**最小改动插入点**就是：

- 在 `v_t = denoise_runner.infer(...)` 和 `x_t = x_t + dt * v_t` 之间，
- 用一个和 PyTorch 同风格的 `denoise_step_partial_call` + `RTCProcessor.denoise_step(...)` 包起来。

## 4. 现有 RTCProcessor 的 callable contract 本身就适合外部 runtime

`RTCProcessor.denoise_step(...)` 的接口是：

- 输入 `x_t`
- 输入 `prev_chunk_left_over / inference_delay / time / execution_horizon`
- 输入一个 `original_denoise_step_partial`

文件：

- `src/lerobot/policies/rtc/modeling_rtc.py:116-248`

这意味着它并不要求必须调用 PyTorch module 本体。只要 adapter 能提供一个：

`Callable[[Tensor], Tensor]`

就可以复用。

对 TRT/ONNX 来说，这个 callable 完全可以是：

- 捕获 `shared_denoise_inputs`
- 捕获当前 step 的 `timestep`
- 内部调用 `denoise_runner.infer(...)`
- 返回 `v_t`

也就是说，**从接口设计上看，RTCProcessor 已经为非 engine 内部集成预留了 host-side wrapper 形态**。

## 5. 真正麻烦的不是 denoise 修正，而是 runtime state

RTC 需要的核心额外信息不是模型权重，而是运行时状态：

- `prev_chunk_left_over`
- `inference_delay`
- `execution_horizon`

这三者都不在 engine 输入里。

而且它们也不属于 prefix_cache / denoise_step 的纯函数输入，而是**跨 chunk、跨控制周期的 runtime state**。

因此，这部分不适合塞进 adapter 的 engine contract 校验逻辑里，更适合放在 launcher 或一个 shared runtime controller 中维护。

---

## 1) 最小可改代码面

我建议把“最小可改代码面”分成两层看：

## A. 核心最小改动面：只动 adapter + 配置

严格从 RTC 算法接入角度，最小改动面其实很小，核心只需要：

### `scripts/trt_pi_adapter.py`

- 增加 `_rtc_enabled()` / `self.rtc_processor`
- 在 `predict_action_chunk()` 的 denoise loop 中复用 `RTCProcessor`
- 让 `predict_action_chunk(..., **kwargs)` 真正支持：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- 在 `runtime_summary()` / `run_preflight()` 中暴露 RTC 配置和 RTC 语义检查结果

### `scripts/onnx_pi_adapter.py`

- 做与 TRT adapter 对称的 RTC 接入
- 最好补一个 `run_preflight()`，至少做到和 TRT adapter 同等级别的语义预热
- 如果 `RTC` 打开，则要求 `denoise_accepts_timestep == True`

### `src/lerobot/policies/pi05/configuration_pi05.py`

- 这里已经有 `rtc_config: RTCConfig | None = None`
- 理论上可以不改，或者只补少量文档 / 验证逻辑

### `src/lerobot/policies/rtc/configuration_rtc.py`

- 现有字段已经够用：
  - `enabled`
  - `prefix_attention_schedule`
  - `max_guidance_weight`
  - `execution_horizon`
  - `debug/debug_maxlen`
- 理论上也可以不改

### `src/lerobot/policies/rtc/modeling_rtc.py`

- 理论上也可以不改
- 可选的小优化是新增一个更显式的 host-side helper，例如 `apply_guidance_from_v_t(...)`
- 但这不是必须

## B. 真正能跑起来的最小落地面：还得动 launcher

如果目标不是“adapter 内能接 RTC kwargs”，而是“real robot runtime 真能用 RTC”，那只改上述六个文件是不够的。

因为谁来提供：

- `prev_chunk_left_over`
- `inference_delay`
- chunk merge / replace 逻辑

答案是：**launcher 或 shared runtime controller**。

因此，真实最小落地面还需要至少动：

- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`

理由：

- TRT launcher 现在走 `predict_action(...) -> policy.select_action(...)` 路径，本身不适合 RTC；
- ONNX launcher 虽然已经是 chunk 驱动，但 queue 只是一个 `deque`，没有保存 original chunk leftover，也没有 RTC delay 语义。

## 推荐的最小变更边界

如果现在要尽量少改，我建议：

1. **先改 adapter，让 chunk-level RTC contract 成立**
2. **再改 launcher，让它提供 runtime state**
3. **不要先做大规模 base class / 框架级重构**

---

## 2) 哪些逻辑可以直接复用

## 2.1 可以直接复用的配置逻辑

### `PI05Config.rtc_config`

文件：

- `src/lerobot/policies/pi05/configuration_pi05.py:50-52`

这已经是最自然的配置入口，不需要为 TRT/ONNX 再造一套 `trt_rtc_config` / `onnx_rtc_config`。

### `RTCConfig`

文件：

- `src/lerobot/policies/rtc/configuration_rtc.py:29-55`

这些字段本身就是 runtime 配置，不依赖 PyTorch 模型结构，完全适用于 TRT/ONNX。

## 2.2 可以直接复用的 RTC 数学逻辑

### `RTCProcessor.denoise_step`

文件：

- `src/lerobot/policies/rtc/modeling_rtc.py:116-248`

建议直接复用，不要在 TRT adapter 和 ONNX adapter 各写一版 RTC 修正数学。

### `RTCProcessor.get_prefix_weights`

文件：

- `src/lerobot/policies/rtc/modeling_rtc.py:250-297`

prefix attention schedule 的实现已经在这里，TRT/ONNX 不应复制一份。

## 2.3 可以直接复用的 PI05 接口契约

### `ActionSelectKwargs`

文件：

- `src/lerobot/policies/pi05/modeling_pi05.py:56-60`

这就是最合适的 adapter 输入契约。建议 TRT/ONNX adapter 直接对齐这个 kwargs 集合。

### `PI05Pytorch.sample_actions()` 的 closure 形态

文件：

- `src/lerobot/policies/pi05/modeling_pi05.py:799-823`

这个模式可以几乎原样平移到 adapter：

- 定义 `denoise_step_partial_call(input_x_t, current_timestep=...)`
- `RTCProcessor` 只拿这个 closure
- closure 内部换成 TRT/ONNX `infer(...)`

## 2.4 可以直接复用的队列逻辑

虽然不在本次重点文件列表里，但本地仓库已经有现成可复用的队列：

### `rtc/action_queue.py`

文件：

- `src/lerobot/policies/rtc/action_queue.py`

它已经解决了两个 RTC runtime 的核心问题：

- 同时维护 `original_queue` 和 `processed_queue`
- 提供 `get_left_over()`
- 提供 `merge(..., real_delay, action_index_before_inference)`

这正好比当前 ONNX launcher 的 `deque` 更符合 RTC 语义。

我的判断是：**这个文件比在 adapter 里自己维护 `_action_queue` 更应该复用**。

## 2.5 可以直接复用的 preflight 思路

### TRT adapter 的 `run_preflight()`

文件：

- `scripts/trt_pi_adapter.py:312-380`

它已经有：

- prefix 跑通检查
- timestep live-input 语义检查
- warmup chunk 推理
- finiteness / shape 检查

这个模式完全可以扩展成 RTC preflight，也可以移植到 ONNX adapter。

---

## 3) 哪些地方要新加 adapter/runtime state

## 3.1 adapter 内建议新增的状态

这里我建议**只加最少状态**，不要把复杂时序状态都塞进 adapter。

### 适合放在 adapter 内的状态

- `self.rtc_processor: RTCProcessor | None`
- `self._rtc_enabled(): bool`
- `self.denoise_accepts_timestep` 或 TRT 对应恒真检查

### 可选但不一定要放 adapter 内

- `self._rtc_last_debug_steps`
- `self._rtc_preflight_summary`

### 不建议放 adapter 内的状态

- `real_delay`
- `action_index_before_inference`
- async future / submission timestamp
- processed/original action queue 的 merge 状态

这些更像 launcher runtime state，而不是 policy adapter state。

## 3.2 建议新增的 runtime state

我建议引入一个显式的 RTC runtime state，而不是继续靠零散变量和 `_action_queue`。

最小需要的状态包括：

### `prev_chunk_left_over`

- 语义上应该来自 **original action chunk** 的剩余部分
- 不应该来自 postprocess 后的 robot action
- 这点很重要，因为 postprocess 可能做量纲变换 / 归一化逆变换，RTC 修正应在 policy chunk 空间完成

### `real_delay / inference_delay`

- 语义上应表示“本次 chunk 计算期间，上一批 action 实际被消费了多少步”
- 更稳妥的来源是 **action index / queue 消费数**
- 不建议只用 wall-clock latency 推算

### `execution_horizon`

- 可以默认来自 `rtc_config.execution_horizon`
- 允许 launcher 侧在极端场景下 override

### `original_queue + processed_queue`

- 最好直接复用 `ActionQueue`
- 因为它正是为 RTC leftover 和 merge 设计的

## 3.3 ONNX 异步预取路径额外需要的状态

ONNX launcher 现在有 `AsyncChunkPrefetcher`。如果接 RTC，我认为还需要给每次异步提交绑定一个 submission context：

- `left_over_snapshot`
- `action_index_before_inference`
- `observation_frame_snapshot`
- 可选的 `submitted_at_s`

原因是：

- RTC 的 `prev_chunk_left_over` 必须对应“提交这次 chunk 计算那一刻”的 queue 状态
- 不能在 future 完成时再临时取 leftover，否则语义已经漂了

换句话说，**ONNX 的异步预取如果接 RTC，必须连同 leftover snapshot 一起提交**。

## 3.4 TRT 同步路径额外需要的状态

TRT launcher 现在每步调用 `predict_action(...)`，内部继续走 adapter 的 `_action_queue`。

这条路径对普通 chunk inference 可以，但对 RTC 不自然，因为：

- `predict_action()` 只知道单步 action，不知道 chunk 开始时机
- `control_utils.predict_action()` 也没有 RTC kwargs 入口
- `PI05Policy` 本身已经说 RTC 不支持 `select_action`

所以 TRT runtime 更合理的状态边界是：

- launcher 自己维护 chunk queue / leftover
- adapter 只负责 `predict_action_chunk(...)`

也就是说，TRT launcher 最终应该更接近 ONNX launcher 的 chunk 驱动模式，而不是继续依赖 adapter 私有 `_action_queue`。

---

## 4) 配置和 CLI 怎么贯通

## 4.1 配置入口建议只认 `PI05Config.rtc_config`

这是最重要的配置原则。

建议：

- checkpoint config 中如果已有 `rtc_config`，TRT/ONNX 直接尊重；
- CLI 只做 override；
- runtime summary / preflight 打印最终生效值。

不要做：

- `rtc_config` 一套
- `--trt-rtc-*` 再来一套
- `--onnx-rtc-*` 再来一套

否则很快会和 PyTorch runtime 漂移。

## 4.2 建议新增的 CLI 覆盖项

建议在 **Torch / ONNX / TRT 三套 launcher 一起加**，名称保持一致：

- `--policy-rtc-enabled`
- `--policy-rtc-prefix-attention-schedule`
- `--policy-rtc-max-guidance-weight`
- `--policy-rtc-execution-horizon`
- `--policy-rtc-debug`
- `--policy-rtc-debug-maxlen`

如果不想加这么多，最少也该有：

- `--policy-rtc-enabled`
- `--policy-rtc-execution-horizon`
- `--policy-rtc-max-guidance-weight`

## 4.3 `apply_pi_runtime_overrides(...)` 应该成为统一注入点

当前 ONNX/TRT launcher 都有自己的：

- `apply_pi_runtime_overrides(...)`

建议在这里统一处理：

1. 如果 `policy_cfg.rtc_config is None` 且用户显式开启 RTC，则创建 `RTCConfig`
2. 把 CLI 覆盖值写回 `policy_cfg.rtc_config`
3. 打印最终配置

这样 adapter 初始化时就能直接拿到完整 `policy_cfg.rtc_config`。

## 4.4 runtime state 不应通过 CLI 传

像下面这些不应该做成 CLI：

- `inference_delay`
- `prev_chunk_left_over`
- `action_index_before_inference`

这些都应由 launcher 在真实运行时计算。

CLI 只应该配置“策略参数”，不应该配置“某一步运行时状态”。

## 4.5 preflight / summary 应补 RTC 可见性

建议 ONNX/TRT adapter 的 `runtime_summary()` 至少增加：

- `rtc_enabled`
- `rtc_prefix_attention_schedule`
- `rtc_execution_horizon`
- `rtc_max_guidance_weight`

如果 RTC 打开，preflight 里建议多做一个轻量语义检查：

- 用固定 `x_t`
- 人造一个非空 `prev_chunk_left_over`
- 设置非零 `inference_delay`
- 检查 “RTC 开 / 关” 两次输出是否不同，且结果有限

这不要求 engine 改 boundary，但能防止接了半套 RTC 仍然静默退化成普通 denoise。

---

## 5) 不建议做的事

## 5.1 不建议改 exporter / engine boundary

不要为了接 RTC 去改：

- ONNX denoise 子图输入
- TRT denoise engine 输入
- prefix_cache 输出格式

例如不建议新增：

- `prev_chunk_left_over`
- `guidance_weight`
- `inference_delay`

这些都应该留在 host-side。

原因很简单：

- 这会把 runtime 调度问题错误地下沉到 engine contract；
- 破坏当前 stage2/stage3/stage4/stage5 的稳定边界；
- 会显著扩大验证面。

## 5.2 不建议在 TRT/ONNX 两个 adapter 里各写一套 RTC 数学

这会带来三个问题：

- 和 PyTorch 行为漂移
- ONNX/TRT 两套实现彼此再漂移
- debug 成本翻倍

应该优先复用：

- `RTCConfig`
- `RTCProcessor`
- `ActionSelectKwargs`

## 5.3 不建议把 RTC 强行塞进 `select_action()` / `predict_action()`

原因有三条：

1. `PI05Policy` 已经明确说 RTC 不支持 `select_action`
2. `control_utils.predict_action()` 没有 RTC kwargs 通道
3. RTC 语义天然依赖 chunk-level leftover 和 delay，不是单步 action API 能自然表达的

因此，**不要试图给 `predict_action()` 偷偷加隐藏状态来“兼容 RTC”**。

## 5.4 不建议继续用普通 `deque` 兼容 RTC leftover

当前 ONNX launcher 的 `deque` 只保存 postprocessed actions。

这对普通 rollout 足够，但对 RTC 不够，因为 RTC 需要：

- original chunk
- processed chunk
- action 消费索引

所以不建议在 launcher 里继续堆 if/else 扩展 `deque`。应优先复用 `rtc/action_queue.py` 或者一个明确的 runtime state dataclass。

## 5.5 不建议把 RTC leftover 建立在 postprocessed robot action 上

RTC 修正发生在 policy chunk 空间，逻辑上应基于：

- model 原始输出 chunk

而不是：

- 经过 postprocessor
- 甚至经过 smooth / clamp 后的 robot command

否则会把 robot safety 层和 policy denoise 层混在一起。

## 5.6 不建议先做大规模 adapter 抽象重构

虽然 `trt_pi_adapter.py` 和 `onnx_pi_adapter.py` 有明显重复，但如果当前目标只是把 RTC 接进 runtime，我不建议先做：

- 一个巨大的 shared base adapter
- 大量抽公共类
- 顺手改完所有预热 / summary / preprocess 结构

因为这会把“RTC 接入”变成“通用 runtime 框架重构”，风险和 review 面都太大。

更现实的做法是：

- 先在两个 adapter 各自加一个小的 shared helper 或相同结构的改动
- 等 RTC 跑通后再考虑抽公共层

---

## 建议的改造落点

## 推荐方案

我认为最稳妥的方案是：

### 第一层：adapter 只负责“RTC-aware denoise loop”

在 `trt_pi_adapter.py` / `onnx_pi_adapter.py` 内：

- 保持现有 prefix / denoise engine 输入输出不变
- 只把 denoise loop 改成与 PyTorch `sample_actions()` 同型
- `predict_action_chunk(..., **kwargs)` 开始真正支持 RTC kwargs

建议新增一个很小的辅助方法，形如：

`_run_denoise_loop(x_t, shared_denoise_inputs, batch_size, num_steps, rtc_kwargs)`

这样 TRT/ONNX 两边都更清晰。

### 第二层：launcher 负责“leftover / delay / merge”

在 real-run launcher 里：

- 不再把 RTC 逻辑塞进 adapter 私有 `_action_queue`
- 显式维护 `ActionQueue`
- 每次生成 chunk 时传入：
  - `prev_chunk_left_over=action_queue.get_left_over()`
  - `inference_delay=real_delay`
  - `execution_horizon=policy_cfg.rtc_config.execution_horizon`

### 第三层：CLI 只覆盖配置，不表达运行时状态

这样最终边界会比较清楚：

- config 决定 RTC 参数
- launcher 决定 runtime state
- adapter 决定如何把 RTC 应用到 denoise loop
- engine 继续保持纯子图边界

---

## 针对 TRT / ONNX 的差异化建议

## TRT

TRT 现在更像“同步单步 select_action + adapter 内部 deque”。

如果要接 RTC，我建议：

- **不要沿用当前 `predict_action()` 路线**
- 改成和 ONNX launcher 一样显式 chunk 驱动

原因：

- 这更符合 PI05 PyTorch 的 RTC 语义
- 也更容易拿到 `inference_delay`
- 不需要去改 `control_utils.predict_action()`

## ONNX

ONNX 已经是 chunk 驱动，并且有异步预取。

所以 ONNX 更适合先落地 RTC，但要特别注意：

- `maybe_submit()` 时就要固定 `left_over_snapshot`
- future 完成后只 merge 结果，不要重新取 leftover

否则 RTC 修正会和实际 chunk 提交时机错位。

---

## 结论

从本地代码结构看，在 **不改 TRT engine / ONNX subgraph 边界** 的条件下，把 RTC 引入 PI05 TRT/ONNX runtime 是可行的，而且边界其实已经比较清楚：

- **模型 / 配置层**：现成有 `PI05Config.rtc_config` 和 `RTCProcessor`
- **adapter 层**：现成有 prefix + denoise loop，正好是最小 RTC 插入点
- **launcher 层**：现成有 chunk queue / async prefetch 雏形，但要补 RTC runtime state 语义

我最推荐的路线是：

1. 先在 `trt_pi_adapter.py` / `onnx_pi_adapter.py` 接上 `RTCProcessor`
2. 再在 TRT/ONNX launcher 中用 `ActionQueue` 取代零散 `_action_queue` / `deque`
3. CLI 统一映射到 `policy_cfg.rtc_config`
4. 不改 engine boundary，不改 exporter，不重写 RTC 数学，不把 RTC 强塞进 `select_action()`

如果只追求“最小代码面”，核心改动主要还是 adapter；  
如果追求“真实 runtime 可用且语义正确”，则 launcher 的 queue/delay 状态管理是必须补上的第二步。
