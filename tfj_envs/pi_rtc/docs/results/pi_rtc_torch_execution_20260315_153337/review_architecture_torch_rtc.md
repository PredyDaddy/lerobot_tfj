# Torch-RTC Architecture Review

## 审查范围

本次仅做架构复审，不做代码修改。审查材料：

- `docs/results/pi_rtc_torch_execution_20260315_153337/worker_a_torch_rtc.md`
- `docs/results/pi_rtc_torch_execution_20260315_153337/worker_b_torch_tests.md`
- `scripts/run_pi05_torch_infer_so101.py`
- `scripts/pi05_chunk_runtime.py`
- `tests/test_worker_b_torch_rtc_contracts.py`

额外复核：

- 复跑 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_b_torch_rtc_contracts.py -q`，结果 `4 passed in 2.11s`
- 执行 `python -m py_compile scripts/run_pi05_torch_infer_so101.py scripts/pi05_chunk_runtime.py tests/test_worker_b_torch_rtc_contracts.py`
- 执行最小配置解析脚本，确认默认情况下即使 checkpoint `rtc_config.enabled=True`，launcher 仍会解析为 `enabled=False`

---

## 结论

结论：**纯 PyTorch 的 PI0.5 实时入口已经不是“名义接入 RTC”，而是已经切到显式 chunk runtime，并且 RTC 关键输入已经真正进入 `policy.predict_action_chunk(...) -> RTCProcessor` 这条执行链。**

但同时，结论也必须加上边界：**当前状态更适合定义为“真正可用的第一阶段实现 / 可进入受控 smoke 的版本”，而不是“已经验证闭环完成的上机版”。** 它已经明显超过半接线状态，但还没有完成主循环时序、真实 GPU/相机/串口节拍、以及 RTC 实际效果的端到端验证。

换句话说：

- 从代码连线角度看，**已接上 RTC**
- 从验证闭环角度看，**还未闭合**

---

## 重点核对结果

### 1. 是否已从旧 `predict_action(...)` 路径切到显式 chunk runtime

结论：**是，已切换。**

依据：

- `scripts/run_pi05_torch_infer_so101.py` 不再导入 `lerobot.utils.control_utils.predict_action`
- 主循环改为显式使用：
  - `AsyncChunkPrefetcher`：见 `scripts/run_pi05_torch_infer_so101.py` 第 715 行
  - `ActionQueue`：见第 725 行
  - `build_chunk_predict_kwargs(...)`：见第 747、836、894 行
  - `merge_chunk_prediction_result(...)`：见第 753、804、849、901 行
  - `action_queue.get()` 消费动作：见第 919 行
- 旧的 `predict_action(...)` 实际上仍存在于共享库 `src/lerobot/utils/control_utils.py`，其内部仍走 `policy.select_action(...)`；但 torch launcher 已不再依赖该路径
- `tests/test_worker_b_torch_rtc_contracts.py` 第 67-86 行通过 AST 静态检查明确禁止 launcher 再调用 legacy `predict_action(...)`

补充判断：

- `policy.reset()` 仍会初始化 policy 内部 `_action_queue`，但当前 launcher 的 rollout 已经不再消费 policy 私有队列，而是使用外部 `ActionQueue`
- 因此这次切换不是“包一层名字变了”，而是 rollout 机制确实改成了 chunk queue 驱动

### 2. 默认 RTC off 是否成立

结论：**成立。**

依据：

- `scripts/run_pi05_torch_infer_so101.py` 第 190-200 行 `_rtc_override_requested(...)` 将任意 `--rtc-*` override 视为显式请求
- 第 203-256 行 `resolve_rtc_runtime_config(...)` 中：
  - `enabled = bool(args.rtc_enable or override_applied)`
  - 这意味着即使 checkpoint 自带 `rtc_config.enabled=True`，若 CLI 未显式启用或 override，launcher 仍会把运行时 RTC 解析为 `False`
- 第 466-471 行 `print_summary(...)` 还专门对“checkpoint 开着但 launcher 默认仍关着”的情况打印 warning

动态复核结果：

- 对一个带 `RTCConfig(enabled=True, ...)` 的伪 `policy_cfg` 调用 `resolve_rtc_runtime_config(...)`，得到：
  - `enabled=False`
  - `checkpoint_enabled=True`
  - `override_applied=False`

需要注意：

- 这不是“沿用 checkpoint 默认值”，而是**launcher 有意覆盖成默认 off**
- 该行为满足本次审查要求，但也意味着操作员若以为“checkpoint 开着，launcher 默认就跟着开”，会上线时产生认知偏差

### 3. `ActionQueue` / `real_delay` / `refill_mode` / `sync_refill_count` 是否都进入 torch 主循环

结论：**是，都已经进入 torch 主循环；其中 `ActionQueue` 和 `real_delay` 是实义执行变量，`refill_mode` 和 `sync_refill_count` 是主循环内的控制/观测变量。**

依据：

- `ActionQueue`
  - 第 725 行实例化
  - 第 749、838、896 行参与 `build_chunk_predict_kwargs(...)`，提供 leftover / execution horizon 语义
  - 第 753、804、849、901 行通过 `merge_chunk_prediction_result(...)` 合并 chunk
  - 第 832、844、863、880、919 行参与 refill 判定和动作消费
- `real_delay`
  - 初始同步装填时显式为 `0`：第 753-758 行
  - 异步 collect / wait 时由 queue index 差值推导：第 803-806 行、第 848-851 行
  - 同步 refill 时显式为 `0`：第 900-907 行
  - 日志周期输出：第 962-963 行
- `refill_mode`
  - `initial_sync`：第 760 行
  - `async_collect`：第 807 行
  - `async_wait`：第 852 行
  - `hold_pending_async`：第 867 行
  - `sync_refill`：第 883 行
  - 周期日志输出：第 961 行
- `sync_refill_count`
  - 第 732 行初始化
  - 第 881 行递增
  - 第 885-888 行 warning 输出
  - 第 960 行周期日志输出

架构含义：

- `ActionQueue` 不是只拿来记日志，而是已经成为 rollout 真正的动作缓冲与 leftover 来源
- `real_delay` 不是只在 CLI 或 helper 中存在，而是已经决定 `ActionQueue.merge(...)` 的实际截断语义
- `refill_mode` / `sync_refill_count` 不是框架层字段，但已成为 torch 主循环对运行状态进行观测和告警的第一层信号

### 4. 当前 torch-RTC 是“真正可用的第一阶段同步版”还是只是半接线状态

结论：**更接近“真正可用的第一阶段版本”，不是半接线状态。**

理由如下：

- RTC 关键 kwargs 并未停留在 launcher 本地，而是实际透传到了模型：
  - `TorchChunkPolicyRuntime.predict_action_chunk(...)` 第 94-100 行直接转调 `policy.predict_action_chunk(...)`
  - `predict_processed_action_chunk(...)` 第 583-585 行把 `predict_kwargs` 传给 `policy.predict_action_chunk(...)`
  - `PI05Policy.predict_action_chunk(...)` 在 `src/lerobot/policies/pi05/modeling_pi05.py` 第 1232-1241 行把 kwargs 继续传给 `self.model.sample_actions(...)`
  - `sample_actions(...)` 在第 809-821 行读取 `inference_delay` / `prev_chunk_left_over` / `execution_horizon`
  - RTC guidance 最终落在 `RTCProcessor.denoise_step(...)`，见 `src/lerobot/policies/rtc/modeling_rtc.py` 第 124-248 行
- 运行态 queue 也并非伪装接线：
  - `ActionQueue.merge(...)` 在 RTC enabled 时走 `_replace_actions_queue(...)`，按 `real_delay` 丢弃已过时前缀，见 `src/lerobot/policies/rtc/action_queue.py` 第 150-168 行
  - leftover 通过 `ActionQueue.get_left_over()` 回流到下一次 `build_chunk_predict_kwargs(...)`，见 `action_queue.py` 第 113-126 行和 `pi05_chunk_runtime.py` 第 388-414 行

因此，从“参数是否真正进入 RTC 算法”和“queue 是否真正影响 rollout”这两个关键判断看，当前实现已经越过“半接线”阶段。

但为什么我仍不把它定义为“已完成上机版”：

- 现有测试没有覆盖真实主循环时序
- 没有证明在真实 PI0.5 checkpoint + 真实 GPU + 相机/串口节拍下，`hold_pending_async` / `async_wait` / `sync_refill` 的切换稳定且安全
- 没有证明启用 RTC 后的输出行为相对 RTC-off 具备预期改善，而不是只做到“参数传到了模型里”

因此最准确的定性是：

- **不是半接线**
- **是第一阶段可用实现**
- **但还不是验证闭环完成的上机完成态**

### 5. 还剩哪些阻断项会影响后续上机

以下问题仍未闭合，并会直接影响后续上机节奏：

1. **缺少主循环状态机级测试**
   - `tests/test_worker_b_torch_rtc_contracts.py` 目前只覆盖 parser、AST、helper 和假 `ActionQueue`
   - 没有用真实 `ActionQueue` + `AsyncChunkPrefetcher` 模拟 `async_collect` / `async_wait` / `hold_pending_async` / `sync_refill` 的时序切换

2. **缺少真实 PI0.5 + RTC-on 的无硬件 smoke**
   - 当前没有证明在真实 checkpoint 上，`predict_action_chunk(..., inference_delay=..., prev_chunk_left_over=..., execution_horizon=...)` 可以稳定走通
   - 也没有覆盖 CUDA AMP、线程化预取、真实 device tensor leftover 的长期稳定性

3. **缺少“RTC 真有效果”层面的验证**
   - 当前代码能证明“RTC 参数被消费”
   - 但还不能证明“RTC 输出质量/连贯性达到了设计预期”
   - 对上机而言，前者是接线完成，后者才是策略有效

4. **默认 RTC off 会造成运维认知偏差**
   - 这条不是代码 bug
   - 但如果现场按“checkpoint 开了 RTC，所以 launcher 默认也开”去理解，会直接导致错误的 smoke 结论

5. **hold / sync refill 仍可能暴露真实时序问题**
   - 当前实现的安全策略是：
     - 若队列空但异步仍在跑，则 hold 当前姿态
     - 若队列空且无异步 future，则同步补 chunk
   - 这是合理的第一阶段保护
   - 但是否会频繁进入 hold / sync refill，仍取决于真实 chunk latency、相机 fps、串口发送抖动和 GPU 负载

---

## 已闭合问题

1. **旧 `predict_action(...)` 路径已从 torch launcher 主链路移除**
2. **显式 chunk runtime 已经成为主执行路径**
3. **RTC 关键参数已真正透传到 PI05 模型和 RTCProcessor**
4. **`ActionQueue` 已进入主循环并承担 leftover / queue merge / rollout 职责**
5. **默认 RTC off 行为已落实，不是口头约定**
6. **`real_delay` / `refill_mode` / `sync_refill_count` 已纳入主循环日志与控制语义**
7. **最小契约测试在当前工作区可复跑通过**

---

## 未闭合问题

1. **没有覆盖真实主循环时序的自动化测试**
2. **没有真实 checkpoint + RTC-on 的无硬件推理 smoke 证据**
3. **没有真实机器人节拍下的 `real_delay` / refill 行为验证**
4. **没有 RTC-on 相对 RTC-off 的效果验证或退化分析**
5. **没有证明异步线程 + CUDA AMP 长时间运行的稳定性**

---

## 风险项

1. **时序风险**
   - 如果 chunk latency 高于当前 `n_action_steps` 能覆盖的窗口，系统会频繁落入 `hold_pending_async` 或 `sync_refill`
   - 这不会等价于 RTC 失效，但会显著影响控制体验和上机安全感知

2. **认知风险**
   - 默认 RTC off 是 launcher 行为，不是 checkpoint 行为
   - 现场若忘记显式传 `--rtc-enable` 或其它 `--rtc-*` override，可能误把 RTC-off 的结果当成 RTC-on smoke

3. **验证盲区风险**
   - 当前测试证明的是“代码接线”和“helper 语义”
   - 不是“真实 runtime 一定稳定”

4. **日志解释风险**
   - `sync_refill` 路径上的 `real_delay=0` 已有 warning 说明，但若只看数值不看 `refill_mode`，仍可能误读为“异步 overlap 健康”

---

## 建议

1. **建议进入上机前 smoke，但必须是受控 smoke，不应直接视为功能验收**
2. **smoke 时必须显式传入 `--rtc-enable` 或等价 `--rtc-*` override，并在日志中确认 `rtc_enabled=True`**
3. **第一轮 smoke 应优先观察 `refill_mode`、`sync_refill_count`、`hold_step_count`、`queue_underrun_count`、`real_delay`，不要只看“机器人能动”**
4. **第一轮 smoke 应设置保守动作保护**
   - 保守 `--joint-delta-limit`
   - 保守 `--gripper-delta-limit`
   - 开启适度 `--joint-action-alpha`
5. **在正式上机前，应补一个无硬件主循环状态机测试**
   - 至少覆盖 `async_collect`
   - `async_wait`
   - `hold_pending_async`
   - `sync_refill`
   - 以及 `real_delay` / queue size 的期望变化
6. **应补一个真实 checkpoint 的 RTC-on 本地 smoke**
   - 不接机器人
   - 仅验证 `policy.predict_action_chunk(...)` 在 RTC-on 时可稳定吃下 leftover / delay / horizon

---

## 是否建议进入 torch-RTC 的上机前 smoke

结论：**建议进入，但仅建议进入“上机前 smoke”，不建议把当前状态判定为可直接进入正式上机回归。**

我的判断标准是：

- 代码接线层面已经足够证明它不是半接线
- 保护策略也已具备第一阶段可用性
- 但验证缺口依然集中在最关键的实时行为层

因此建议：

- **可以进入受控 smoke**
- **不建议跳过 smoke 直接进入正式上机**
- **smoke 的验收重点应放在 refill/underrun 时序是否健康，而不是只看是否跑起来**
