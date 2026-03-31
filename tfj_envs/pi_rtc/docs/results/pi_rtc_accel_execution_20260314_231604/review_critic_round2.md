# RTC 改造 Round 2 挑刺审查

结论：当前版本还不满足“RTC 主路径已打通、日志可无歧义解释、CLI 行为一致”的上线标准。最关键的问题不是某个函数写错，而是“共享 helper 已接好”和“RTC 行为可被可靠观测”这两件事，在现状里都没有被真正证明。

本审查以当前代码状态为准，不以 worker 自述为准。审查范围覆盖：

- `docs/results/pi_rtc_accel_execution_20260314_231604/worker_a_shared_helper_round2.md`
- `docs/results/pi_rtc_accel_execution_20260314_231604/worker_b_trt_launcher_round2.md`
- `docs/results/pi_rtc_accel_execution_20260314_231604/worker_c_onnx_launcher_round2.md`
- `scripts/pi05_chunk_runtime.py`
- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`

## 最严重问题

1. 共享 helper 的新 RTC API 实际没有进入任何 launcher 主执行路径，所谓“共享化”只停留在模块存在，不是运行闭环。

证据：

- `scripts/pi05_chunk_runtime.py:337-384` 定义了共享 `build_chunk_predict_kwargs(...)`。
- `scripts/pi05_chunk_runtime.py:429-467` 定义了共享 `merge_chunk_prediction_result(...)`。
- `scripts/run_pi05_trt_infer_so101.py:33-38` 只从共享层导入了 `AsyncChunkPrefetcher`、`ChunkPredictionResult`、`compute_real_delay`、`estimate_prefetch_threshold`，没有导入新 helper。
- `scripts/run_pi05_trt_infer_so101.py:1259-1295` 仍保留本地 `build_chunk_predict_kwargs(...)` / `merge_prediction_chunk(...)`，实际调用点在 `1648-1652`、`1730-1734`、`1783-1787`。
- `scripts/run_pi05_onnx_infer_so101.py:32` 同样没有导入共享 `build_chunk_predict_kwargs(...)` / `merge_chunk_prediction_result(...)`。
- `scripts/run_pi05_onnx_infer_so101.py:796-828` 仍保留本地 `build_chunk_predict_kwargs(...)` / `merge_completed_chunk(...)`，实际调用点在 `945-949`、`1028-1032`、`1068-1072`。
- 代码库内 `build_chunk_predict_kwargs(` 的调用结果显示，launcher 调到的全是各自本地函数，`scripts/pi05_chunk_runtime.py` 里的同名 helper 没有真实调用方。

风险：

- 这正是“看起来都接好了，但主路径其实没走到”的典型情况。
- worker A 的共享 API 只做了 stub 级自检，没有任何 TRT/ONNX launcher 的端到端实证。
- 只要 launcher 本地副本继续存在，后续修复会再次双线漂移，shared helper 会继续处于“文档存在、运行不经过”的假共享状态。

阻断判断：

- 这是上线阻断项。当前不能宣称“RTC glue 已共享化”。

2. ONNX launcher 的 CLI 对 `0` 值做了静默吞掉处理，和 TRT launcher 不一致，且会直接改变运行语义。

证据：

- `scripts/run_pi05_onnx_infer_so101.py:106-121` 中：
  - `parse_optional_int("0") -> None`
  - `parse_optional_float("0") -> None`
- `scripts/run_pi05_trt_infer_so101.py:145-160` 中：
  - `parse_optional_int("0") -> 0`
  - `parse_optional_float("0") -> 0.0`
- 本地最小复现已经验证：
  - ONNX: `parse_optional_int("0") = None`，`parse_optional_float("0") = None`
  - TRT: `parse_optional_int("0") = 0`，`parse_optional_float("0") = 0.0`

受影响参数不是边角料，而是实打实会影响运行和安全边界的参数：

- `--prefetch-threshold 0` 在 ONNX 里会被当成未设置，退回 `<latency-aware>`，而不是明确的 0。
- `--policy-num-inference-steps 0`、`--policy-n-action-steps 0` 在 ONNX 里不会报错，而会被静默当成“没传”。
- `--joint-action-alpha 0`、`--joint-delta-limit 0`、`--robot-max-relative-target 0` 在 ONNX 里也会被静默当成“没传”，而不是被拒绝或按 0 处理。

风险：

- 用户以为自己传入了合法边界值或故意传了非法值以触发校验，ONNX runtime 实际上悄悄换成了另一套语义。
- 这会直接制造 CLI 兼容假象：同一命令在 TRT 和 ONNX 上不是同一种含义。
- 这是 silent misconfiguration，不是普通 help 文案问题。

阻断判断：

- 这是上线阻断项。至少要先把 ONNX/TRT 的 CLI 语义收敛到一致，不能让 `0` 在一边是数值、另一边是“没传”。

3. 共享 helper 新 API 目前过于宽松，存在“调用错了但不会立刻炸，只会静默退回非 RTC 或静默记成 0”的设计风险。

证据一：

- `scripts/pi05_chunk_runtime.py:351-356` 中，`build_chunk_predict_kwargs(...)` 只要 `_resolve_rtc_enabled(...)` 结果为 false，就直接返回 `{}`。
- 本地最小复现已经验证：
  - `build_chunk_predict_kwargs(prev_chunk_left_over=tensor, inference_delay=1, execution_horizon=5)` 返回 `{}`。
  - 只有显式加 `rtc_enabled=True` 才返回真正的 RTC kwargs。

这意味着：

- 调用方即便已经显式提供了 `prev_chunk_left_over`、`inference_delay`、`execution_horizon`，只要忘了传 `rtc_enabled=True`，RTC 也会被整包静默吞掉。
- 这正是“从代码表面看像是把参数都接过去了，实际上 helper 直接把它们抹掉”的高风险 API 形状。

证据二：

- `scripts/pi05_chunk_runtime.py:442-458` 中，`merge_chunk_prediction_result(...)` 在拿不到 `action_index_after_inference`、队列也没有 `get_action_index()`、prediction 自身也没有 `real_delay` 时，会静默回落到 `0`。
- 本地最小复现已经验证：一个仅实现了 `merge(...)`、未实现 `get_action_index()` 的 queue 对象，会得到 `returned_delay=0`。

这意味着：

- 这个 helper 会把“调用方协议不完整”伪装成“真实 delay 就是 0”。
- 一旦后续别的调用方用了这个 API，日志层很难再区分“真 0”和“算不出来被默认成 0”。

阻断判断：

- 这不是当前 launcher 的已触发故障，但这是共享 API 自身的结构性风险。若准备让别的调用方复用它，必须先补契约测试，否则下一轮会继续出现“看着接上了，实际上默默退化”的问题。

## 次要问题

1. grace wait 逻辑仍有 timeout 边界竞态，可能多打一拍 hold。

证据：

- TRT: `scripts/run_pi05_trt_infer_so101.py:1739-1760`
- ONNX: `scripts/run_pi05_onnx_infer_so101.py:1035-1053`

问题形态：

- `wait_for_result(timeout)` 超时返回后，代码没有立刻再做一次 `maybe_collect()`。
- 后续分支判断使用的是 `has_future()`，不是 `has_pending()`。
- 如果 future 恰好在 timeout 刚过后完成，那么当前循环仍可能进入 hold 路径，多发一个 hold step，下个循环才 collect 结果。

风险：

- 这不会导致数据丢失，但会造成一次不必要的停顿。
- 对日志和计数器来说，这又会把“刚好完成的 chunk”误记成 underrun/hold。

2. checkpoint 自带 `rtc_config.enabled=true` 并不意味着 launcher runtime 会真的开 RTC，这条路径依然需要操作者额外知道 CLI 规则。

证据：

- TRT: `scripts/run_pi05_trt_infer_so101.py:243-286`，并在 `1347-1352` 只打印 warning。
- ONNX: `scripts/run_pi05_onnx_infer_so101.py:201-244`，并在 `629-631` 只打印 warning。

风险：

- 加载 RTC-enabled checkpoint 并不会自动进入 RTC runtime。
- 这不是实现错误，但它是非常强的操作风险。很多人天然会把“checkpoint 里 enabled=true”理解成“运行时默认就会启用”。
- 当前只是打印 warning，不是 fail-fast，也没有额外验证保证操作者真的看到了这条 warning。

## 误导性信号

1. ONNX 周期日志仍然没有 `refill_mode`，`rtc_enabled` + `real_delay` 这两个字段不足以解释实际 refill 路径。

证据：

- ONNX 周期日志字段在 `scripts/run_pi05_onnx_infer_so101.py:1119-1133`。
- ONNX sync refill 分支在 `1059-1079`，这里会把 `real_delay` 直接记成 `0`。

问题：

- 在 ONNX 里，`real_delay=0` 既可能是健康路径，也可能是阻塞式 `sync_refill`。
- 没有 `refill_mode`，就无法从周期日志区分“RTC overlap 很好”还是“异步路径已经失效、当前在同步补货”。

2. `queue_underrun_count` / `hold_step_count` 的名字会让人以为是在统计所有队列耗尽事件，但实际只统计了 hold 路径，没有统计 sync refill。

证据：

- TRT 只在 `hold_pending_async` 路径计数：`scripts/run_pi05_trt_infer_so101.py:1755-1758`
- TRT 的 `sync_refill` 路径不计数：`1773-1791`
- ONNX 只在 hold 路径计数：`scripts/run_pi05_onnx_infer_so101.py:1048-1052`
- ONNX 的 sync refill 路径不计数：`1059-1079`

问题：

- “队列空了但选择同步补货”也是队列耗尽，只是没有 hold。
- 现在的命名和计数口径会让人误以为 `queue_underrun_count=0` 就代表异步供给一直健康，实际上可能已经发生过多次阻塞式补货。

3. TRT 周期日志没有把 `rtc_enabled` 和 `real_delay` 放在同一条状态线上，`real_delay` 容易脱离上下文被误读。

证据：

- TRT 启动摘要会打印 `rtc_enabled`：`scripts/run_pi05_trt_infer_so101.py:1680-1683`
- TRT 周期日志没有 `rtc_enabled`：`1833-1846`

问题：

- `real_delay` 在 TRT 周期日志里是上下文缺失的。
- 在 RTC 关闭时，代码仍会计算 `real_delay` 这个观察值，但 `ActionQueue` 的 merge 语义并不会据此做 RTC replace。只看周期日志，容易把“观察到 delay”误读成“RTC 正在生效”。

4. ONNX 的 checkpoint warning 文案本身也不准确，仍然把 RTC 打开方式描述得过窄。

证据：

- `scripts/run_pi05_onnx_infer_so101.py:630-631` 只写了 `without --rtc-enable`

问题：

- ONNX 实际上还支持 `--rtc-enabled` 兼容别名，以及其他 `--rtc-*` override 触发启用。
- warning 文案没有把真实规则讲清楚，继续扩大了运维歧义。

5. worker 自述会让人以为“共享 helper 已经被 launcher 复用”，但当前代码并不支持这个结论。

问题：

- 这不是代码运行时信号，而是 rollout 信号误导。
- 当前代码状态仍然是“共享模块存在，但关键 RTC kwargs 构造和 merge 逻辑仍分散在各 launcher 本地副本里”。

## 上线前必须补的验证

1. 必须补一条“共享 helper 真实接线”验证。

要求：

- 要么 launcher 真实导入并调用 `scripts/pi05_chunk_runtime.py` 的 `build_chunk_predict_kwargs(...)` / `merge_chunk_prediction_result(...)`，并用测试锁死。
- 要么明确承认本轮没有完成共享化，撤回相关口径，避免继续把“有模块”说成“已接通”。

2. 必须补 ONNX/TRT CLI 契约一致性测试，重点覆盖 `0` 值。

最低覆盖项：

- `--prefetch-threshold 0`
- `--policy-n-action-steps 0`
- `--policy-num-inference-steps 0`
- `--joint-action-alpha 0`
- `--joint-delta-limit 0`
- `--robot-max-relative-target 0`
- `--rtc-enable`
- `--rtc-enabled`
- 仅传 `--rtc-execution-horizon 8`

验证目标：

- ONNX/TRT 对相同参数给出相同的解析结果、相同的报错策略，不能再出现一边吞 0、一边保留 0。

3. 必须补 RTC 路径真值表测试，不允许只做 `py_compile` 和 `--help`。

最低要覆盖的路径：

- RTC off
- RTC on + `async_collect`
- RTC on + `async_wait`
- RTC on + `hold_pending_async`
- RTC on + `sync_refill`

验证目标：

- 每条路径都要能确定 `predict_kwargs`、queue merge 语义、`real_delay`、`refill_mode`、`queue_underrun_count`、`hold_step_count` 是否符合预期。

4. 必须补“checkpoint 自带 RTC enabled”真值表测试。

最低组合：

- checkpoint `rtc_config.enabled=true`，无任何 CLI
- checkpoint `rtc_config.enabled=true`，传 `--rtc-enable`
- checkpoint `rtc_config.enabled=true`，只传 `--rtc-execution-horizon`

验证目标：

- `ResolvedRTCRuntimeConfig`
- adapter 内部 `_rtc_enabled()` 实际状态
- 首次/后续 chunk 的 `predict_kwargs`
- 周期日志里的 RTC 状态信号

这四者必须一致，不能只是在 summary 里“看起来 resolved 对了”。

5. 必须补 shared helper 契约测试，专门打现在这些静默退化点。

最低覆盖项：

- 显式传了 `prev_chunk_left_over/inference_delay/execution_horizon`，但没传 `rtc_enabled=True`
- 同时传 `predicted_delay_steps` 和 `inference_delay`
- queue 只有 `merge(...)` 没有 `get_action_index()`

验证目标：

- 这些情况要么 fail-fast，要么至少有明确告警；不能继续静默回落成 `{}` 或 `real_delay=0`。

6. 必须至少做一次强制慢推理场景的实跑或可回放 mock，专门压出 hold 和 sync refill。

验证目标：

- 不是只验证“能跑”，而是验证日志有没有把坏路径诚实地报出来。
- 如果 ONNX 仍然不输出 `refill_mode`，那就不能声称“日志足够解释 RTC 行为”。

## 审查结论

按最苛刻标准，这一轮 RTC 改造还停留在“局部逻辑改善”和“部分字段补齐”，没有达到“主路径共享化完成、CLI 契约一致、日志可用于无歧义判责”的程度。

如果现在上线，最大风险不是立刻崩，而是更糟的那种情况：

- 主路径并没有真的走共享 helper，但材料会让人以为已经走了。
- ONNX CLI 会静默吞掉一部分用户输入，但表面上不会报错。
- 日志里会出现看似合理的 `real_delay` / `queue_underrun_count` / `rtc_enabled`，但它们不足以准确说明当时到底处于哪条 refill 路径。

这类问题最容易把联调时间浪费在错误方向上，因此我不建议按“本轮 RTC 改造已收敛”的口径推进上线。
