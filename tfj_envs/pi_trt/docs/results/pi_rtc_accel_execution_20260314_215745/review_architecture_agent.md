# PI05 RTC 架构审查报告

## 结论

本次实现属于“部分完成，未达到设计/计划闭环”的状态。

- `TRT` 主链已经基本满足技术方案的主方向：
  - 默认 `RTC off`
  - 入口已经从 `select_action()` 切到显式 chunk runtime
  - 已接入 `ActionQueue`
  - 已把 `prev_chunk_left_over / inference_delay / execution_horizon` 传入 adapter
- 但从“设计文档 + 实施计划 + 当前代码”三方对齐来看，整体仍不能判定为“TRT/ONNX 两条链路都完成 RTC 接入”。
- 最主要的缺口在 `ONNX launcher`：
  - 没有完成 RTC CLI 贯通
  - 没有接入 `ActionQueue`
  - 没有 real delay merge
  - 没有复用共享 `pi05_chunk_runtime.py`
- 此外，`TRT/ONNX` 两条 launcher 都存在 `sync_refill_timeout_s` 等待逻辑与设计语义不完全一致的问题，会削弱 runtime hiding 效果。

综合判断：

- 若只看 `TRT launcher`，实现已经接近设计目标。
- 若按实施计划要求审查整个交付，当前版本不应被标记为“全部完成”。

## 通过项

- `adapter` 层 RTC 接入方向正确，符合技术方案第 5.1 节与 Plan 的 Commit B。
  - `scripts/trt_pi_adapter.py:86` 持有 `RTCProcessor`
  - `scripts/trt_pi_adapter.py:241` 定义 `_rtc_enabled()`
  - `scripts/trt_pi_adapter.py:510` 开始解析 `prev_chunk_left_over / inference_delay / execution_horizon`
  - `scripts/trt_pi_adapter.py:578` 在 denoise loop 中通过 `rtc_processor.denoise_step(...)` 包裹 TRT 单步 denoise
  - `scripts/onnx_pi_adapter.py:97`
  - `scripts/onnx_pi_adapter.py:170`
  - `scripts/onnx_pi_adapter.py:315`
  - `scripts/onnx_pi_adapter.py:384`
  - 这说明两条 adapter 都已经具备 RTC 语义入口，且没有把 `ActionQueue` 错塞进 adapter，边界判断是对的。

- `TRT launcher` 已经满足“不要再走 `select_action()`”这一核心架构要求，符合技术方案第 7.1 节与 Plan 的 Commit C。
  - `scripts/run_pi05_trt_infer_so101.py` 中没有再走 `control_utils.predict_action()` / `policy.select_action()`
  - `scripts/run_pi05_trt_infer_so101.py:1357` 显式走 `predict_action_chunk(...)`
  - `scripts/run_pi05_trt_infer_so101.py:1842` 使用 `ActionQueue`
  - `scripts/run_pi05_trt_infer_so101.py:1403` 定义 `merge_completed_chunk(...)`

- `TRT launcher` 默认 `RTC off`，符合技术方案第 6.3 节。
  - `scripts/run_pi05_trt_infer_so101.py:247` 到 `:300` 通过 `resolve_rtc_runtime_config(...)` 在 runtime 层统一解析
  - `scripts/run_pi05_trt_infer_so101.py:257` 只有显式 `--rtc-*` 参数才启用
  - `scripts/run_pi05_trt_infer_so101.py:1564` 在 checkpoint 自带 RTC 但 launcher 默认关闭时给出 warning
  - 这条链路满足“保留 baseline，不默认偷偷开 RTC”的设计要求。

- `TRT launcher` 保留了安全逻辑在 launcher，而不是塞到 adapter 或 queue 内，符合技术方案第 7.3 节。
  - `scripts/run_pi05_trt_infer_so101.py:1653` smoothing
  - `scripts/run_pi05_trt_infer_so101.py:1688` delta clamp
  - `scripts/run_pi05_trt_infer_so101.py:1725` finite check
  - `scripts/run_pi05_trt_infer_so101.py:2010` 之后仍在 send_action 前执行这些逻辑

- leftover 来源在 `TRT` 链路上是对的，符合技术方案第 7.4 节。
  - `scripts/run_pi05_trt_infer_so101.py:1344` 通过 `ActionQueue.get_left_over()` 取 leftover
  - `scripts/run_pi05_trt_infer_so101.py:1405` merge 时同时传入 `original_actions` 与 `processed_actions`
  - `src/lerobot/policies/rtc/action_queue.py:113` 到 `:126` 明确 leftover 来自 `original_queue`

- 共享 helper 文件已经创建，方向本身符合 Plan 的 Commit A。
  - `scripts/pi05_chunk_runtime.py:29` 定义了统一 `ChunkPredictionResult`
  - `scripts/pi05_chunk_runtime.py:117` 定义统一 `predict_processed_action_chunk(...)`
  - `scripts/pi05_chunk_runtime.py:200` 定义统一 `AsyncChunkPrefetcher`

## 问题项

- 阻塞级问题 1：`ONNX launcher` 没有完成 RTC 接入，和设计文档/实施计划明显不一致。
  - 技术方案第 5.3 节明确把 `run_pi05_onnx_infer_so101.py` 列为 launcher 层改造目标。
  - Plan 的 Commit D 和 Commit E 明确要求：
    - 共享 runtime helper
    - 对齐 queue / metrics 语义
    - ONNX launcher 侧 RTCConfig 贯通
  - 当前实现中：
    - `scripts/run_pi05_onnx_infer_so101.py:114` 到 `:206` 的 parser 没有任何 `--rtc-*` 参数
    - `scripts/run_pi05_onnx_infer_so101.py:411` 到 `:429` 只处理 `n_action_steps / num_inference_steps`，没有解析或注入 `RTCConfig`
    - `scripts/run_pi05_onnx_infer_so101.py:884` 仍使用本地 `deque`
    - `scripts/run_pi05_onnx_infer_so101.py:942` 到 `:999` 只是旧式 prefetch + sync refill，没有 `ActionQueue.merge(...)`
    - `scripts/run_pi05_onnx_infer_so101.py:1033` 到 `:1044` 日志里没有 `real_delay`
  - 与此相对，`onnx_pi_adapter.py` 已经支持 RTC kwargs，但 launcher 从未提供这些输入。
  - 结果是：`ONNX` 链路并没有真正完成“RTC-aware runtime”。

- 严重问题 2：`ONNX launcher` 不能保证“默认 RTC 关闭”，与设计第 6.3 节不一致。
  - 设计要求是“第一阶段默认仍保持 RTC off，只有显式 RTC CLI 才进入 RTC-aware runtime”。
  - 但当前 `ONNX launcher` 没有任何 runtime 级 RTC override：
    - `scripts/run_pi05_onnx_infer_so101.py:432` 到 `:438` 直接加载 checkpoint config
    - `scripts/run_pi05_onnx_infer_so101.py:411` 到 `:429` 没有重写 `policy_cfg.rtc_config`
    - `scripts/onnx_pi_adapter.py:97` 到 `:99` 若 checkpoint 带 `rtc_config`，就会实例化 `RTCProcessor`
    - `scripts/onnx_pi_adapter.py:170` 到 `:171` 只看 config 是否 enabled
  - 这意味着 ONNX 路径是否开 RTC 取决于 checkpoint 内容，而不是 launcher 显式控制。
  - 这直接破坏了“同一套 engine / 同一套 launcher，只切 runtime orchestration”的基线设计。

- 严重问题 3：两条 launcher 都没有真正接入共享 `pi05_chunk_runtime.py`，Step 0 schema 也没有闭环。
  - Plan 的 Step 0 明确要求先固定统一 runtime schema，Commit A 明确要求抽 shared helper。
  - 但当前代码里：
    - `scripts/run_pi05_trt_infer_so101.py:129` 自己又定义了一套 `ChunkPredictionResult`
    - `scripts/run_pi05_trt_infer_so101.py:1414` 自己又定义 `AsyncChunkPrefetcher`
    - `scripts/run_pi05_onnx_infer_so101.py:73` 自己又定义了一套更简化的 `ChunkPredictionResult`
    - `scripts/run_pi05_onnx_infer_so101.py:722` 自己又定义 `AsyncChunkPrefetcher`
    - 两个 launcher 都没有 import `scripts/pi05_chunk_runtime.py`
  - 结果是三套 schema 并存：
    - helper 版
    - TRT launcher 本地版
    - ONNX launcher 本地版
  - 这和 Step 0“后续 `TRT/ONNX launcher` 和 benchmark 统一消费一份 schema”的目标不一致。

- 严重问题 4：`sync_refill_timeout_s` 的等待逻辑与注释/帮助语义不一致，可能导致不必要的 hold 或 sync fallback。
  - `scripts/run_pi05_trt_infer_so101.py:474` 到 `:477` 的帮助文案说它是“collecting a just-finished async chunk before a synchronous refill”的 grace period。
  - 但主循环里写成了：
    - `scripts/run_pi05_trt_infer_so101.py:1949`
    - 仅在 `action_queue.empty() and not prefetcher.has_pending()` 时才 `wait_for_result(...)`
  - `scripts/run_pi05_onnx_infer_so101.py:172` 到 `:175` 也有同样语义说明。
  - 但它的主循环同样写成：
    - `scripts/run_pi05_onnx_infer_so101.py:965`
    - 仅在 `not prefetcher.has_pending()` 时才等
  - 这会导致真正“future 还在 pending，但可能在 timeout 窗口内完成”的场景根本不等待，直接进入：
    - TRT 的 hold fallback 或 sync refill 分支
    - ONNX 的 hold fallback 或 sync refill 分支
  - 从架构语义上看，这个分支条件是反的，`sync_refill_timeout_s` 大概率没有发挥设计预期的作用。

- 中等级问题 5：`TRT` 虽已完成主链改造，但 CLI 命名与文档/计划不一致。
  - 设计文档第 6.2 节与 Plan 的 Commit E 写的是 `--rtc-enabled`
  - 代码实现为 `scripts/run_pi05_trt_infer_so101.py:198` 的 `--rtc-enable`
  - 虽不影响代码运行，但属于交付接口与设计文档不一致，后续文档和操作命令容易错位。

- 中等级问题 6：开发执行链路的证据不完整，Worker C 报告没有形成有效交付闭环。
  - `docs/results/pi_rtc_accel_execution_20260314_215745/worker_c_launchers_report.md:1` 到 `:16` 只有占位说明
  - 没有实际改动说明
  - 没有自检结果
  - 没有对 TRT/ONNX 两条 launcher 完成度的独立说明
  - 从“执行清单一致性”的角度，这意味着 launcher 改造缺少正式交付证据。

## 风险项

- `TRT/ONNX` runtime 已经开始分叉。
  - `TRT` 走 `ActionQueue + RTC runtime config + merge_completed_chunk`
  - `ONNX` 仍停留在旧式 `deque + async prefetch`
  - 继续演进下去会让两条链路越来越难比较，违背“共享 runtime helper、统一 runtime 概念”的初衷。

- ONNX 路径的默认行为存在不可预期性。
  - 由于它不做 runtime override，若 checkpoint 带 RTC 配置，ONNX 行为可能直接变成 RTC on
  - 这会让 benchmark、问题复现、线上命令都缺乏稳定基线

- `sync_refill_timeout_s` 逻辑偏差会放大性能判断误差。
  - 如果 pending future 本可在 timeout 内返回，但代码没有等，就会额外产生 hold step 或 sync refill
  - 这会让后续控制环 benchmark 夸大 underrun / hold，误判 RTC 真实收益

- 共享 helper 未被采用，后续 benchmark 很容易继续各写各的。
  - 这与 Plan Step 0 已经明确规避的风险一致
  - 后续若补 benchmark，字段口径很可能继续漂移

- 本次审查未做硬件运行验证。
  - 这是按任务要求刻意不碰硬件
  - 因此结论只覆盖架构一致性和静态执行路径，不覆盖真机时序表现

## 建议

- 第一优先级先补 `ONNX launcher`，否则不要把本轮交付标记为“TRT/ONNX 双链路 RTC 接入完成”。
  - 至少补齐：
    - RTC CLI
    - runtime override
    - `ActionQueue` 或与之等价的 original/processed 双队列结构
    - `real_delay` merge
    - 对齐 TRT 的核心指标输出

- 第二优先级收敛到共享 `pi05_chunk_runtime.py`。
  - 建议让 `TRT/ONNX launcher` 都直接 import helper
  - 不要再在 launcher 里保留本地 `ChunkPredictionResult / AsyncChunkPrefetcher / estimate_prefetch_threshold`
  - 否则 Step 0 的统一 schema 事实上没有落地

- 第三优先级修正 `sync_refill_timeout_s` 分支条件。
  - 设计语义应该是：
    - queue 空
    - 但已经存在 future
    - 先给 future 一个 timeout 窗口
    - 再决定 hold 或 sync refill
  - 当前实现没有做到这一点

- 第四优先级统一文档与 CLI 命名。
  - 选一个最终名称：
    - `--rtc-enable`
    - 或 `--rtc-enabled`
  - 然后同步设计文档、README、命令示例，避免后续操作层误用

- 第五优先级补齐 Worker C 报告。
  - 至少要写清：
    - TRT launcher 实际改了什么
    - ONNX launcher 实际改了什么
    - 哪些计划项完成
    - 哪些计划项未完成
  - 当前 launcher 代码与交付报告证据链不对齐

## 总体判断

当前代码库中：

- `TRT RTC-aware chunk runtime` 基本成形
- `ONNX RTC runtime` 仍未闭环
- `shared runtime helper` 已创建但未真正落地到 launcher

因此本审查结论是：

- 不建议把当前状态认定为“按设计完整交付”
- 建议认定为“TRT 主链基本完成，ONNX 和共享化收尾未完成，需继续整改后再做下一轮评审”
