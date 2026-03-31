# Torch RTC 补位挑刺审查报告

审查范围：
- `docs/results/pi_rtc_torch_execution_20260315_153337/worker_a_torch_rtc.md`
- `docs/results/pi_rtc_torch_execution_20260315_153337/worker_b_torch_tests.md`
- `scripts/run_pi05_torch_infer_so101.py`
- `scripts/pi05_chunk_runtime.py`
- `tests/test_worker_b_torch_rtc_contracts.py`

审查结论：
- 当前纯 PyTorch RTC 入口还不具备“已可进入 torch-RTC 上机前 smoke”的最低可信度。
- 不是“还有点风险”的级别，而是存在一个足以让 `--rtc-enable` 在真正进入 RTC correction 时直接失效的硬冲突；同时，现有日志和材料会让联调人员把“RTC 开关接上了”误读成“RTC 语义已经稳定进入主循环”。

## 最严重问题

- `scripts/run_pi05_torch_infer_so101.py:87-100` 的 `TorchChunkPolicyRuntime.predict_action_chunk()` 把整段策略调用包在 `torch.inference_mode()` 中；但 PI05 的 RTC 真正生效时，会在 `src/lerobot/policies/pi05/modeling_pi05.py:809-821` 进入 `RTCProcessor.denoise_step(...)`，而该实现依赖 `src/lerobot/policies/rtc/modeling_rtc.py:211-218` 的 `torch.enable_grad()` 和 `torch.autograd.grad(...)` 计算 correction。两者语义硬冲突。我做了最小 PyTorch 语义复现，在当前环境 `torch 2.7.1+cu126` 下，`inference_mode` 嵌套 `enable_grad` 会直接报 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`。这意味着当前 launcher 即使表面接通 RTC，也很可能在第一个真正带 prefix leftover 的 chunk 上炸掉。最危险的点是：warmup 初始 chunk 仍可能正常，问题会延后到“看起来已经跑起来了”的第二段或后续异步 chunk 才暴露。

- 现在仍然存在“看起来接了，实际上 RTC 语义没真正进入主循环”的路径，而且不止一条。`scripts/pi05_chunk_runtime.py:388-414` 只有在 `ActionQueue.get_left_over()` 返回非空 leftover 时才会把有效 prefix 送入模型；一旦 leftover 为空，`prev_chunk_left_over` 会被归一化成 `None`。而 `src/lerobot/policies/rtc/modeling_rtc.py:166-169` 明确规定：`prev_chunk_left_over is None` 时直接走原始 denoise，不做 RTC guidance。当前主循环的 `initial_sync` 和 `sync_refill` 都是阻塞补货路径，队列耗尽后极易出现 `rtc_enabled=true`、`refill_mode=sync_refill`、`real_delay=0`，但模型端实际上根本没有进入 RTC correction，只是在 RTC 配置打开的前提下重新做了一次普通 chunk 生成。也就是说，“同步 refill 版 RTC”这个说法本身就带误导色彩。

- 现有材料没有一条证据真正执行过“RTC enabled + 非空 leftover + policy.predict_action_chunk(... RTC kwargs ...)”。`worker_a_torch_rtc.md:44-67` 的所谓自检只有 `py_compile`、`--help` 和 parser 解析；`worker_b_torch_tests.md:19-31` 也明确承认没有覆盖真实 `ActionQueue` / `RTCProcessor` 对接、没有覆盖 async prefetch 时序、没有覆盖 CUDA AMP。`tests/test_worker_b_torch_rtc_contracts.py:39-177` 的 4 个测试，本质上只是 parser、AST、helper 拼装和 fake queue spy。它们足以证明“参数名和 helper 接口还在”，远远不够证明“RTC 语义已经在线路里跑通”。

## 次要问题

- `policy_use_amp` 现在非常容易制造“已经具备 CUDA 优化运行语义”的错觉。`scripts/run_pi05_torch_infer_so101.py:402` 只是把这个 flag 写进 `policy_cfg.use_amp`，真正使用它的只有 launcher 本地 wrapper `scripts/run_pi05_torch_infer_so101.py:688-693`，而不是 PI05 policy 自身。也就是说，这不是一个已经被 PI05 主实现天然支持并验证过的策略能力，只是 launcher 额外包了一层 autocast。现有材料没有任何一次真实 chunk 执行去证明 AMP on/off 都能稳定跑。

- `policy_use_amp` 的日志还是“声明式”的，不是“生效式”的。`scripts/run_pi05_torch_infer_so101.py:459-465` 和 `scripts/run_pi05_torch_infer_so101.py:433-437` 会打印 `use_amp=True`，但 `TorchChunkPolicyRuntime` 实际只在 `device.type == "cuda"` 时才进入 autocast。换句话说，在非 CUDA 设备上它完全是 no-op，但日志不会说明“这个 AMP 现在并未生效”。

- 当前日志里最关键的延迟量其实不是一个。模型调用时吃进去的是 `predicted_delay_steps`，来源于 `scripts/run_pi05_torch_infer_so101.py:824-840` 的估计值；而日志打印的是 merge 之后回填的 `real_delay`，来源于 `scripts/pi05_chunk_runtime.py:481-548` 的后验计算。这两个值语义完全不同，但日志只暴露后者，不暴露前者。联调时如果发现“RTC correction 效果怪异”，操作者无法知道模型当时到底以为自己的 `inference_delay` 是多少。

- `real_delay` 在 `rtc_enabled=false` 时仍然会被计算并打印，但 `ActionQueue` 的 non-RTC 路径 `src/lerobot/policies/rtc/action_queue.py:150-155` / `src/lerobot/policies/rtc/action_queue.py:176-197` 实际会忽略这个 delay，只做 append continuity。也就是说，`real_delay` 在非 RTC 模式下只是观测值，不是控制语义；当前日志没有明确告诉操作者这一点。

- 周期日志中的 `refill_mode` 和 `real_delay` 其实都是“最近一次 chunk 合并/补货事件”的残留状态，不是“这一拍主循环当前正在走的路径”。对应变量在 `scripts/run_pi05_torch_infer_so101.py:732-734` 初始化，在 `scripts/run_pi05_torch_infer_so101.py:801-808`、`scripts/run_pi05_torch_infer_so101.py:847-852`、`scripts/run_pi05_torch_infer_so101.py:867`、`scripts/run_pi05_torch_infer_so101.py:883` 更新，但日志在 `scripts/run_pi05_torch_infer_so101.py:950-967` 只是周期性打印最后一次值。这会让人把“last event”误读成“current state”。

- `preflight_policy()` 的输出语气过满。`scripts/run_pi05_torch_infer_so101.py:425-444` 打印 `PI05 PyTorch policy OK: device=..., use_amp=..., rtc_enabled=...`，但这个阶段只验证了模型能加载，根本没有执行一拍 `predict_action_chunk`，更没有执行 RTC-guided chunk。这种日志很容易被现场人员当成“RTC 至少已经过一拍 smoke”。

- `AsyncChunkPrefetcher` 的线程化 chunk 推理确实已接入主循环，但现有验证完全没有触及“真实 policy + 真实 pre/postprocessor + 后台线程 + CUDA/AMP”的组合。`worker_a_torch_rtc.md:69-73` 已经承认这点，但报告正文前半段的表述仍然会让人低估这个缺口。

## 误导性信号

- `worker_a_torch_rtc.md:22-42` 把 RTC 接线路径写得像是“入口、主循环、queue、发送路径都已闭环”，但后面的自检 `worker_a_torch_rtc.md:44-67` 只证明 CLI 和 parser 层面没有坏，完全没有证明 RTC-guided chunk 真执行过。

- `rtc_enabled=true` 现在最多只代表“本轮 runtime 允许构造 RTC kwargs”，不代表“本轮 chunk 一定发生了 RTC guidance”。只要 leftover 为空，模型就会退回普通 denoise。当前日志没有任何一项直接告诉你：这次 `predict_action_chunk(...)` 到底有没有拿到非空 `prev_chunk_left_over`。

- `refill_mode=sync_refill` 和 `real_delay=0` 很容易被读成“同步 refill 很干净、实时延迟控制良好”。严格来说，这个组合更接近“队列已经见底，只能阻塞式重算；而且这条路径上多半没有 RTC prefix guidance”，不是健康的 async overlap 信号。

- `use_amp=True` 容易被误读为“AMP 已生效且已验证”，但当前它既可能在非 CUDA 下根本没生效，也没有任何一次实际 chunk 级 smoke 去证明它与 RTC 主路径相容。

- `4 passed` 容易被误读为“torch RTC 契约已基本齐”，但 `tests/test_worker_b_torch_rtc_contracts.py` 实际没有一条测试触达真实 PI05 policy 的 RTC 路径，更没有触达 `RTCProcessor.denoise_step(...)`。

## 上线前必须补的验证

- 必须先做一个无硬件、真 checkpoint 的“两段 chunk”最小 smoke：第一段 warmup，第二段强制带非空 leftover、`rtc_enabled=true`，并且真正执行到 `policy.predict_action_chunk(... prev_chunk_left_over=..., inference_delay=..., execution_horizon=...)`。只有它能证明 RTC correction 没被 launcher 包装层吞掉。

- 必须单独验证 `torch.inference_mode` / `torch.enable_grad` / RTC correction 的兼容性结论，而不是再靠 parser smoke 或 helper test 代替。当前我已经在本地最小复现里证明两者冲突，除非这点被用真实 PI05 chunk 路径推翻，否则不能进入上机前 smoke。

- 必须补一个 `policy_use_amp` 维度的最小矩阵：`rtc_enabled` on/off × `policy_use_amp` on/off，至少各跑一拍真实 `predict_action_chunk`。目标不是看吞吐，而是确认“能跑、不报错、输出 finite、行为语义与日志一致”。

- 必须补一个 `sync_refill` 强制路径验证，并把结论写清楚：当队列耗尽时，这条路径到底是不是 RTC-guided generation；如果不是，就不能再用“同步 refill 版 RTC 已接通”这种表述。

- 必须补一个日志可信度验证：至少要能从日志直接判断 `rtc_enabled` 是配置态还是生效态、模型实际收到的 `inference_delay` 是多少、`prev_chunk_left_over` 是否非空、当前 `refill_mode` 是当前态还是最近事件、`real_delay` 是测量值还是控制输入。现在这些信息不足以支撑现场联调。

- 必须补一个真实 `ActionQueue` + `AsyncChunkPrefetcher` 的时序 smoke，覆盖 `async_collect`、`async_wait`、`hold_pending_async`、`sync_refill` 四条关键分支。`worker_b_torch_tests.md:26-31` 已经明确承认这部分没测到，所以不能把“4 passed”当成时序可信度。

## 是否建议进入 torch-RTC 上机前 smoke

- 不建议。

- 原因不是“还差一些润色”，而是“当前存在高概率在真正 RTC correction 首次发生时直接出错的硬语义冲突”，并且日志/材料还会把配置接线成功误导成 RTC 语义已经进入主循环。

- 只有在下面两个前置条件都满足后，才有资格谈“进入 torch-RTC 上机前 smoke”：
  1. 已经用真 checkpoint 跑通过至少一个带非空 leftover 的 RTC-guided 第二段 chunk。
  2. 已经明确证明 `policy_use_amp`、外部 `ActionQueue`、同步 refill、`real_delay=0` 这些信号不会再把联调人员带到错误结论上。
