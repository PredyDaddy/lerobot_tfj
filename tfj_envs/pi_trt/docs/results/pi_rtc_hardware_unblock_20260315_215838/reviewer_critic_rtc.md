# Reviewer Critic RTC

## 执行结论

结论先说死：

- 不建议把当前状态表述成“PI05 + RTC 已可直接进入带动作的上机 smoke”。
- 可以接受的最远结论，只能到“grad context 这一类硬 blocker 已被局部收口，并且有了最小无硬件回归保护”。
- 现在最多建议进入“无动作 / 离机 / 基准台”级 smoke，不建议直接进入真实 `robot.send_action(...)` 的 torch-RTC motion smoke。

我本轮额外做了两件事来确认这一点：

1. 复核了当前 tree 的最新实现，而不是沿用上一轮文档。
2. 复跑了 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tfj_envs/pi_trt/tests/test_worker_c_torch_rtc_grad_context.py -q`，结果为 `3 passed in 2.25s`。

## 先澄清一个容易误导的点

“当前 launcher 已修，但 policy 层 `@torch.no_grad()` 仍在，所以现在还是同一个 blocker” 这个说法，按当前 tree 看已经不精确了。

- launcher 侧现在已经明确把 `RTC off` 和 `RTC on` 分开处理：
  - `RTC off` 仍走 `torch.inference_mode()`。
  - `RTC on` 改为 `torch.enable_grad()`，并且如果外层已经先进入 `torch.inference_mode()`，会直接 fail-fast。
  证据：`[run_pi05_torch_infer_so101.py:94](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L94)`、`[run_pi05_torch_infer_so101.py:98](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L98)`、`[run_pi05_torch_infer_so101.py:111](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L111)`。
- policy 侧当前也不再是“RTC 开启时整条 `predict_action_chunk` 都被 `@torch.no_grad()` 包死”的旧形态：
  - `predict_action_chunk(...)` 本身已经不是 `@torch.no_grad()` 装饰；RTC 开启时使用 `nullcontext()`，RTC 关闭时才用 `torch.no_grad()`。
  - `sample_actions(...)` 里只把 prefix 预计算包在 `torch.no_grad()` 中；真正需要给 `RTCProcessor.autograd.grad(...)` 提供图的 denoise partial，在 `rtc_guidance_enabled` 时会显式切到 `torch.enable_grad()`。
  证据：`[modeling_pi05.py:1237](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L1237)`、`[modeling_pi05.py:1241](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L1241)`、`[modeling_pi05.py:776](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L776)`、`[modeling_pi05.py:794](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L794)`、`[modeling_pi05.py:804](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L804)`。
- `RTCProcessor` 仍然依赖 `torch.autograd.grad(...)`，所以只要谁把 RTC-on 路径重新包回 `torch.inference_mode()`，就会直接坏。
  证据：`[modeling_rtc.py:211](/data/tfj/lerobot_tfj/src/lerobot/policies/rtc/modeling_rtc.py#L211)`、`[modeling_rtc.py:218](/data/tfj/lerobot_tfj/src/lerobot/policies/rtc/modeling_rtc.py#L218)`、`[test_worker_c_torch_rtc_grad_context.py:144](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_worker_c_torch_rtc_grad_context.py#L144)`。

硬结论：

- “只修 launcher 仍然会失败” 这句话如果指的是“只修 CLI / queue / refill / 日志，不处理 policy 侧 RTC 求导窗口”，结论是对的。
- 但按当前 tree，launcher 和 policy 两边都已经做了 grad-context 收口，所以这个旧 blocker 已经不是当前最严重问题。
- 当前真正的问题，是有人很容易把“grad blocker 局部修掉了”误读成“PI05 + RTC 已够资格直接上机”。

## 最严重问题

### 1. 缺少一个“跑完整 RTC 主循环但绝不发机器人动作”的安全中间层

这是我认为当前最妨碍保守上机的真实问题。

- `--dry-run` 在任何 preflight 和硬件访问之前就退出，根本碰不到 cameras、policy preflight、prefetch 主循环。
  证据：`[run_pi05_torch_infer_so101.py:682](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L682)`。
- `--preflight-only` 也会在 `robot.connect()` 和 async prefetch 主循环之前退出，仍然碰不到 `AsyncChunkPrefetcher`、`ActionQueue`、`hold_pending_async`、`sync_refill` 这些真正会在现场出问题的部分。
  证据：`[run_pi05_torch_infer_so101.py:694](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L694)`、`[run_pi05_torch_infer_so101.py:718](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L718)`。

这意味着现在从 `--preflight-only` 到“真的连机器人并 `send_action`”之间，缺了一整层应该存在的无动作验证。对真实机械臂 bring-up 来说，这个跳跃过大。

### 2. 异步 prefetch 主循环的真实状态机没有被当前验证覆盖

当前已经有的回归保护，主要锁的是 grad context 契约，不是时序状态机。

- 当前主循环里真正决定是否稳、是否卡、是否误判 RTC 工作正常的分支，是：
  - `async_collect`
  - `async_wait`
  - `hold_pending_async`
  - `sync_refill`
  证据：`[run_pi05_torch_infer_so101.py:820](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L820)`、`[run_pi05_torch_infer_so101.py:863](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L863)`、`[run_pi05_torch_infer_so101.py:899](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L899)`。
- 但这轮新增测试只覆盖了 grad context，不覆盖 `AsyncChunkPrefetcher` 主循环状态机。
  证据：`[worker_tests_validation.md:13](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_hardware_unblock_20260315_215838/worker_tests_validation.md#L13)`、`[worker_tests_validation.md:69](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_hardware_unblock_20260315_215838/worker_tests_validation.md#L69)`。
- `AsyncChunkPrefetcher` 仍然只是一个 `ThreadPoolExecutor(max_workers=1)` 包装，没有针对 CUDA + PI05 真 checkpoint + 实际相机节拍的专项验证。
  证据：`[pi05_chunk_runtime.py:644](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L644)`、`[pi05_chunk_runtime.py:664](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L664)`。

换句话说，当前通过的不是“PI05 + RTC 主循环已验证”，而是“RTC grad 上下文边界已验证”。

### 3. `policy_use_amp` 仍然是残余风险，不是可放心打开的默认项

我不把 AMP 归类成当前的硬 blocker，但我会明确把它列为上机前不能跳过的风险项。

- launcher 在 RTC-on 路径里，仍然把 `torch.autocast(...)` 叠在 grad-enabled 的上下文外层。
  证据：`[run_pi05_torch_infer_so101.py:111](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L111)`、`[run_pi05_torch_infer_so101.py:114](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L114)`。
- policy 侧 denoise partial 在 RTC guidance 有效时会重新开启 grad，所以 `autograd.grad(...)` 现在是在 autocast 选出的 dtype 语义上工作，而不是纯 FP32 语义。
  证据：`[modeling_pi05.py:801](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L801)`、`[modeling_pi05.py:804](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L804)`。
- 当前这轮验证明确没有覆盖 CUDA、AMP、真实 checkpoint。
  证据：`[worker_tests_validation.md:71](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_hardware_unblock_20260315_215838/worker_tests_validation.md#L71)`。

所以对 AMP 的正确结论不是“会继续把 grad 彻底关掉”，而是：

- 现在看不到 grad wiring 被 AMP 直接压死的证据。
- 但 AMP 仍可能改变 RTC correction 的数值稳定性、延迟统计、prefetch 阈值估计和实际 chunk ready 时序。
- 因此最保守 bring-up 顺序必须先 `RTC on + AMP off`，最后才允许 `RTC on + AMP on`。

### 4. 运行日志仍然可能低估 stall 严重性

日志现在已经比之前好，但还没有好到“足够让现场人员完全不误判”。

- `sync_refill` 的 warning 已经明确声明：这里的 `real_delay=0` 只是同步阻塞补货语义，不代表健康异步 overlap。
  证据：`[run_pi05_torch_infer_so101.py:903](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L903)`。
- 但在 `hold_pending_async` 路径下，系统会持续发送 hold action，同时 `last_real_delay` 不会随 hold step 增长而更新，它只会保留上一次 merge 时的值。
  证据：`[run_pi05_torch_infer_so101.py:881](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L881)`、`[run_pi05_torch_infer_so101.py:968](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L968)`。

这会带来一个现场误读：日志里可能同时出现较小的 `real_delay`，但系统实际上已经靠 hold action 在拖时间。也就是说，`hold_step_count` 和 `queue_underrun_count` 在当前日志里比 `real_delay` 更接近真实风险指标。

## 次要问题

### 1. 当前无硬件验证仍然以 stub / CPU 为主

- `test_worker_c_torch_rtc_grad_context.py` 很有价值，但它本质上是 stub policy + 真 `RTCProcessor` 的边界测试。
  证据：`[test_worker_c_torch_rtc_grad_context.py:40](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_worker_c_torch_rtc_grad_context.py#L40)`。
- 仓内确实存在更接近真实 policy 的 PI05 RTC 测试，但它们是本地 CUDA / OpenPI 依赖测试，不是这轮实际跑过的 launcher 证据。
  证据：`[test_pi05_rtc.py:92](/data/tfj/lerobot_tfj/tests/policies/pi0_pi05/test_pi05_rtc.py#L92)`、`[test_pi05_rtc.py:150](/data/tfj/lerobot_tfj/tests/policies/pi0_pi05/test_pi05_rtc.py#L150)`。

这意味着当前“grad blocker 已修”的可信度高于“整条 PI05 + launcher + async prefetch + hardware 路径已稳”的可信度。

### 2. 外部 `ActionQueue` 方案已经接上，但没有形成真实硬件时序证据

- 当前 torch launcher 明确使用外部 `ActionQueue`，不依赖 policy 私有 `_action_queue`。
  证据：`[run_pi05_torch_infer_so101.py:744](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py#L744)`。
- `ActionQueue.merge(...)` 的 RTC 语义本身在单元测试层面没大问题，但这不等于“实际相机/串口/PI05 延迟下的主循环行为已验证”。
  证据：`[action_queue.py:128](/data/tfj/lerobot_tfj/src/lerobot/policies/rtc/action_queue.py#L128)`、`[worker_tests_validation.md:74](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_hardware_unblock_20260315_215838/worker_tests_validation.md#L74)`。

## 误导性信号

我认为下面这些表述在联调现场最危险：

### 1. “launcher 已修，所以 torch-RTC 可上机”

这句话现在最容易把人带偏。更准确的说法应该是：

- launcher 的 grad-context 硬冲突已经被针对性修掉，并且有了最小回归测试。
- 但 async prefetch 主循环、AMP、真实 checkpoint、真实相机节拍、真实动作发送，还没有形成足够保守的闭环证据。

### 2. “policy 层 `@torch.no_grad()` 还在，所以现在还是老问题”

这句话也不够精确。按当前 tree：

- `select_action(...)` 仍是 `@torch.no_grad()`，但 RTC 本来就不支持 `select_action(...)`。
  证据：`[modeling_pi05.py:1220](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L1220)`、`[modeling_pi05.py:1223](/data/tfj/lerobot_tfj/src/lerobot/policies/pi05/modeling_pi05.py#L1223)`。
- 真正的 RTC 入口 `predict_action_chunk(...)` 当前已经不是“整条 no_grad”。

如果继续用“policy 还是 no_grad”来概括，会掩盖当前更重要的事实：现在最大的剩余风险已经不是 grad 上下文，而是缺少保守的无动作主循环验证层。

### 3. “`3 passed` 就说明 RTC 已可 smoke”

不成立。

- 当前 `3 passed` 只说明 grad context 边界没回退。
- 它没有证明 PI05 真 checkpoint 能在异步 prefetch 线程里长期稳定，也没有证明 `refill_mode`、`hold_pending_async`、`sync_refill` 在真实相机节拍下行为可接受。

## 最保守的上机顺序

这里我只给最保守顺序，不给乐观顺序。

### Stage 0: 合同和上下文边界

先跑这两个最小门槛，没过就不要谈上机：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest \
  tfj_envs/pi_trt/tests/test_worker_c_torch_rtc_grad_context.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest \
  tfj_envs/pi_trt/tests/test_worker_b_torch_rtc_contracts.py -q
```

### Stage 1: 纯导入 / 配置 / tokenizer 路径

```bash
python tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py \
  --dry-run \
  --policy-device cuda \
  --rtc-enable
```

目标不是“RTC 可跑”，而是排除 checkpoint、tokenizer、config override、参数解析这些低级错误。

### Stage 2: 真机环境但仍不进入主循环

```bash
python tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py \
  --policy-device cuda \
  --rtc-enable \
  --preflight-only
```

如果 Stage 2 都不过，后面不用继续。

### Stage 3: 离机的真实 policy smoke

这一步当前 launcher 没有现成的 “full loop but never send_action” 模式，所以不能跳过，只能用现有离机手段补。

优先选择现有真实 policy RTC 测试或等价脚本，在目标 GPU 上验证：

- `RTC on + AMP off`
- `prev_chunk_left_over` 非空
- `inference_delay` 非零
- 输出全是 finite
- 不出现 autograd / dtype / AMP 相关异常

如果现场环境满足依赖，现有最接近的仓内测试是：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest \
  tests/policies/pi0_pi05/test_pi05_rtc.py -q -k 'inference_with_prev_chunk or validation_rules'
```

这一步通过后，才能说明“policy 本体 + RTC 求导”在目标机上有基本证据。

### Stage 4: 有机器人但先不要直接上 RTC + AMP

先做最保守的 live 顺序：

1. `RTC off + AMP off`
2. `RTC on + AMP off`
3. `RTC on + AMP on`

每一步都只跑很短时间，且必须带最严限制，例如：

- 很小的 `--joint-delta-limit`
- 很小的 `--robot-max-relative-target`
- 很短的 `--run-time-s`
- 操作员手在急停上

### Stage 5: 现场判定看什么

不要只看 `real_delay`。现场优先看：

1. 有没有进入 `hold_pending_async`
2. `queue_underrun_count` 是否增长
3. `sync_refill_count` 是否增长
4. `chunk_latency_s` 是否明显大于控制节拍
5. `RTC on + AMP on` 是否比 `RTC on + AMP off` 更容易触发 hold/refill

## 是否建议进入 torch-RTC 上机前 smoke

我的建议分两层：

- 建议进入“上机前 smoke”中的无动作层：可以。
- 不建议直接进入“真实 `send_action` 的 torch-RTC motion smoke”：现在还不够保守。

真正卡住我的，不是 grad blocker 本身，而是这两个事实同时成立：

1. 当前 tree 的 grad blocker 已局部修掉，最容易让人放松警惕。
2. 但 launcher 仍缺一个完整主循环的 no-send 安全层，导致从 preflight 到 live motion 的跨度过大。

如果一定要问一句最短结论：

- 现在可以说“PI05 + RTC 的 grad 上下文不再是已知硬死点”。
- 不能说“PI05 + RTC 已经足够保守，可以直接进带动作 smoke”。
