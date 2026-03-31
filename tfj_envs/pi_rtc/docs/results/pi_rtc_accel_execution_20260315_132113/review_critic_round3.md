# Round 3 独立挑刺审查

## 结论

- 结论：当前版本仍然不应按“基本可上线”口径推进。
- 是否建议进入上机前 smoke 阶段：不建议。
- 核心原因不是 RTC glue 本身，而是 ONNX 启动路径仍然允许“产物看起来成套、实际并非同一套”的情况继续通过，并且部分 CLI/报告口径继续制造 TRT/ONNX 已经对齐的错觉。
- 如果后续要继续推进，必须先把 ONNX provenance/coherence gate 补到和 TRT 同等级，否则任何 dry-run / preflight / 轻量联调结论都不可靠。

## 最严重问题

### 1. ONNX artifact 仍可被静默混拼，日志却只暴露一个 `ONNX dir`

- 代码位置：`scripts/run_pi05_onnx_infer_so101.py:451`，`scripts/run_pi05_onnx_infer_so101.py:458`，`scripts/run_pi05_onnx_infer_so101.py:477`，`scripts/run_pi05_onnx_infer_so101.py:645`，`scripts/run_pi05_onnx_infer_so101.py:650`
- 问题本质：`resolve_onnx_artifacts(...)` 会分别为三个子图独立补洞，只要文件存在就接受；它没有要求三个 ONNX 来自同一目录、同一 run_dir、同一 stage2 报告。随后 `onnx_dir` 直接取 `vision_encoder` 的父目录，`print_summary()` 也只打印这一个目录。
- 这会直接制造“看起来共享了，实际上没共享”的假象。联调人员看到单个 `ONNX dir`，很容易误以为 vision/prefix/denoise 来自同一套导出产物。
- 本地最小复现已经证明这不是理论风险，而是当前实现允许的行为：

```json
{
  "onnx_dir": "/tmp/tmp7nd_eroy/bundle/onnx",
  "vision": "/tmp/tmp7nd_eroy/bundle/onnx/pi_shared_vision_encoder.onnx",
  "prefix": "/tmp/tmp7nd_eroy/bundle/artifacts/onnx/pi_shared_prefix_cache.onnx",
  "denoise": "/tmp/tmp7nd_eroy/bundle/artifacts/onnx/pi05_denoise_step.onnx",
  "all_same_parent": false
}
```

- 这类混拼一旦进入真机链路，preprocessor/postprocessor/RTC/日志都可能仍然“正常”，但行为已经不再代表任一单一导出 run，最容易误导上线判断。

### 2. ONNX `policy_dir` 与 `stage2_policy_dir` 不一致时只告警不阻断

- 代码位置：`scripts/run_pi05_onnx_infer_so101.py:495`，`scripts/run_pi05_onnx_infer_so101.py:501`，`scripts/run_pi05_onnx_infer_so101.py:678`，`scripts/run_pi05_onnx_infer_so101.py:823`，`scripts/run_pi05_onnx_infer_so101.py:829`
- 问题本质：ONNX 启动器会从 `policy_path` 加载 config / preprocessor / postprocessor / tokenizer，但 stage2 报告记录的 `policy_dir` 即使不一致，也只是打一条 warning，流程继续。
- 这意味着：
  - 前处理和后处理可以来自 checkpoint A。
  - ONNX 子图可以来自 checkpoint B。
  - 启动过程依旧可能打印 `PI05 ONNX policy OK`，继续 preflight，甚至继续进入机器人连接阶段。
- 本地最小复现已经证明 mismatch 会被直接接受：

```json
{
  "policy_dir": "/tmp/tmp2pn7fo_1/policy_a",
  "stage2_policy_dir": "/tmp/tmp2pn7fo_1/policy_b",
  "mismatch_allowed": true
}
```

- 这是当前版本里最危险的“假阳性”来源之一，因为它会让人以为自己正在验证同一 checkpoint 的完整链路，实际不是。

### 3. ONNX 路径没有与 TRT 对等的 provenance / verification gate，`policy OK` 不能当作上线信号

- 代码位置：`scripts/run_pi05_onnx_infer_so101.py:445`，`scripts/run_pi05_onnx_infer_so101.py:495`，`scripts/run_pi05_onnx_infer_so101.py:595`，`scripts/run_pi05_onnx_infer_so101.py:610`
- 对照位置：`scripts/run_pi05_trt_infer_so101.py:794`，`scripts/run_pi05_trt_infer_so101.py:1133`，`scripts/run_pi05_trt_infer_so101.py:1140`
- TRT 启动器会对 metadata、checkpoint_dir、build report、stage4/stage5 report、engine dir coherence 做阻断式检查；ONNX 启动器只要三个 ONNX 文件能找到、session contract 能过，就会打印 `PI05 ONNX policy OK`。
- 当前 ONNX 路径没有检查：
  - stage2/stage3 导出与验证是否 pass
  - report 中记录的 policy_dir/run_dir 是否和当前请求一致
  - 三个 ONNX 是否来自同一导出集合
- 所以 ONNX 的 `preflight-only`、`dry-run`、`policy OK` 目前都不能被当作“基本可上线”的证据，只能说明“文件能打开，session 名字对得上”。

## 次要问题

### 1. TRT / ONNX 的 `0` 值 CLI 语义仍未真正对齐，worker C 报告对此有误导

- 报告位置：`docs/results/pi_rtc_accel_execution_20260315_132113/worker_c_onnx_round3.md:53`，`docs/results/pi_rtc_accel_execution_20260315_132113/worker_c_onnx_round3.md:57`
- 代码位置：`scripts/run_pi05_onnx_infer_so101.py:727`，`scripts/run_pi05_onnx_infer_so101.py:731`，`scripts/run_pi05_trt_infer_so101.py:1500`，`scripts/run_pi05_trt_infer_so101.py:1503`
- ONNX 路径里，`--joint-delta-limit 0` / `--gripper-delta-limit 0` 会保留下来，并在 clamp 阶段被解释为“关闭限幅”。
- TRT 路径里，同样的参数会在 main 入口直接报错，因为它要求 `> 0`。
- 也就是说，parser 层虽然都不再把 `"0"` 吞成 `None`，但运行时语义并没有对齐。任何“TRT/ONNX 现在一致”的结论都是过度表述。

### 2. helper 新契约仍有剩余模糊地带，显式 `None` 仍无法与“未传参”区分

- 代码位置：`scripts/pi05_chunk_runtime.py:29`，`scripts/pi05_chunk_runtime.py:337`，`scripts/pi05_chunk_runtime.py:353`
- 报告位置：`docs/results/pi_rtc_accel_execution_20260315_132113/worker_a2_helper_round3_retry.md:149`，`docs/results/pi_rtc_accel_execution_20260315_132113/worker_a2_helper_round3_retry.md:181`
- 当前 helper 只给 `prev_chunk_left_over` 配了 `_UNSET` sentinel，`inference_delay` / `execution_horizon` 仍然用 `None` 表达“没传”。
- 结果是“显式传了 `None`”和“根本没传”在 helper 看起来完全一样，所谓“显式提供 RTC-only 输入时 fail fast”的契约并不完整。
- 本地最小复现：

```python
build_chunk_predict_kwargs(rtc_enabled=False, inference_delay=None) == {}
build_chunk_predict_kwargs(rtc_enabled=False, execution_horizon=None) == {}
```

- 这不是当前 TRT/ONNX 主路径的立即阻断项，但它说明 shared helper 的新契约还没有完全闭合，后续新 caller 仍可能踩进“我明明显式传了值，为什么没报错”的灰区。

### 3. ONNX 运行路径仍少一层 TRT 已有的发送前有限值保护

- 代码位置：`scripts/run_pi05_trt_infer_so101.py:1473`，`scripts/run_pi05_trt_infer_so101.py:1799`，`scripts/run_pi05_onnx_infer_so101.py:1107`
- TRT 在 `robot.send_action(...)` 前调用了 `assert_finite_robot_action(...)`；ONNX 没有对应保护。
- 这会让“两个 launcher 只是后端不同、其余安全口径一致”的判断失真。它不直接证明当前 ONNX 一定会发出非法值，但它确实说明当前安全边界并不对齐。

## 误导性信号

- `scripts/run_pi05_onnx_infer_so101.py:650` 打印的 `ONNX dir` 不是完整 provenance，只是 `vision_encoder` 的父目录；在当前实现下，它可能掩盖 prefix/denoise 来自别处。
- `scripts/run_pi05_onnx_infer_so101.py:678` 的 checkpoint mismatch 只是 warning，不会阻止后续 `PI05 ONNX policy OK`，这会放大“只是小告警、不影响联调”的错觉。
- `scripts/run_pi05_onnx_infer_so101.py:610` 的 `PI05 ONNX policy OK` 只代表 session contract 可加载，不代表来源一致、更不代表 stage3 数值验证通过。
- `docs/results/pi_rtc_accel_execution_20260315_132113/worker_c_onnx_round3.md:53` 把 ONNX 的 `"0"` 语义说成“与 TRT 当前行为对齐”，这在运行时层面不成立。
- `docs/results/pi_rtc_accel_execution_20260315_132113/worker_b_trt_round3.md:67` 仍写着 shared helper “falls back to 0 if no usable delay signal is available”，但当前 helper 代码已经改成抛 `ValueError`，说明材料自身也存在时序错位，不能直接拿来当最终事实来源。
- `docs/results/pi_rtc_accel_execution_20260315_132113/worker_a2_helper_round3_retry.md:149` 对“显式提供 RTC-only 输入就 fail fast”的表述比实际实现更绝对，显式 `None` 仍是例外灰区。

## 上线前必须补的验证

1. 必须给 ONNX 启动器补阻断式 provenance/coherence gate。
   要求至少验证 `policy_dir` 一致、三个 ONNX 属于同一导出集合、stage2/stage3 报告状态可追溯，不能再只 warning。

2. 必须补负向验证，而不是只补 happy path。
   需要明确证明以下场景会 hard fail：
   - stage2 `policy_dir` 与 `--policy-path` 不一致
   - 三个 ONNX 不在同一集合
   - report 缺失或状态不是 pass

3. 必须补跨 launcher 的 CLI 语义一致性验证。
   重点覆盖 `--joint-delta-limit 0`、`--gripper-delta-limit 0`、`--robot-max-relative-target 0`、RTC override 组合，以及 warning/log 文案是否真的表达了同一语义。

4. 必须补 shared helper 的契约边界测试。
   至少覆盖：
   - RTC-off + 显式 RTC-only 输入
   - 显式 `None` 与未传参
   - 依赖 `prediction.real_delay`
   - 缺失 action index 时是否稳定 fail fast

5. 必须补一次真实时序口径验证。
   需要在实际设备节拍下确认 `refill_mode`、`real_delay`、`hold_step_count`、`sync_refill_count` 的组合确实能映射到真实运行状态，否则现有日志只能算“看起来更详细”，还不能算“可判责”。

## 是否建议进入上机前 smoke 阶段

- 结论：不建议。
- 原因：当前最关键的问题不是“偶发小 bug”，而是 ONNX 路径仍然能在 provenance 不闭合的情况下给出正向信号。这会直接污染 smoke 结果，导致团队把一套混拼或错配产物误判成“已经可上机验证”。
- 只有在以下前提全部满足后，才建议进入上机前 smoke：
  - ONNX provenance/coherence 改为阻断式校验
  - TRT/ONNX CLI 关键语义收口
  - helper 契约边界有自动化覆盖
  - 至少完成一次针对 mismatch / mixed artifacts 的负向 smoke

## 审查口径补充

- 本审查没有修改任何实现代码。
- 本审查额外做了最小复现实验，只为确认高风险项确实是当前代码允许的行为，不是纯主观猜测。
