# PI05 RTC Round 4 Architecture Review

本报告以当前代码状态为准，不以 worker 自述为准。复核范围限定为以下材料：

- [review_critic_round3.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_accel_execution_20260315_132113/review_critic_round3.md)
- [worker_a_onnx_provenance_round4.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_accel_execution_20260315_140805/worker_a_onnx_provenance_round4.md)
- [worker_b_tests_round4.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_accel_execution_20260315_140805/worker_b_tests_round4.md)
- [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py)
- [pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py)
- [test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py)

本次额外独立执行了：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex python -m pytest --confcutdir=/data/tfj/lerobot_tfj/tfj_envs/pi_trt tests/test_round4_contracts.py -q`，结果 `8 passed in 2.14s`
- 最小负向实验 1：构造 mixed ONNX artifacts，显式传 `stage2_report_path`，`resolve_onnx_artifacts(...)` 返回 `ValueError: Refusing to launch PI05 ONNX runtime without coherent stage2/stage3 provenance`
- 最小负向实验 2：解析 `--joint-delta-limit 0`，当前 ONNX launcher 的运行前 guard 会进入 fail-fast 分支
- 最小入口实验：直接把 `--onnx-path` 指向单个 `.onnx` 文件且不显式传 `--onnx-stage2-report-path`，当前实现会抛 `JSONDecodeError`，这一点是本轮新发现的未闭合问题

## 结论

- Round 3 critic 点名的主阻断项已经大体闭合：ONNX provenance/coherence 现在确实进入了阻断式 gate；mixed artifacts / `policy_dir` mismatch 的假阳性路径，在标准入口口径下已被 hard-fail 收口；`0` 值 CLI 尾项与发送前 finite guard 也已经补到运行前/发送前安全边界。
- 当前版本不应再按 Round 3 的“不建议进入 smoke”原判直接沿用，但也不能写成“全部整改完成”或“可按基本可上线口径签收”。
- 现在最准确的项目状态标签是：`受控 pre-smoke 候选（Conditional Go），非最终签收`。
- 是否建议进入上机前 smoke：`有条件建议进入`。

条件边界：

- 仅建议按 ONNX 目录或显式 `stage2_export_onnx.json` 入口推进，不建议在本轮 smoke 中使用“单个 `.onnx` 文件 + 自动发现 report”的入口。
- smoke 目标应限定为“验证当前 launcher/runtime boundary 与日志解释链是否按预期工作”，而不是宣称 artifact/CLI/test 全面闭合。

## 已闭合问题

- ONNX provenance/coherence 已变为阻断式 gate，而不是 warning 式提示。当前 `resolve_onnx_artifacts(...)` 会在缺失 `stage2/stage3` 报告、gate 非 `pass`、`policy_dir/run_dir/onnx_dir` 不一致、三件套不在同一 coherent 目录、`stage3` 记录与 `stage2` provenance 不一致等情况下直接 `ValueError` 阻断，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L522)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L712)。
- Round 3 critic 指出的 mixed artifacts 假阳性，在标准入口下已闭合。核心阻断发生在 coherent 目录判定与 `stage3` 对 `stage2` 回指一致性检查，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L626)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L645)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L666)。我独立构造 mixed artifact 并显式提供 `stage2_report_path` 后，当前代码确实返回了硬失败。
- Round 3 critic 指出的 `policy_dir` mismatch 假阳性已闭合。`validate_paths(...)` 现在会对 `stage2_policy_dir` 和 `stage3_policy_dir` 分别与请求的 `--policy-path` 做阻断式一致性检查，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L774)。
- summary/logging 已不再用单个误导性的 `ONNX dir` 伪装 provenance 闭合。当前 summary 会显式打印 `stage2/stage3` 报告路径、gate 状态、`policy_dir/run_dir/onnx_dir` 与三个子图实际路径，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L910)。
- TRT/ONNX 的 `0` 值 CLI 尾项，至少在 critic 点名的运行前 guard 上已经收口。ONNX `main()` 现在会像 TRT 一样拒绝 `--joint-delta-limit <= 0`、`--gripper-delta-limit <= 0`、`--robot-max-relative-target <= 0`，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L1116)。我独立解析 `--joint-delta-limit 0` 时，也确认当前代码会进入 fail-fast 分支。
- ONNX 运行路径已补上 critic 点名的发送前 finite guard。`assert_finite_robot_action(...)` 在 `robot.send_action(...)` 之前执行，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L1086)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L1427)。
- shared helper 的最小 fail-fast 契约已经足够支撑当前 launcher 主路径，不再像 Round 3 一样存在明显静默退化。`build_chunk_predict_kwargs(...)` 对 RTC-off + 显式 RTC-only 输入会直接报错，`merge_chunk_prediction_result(...)` 对缺失可靠 `real_delay` 会直接报错，见 [pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L353)、[pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L459)、[pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L519)。

## 未闭合问题

- 文档承诺支持的“单个 `.onnx` 文件入口”当前并不成立。`--onnx-path` 的 help 明确写着可接受 “ONNX directory, a stage2_export_onnx json, or one ONNX file”，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L349)。但 `_resolve_report_candidates(...)` 在 file 模式下把传入文件本身列为第一个 `stage2` candidate，`_resolve_stage2_report_path(...)` 只要看到它是普通文件就直接返回，随后 `read_json(...)` 会把 `.onnx` 当 JSON 读，导致 `JSONDecodeError`，见 [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L411)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L467)、[run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py#L724)。这说明 worker B 自己标记的“单文件 `.onnx` 分支未覆盖”不是形式问题，而是真实缺口。
- helper 契约的“显式 `None` 与未传参”灰区仍未闭合。当前 `_UNSET` 只用于 `prev_chunk_left_over`；`inference_delay` / `execution_horizon` 仍然以 `None` 表示“未提供”，所以显式 `None` 与未传参在 helper 看起来仍是同一语义，见 [pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L344)、[pi05_chunk_runtime.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi05_chunk_runtime.py#L353)。这不阻断当前两个 launcher 进入 pre-smoke，但 shared helper 契约不能算完全封口。
- 负向测试还没有覆盖全部关键入口。当前测试主要覆盖 mixed artifacts、`policy_dir` mismatch、RTC-off 显式 RTC-only 输入、缺失 `real_delay` 和 `0` 字符串解析，见 [test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L100)、[test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L136)、[test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L173)、[test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L188)、[test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L215)。但它们全部通过显式 `onnx_stage2_report_path` 驱动 `validate_paths(...)`，见 [test_round4_contracts.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tests/test_round4_contracts.py#L42)，因此自动 report 发现入口、单文件 `.onnx` 入口、`stage2/stage3` gate 非 `pass`、发送前 finite guard、以及 `<=0` CLI guard 都没有被自动化锁住。

## 风险项

- 本轮 readiness 更接近“代码层面可以做受控 smoke”，不是“入口和契约已经全面可靠”。如果 smoke 执行人按 help 使用单文件 `.onnx` 入口，却没有显式给 `--onnx-stage2-report-path`，当前会在 provenance gate 之前就得到一个 `JSONDecodeError`，这会制造额外排障噪音。
- 现有单测对 provenance gate 的覆盖还偏窄，更多是证明“主阻断项已经接上”，还不足以证明“所有报告异常分支都已稳定 fail-fast”。在后续多人并行修改阶段，这类未覆盖入口更容易再次回归。
- 当前审查没有做真实硬件 smoke，也没有对现有真实 ONNX bundle 执行一轮正向 dry-run/preflight。也就是说，本报告只证明“结构上已经具备进入受控 pre-smoke 的条件”，不证明“现网常用 artifact 一定都能直接通过新 gate”。
- 历史导出产物若缺少 `stage2/stage3` gate 状态字段，现在可能会被 launcher 明确拒绝。这是正确收紧，但会影响旧 bundle 的可用性预期。

## 建议

- 项目状态对外建议写成：`Round 4 已闭合 Round 3 的主阻断项，可进入受控上机前 smoke；仍保留一个单文件 ONNX 入口缺陷和若干测试覆盖缺口`。不要写成“全部问题已闭合”。
- 进入 pre-smoke 时，操作约束 1：只使用 ONNX 目录或显式 `stage2_export_onnx.json` 入口。
- 进入 pre-smoke 时，操作约束 2：不使用“单个 `.onnx` 文件 + 自动发现 report”入口。
- 进入 pre-smoke 时，操作约束 3：smoke 前先跑一次当前的 `tests/test_round4_contracts.py`。
- 下一轮最优先补齐自动化覆盖 1：单文件 `.onnx` 入口。
- 下一轮最优先补齐自动化覆盖 2：`stage2/stage3` 缺失或 gate 非 `pass`。
- 下一轮最优先补齐自动化覆盖 3：`<=0` CLI guard。
- 下一轮最优先补齐自动化覆盖 4：`assert_finite_robot_action(...)`。
- 下一轮最优先补齐自动化覆盖 5：helper 的显式 `None` / 未传参边界。
- 如果团队希望把“单个 `.onnx` 文件入口”继续作为正式 CLI 契约，就必须修正 report 发现顺序；否则应直接从 help 文案和调用约束里去掉这一承诺，避免产生虚假的可用入口。

## 是否建议进入上机前 smoke

- 结论：`有条件建议进入`。
- 推荐前提 1：按受控流程只走 ONNX 目录或显式 `stage2` report 入口。
- 推荐前提 2：把本报告中的单文件入口缺陷作为已知限制写进 smoke 执行说明。
- 推荐前提 3：不把这次 smoke 解读成“全面签收”，只把它作为下一阶段真实时序/硬件边界验证。
- 不建议的表述 1：“已达到基本可上线”。
- 不建议的表述 2：“artifact/CLI/test 已完全闭合”。
- 不建议的表述 3：“任意 ONNX 入口都已可安全使用”。
