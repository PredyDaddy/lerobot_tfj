# Round 4 Critic Review

## 结论先行

不建议进入上机前 smoke。

Round 4 的确把一批明显的“混拼三件套就继续跑”问题收紧了，但现在的证据仍然只足以说明：

- launcher 会拒绝一部分 provenance 不自洽的输入；
- 一些 RTC helper 契约开始 fail-fast；
- CLI 参数 `0` 值和 finite action guard 有所收口。

它还不能证明：

- 操作者指定的 `--onnx-path` 就是实际运行的 artifact 集合；
- 当前磁盘上的 ONNX 三件套仍然是 Stage 3 验证过的那一份；
- 目标 policy/preprocessor/tokenizer/provider 组合已经能在真实 artifact 上完成一次最小闭环的 chunk 推理。

## 最严重问题

### 1. ONNX provenance gate 仍然没有把“操作者选择的目录”绑定成硬约束

这轮的 gate 主要验证的是 `stage2/stage3` 报告内部是否自洽，不是“CLI 传入的 `--onnx-path` 就必须等于最终将要加载的目录”。

证据：

- `scripts/run_pi05_onnx_infer_so101.py:467-472` 中，显式 `--onnx-stage2-report-path` 会被直接接受。
- `scripts/run_pi05_onnx_infer_so101.py:688-691` 只对“显式单个 `.onnx` 文件不属于 coherent set”做阻断。
- 对“显式传入的是目录，但该目录不存在 / 与 stage2/stage3 解析出的真实 artifact 目录不同”的情况，没有等价阻断。

我本地复现了这一点：

- 使用临时目录构造一套自洽的 `stage2/stage3` 报告与三件套；
- 把 `onnx_path` 故意传成一个不存在的 `bogus_dir`；
- 同时显式传 `onnx_stage2_report_path` 指向那套自洽报告；
- `validate_paths(...)` 仍然通过，并解析出真正的 `resolved onnx_dir=/tmp/.../run/onnx`。

复现结论：

- `validate_paths accepted mismatched directory input`
- `requested onnx_path=/tmp/.../bogus_dir`
- `resolved onnx_dir=/tmp/.../run/onnx`

这意味着当前 launcher 仍然允许“用户以为自己在验证/运行 A 目录，实际运行的是 B 目录”。这不是纯文案问题，而是 provenance 绑定还没闭合。它会直接误导 smoke，尤其是在人工切换 run 目录、回填 `--onnx-stage2-report-path`、或目录命名相近时。

### 2. provenance 仍然是“路径级”而不是“内容级”，原地替换 artifact 后 gate 仍可放行

`assess_onnx_artifact_safety(...)` 目前做的是：

- 校验 `stage2/stage3` stage 名和 gate 状态；
- 校验 `policy_dir/run_dir/onnx_dir` 一致；
- 校验 `onnx_paths` / `artifact_paths` 指向同一路径；
- 校验文件存在。

对应代码在 `scripts/run_pi05_onnx_infer_so101.py:604-692`。

这里没有任何对当前磁盘文件内容的绑定检查：

- 没有 hash / digest 校验；
- 没有 size 校验；
- 没有 mtime 或内容签名校验。

而实际 Stage 2 报告里至少已经有 `onnx_file_sizes`，见 `docs/results/pi_model_fp16_20260314_172759/stage2_export_onnx.json:19-23`。但 launcher 当前完全没有读取这些信息。代码里对 `onnx_file_sizes` / `digest` / `checksum` 也没有任何使用痕迹。

结果是：

- 只要路径不变，文件被原地替换后仍可能通过 Round 4 gate；
- 操作者看到的仍是“同一个报告、同一路径、gate=pass”，但那已经不保证是 Stage 3 验证过的内容。

这条是我认为当前最硬的剩余 provenance 缺口。它足以让人误判“这一套已经过验证，可以上机前 smoke”。

### 3. 仍然没有“真实 artifact 上一次最小正向闭环推理”的证据，当前成功信号过强

`preflight_onnx_adapter(...)` 在 `scripts/run_pi05_onnx_infer_so101.py:891-907` 里做的事情是：

- 构造 `OnnxPi05PolicyAdapter`；
- `eval()`；
- 打印 `PI05 ONNX policy OK`；
- 打印 runtime summary 和各子图输入输出名。

但它没有做：

- 一次 `predict_action_chunk(...)`；
- 一次 `predict_sync(...)`；
- 一次从真实 preprocessor 输出到 ONNX runtime 的最小正向调用。

而 `OnnxPi05PolicyAdapter.__init__` 在 `scripts/onnx_pi_adapter.py:82-105` 里也只是：

- 加载三个 ORT session；
- 跑 `_validate_session_contract()`；
- `reset()`。

也就是说，当前 `PI05 ONNX policy OK` 本质上只等价于“session 能打开，且输入输出名符合预期”，不等价于“首帧能推理”“首个 chunk 能产出”“preprocessor/tokenizer/runtime 边界已打通”。

结合两份 Round 4 材料看，这个缺口没有被补上：

- Worker A 自检只有 `py_compile`、`--help` 和合成负向样例，见 `worker_a_onnx_provenance_round4.md:52-216`。
- Worker B 测试全部基于 stub 文件与合成 JSON，见 `tests/test_round4_contracts.py:23-32`、`tests/test_round4_contracts.py:100-225`。

这意味着“成功信号”已经被打印出来，但“最关键的正向证据”仍然不存在。对于上机前 smoke，这个差距是致命的。

## 次要问题

### 1. `stage3 overall_status != pass` 被降级成 note，而且最终以 `info` 打印，风险信号过弱

代码在 `scripts/run_pi05_onnx_infer_so101.py:557-562` 把 `stage3_overall_status != pass && stage3_gate_status == pass` 仅记为 `notes`，随后在 `scripts/run_pi05_onnx_infer_so101.py:980-981` 用 `info(...)` 打印。

这会制造明显的假阳性，因为仓库里就有现实样本：

- `docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json:3` 是 `overall_status: "warn"`；
- 但 `docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json:3331-3359` 的 `stage3_acceptance.status` 仍是 `pass`；
- 同一份报告里还有实际 warn 项，例如 `mean_abs_diff 0.023193 > 0.005`，见 `docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json:640-648`；
- 汇总 section 也明确是 `status: "warn"`，见 `docs/results/pi_model_fp16_20260314_172759/stage3_verify_onnx.json:3016-3019`。

当前 launcher 会继续跑，并且最终仍会打印 `PI05 ONNX policy OK`。如果操作者没有逐行细看 summary，很容易把它理解成“验证已过，可以 smoke”。

### 2. `--preflight-only --skip-onnx-preflight` 仍然可以返回成功，并打印“Preflight completed”

路径在 `scripts/run_pi05_onnx_infer_so101.py:1156-1174`：

- `skip_onnx_preflight` 为真时，根本不会构造 `onnx_policy`；
- 但只要同时传了 `--preflight-only`，程序仍会打印 `Preflight completed. Exiting before robot connect.` 并返回 `0`。

这条不是默认路径，但它对 runbook/人工 smoke 非常危险，因为日志字面上没有强调“本次 preflight 根本没有做 ONNX runtime load”。如果有人把这条命令当成“上线前 smoke 的快速绿灯”，会直接造成误判。

### 3. 当前 contract tests 证明的是“路径/契约分支”，不是“artifact 真能跑”

我按文档命令重跑了测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex python -m pytest --confcutdir=/data/tfj/lerobot_tfj/tfj_envs/pi_trt tests/test_round4_contracts.py -q
```

结果是 `8 passed in 2.06s`，与 `worker_b_tests_round4.md:17-19` 一致。

但这些测试的局限也很清楚：

- 测试用的 ONNX 只是 `_write_stub_file(..., "stub\\n")` 写出的文本文件，见 `tests/test_round4_contracts.py:29-32`；
- 没有任何真实 ORT session load；
- 没有任何真实 ONNX provider 选择；
- 没有任何真实 preprocessor/tokenizer/runtime 边界。

所以这 8 个测试只能说明“Round 4 合同约束没回退”，不能说明“可以进入上机前 smoke”。

## 误导性信号

- `PI05 ONNX policy OK` 这个文案过强。按当前实现，它只代表 adapter/session contract 通过，不代表实际推理链路通过。
- summary 同时打印了 `ONNX path` 和 `resolved artifact paths`，但由于目录不绑定硬校验，操作者仍然可能误以为“我传入的那个目录已经被强制采用”。
- `stage3 overall_status=warn` 最终只作为 `info` note 打印，容易淹没在正常 summary 中；而真实仓库里已经存在这种 warn 样例。
- `Preflight completed` 这个成功语句在 `--skip-onnx-preflight` + `--preflight-only` 组合下仍会出现，字面上足以制造假阳性。
- Worker A 文档里“ONNX 现在也在 launcher 入口做阻断式校验，不再允许‘文件能打开就继续’”这句话仍然偏强，见 `worker_a_onnx_provenance_round4.md:47-50`。就当前实现看，它最多只能说“不再允许一部分明显的 provenance 混拼”，还不能说“已经把操作者选择、磁盘现状、已验证产物三者完全绑定”。

## 上线前必须补的验证

### 1. 必须补一个负向用例：显式 report + 错误目录必须 hard-fail

至少要覆盖以下场景之一，并把它写进 tests：

- `--onnx-stage2-report-path` 指向一套自洽报告，但 `--onnx-path` 是不存在目录；
- `--onnx-stage2-report-path` 指向一套自洽报告，但 `--onnx-path` 是另一套 run 的目录。

当前这是我已经复现能通过的路径，所以这条不是“建议增强”，而是必须补的 blocker。

### 2. 必须补 artifact 内容级绑定

最低标准也应做到下面之一：

- launcher 起跑前比对当前三件套的 size 与 Stage 2 记录一致；
- 更稳妥的是在 Stage 2/3 报告里落 hash/digest，并在 launcher 启动前校验。

在没有内容级绑定之前，路径级 provenance 不能当成“已验证 artifact 仍然有效”的证据。

### 3. 必须补一条真实 artifact 的最小正向验证

不是 stub，不是 `--help`，也不是只 load session。

最低限度要证明：

- 指定的 policy checkpoint；
- 指定的 tokenizer / preprocessor；
- 指定的 ONNX 三件套；
- 指定的 provider；

在当前环境里能完成一次最小正向 `predict_action_chunk` 或 `predict_sync`。

没有这条正向证据，就不该进入上机前 smoke。

### 4. 必须明确处理 `overall_status=warn`

当前至少需要做到下面之一：

- 把 `overall_status != pass` 提升为醒目的 warning，而不是 `info note`；
- 或者在 smoke runbook 里明确规定：只要 `overall_status != pass`，默认不得进入上机前 smoke，除非人工签字确认具体 warn 原因。

否则“acceptance pass + overall warn + policy OK”这组三重信号放在一起，极易制造假阳性。

### 5. 必须补一条“preflight 语义”验证

如果还保留 `--skip-onnx-preflight` 与 `--preflight-only` 组合，就必须明确验证并记录：

- 这条命令不算 ONNX-ready；
- 这条命令不应该在 runbook 中被当成 smoke 绿灯；
- 最好有单测或脚本级断言，防止以后又把它写成成功完成的 preflight。

## 是否建议进入上机前 smoke

不建议。

原因不是 Round 4 没有改进，而是它当前仍停在“收紧一部分负向入口”的阶段，还没有达到“可以信任这就是将要上机的那组三件套，并且它至少跑通过一次真实正向最小链路”的阶段。

更直白地说：当前状态更像“能减少误用的 launcher”，还不是“已具备上机前 smoke 最低可信度的 launcher”。
