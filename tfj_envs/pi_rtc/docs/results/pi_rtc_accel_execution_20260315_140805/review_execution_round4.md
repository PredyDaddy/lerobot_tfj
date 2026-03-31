# Round 4 Execution Review

## 验证范围

- `scripts/run_pi05_onnx_infer_so101.py`
- `scripts/pi05_chunk_runtime.py`
- `tests/test_round4_contracts.py`
- 验证类型仅限执行验证和只读检查，不修改目标代码。

## 执行命令

### 必执行命令

```bash
python -m py_compile scripts/run_pi05_onnx_infer_so101.py scripts/pi05_chunk_runtime.py
python scripts/run_pi05_onnx_infer_so101.py --help
python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help
python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex python -m pytest --confcutdir=/data/tfj/lerobot_tfj/tfj_envs/pi_trt tests/test_round4_contracts.py -q
```

### 只读检查命令

```bash
rg -n "stage2_export_onnx gate must be 'pass'|stage3_verify_onnx gate must be 'pass'|Refusing to launch PI05 ONNX runtime|Policy path does not match stage2_export_onnx policy_dir|Policy path does not match stage3_verify_onnx policy_dir|ONNX provenance onnx_dir|ONNX artifact |runtime_summary\\(|describe_engines\\(" scripts/run_pi05_onnx_infer_so101.py
sed -n '920,990p' scripts/run_pi05_onnx_infer_so101.py
sed -n '1000,1160p' scripts/run_pi05_onnx_infer_so101.py
sed -n '320,560p' scripts/pi05_chunk_runtime.py
sed -n '1438,1512p' scripts/run_pi05_trt_infer_so101.py
rg -n "test_validate_paths_rejects_mixed_onnx_artifacts_from_different_runs|test_validate_paths_rejects_stage2_policy_dir_mismatch|test_build_chunk_predict_kwargs_fails_fast_for_explicit_rtc_inputs_when_disabled|test_merge_chunk_prediction_result_requires_real_delay_signal|test_parse_optional_zero_strings_preserve_numeric_zero" tests/test_round4_contracts.py
diff -u <(sed -n '1015,1034p' scripts/run_pi05_onnx_infer_so101.py) <(sed -n '1443,1459p' scripts/run_pi05_trt_infer_so101.py)
diff -u <(sed -n '1133,1141p' scripts/run_pi05_onnx_infer_so101.py) <(sed -n '1500,1508p' scripts/run_pi05_trt_infer_so101.py)
```

## 结果

### 命令执行结果

- `python -m py_compile ...` 成功，退出码 `0`，无输出。
- `python scripts/run_pi05_onnx_infer_so101.py --help` 成功，退出码 `0`。
- `python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help` 成功，退出码 `0`。
- `python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help` 成功，退出码 `0`。
- 三个 help 输出一致，均显示 `--rtc-enable, --rtc-enabled` 兼容别名。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex python -m pytest ... -q` 成功，结果为 `8 passed in 2.07s`。

### 只读检查结果

- ONNX summary 已不再只输出单个 ONNX dir。
  - `scripts/run_pi05_onnx_infer_so101.py:900-905` 在 preflight 中输出 runtime summary 和每个子图的 inputs/outputs/providers。
  - `scripts/run_pi05_onnx_infer_so101.py:938-952` 在 summary 中输出 stage2 report、stage3 report、两个 gate、`overall_status`、`policy_dir`、`run_dir`、`onnx_dir`，并逐个输出 `vision_encoder` / `prefix_cache` / `denoise_step` 的 artifact 路径。
- mixed artifacts / policy mismatch / stage2-stage3 gate 是 fail-closed，不是 warning 后继续。
  - `scripts/run_pi05_onnx_infer_so101.py:533-690` 将 stage2/stage3 gate 非 `pass`、policy/run_dir/onnx_dir 不一致、artifact path 不一致、mixed ONNX dir 等情况累积到 `blocking_reasons`。
  - `scripts/run_pi05_onnx_infer_so101.py:717-748` 在 provenance 不安全时直接 `FileNotFoundError` / `TypeError` / `ValueError`，并以 `Refusing to launch PI05 ONNX runtime without coherent stage2/stage3 provenance` 终止。
  - `scripts/run_pi05_onnx_infer_so101.py:780-789` 在 stage2/stage3 `policy_dir` 与传入 `policy_path` 不一致时直接 `ValueError`。
  - `tests/test_round4_contracts.py:100-170` 的两条测试分别覆盖 mixed artifacts hard-fail 和 stage2 policy mismatch hard-fail，并且本次执行通过。
- `joint-delta-limit 0` / `gripper-delta-limit 0` / `robot-max-relative-target 0` 的运行期 guard 与 TRT 对齐。
  - `scripts/run_pi05_onnx_infer_so101.py:1133-1141` 对 `--joint-delta-limit`、`--gripper-delta-limit`、`--robot-max-relative-target` 在传入时统一要求 `> 0`，否则直接 `ValueError`。
  - `scripts/run_pi05_trt_infer_so101.py:1500-1508` 保持相同 guard，`diff` 结果为无差异。
  - `scripts/run_pi05_onnx_infer_so101.py:1015-1034` 与 `scripts/run_pi05_trt_infer_so101.py:1443-1459` 的 clamp 逻辑语义一致。`diff` 仅显示 gripper limit 赋值写法从 `if/else` 改为三元表达式，行为未变化，且两边都对 `limit <= 0` 视为不启用 clamp。
- `scripts/pi05_chunk_runtime.py` 的 RTC/runtime fail-fast 也符合 Round 4 目标。
  - `scripts/pi05_chunk_runtime.py:353-385` 在 `rtc_enabled=False` 时，显式传入 `prev_chunk_left_over` / `inference_delay` / `execution_horizon` 会直接报错，不会静默继续。
  - `scripts/pi05_chunk_runtime.py:459-548` 在无法可靠解析 `real_delay` 时直接 `ValueError`，不会调用 `action_queue.merge(...)`。
  - `tests/test_round4_contracts.py:174-222` 覆盖了上述两类行为和 `"0"` 解析保持数值零的契约，并已通过。

## 失败项 / 异常

- 必执行命令没有失败项。
- 一条临时探索用 `rg` 命令首次执行时出现 shell quoting 错误，随后已重跑并拿到需要的只读证据；不影响验证结论。
- 一条 `diff` 命令返回退出码 `1`，原因是文本写法存在等价差异，不代表行为失败。
  - 差异仅为 `if key == "gripper.pos"` 的 `if/else` 与三元表达式写法不同，语义一致。
- 本轮测试集没有直接执行 CLI 负向用例去验证 `--joint-delta-limit 0` / `--gripper-delta-limit 0` / `--robot-max-relative-target 0` 的报错路径。
  - 这一点本次是通过只读代码比对与 TRT 对齐来确认，不是通过 pytest 直接覆盖。

## ONNX 结论

- 结论为通过。
- launcher 的 ONNX summary 已扩展为 provenance-aware 输出，不再只有单个 `onnx_dir`。
- mixed artifacts、policy mismatch、stage2/stage3 gate 非 `pass` 均为 fail-closed。
- 需要保留一个细节说明：
  - `scripts/run_pi05_onnx_infer_so101.py:557-562` 对 `stage3_overall_status != pass` 且 gate 已 `pass` 的情况只记为 note，不作为 blocker。
  - 因此本轮可以确认的是 gate fail-closed，而不是所有 `overall_status` 异常都 hard-fail。

## tests 结论

- `tests/test_round4_contracts.py` 本次实跑通过，结果为 `8 passed in 2.07s`。
- 已覆盖并验证：
  - mixed ONNX artifacts / provenance mismatch hard-fail
  - stage2 `policy_dir` mismatch hard-fail
  - RTC-off + 显式 RTC-only 输入 fail-fast
  - merge 时缺失 `real_delay` 线索 fail-fast
  - `"0"` 解析保持数值零
- 尚未直接覆盖：
  - stage2/stage3 gate 非 `pass` 的独立负向测试
  - CLI 层面对 `--joint-delta-limit 0` / `--gripper-delta-limit 0` / `--robot-max-relative-target 0` 的负向测试

## 总体结论

- 本轮对指定范围的执行验证结果为通过。
- 你要求的 1-5 号命令均成功执行，6 号只读检查也支持目标结论。
- 当前代码状态下，可以确认：
  - ONNX summary 不再只是单个 ONNX dir。
  - mixed artifacts / policy mismatch / stage2-stage3 gate 走 fail-closed。
  - 零值 delta-limit 与 `robot-max-relative-target` guard 已与 TRT launcher 对齐。
- 剩余风险主要是测试覆盖面而不是当前观察到的行为错误。
