# Worker B Round 4 Tests

## 新增测试文件

- `tests/test_round4_contracts.py`

## 执行命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n lerobot_flex \
python -m pytest \
  --confcutdir=/data/tfj/lerobot_tfj/tfj_envs/pi_trt \
  tests/test_round4_contracts.py -q
```

## 通过情况

- 结果：`8 passed in 2.10s`
- 已覆盖最小目标：
  - ONNX mixed artifacts / provenance mismatch 至少一个 hard-fail 用例
  - stage2 report 的 `policy_dir` 与传入 `policy_path` 不一致时 hard-fail
  - `build_chunk_predict_kwargs` 在 RTC-off + 显式 RTC-only 输入时 fail-fast
  - `merge_chunk_prediction_result` 在缺失 delay 线索时 fail-fast
  - `parse_optional_int("0") == 0`
  - `parse_optional_float("0") == 0.0`

## 仍未覆盖的点

- stage2/stage3 report 缺失、gate 非 `pass`、`overall_status` 异常等更多 provenance 负向分支还没单独拆开验证。
- `--onnx-path` 显式指向单个 `.onnx` 文件但不属于 coherent artifact set 的分支未覆盖。
- helper 契约里“显式 `None` 和未传参的语义差异”还没有单测，critic 提到的灰区仍在。
- 真机相关时序口径、发送前 finite action 保护、相机/robot preflight 都未覆盖，本轮保持轻量无硬件。
