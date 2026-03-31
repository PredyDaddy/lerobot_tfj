# Torch RTC Execution Review

## 验证范围

- 工作目录：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt`
- 验证时间：`2026-03-15 16:06:07 CST`
- 本次仅做执行验证和只读检查，不修改以下目标代码：
  - `scripts/run_pi05_torch_infer_so101.py`
  - `tests/test_worker_b_torch_rtc_contracts.py`

## 执行命令

```bash
python -m py_compile scripts/run_pi05_torch_infer_so101.py
python scripts/run_pi05_torch_infer_so101.py --help
python scripts/run_pi05_torch_infer_so101.py --rtc-enable --help
python scripts/run_pi05_torch_infer_so101.py --rtc-enabled --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_b_torch_rtc_contracts.py -q
rg -n "\bpredict_action\b" scripts/run_pi05_torch_infer_so101.py
rg -n "\bActionQueue\b|\bbuild_chunk_predict_kwargs\b|\bmerge_chunk_prediction_result\b|rtc_enabled|real_delay|refill_mode|sync_refill_count" scripts/run_pi05_torch_infer_so101.py
nl -ba scripts/run_pi05_torch_infer_so101.py | sed -n '24,55p'
nl -ba scripts/run_pi05_torch_infer_so101.py | sed -n '720,910p'
nl -ba scripts/run_pi05_torch_infer_so101.py | sed -n '948,968p'
nl -ba tests/test_worker_b_torch_rtc_contracts.py | sed -n '60,178p'
```

## 结果

### 1. `py_compile`

- 命令：`python -m py_compile scripts/run_pi05_torch_infer_so101.py`
- 退出码：`0`
- 输出：无输出
- 结论：脚本可通过 Python 语法编译检查

### 2. `--help`

- 命令：`python scripts/run_pi05_torch_infer_so101.py --help`
- 退出码：`0`
- 关键结果：
  - 正常打印 `usage: run_pi05_torch_infer_so101.py ...`
  - 帮助文本中存在 `--rtc-enable, --rtc-enabled`
  - 帮助描述明确说明 `--rtc-enabled` 为兼容别名

### 3. `--rtc-enable --help`

- 命令：`python scripts/run_pi05_torch_infer_so101.py --rtc-enable --help`
- 退出码：`0`
- 关键结果：
  - 正常打印帮助信息
  - CLI 能接受 `--rtc-enable`
  - 帮助输出与基础 `--help` 一致，未出现参数解析错误

### 4. `--rtc-enabled --help`

- 命令：`python scripts/run_pi05_torch_infer_so101.py --rtc-enabled --help`
- 退出码：`0`
- 关键结果：
  - 正常打印帮助信息
  - CLI 能接受 `--rtc-enabled`
  - 与 `--rtc-enable` 一样可以进入帮助路径，兼容别名生效

### 5. `pytest`

- 命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_b_torch_rtc_contracts.py -q`
- 退出码：`0`
- 原始摘要：`.... [100%]`
- 原始结论：`4 passed in 2.07s`

### 6. `rg` / 只读检查

#### 6.1 不再 import/use `predict_action(...)`

- 命令：`rg -n "\bpredict_action\b" scripts/run_pi05_torch_infer_so101.py`
- 退出码：`1`
- 结果：无匹配
- 结论：目标 torch launcher 中未发现 `predict_action` 的独立标识符导入或调用

补充只读核对：

- `tests/test_worker_b_torch_rtc_contracts.py:67-86` 含 AST 检查，显式禁止：
  - `from lerobot.utils.control_utils import predict_action`
  - `predict_action(...)`

#### 6.2 已导入并使用 `ActionQueue` / `build_chunk_predict_kwargs` / `merge_chunk_prediction_result`

- `scripts/run_pi05_torch_infer_so101.py:32-38`
  - 导入 `build_chunk_predict_kwargs`
  - 导入 `merge_chunk_prediction_result`
- `scripts/run_pi05_torch_infer_so101.py:47`
  - 导入 `ActionQueue`
- `scripts/run_pi05_torch_infer_so101.py:725`
  - 实例化 `action_queue = ActionQueue(rtc_runtime.config)`
- `scripts/run_pi05_torch_infer_so101.py:747-751`
  - 初始 chunk 调用 `build_chunk_predict_kwargs(...)`
- `scripts/run_pi05_torch_infer_so101.py:753-758`
  - 初始 chunk 调用 `merge_chunk_prediction_result(...)`
- `scripts/run_pi05_torch_infer_so101.py:836-840`
  - 异步提交路径调用 `build_chunk_predict_kwargs(...)`
- `scripts/run_pi05_torch_infer_so101.py:848-850`
  - async wait 路径调用 `merge_chunk_prediction_result(...)`
- `scripts/run_pi05_torch_infer_so101.py:894-898`
  - sync refill 路径调用 `build_chunk_predict_kwargs(...)`
- `scripts/run_pi05_torch_infer_so101.py:900-906`
  - sync refill 路径调用 `merge_chunk_prediction_result(...)`

#### 6.3 周期日志或 warning 中包含 `rtc_enabled`、`real_delay`、`refill_mode`、`sync_refill_count`

- warning 路径：
  - `scripts/run_pi05_torch_infer_so101.py:884-888`
  - 文本包含 `refill_mode=sync_refill`
  - 文本包含 `sync_refill_count=...`
  - 文本包含 `rtc_enabled=...`
  - 文本包含 ``real_delay=0``
- 周期日志路径：
  - `scripts/run_pi05_torch_infer_so101.py:952-967`
  - 文本包含 `sync_refill_count={sync_refill_count}`
  - 文本包含 `refill_mode={last_refill_mode}`
  - 文本包含 `rtc_enabled={rtc_runtime.config.enabled}`
  - 文本包含 `real_delay={last_real_delay}`

## 失败项 / 异常

- 无执行失败
- 无语法错误
- 无 CLI 参数解析错误
- 无 pytest 失败
- 范围说明：本次没有执行真实机器人、相机、串口、checkpoint 推理闭环；`scripts/run_pi05_torch_infer_so101.py` 仅验证了帮助路径和静态接线，不构成真实硬件运行背书

## torch-RTC 结论

- 在本次要求的执行验证范围内，torch launcher 已切换到基于 `ActionQueue` 和 shared chunk helper 的 RTC 路径
- `predict_action(...)` 的 legacy import/call 未在 launcher 中发现
- `--rtc-enable` 与 `--rtc-enabled` 都可被 CLI 接受，其中 `--rtc-enabled` 作为兼容别名在帮助文本中明确可见
- 周期日志和同步 refill warning 已暴露 `rtc_enabled`、`real_delay`、`refill_mode`、`sync_refill_count`

## tests 结论

- `tests/test_worker_b_torch_rtc_contracts.py` 全部通过
- 当前测试已覆盖：
  - RTC 相关 CLI 参数解析
  - launcher 源码不依赖 legacy `predict_action(...)`
  - `build_chunk_predict_kwargs(...)` 的 RTC-off / RTC-on 分支
  - `merge_chunk_prediction_result(...)` 向 `ActionQueue.merge(...)` 传递计算后的 `real_delay`

## 总体结论

- 本次验证通过
- 对用户要求的 6 项检查，执行结果均为通过或满足预期
- 结论限于脚本可编译、CLI 帮助路径、指定单测和只读源码契约；不包含真实 SO101 硬件推理时序的端到端运行结论
