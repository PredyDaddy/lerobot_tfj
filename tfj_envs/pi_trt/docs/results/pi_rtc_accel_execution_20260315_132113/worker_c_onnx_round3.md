# Worker C ONNX Round 3

## 改动摘要

- 修改 `scripts/run_pi05_onnx_infer_so101.py`，把 `parse_optional_int(...)` / `parse_optional_float(...)` 的 `"0"` 特判去掉，和 TRT launcher 保持一致。
- 删除 launcher 本地 RTC glue 主路径，直接导入并调用 shared helper `build_chunk_predict_kwargs(...)` / `merge_chunk_prediction_result(...)`。
- 保持 RTC 默认关闭和现有 CLI 不变，只补充 `--rtc-enable` help 文案，明确 `--rtc-enabled` 是兼容别名。
- 修正 checkpoint warning 文案，不再把 RTC 打开条件错误地收窄成只有 `--rtc-enable`。
- 周期日志新增 `refill_mode` 与 `sync_refill_count`，并补充 hold / sync refill warning，使 `real_delay=0` 的语义可判责。

## 自检命令

```bash
python -m py_compile scripts/run_pi05_onnx_infer_so101.py
python scripts/run_pi05_onnx_infer_so101.py --help | rg -- '--rtc-enable|--rtc-enabled|prefetch-threshold|sync-refill-timeout-s'
python - <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path('scripts/run_pi05_onnx_infer_so101.py').resolve()
spec = importlib.util.spec_from_file_location('run_pi05_onnx_infer_so101', module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.parse_optional_int('0') == 0
assert module.parse_optional_float('0') == 0.0

args = module.build_parser().parse_args([
    '--prefetch-threshold', '0',
    '--policy-n-action-steps', '0',
    '--joint-delta-limit', '0',
    '--rtc-enabled',
])
assert args.prefetch_threshold == 0
assert args.policy_n_action_steps == 0
assert args.joint_delta_limit == 0.0
assert args.rtc_enable is True
print('smoke_ok')
PY
```

结果：

- `py_compile` 通过。
- `--help` 输出确认仍暴露 `--rtc-enable, --rtc-enabled`、`--prefetch-threshold`、`--sync-refill-timeout-s`。
- 轻量 smoke 通过，确认 `'0'` 不再被静默解析成 `None`。

## 0 值语义修正说明

- 现在 `parse_optional_int("0") == 0`，`parse_optional_float("0") == 0.0`。
- 这避免了 ONNX launcher 把显式传入的 `0` 静默吞回 `None`，与 TRT 当前行为对齐。
- 直接影响的 CLI 语义：
  - `--prefetch-threshold 0` 现在会保留为 `0`，不再回落成“自动估算”。
  - `--policy-n-action-steps 0` 现在会保留为 `0`，随后在 runtime override 校验阶段报非法值，而不是静默回落到 checkpoint 默认值。
  - `--joint-delta-limit 0` / `--gripper-delta-limit 0` 现在会保留为 `0.0`，含义是显式关闭该限幅，而不是未设置。

## Shared Helper 接线说明

- ONNX launcher 现在直接从 `scripts/pi05_chunk_runtime.py` 导入：
  - `build_chunk_predict_kwargs(...)`
  - `merge_chunk_prediction_result(...)`
- 已移除 launcher 本地的 `build_chunk_predict_kwargs(...)` / `merge_completed_chunk(...)` 主路径，避免继续维护重复 RTC glue。
- 当前接线方式：
  - startup initial chunk、async submit、sync refill 都调用 shared `build_chunk_predict_kwargs(...)`。
  - async collect / async wait 直接调用 shared `merge_chunk_prediction_result(...)`，由 shared helper 基于 `ActionQueue.get_action_index()` 计算真实 delay。
  - initial sync / sync refill 仍通过 shared merge helper 合并，但显式传入 `real_delay=0`，保留“阻塞式补货”的既有语义。
- `resolve_rtc_runtime_config(...)` 未改动，因此 RTC 默认仍然是 off；只有 `--rtc-enable/--rtc-enabled` 或其他 `--rtc-*` override 才会打开。

## 日志口径变化

- 周期日志新增：
  - `refill_mode`
  - `sync_refill_count`
- `refill_mode` 当前会输出以下路径标签：
  - `initial_sync`
  - `async_collect`
  - `async_wait`
  - `hold_pending_async`
  - `sync_refill`
- 这样当日志里出现 `real_delay=0` 时，可以区分：
  - 健康异步路径：`refill_mode=async_collect` 或 `async_wait`
  - 阻塞式补货路径：`refill_mode=sync_refill`
- `queue_underrun_count` / `hold_step_count` 继续只统计 hold 路径；本轮补出的 `sync_refill_count` 用来显式覆盖此前容易被漏读的 sync refill 路径。
- hold warning 现在会补充“是否已经等过 `sync_refill_timeout_s`”的信息。
- sync refill warning 现在明确说明：该路径上的 `real_delay=0` 只代表阻塞式 refill 语义，不代表 overlap 健康。

## 剩余风险

- 本轮只做了无硬件检查，没有做真实 robot loop 或 checkpoint 联调，无法证明 hold / sync refill 在真机时序下的表现。
- `refill_mode` / `sync_refill_count` 的新口径没有状态机级自动化测试；后续如果主循环再重构，仍有回归风险。
- shared helper 正在被并行维护；虽然本轮已优先接到 shared helper，但如果 helper 的函数签名在后续 worker 提交中变化，需要重新跑这组 smoke 检查。
- TRT launcher 是否同步补齐相同口径，不在本 worker 代码所有权范围内。
