# Round 3 执行验证报告

执行时间: 2026-03-15

工作目录: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt`

仓库提交: `24acade`

验证角色: 独立执行验证员

## 验证范围

本轮仅做只读执行验证与契约检查，不修改以下 3 个脚本实现:

- `scripts/pi05_chunk_runtime.py`
- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`

验证目标:

- 验证 3 个脚本可通过语法编译。
- 验证 TRT/ONNX 两个 launcher 在无硬件条件下至少可完成 import + argparse `--help` 路径。
- 验证 RTC CLI 主开关 `--rtc-enable` 与兼容别名 `--rtc-enabled` 均被接受。
- 验证 ONNX 侧 parser 契约:
  `parse_optional_int('0') == 0`，
  `parse_optional_float('0') == 0.0`。
- 验证 shared helper 在 RTC-off 的 launcher 风格调用下不报错。
- 用 `rg` 检查 TRT/ONNX 两侧是否都直接导入并调用 `build_chunk_predict_kwargs` / `merge_chunk_prediction_result`，以及日志里是否包含 `refill_mode` / `sync_refill_count` / `rtc_enabled` 关键信号。

说明:

- 本报告验证的是“无硬件可执行性”和“关键契约是否成立”。
- 本报告不验证真实相机、真实机器人、真实 TensorRT engine、真实 ONNX artifact、真实 GPU 推理是否可运行。

## 执行命令

### 1. 语法编译

```bash
python -m py_compile scripts/pi05_chunk_runtime.py scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py
```

### 2. TRT launcher help

```bash
python scripts/run_pi05_trt_infer_so101.py --help
```

### 3. TRT launcher RTC flag help

```bash
python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help
```

### 4. TRT launcher RTC alias help

```bash
python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help
```

### 5. ONNX launcher help

```bash
python scripts/run_pi05_onnx_infer_so101.py --help
```

### 6. ONNX launcher RTC flag help

```bash
python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help
```

### 7. ONNX launcher RTC alias help

```bash
python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help
```

### 8. 轻量 parser/smoke

```bash
python - <<'PY'
import json
import sys
from pathlib import Path

import torch

repo = Path('/data/tfj/lerobot_tfj/tfj_envs/pi_trt')
sys.path.insert(0, str(repo / 'scripts'))

from run_pi05_onnx_infer_so101 import parse_optional_float, parse_optional_int
from pi05_chunk_runtime import (
    ChunkPredictionResult,
    build_chunk_predict_kwargs,
    merge_chunk_prediction_result,
)

class DummyRTCConfig:
    def __init__(self, enabled=False, execution_horizon=4):
        self.enabled = enabled
        self.execution_horizon = execution_horizon

class DummyRTCRuntime:
    def __init__(self, config):
        self.config = config

class DummyActionQueue:
    def __init__(self):
        self.calls = []
    def get_left_over(self):
        return None
    def get_action_index(self):
        return 6
    def merge(self, original_actions, processed_actions, *, real_delay, action_index_before_inference):
        self.calls.append({
            'original_shape': tuple(original_actions.shape),
            'processed_shape': tuple(processed_actions.shape),
            'real_delay': int(real_delay),
            'action_index_before_inference': int(action_index_before_inference),
        })

parse_int = parse_optional_int('0')
parse_float = parse_optional_float('0')
assert parse_int == 0, parse_int
assert parse_float == 0.0, parse_float

queue = DummyActionQueue()
rtc_runtime = DummyRTCRuntime(DummyRTCConfig(enabled=False, execution_horizon=8))
rtc_off_kwargs = build_chunk_predict_kwargs(
    rtc_runtime=rtc_runtime,
    action_queue=queue,
    predicted_delay_steps=0,
)
assert rtc_off_kwargs == {}, rtc_off_kwargs

prediction = ChunkPredictionResult(
    original_actions=torch.zeros((2, 3), dtype=torch.float32),
    processed_actions=[torch.zeros(3, dtype=torch.float32), torch.ones(3, dtype=torch.float32)],
    preprocess_time_s=0.0,
    inference_time_s=0.0,
    postprocess_time_s=0.0,
    processed_actions_tensor=torch.stack(
        [torch.zeros(3, dtype=torch.float32), torch.ones(3, dtype=torch.float32)], dim=0
    ),
    action_index_before_inference=5,
)
real_delay = merge_chunk_prediction_result(queue, prediction)
assert real_delay == 1, real_delay
assert len(queue.calls) == 1, queue.calls

print(json.dumps({
    'parse_optional_int_0': parse_int,
    'parse_optional_float_0': parse_float,
    'rtc_off_kwargs': rtc_off_kwargs,
    'merge_real_delay': real_delay,
    'merge_call': queue.calls[0],
}, indent=2, sort_keys=True))
PY
```

### 9. `rg` 契约检查

```bash
rg -n "build_chunk_predict_kwargs|merge_chunk_prediction_result|refill_mode|sync_refill_count|rtc_enabled" \
  scripts/run_pi05_trt_infer_so101.py \
  scripts/run_pi05_onnx_infer_so101.py \
  scripts/pi05_chunk_runtime.py
```

```bash
rg -n "def build_chunk_predict_kwargs|def merge_chunk_prediction_result|def launch|rtc" scripts/pi05_chunk_runtime.py
```

补充查看调用点与日志片段:

```bash
sed -n '1590,1770p' scripts/run_pi05_trt_infer_so101.py
sed -n '916,1078p' scripts/run_pi05_onnx_infer_so101.py
sed -n '1798,1820p' scripts/run_pi05_trt_infer_so101.py
sed -n '1114,1128p' scripts/run_pi05_onnx_infer_so101.py
sed -n '220,520p' scripts/pi05_chunk_runtime.py
sed -n '520,700p' scripts/pi05_chunk_runtime.py
```

## 结果

### 1. 语法编译结果

- 退出码: `0`
- 标准输出: 空
- 结论: `scripts/pi05_chunk_runtime.py`、`scripts/run_pi05_trt_infer_so101.py`、`scripts/run_pi05_onnx_infer_so101.py` 均通过 Python 语法编译。

### 2. TRT `--help` 结果

- `python scripts/run_pi05_trt_infer_so101.py --help` 退出码 `0`
- `python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help` 退出码 `0`
- `python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help` 退出码 `0`
- 三次输出均成功打印 usage 和 options，没有触发硬件访问错误。
- help 文本中明确暴露:
  `--rtc-enable, --rtc-enabled`
- help 文本中明确包含:
  `--prefetch-threshold`
  `--sync-refill-timeout-s`
  `--rtc-execution-horizon`
  `--rtc-max-guidance-weight`
  `--rtc-prefix-attention-schedule`
  `--rtc-debug`
  `--rtc-debug-maxlen`

结论:

- TRT launcher 的无硬件 `import + argparse --help` 路径可执行。
- RTC 主开关和兼容别名均生效。

### 3. ONNX `--help` 结果

- `python scripts/run_pi05_onnx_infer_so101.py --help` 退出码 `0`
- `python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help` 退出码 `0`
- `python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help` 退出码 `0`
- 三次输出均成功打印 usage 和 options，没有触发硬件访问错误。
- help 文本中明确暴露:
  `--rtc-enable, --rtc-enabled`
- help 文本中明确包含:
  `--prefetch-threshold`
  `--sync-refill-timeout-s`
  `--rtc-execution-horizon`
  `--rtc-max-guidance-weight`
  `--rtc-prefix-attention-schedule`
  `--rtc-debug`
  `--rtc-debug-maxlen`

结论:

- ONNX launcher 的无硬件 `import + argparse --help` 路径可执行。
- RTC 主开关和兼容别名均生效。

### 4. parser/smoke 结果

执行输出:

```json
{
  "merge_call": {
    "action_index_before_inference": 5,
    "original_shape": [
      2,
      3
    ],
    "processed_shape": [
      2,
      3
    ],
    "real_delay": 1
  },
  "merge_real_delay": 1,
  "parse_optional_float_0": 0.0,
  "parse_optional_int_0": 0,
  "rtc_off_kwargs": {}
}
```

验证结论:

- ONNX `parse_optional_int('0') == 0`
- ONNX `parse_optional_float('0') == 0.0`
- shared helper 的 RTC-off launcher 风格调用
  `build_chunk_predict_kwargs(rtc_runtime=DummyRTCRuntime(enabled=False), action_queue=DummyActionQueue(), predicted_delay_steps=0)`
  返回 `{}`，不报错
- `merge_chunk_prediction_result(...)` 在最小假对象条件下成功调用 `action_queue.merge(...)`
  并返回 `real_delay == 1`

### 5. `rg` 契约检查结果

#### shared helper 定义位置

- `scripts/pi05_chunk_runtime.py:353` 定义 `build_chunk_predict_kwargs`
- `scripts/pi05_chunk_runtime.py:519` 定义 `merge_chunk_prediction_result`

#### TRT 直接导入与调用

- 导入:
  `scripts/run_pi05_trt_infer_so101.py:35`
  `scripts/run_pi05_trt_infer_so101.py:37`
- `build_chunk_predict_kwargs(...)` 调用:
  `scripts/run_pi05_trt_infer_so101.py:1610`
  `scripts/run_pi05_trt_infer_so101.py:1696`
  `scripts/run_pi05_trt_infer_so101.py:1755`
- `merge_chunk_prediction_result(...)` 调用:
  `scripts/run_pi05_trt_infer_so101.py:1617`
  `scripts/run_pi05_trt_infer_so101.py:1665`
  `scripts/run_pi05_trt_infer_so101.py:1710`
  `scripts/run_pi05_trt_infer_so101.py:1762`

#### ONNX 直接导入与调用

- 导入:
  `scripts/run_pi05_onnx_infer_so101.py:34`
  `scripts/run_pi05_onnx_infer_so101.py:36`
- `build_chunk_predict_kwargs(...)` 调用:
  `scripts/run_pi05_onnx_infer_so101.py:923`
  `scripts/run_pi05_onnx_infer_so101.py:1009`
  `scripts/run_pi05_onnx_infer_so101.py:1060`
- `merge_chunk_prediction_result(...)` 调用:
  `scripts/run_pi05_onnx_infer_so101.py:929`
  `scripts/run_pi05_onnx_infer_so101.py:979`
  `scripts/run_pi05_onnx_infer_so101.py:1021`
  `scripts/run_pi05_onnx_infer_so101.py:1066`

#### 日志关键信号

TRT 侧:

- `rtc_enabled`:
  `scripts/run_pi05_trt_infer_so101.py:1647`
  `scripts/run_pi05_trt_infer_so101.py:1747`
  `scripts/run_pi05_trt_infer_so101.py:1809`
- `sync_refill_count`:
  `scripts/run_pi05_trt_infer_so101.py:1747`
  `scripts/run_pi05_trt_infer_so101.py:1816`
- `refill_mode`:
  `scripts/run_pi05_trt_infer_so101.py:1746`
  `scripts/run_pi05_trt_infer_so101.py:1817`

ONNX 侧:

- `rtc_enabled`:
  `scripts/run_pi05_onnx_infer_so101.py:962`
  `scripts/run_pi05_onnx_infer_so101.py:1124`
- `sync_refill_count`:
  `scripts/run_pi05_onnx_infer_so101.py:1122`
- `refill_mode`:
  `scripts/run_pi05_onnx_infer_so101.py:1123`

结论:

- TRT 与 ONNX 两侧都直接导入并直接调用了 shared helper:
  `build_chunk_predict_kwargs` 和 `merge_chunk_prediction_result`
- TRT 与 ONNX 两侧日志都包含 `refill_mode` / `sync_refill_count` / `rtc_enabled` 关键信号

## 失败项 / 异常

- 无命令失败
- 无 import 异常
- 无 argparse 异常
- 无 parser/smoke 断言失败

边界说明:

- 本轮没有执行真实 `run` 路径，因此未覆盖:
  相机枚举、
  机器人串口连接、
  TensorRT engine 加载、
  ONNX Runtime session 创建、
  真正的 policy chunk 推理与动作发送

## TRT 结论

- `scripts/run_pi05_trt_infer_so101.py` 在当前环境下可完成无硬件 `--help` 执行，退出码均为 `0`
- `--rtc-enable` 与 `--rtc-enabled` 均被接受
- 直接导入并使用 shared helper 的契约成立
- 日志中可观测到 `refill_mode` / `sync_refill_count` / `rtc_enabled` 关键信号

判定:

- TRT launcher 满足本轮“无硬件可执行性 + 关键契约”验证要求

## ONNX 结论

- `scripts/run_pi05_onnx_infer_so101.py` 在当前环境下可完成无硬件 `--help` 执行，退出码均为 `0`
- `--rtc-enable` 与 `--rtc-enabled` 均被接受
- `parse_optional_int('0') == 0` 与 `parse_optional_float('0') == 0.0` 契约成立
- 直接导入并使用 shared helper 的契约成立
- 日志中可观测到 `refill_mode` / `sync_refill_count` / `rtc_enabled` 关键信号

判定:

- ONNX launcher 满足本轮“无硬件可执行性 + 关键契约”验证要求

## helper 结论

- `scripts/pi05_chunk_runtime.py` 已定义 `build_chunk_predict_kwargs` 和 `merge_chunk_prediction_result`
- 在 RTC-off 的 launcher 风格调用下，
  `build_chunk_predict_kwargs(...)` 返回空字典 `{}`，未错误注入 RTC-only 参数
- `merge_chunk_prediction_result(...)` 可在最小假对象条件下完成 merge 并给出 `real_delay`

判定:

- shared helper 满足本轮验证的最小执行契约

## 总体结论

本轮要求的 1-9 项验证均已执行并通过。

在当前环境下，这 3 个脚本满足以下 Round 3 判定:

- 语法层面可编译
- TRT/ONNX launcher 均具备无硬件 `--help` 可执行性
- RTC 主开关与兼容别名可用
- ONNX parser 零值契约成立
- shared helper 的 RTC-off launcher 风格调用不报错
- TRT/ONNX 两侧均已直接接入 shared helper，且日志包含本轮要求的关键运行信号

因此，本轮执行验证结论为: 通过。
