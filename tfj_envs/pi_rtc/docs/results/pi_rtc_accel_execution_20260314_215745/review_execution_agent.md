# Review Execution Agent

## 验证范围

- `scripts/pi05_chunk_runtime.py`
- `scripts/trt_pi_adapter.py`
- `scripts/onnx_pi_adapter.py`
- `scripts/run_pi05_trt_infer_so101.py`
- `scripts/run_pi05_onnx_infer_so101.py`

说明：

- 本次只做静态核查和无硬件命令验证。
- 未运行机器人、相机、串口相关命令。
- 未回退任何已有改动。

## 执行命令

### 1. 工作区状态核查

```bash
git status --short
```

结果：

- 退出码：`0`
- 工作区存在大量既有改动和未跟踪文件。
- 本次验证未回退这些改动，只在目标报告路径写入结果。

### 2. `pi05_chunk_runtime.py` 静态阅读

```bash
nl -ba scripts/pi05_chunk_runtime.py | sed -n '1,260p'
```

结果：

- 退出码：`0`
- 文件存在，定义了 `ChunkPredictionResult`、`prepare_policy_observation(...)`、`postprocess_action_chunk(...)`、`predict_processed_action_chunk(...)`、`estimate_prefetch_threshold(...)` 和 `AsyncChunkPrefetcher`。
- 该文件提供了可复用的 chunk runtime helper，但这里只是静态存在性确认。

### 3. `trt_pi_adapter.py` 前半段静态阅读

```bash
nl -ba scripts/trt_pi_adapter.py | sed -n '1,260p'
```

结果：

- 退出码：`0`
- 文件顶层导入了 `RTCProcessor`。
- `TrtPi05PolicyAdapter.__init__` 中创建了 `self.rtc_processor`。
- 存在 `_rtc_enabled()` 和 `runtime_summary()`，说明 adapter 层具备 RTC 开关与状态汇总。

### 4. `onnx_pi_adapter.py` 前半段静态阅读

```bash
nl -ba scripts/onnx_pi_adapter.py | sed -n '1,260p'
```

结果：

- 退出码：`0`
- 文件顶层导入了 `RTCProcessor`。
- `OnnxPi05PolicyAdapter.__init__` 中创建了 `self.rtc_processor`。
- 存在 `_rtc_enabled()` 和 `runtime_summary()`，说明 adapter 层同样具备 RTC 开关与状态汇总。

### 5. `run_pi05_trt_infer_so101.py` 前半段静态阅读

```bash
nl -ba scripts/run_pi05_trt_infer_so101.py | sed -n '1,320p'
```

结果：

- 退出码：`0`
- 顶层引入了 `RTCAttentionSchedule`、`ActionQueue`、`RTCConfig`。
- 定义了 `ResolvedRTCRuntimeConfig`、`add_rtc_runtime_arguments(...)`、`resolve_rtc_runtime_config(...)`、`estimate_inference_delay_steps(...)`。
- TRT launcher 在参数层和运行时配置层已经显式接入 RTC。

### 6. `run_pi05_onnx_infer_so101.py` 前半段静态阅读

```bash
nl -ba scripts/run_pi05_onnx_infer_so101.py | sed -n '1,320p'
```

结果：

- 退出码：`0`
- 该 launcher 只定义了普通 ONNX 推理参数，没有 RTC CLI 参数定义。
- 前半段没有 `RTCAttentionSchedule`、`RTCConfig`、`ActionQueue` 相关导入或解析逻辑。
- ONNX launcher 在入口参数层未完成 RTC 接线。

### 7. RTC 关键字全量检索

```bash
rg -n "rtc|RTC|ActionQueue|predict_action_chunk|runtime_summary|build_parser|def main|if __name__ == '__main__'" scripts/pi05_chunk_runtime.py scripts/trt_pi_adapter.py scripts/onnx_pi_adapter.py scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py
```

结果：

- 退出码：`0`
- `trt_pi_adapter.py` 和 `onnx_pi_adapter.py` 都命中了 `RTCProcessor`、`_rtc_enabled()`、`predict_action_chunk(...)` 中的 RTC 路径。
- `run_pi05_trt_infer_so101.py` 命中了 RTC CLI、`ActionQueue`、`ResolvedRTCRuntimeConfig` 和 RTC 相关调用点。
- `run_pi05_onnx_infer_so101.py` 只命中了普通 `predict_action_chunk(...)` 调用，没有命中 launcher 层 RTC 接线。

### 8. TRT launcher 中段静态阅读

```bash
nl -ba scripts/run_pi05_trt_infer_so101.py | sed -n '1240,1415p'
```

结果：

- 退出码：`0`
- `apply_pi_runtime_overrides(...)` 最终返回 `resolve_rtc_runtime_config(...)` 的结果。
- `build_chunk_predict_kwargs(...)` 会在 RTC 打开时组装：
  - `prev_chunk_left_over`
  - `inference_delay`
  - `execution_horizon`
- `predict_processed_action_chunk(...)` 把这些 `predict_kwargs` 透传给 `policy.predict_action_chunk(...)`。
- `merge_completed_chunk(...)` 计算并回写 `real_delay`。

### 9. TRT launcher 主循环静态阅读

```bash
nl -ba scripts/run_pi05_trt_infer_so101.py | sed -n '1528,1995p'
```

结果：

- 退出码：`0`
- `print_summary(...)` 会打印 `Resolved RTC config`。
- `main()` 中调用 `apply_pi_runtime_overrides(...)`，随后创建 `ActionQueue(rtc_runtime.config)`。
- 初始 chunk、异步回收、同步补充三条路径都会调用 `build_chunk_predict_kwargs(...)`。
- 主循环里会计算 `predicted_delay_steps`，并在日志中输出 `rtc_enabled`。
- 这说明 TRT launcher 的 RTC launcher 层已经形成参数解析、队列管理、delay 估计和 merge 闭环。

### 10. ONNX launcher 主循环静态阅读

```bash
nl -ba scripts/run_pi05_onnx_infer_so101.py | sed -n '680,980p'
```

结果：

- 退出码：`0`
- `predict_processed_action_chunk(...)` 直接调用 `policy.predict_action_chunk(policy_observation)`，没有额外 RTC kwargs。
- `AsyncChunkPrefetcher` 的 `_submit_impl(...)`、`predict_sync(...)`、`maybe_submit(...)` 也都没有 RTC 参数通道。
- 主循环使用的是 `action_queue: deque[Any] = deque()`，不是 `ActionQueue`。
- 这说明 ONNX launcher 仍停留在旧的 async chunk 路径，没有完成 RTC launcher 层接入。

### 11. 语法编译检查

```bash
python -m py_compile scripts/pi05_chunk_runtime.py scripts/trt_pi_adapter.py scripts/onnx_pi_adapter.py scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py
```

结果：

- 退出码：`0`
- 5 个目标脚本全部通过 `py_compile`。
- 当前没有检测到语法错误。

### 12. TRT launcher `--help` 检查

```bash
python scripts/run_pi05_trt_infer_so101.py --help
```

结果：

- 退出码：`0`
- `--help` 正常输出。
- CLI 中明确出现了 RTC 参数：
  - `--rtc-enable`
  - `--rtc-execution-horizon`
  - `--rtc-max-guidance-weight`
  - `--rtc-prefix-attention-schedule {zeros,ones,linear,exp}`
  - `--rtc-debug`
  - `--rtc-debug-maxlen`
- TRT launcher 已完成 RTC CLI 层接入。

### 13. ONNX launcher `--help` 检查

```bash
python scripts/run_pi05_onnx_infer_so101.py --help
```

结果：

- 退出码：`0`
- `--help` 正常输出。
- 没有出现任何 RTC 参数。
- 仅保留普通 ONNX 推理和 async chunk 相关参数。
- ONNX launcher 未完成 RTC CLI 层接入。

### 14. 首次轻量导入检查

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

paths = [
    Path('scripts/pi05_chunk_runtime.py'),
    Path('scripts/trt_pi_adapter.py'),
    Path('scripts/onnx_pi_adapter.py'),
]
for path in paths:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    print(f'IMPORT_OK {path}')
PY
```

结果：

- 退出码：`1`
- 失败栈停在 `pi05_chunk_runtime.py` 的 `@dataclass` 处理阶段，报错为：
  - `AttributeError: 'NoneType' object has no attribute '__dict__'`
- 这是导入测试夹具问题，不是目标脚本本身的问题。
- 原因是该测试写法没有先把 module 放进 `sys.modules`，与正常 import 机制不一致。

### 15. 修正后的轻量导入检查

```bash
python - <<'PY'
import importlib.util
import sys
from pathlib import Path

paths = [
    Path('scripts/pi05_chunk_runtime.py'),
    Path('scripts/trt_pi_adapter.py'),
    Path('scripts/onnx_pi_adapter.py'),
]
for path in paths:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    print(f'IMPORT_OK {path}')
PY
```

结果：

- 退出码：`0`
- 输出：
  - `IMPORT_OK scripts/pi05_chunk_runtime.py`
  - `IMPORT_OK scripts/trt_pi_adapter.py`
  - `IMPORT_OK scripts/onnx_pi_adapter.py`
- 三个无硬件模块在修正后的轻量导入检查下均可导入。

### 16. launcher RTC 接线聚焦检索

```bash
rg -n "ResolvedRTCRuntimeConfig|add_rtc_runtime_arguments|resolve_rtc_runtime_config|ActionQueue\\(|build_chunk_predict_kwargs|predict_kwargs|rtc_enabled|rtc-" scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py
```

结果：

- 退出码：`0`
- 输出只命中了 `run_pi05_trt_infer_so101.py`，未命中 `run_pi05_onnx_infer_so101.py`。
- 直接证明：
  - TRT launcher 已经有 RTC 参数解析、RTC runtime config、`ActionQueue` 和 `build_chunk_predict_kwargs(...)`
  - ONNX launcher 不具备这些 launcher 层 RTC 关键字和结构

## 结果

### `scripts/pi05_chunk_runtime.py`

- 语法检查通过。
- 轻量导入检查通过。
- 提供了共享的 chunk runtime helper：
  - `ChunkPredictionResult`
  - `prepare_policy_observation(...)`
  - `postprocess_action_chunk(...)`
  - `predict_processed_action_chunk(...)`
  - `AsyncChunkPrefetcher`
- 结论：文件本身可用，但本次检查没有发现 launcher 实际复用它的证据。

### `scripts/trt_pi_adapter.py`

- 语法检查通过。
- 轻量导入检查通过。
- adapter 层已接入 RTC：
  - 初始化 `RTCProcessor`
  - `_rtc_enabled()`
  - `_resolve_rtc_kwargs(...)`
  - `predict_action_chunk(...)` 中通过 `rtc_processor.denoise_step(...)` 走 RTC 路径
- 结论：TRT adapter 的 RTC 能力层已接通。

### `scripts/onnx_pi_adapter.py`

- 语法检查通过。
- 轻量导入检查通过。
- adapter 层也已接入 RTC：
  - 初始化 `RTCProcessor`
  - `_rtc_enabled()`
  - `_resolve_rtc_kwargs(...)`
  - `predict_action_chunk(...)` 中通过 `rtc_processor.denoise_step(...)` 走 RTC 路径
- 结论：ONNX adapter 的 RTC 能力层已接通。

### `scripts/run_pi05_trt_infer_so101.py`

- 语法检查通过。
- `--help` 通过。
- launcher 层包含 RTC CLI、RTC runtime config 解析、`ActionQueue`、delay 估计、`build_chunk_predict_kwargs(...)` 和 `real_delay` merge。
- 结论：TRT launcher 已完成 RTC 接入。

### `scripts/run_pi05_onnx_infer_so101.py`

- 语法检查通过。
- `--help` 通过。
- launcher 层没有 RTC CLI，没有 `ActionQueue`，没有 `ResolvedRTCRuntimeConfig`，也没有向 `policy.predict_action_chunk(...)` 传入 RTC kwargs。
- 结论：ONNX launcher 未完成 RTC 接入。

## 失败项 / 缺口

1. `run_pi05_onnx_infer_so101.py` 未完成 RTC launcher 层接线。
   - 无 RTC CLI。
   - 无 `ActionQueue`。
   - 无 `build_chunk_predict_kwargs(...)`。
   - 无 `prev_chunk_left_over / inference_delay / execution_horizon` 透传。
   - 无 `real_delay` merge。

2. 共享 helper 与 launcher 没有明确复用关系。
   - `pi05_chunk_runtime.py` 已存在共享实现。
   - 但本次静态核查中，没有看到两个 launcher 显式接入该 helper 文件。

3. 首次导入检查命令失败。
   - 失败原因是测试夹具本身没有把模块预先注册到 `sys.modules`。
   - 该失败不构成目标脚本缺陷，修正导入方式后 3 个模块均通过。

4. 本次没有执行任何需要真机的路径。
   - 没有验证相机预检。
   - 没有验证机器人连接。
   - 没有验证串口。
   - 没有验证实时 loop 下的实际 `real_delay` 数值表现。

## 结论

结论明确如下：

- `TRT launcher` 已完成 RTC 接入。
- `ONNX launcher` 未完成 RTC 接入。

更准确地说：

- `scripts/trt_pi_adapter.py` 和 `scripts/onnx_pi_adapter.py` 的 adapter 层都已经具备 RTC 能力。
- 真正完成到 launcher 层闭环接线的只有 `scripts/run_pi05_trt_infer_so101.py`。
- `scripts/run_pi05_onnx_infer_so101.py` 仍是旧的 async chunk launcher 流程，没有把 adapter 层 RTC 能力真正接到入口、队列和 delay 管理层。

本次验证只运行了无硬件命令，没有执行机器人、相机、串口相关操作。
