# Review Execution Round 2

## 验证范围

- 工作目录：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt`
- 验证对象：
  - `scripts/pi05_chunk_runtime.py`
  - `scripts/run_pi05_trt_infer_so101.py`
  - `scripts/run_pi05_onnx_infer_so101.py`
- 验证目标：仅验证“无硬件可执行性”，不做代码修改，不做真机、相机、TensorRT engine、ONNX artifact 联调。
- 参考但未照抄：
  - `docs/results/pi_rtc_accel_execution_20260314_231604/worker_a_shared_helper_round2.md`
  - `docs/results/pi_rtc_accel_execution_20260314_231604/worker_b_trt_launcher_round2.md`
  - `docs/results/pi_rtc_accel_execution_20260314_231604/worker_c_onnx_launcher_round2.md`

补充观察：

- `git status --short` 显示本轮 3 个脚本和当前结果目录均为未跟踪状态；本次验证未对其做回退或内容修改。

## 执行命令

1. `python -m py_compile scripts/pi05_chunk_runtime.py scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py`
2. `python scripts/run_pi05_trt_infer_so101.py --help`
3. `python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help`
4. `python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help`
5. `python scripts/run_pi05_onnx_infer_so101.py --help`
6. `python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help`
7. `python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help`
8. `rg -n "from pi05_chunk_runtime import|ActionQueue|real_delay|rtc_enabled|sync_refill_timeout_s" scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py`

辅助静态核对：

- `rg -n "__all__|ChunkPredictionResult|AsyncChunkPrefetcher|compute_real_delay|estimate_prefetch_threshold" scripts/pi05_chunk_runtime.py`

## 结果

### 1. py_compile

- 退出码：`0`
- 结果：通过。
- 结论：3 个目标脚本在当前 Python 环境下均可完成语法级编译，未出现语法错误或导入期编译异常。

### 2. TRT launcher `--help`

- 命令：`python scripts/run_pi05_trt_infer_so101.py --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 关键点：
  - 帮助中可见 `--prefetch-threshold`
  - 帮助中可见 `--sync-refill-timeout-s`
  - 详细参数区可见 `--rtc-enable, --rtc-enabled`

### 3. TRT launcher `--rtc-enabled --help`

- 命令：`python scripts/run_pi05_trt_infer_so101.py --rtc-enabled --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 结论：兼容别名 `--rtc-enabled` 可被 argparse 正常接受。

### 4. TRT launcher `--rtc-enable --help`

- 命令：`python scripts/run_pi05_trt_infer_so101.py --rtc-enable --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 结论：主参数 `--rtc-enable` 可被 argparse 正常接受。

### 5. ONNX launcher `--help`

- 命令：`python scripts/run_pi05_onnx_infer_so101.py --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 关键点：
  - 帮助中可见 `--prefetch-threshold`
  - 帮助中可见 `--sync-refill-timeout-s`
  - 详细参数区可见 `--rtc-enable, --rtc-enabled`

### 6. ONNX launcher `--rtc-enabled --help`

- 命令：`python scripts/run_pi05_onnx_infer_so101.py --rtc-enabled --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 结论：兼容别名 `--rtc-enabled` 可被 argparse 正常接受。

### 7. ONNX launcher `--rtc-enable --help`

- 命令：`python scripts/run_pi05_onnx_infer_so101.py --rtc-enable --help`
- 退出码：`0`
- 结果：成功打印帮助信息。
- 结论：主参数 `--rtc-enable` 可被 argparse 正常接受。

### 8. `rg` 关键词命中检查

#### shared helper import

- TRT：
  - `scripts/run_pi05_trt_infer_so101.py:33-37` 命中 `from pi05_chunk_runtime import (...)`
- ONNX：
  - `scripts/run_pi05_onnx_infer_so101.py:32` 命中 `from pi05_chunk_runtime import AsyncChunkPrefetcher, ChunkPredictionResult, compute_real_delay, estimate_prefetch_threshold`

#### ActionQueue

- TRT：
  - `scripts/run_pi05_trt_infer_so101.py:46`
  - `scripts/run_pi05_trt_infer_so101.py:1262`
  - `scripts/run_pi05_trt_infer_so101.py:1281`
  - `scripts/run_pi05_trt_infer_so101.py:1630`
- ONNX：
  - `scripts/run_pi05_onnx_infer_so101.py:40`
  - `scripts/run_pi05_onnx_infer_so101.py:799`
  - `scripts/run_pi05_onnx_infer_so101.py:818`
  - `scripts/run_pi05_onnx_infer_so101.py:925`

#### real_delay

- TRT：
  - `scripts/run_pi05_trt_infer_so101.py:1282-1292`
  - `scripts/run_pi05_trt_infer_so101.py:1656`
  - `scripts/run_pi05_trt_infer_so101.py:1701`
  - `scripts/run_pi05_trt_infer_so101.py:1744`
  - `scripts/run_pi05_trt_infer_so101.py:1776`
  - `scripts/run_pi05_trt_infer_so101.py:1790`
  - `scripts/run_pi05_trt_infer_so101.py:1842`
- ONNX：
  - `scripts/run_pi05_onnx_infer_so101.py:818-828`
  - `scripts/run_pi05_onnx_infer_so101.py:932`
  - `scripts/run_pi05_onnx_infer_so101.py:982`
  - `scripts/run_pi05_onnx_infer_so101.py:999`
  - `scripts/run_pi05_onnx_infer_so101.py:1038`
  - `scripts/run_pi05_onnx_infer_so101.py:1076`
  - `scripts/run_pi05_onnx_infer_so101.py:1129`

#### rtc_enabled

- TRT：
  - `scripts/run_pi05_trt_infer_so101.py:291`
  - `scripts/run_pi05_trt_infer_so101.py:297`
  - `scripts/run_pi05_trt_infer_so101.py:1683`
  - `scripts/run_pi05_trt_infer_so101.py:1720`
- ONNX：
  - `scripts/run_pi05_onnx_infer_so101.py:778`
  - `scripts/run_pi05_onnx_infer_so101.py:784`
  - `scripts/run_pi05_onnx_infer_so101.py:982`
  - `scripts/run_pi05_onnx_infer_so101.py:1017`
  - `scripts/run_pi05_onnx_infer_so101.py:1128`

#### sync_refill_timeout_s

- TRT：
  - `scripts/run_pi05_trt_infer_so101.py:1389`
  - `scripts/run_pi05_trt_infer_so101.py:1613-1614`
  - `scripts/run_pi05_trt_infer_so101.py:1682`
  - `scripts/run_pi05_trt_infer_so101.py:1741`
  - `scripts/run_pi05_trt_infer_so101.py:1764-1766`
- ONNX：
  - `scripts/run_pi05_onnx_infer_so101.py:646`
  - `scripts/run_pi05_onnx_infer_so101.py:912-913`
  - `scripts/run_pi05_onnx_infer_so101.py:981`
  - `scripts/run_pi05_onnx_infer_so101.py:1036`

#### shared helper 源文件导出确认

- `scripts/pi05_chunk_runtime.py:419` 命中 `compute_real_delay`
- `scripts/pi05_chunk_runtime.py:541` 命中 `estimate_prefetch_threshold`
- `scripts/pi05_chunk_runtime.py:563` 命中 `AsyncChunkPrefetcher`
- `scripts/pi05_chunk_runtime.py:683-688` 命中 `__all__`，其中包含：
  - `AsyncChunkPrefetcher`
  - `ChunkPredictionResult`
  - `compute_real_delay`
  - `estimate_prefetch_threshold`

结论：TRT/ONNX 两边都已经命中要求的关键字，且共享 helper 所需导出符号在 `pi05_chunk_runtime.py` 中存在。

## 失败项 / 异常

- 强制执行的 8 项命令中，未出现失败项。
- 未出现 Python traceback、参数解析错误、缺失导入导致的启动失败。
- 一个非阻塞观察：
  - `argparse` 的 usage 摘要行只展示主长参数 `--rtc-enable`，但详细参数区展示 `--rtc-enable, --rtc-enabled`，且命令 3、4、6、7 全部退出码为 `0`。这不是失败，只是 argparse 对 alias 的常见展示行为。

## TRT 结论

- `scripts/run_pi05_trt_infer_so101.py` 在当前无硬件环境下通过了帮助入口验证，说明至少在“脚本启动、依赖导入、CLI 参数注册、RTC alias 解析”这一层面可执行。
- `rg` 静态核对显示 TRT 侧已经接入共享 helper import、`ActionQueue`、`real_delay`、`rtc_enabled`、`sync_refill_timeout_s`。
- 基于本次验证，可以确认 TRT launcher 达到无硬件 smoke-test 级可执行；但不能据此替代真机、相机、engine artifact 的运行验证。

## ONNX 结论

- `scripts/run_pi05_onnx_infer_so101.py` 在当前无硬件环境下通过了帮助入口验证，说明至少在“脚本启动、依赖导入、CLI 参数注册、RTC alias 解析”这一层面可执行。
- `rg` 静态核对显示 ONNX 侧已经接入共享 helper import、`ActionQueue`、`real_delay`、`rtc_enabled`、`sync_refill_timeout_s`。
- 基于本次验证，可以确认 ONNX launcher 达到无硬件 smoke-test 级可执行；但不能据此替代真机、相机、ONNX artifact/provider 的运行验证。

## 总体结论

- 本轮指定的 3 个脚本通过了无硬件可执行性验证范围内的全部强制检查。
- 结论边界：
  - 已验证：语法编译、脚本启动、帮助入口、RTC 参数兼容别名、TRT/ONNX 两侧关键接线的静态命中。
  - 未验证：真实 robot/camera、TensorRT engine 加载、ONNX Runtime provider 执行、完整推理循环与时序行为。
- 在“只看当前环境、只做无硬件验证”的前提下，本轮改动可判定为通过。
