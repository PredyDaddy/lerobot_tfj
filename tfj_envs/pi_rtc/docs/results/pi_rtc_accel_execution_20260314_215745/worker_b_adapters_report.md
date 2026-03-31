# Worker B Adapters Report

## 改动摘要

本轮按要求只在 adapter 层接入 RTCProcessor，没有改 engine / ONNX graph 边界，也没有把 `ActionQueue` 塞进 adapter。

已完成的代码改动：

1. `scripts/trt_pi_adapter.py`
   - 在 adapter 内持有 `self.rtc_processor`，当 `config.rtc_config` 存在时实例化 `RTCProcessor`
   - 新增 `_rtc_enabled()` helper
   - 新增 RTC kwargs 解析与输入规范化 helper：
     - `_prepare_prev_chunk_left_over(...)`
     - `_resolve_rtc_kwargs(...)`
   - `predict_action_chunk(...)` 现在消费：
     - `prev_chunk_left_over`
     - `inference_delay`
     - `execution_horizon`
   - 在 denoise loop 内定义单步 `denoise_step_partial_call(...)` closure，并在 RTC on 时通过 `self.rtc_processor.denoise_step(...)` 包装
   - RTC off 时继续直接调用原始单步 denoise closure，保持原有行为路径

2. `scripts/onnx_pi_adapter.py`
   - 同样在 adapter 内持有 `self.rtc_processor`
   - 新增 `_rtc_enabled()` helper
   - 新增 RTC kwargs 解析与 leftover 规范化 helper
   - `predict_action_chunk(...)` 支持消费 `prev_chunk_left_over / inference_delay / execution_horizon`
   - 在 denoise loop 中用 RTCProcessor 包裹单步 ONNX denoise closure
   - RTC off 时仍保留原始单步 ONNX denoise 路径

3. 两个 adapter 都额外补了运行时摘要字段：
   - `rtc_enabled`
   - `rtc_debug_enabled`

## 边界说明

本轮没有做的事：

- 没有修改 TensorRT engine 输入输出名、shape、session contract
- 没有修改 ONNX 模型边界
- 没有引入 `ActionQueue`
- 没有改 launcher / async prefetch / queue merge 逻辑

因此本轮是一个严格的 adapter 内 RTC 接口接入，不是完整 runtime queue 改造。

## 自检命令与结果

1. 语法编译检查

命令：

```bash
python -m py_compile scripts/trt_pi_adapter.py scripts/onnx_pi_adapter.py
```

结果：

- 退出码 `0`
- 两个文件均通过 `py_compile`

2. RTC 结构落点检查

命令：

```bash
rg -n "rtc_processor|def _rtc_enabled|prev_chunk_left_over|inference_delay|execution_horizon|original_denoise_step_partial" scripts/trt_pi_adapter.py scripts/onnx_pi_adapter.py -n -S
```

结果：

- `scripts/trt_pi_adapter.py`
  - `self.rtc_processor` 已落在 `86`
  - `_rtc_enabled()` 已落在 `241`
  - RTC kwargs 解析已落在 `481-527`
  - RTC 包裹 denoise closure 已落在 `567-591`
- `scripts/onnx_pi_adapter.py`
  - `self.rtc_processor` 已落在 `97`
  - `_rtc_enabled()` 已落在 `170`
  - RTC kwargs 解析已落在 `286-332`
  - RTC 包裹 denoise closure 已落在 `372-397`

## 剩余风险

1. 这次只做了 adapter 接口接入，没有做 `ActionQueue` 级别的 runtime 集成。
   - 所以 `predict_action_chunk(...)` 已具备 RTC 输入能力
   - 但完整的 leftover 维护、real delay 计算、queue merge 仍需要 launcher/runtime 层配合

2. TRT / ONNX 路径里的 RTCProcessor 仍然复用了现有 autograd-based guidance wrapper，但单步 denoise 本体来自外部 runner。
   - 这意味着它的数值行为已经接上 RTC 接口
   - 但和纯 PyTorch denoiser 的 RTC guidance 是否完全一致，本轮没有做数值一致性验证

3. 当前实现在传入 `prev_chunk_left_over` 但未传 `inference_delay` 时会显式报错。
   - 这是为了避免在 RTCProcessor 内部落成更晚、更隐晦的类型错误

4. `select_action()` 仍保持 adapter 本地 `deque` 语义。
   - 这是按要求避免把 `ActionQueue` 塞进 adapter
   - 也意味着完整 RTC 实时执行收益仍取决于后续 launcher 侧改造

## 产出范围

本轮代码改动文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/trt_pi_adapter.py`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/onnx_pi_adapter.py`

本轮报告文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_rtc_accel_execution_20260314_215745/worker_b_adapters_report.md`
