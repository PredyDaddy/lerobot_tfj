# Worker Launcher Grad Fix

## 分析

- 问题点位收敛在 `scripts/run_pi05_torch_infer_so101.py` 的 `TorchChunkPolicyRuntime.predict_action_chunk(...)`。
- RTC 开启时，PI0.5 的 RTC guidance 会在 chunk denoising 过程中调用 `torch.autograd.grad(...)`。如果 launcher/runtime 把这段路径包在 `torch.inference_mode()` 里，autograd 会失效。
- RTC 关闭时，launcher 仍应尽量保持原有高效纯推理语义，因此不能简单全局移除 `torch.inference_mode()`。
- 当前最稳妥的边界是：
  - `RTC off`：仍由 launcher wrapper 主动进入 `torch.inference_mode()`。
  - `RTC on`：launcher wrapper 主动切到 `torch.enable_grad()`，并且如果已经检测到外层先进入了 `torch.inference_mode()`，直接 fail-fast，而不是让 RTC 路径静默失效。

## 改动点

- 在 `scripts/run_pi05_torch_infer_so101.py` 的 `TorchChunkPolicyRuntime` 内新增 `_predict_grad_context()`。
- 该 helper 的行为：
  - `RTC off` 时返回 `torch.inference_mode()`，保留原有高效推理语义。
  - `RTC on` 时先检查当前线程是否已经处于 `torch.inference_mode()`。
  - 如果 `RTC on` 且外层已经开启 `torch.inference_mode()`，抛出 `RuntimeError`，错误信息明确要求移除 RTC 路径上的外层 inference wrapper。
  - 如果 `RTC on` 且未处于 inference mode，则返回 `torch.enable_grad()`，保证 `policy.predict_action_chunk(...)` 内部可以正常走 autograd。
- `predict_action_chunk(...)` 现在统一通过 `_predict_grad_context()` 选择 grad/inference 上下文，再叠加原有的 CUDA AMP autocast。

## 自检

执行命令：

```bash
python -m py_compile scripts/run_pi05_torch_infer_so101.py
python - <<'PY'
import sys
import torch
sys.path.insert(0, 'scripts')
from run_pi05_torch_infer_so101 import TorchChunkPolicyRuntime

class _RTCConfig:
    def __init__(self, enabled):
        self.enabled = enabled

class _Config:
    def __init__(self, enabled):
        self.rtc_config = _RTCConfig(enabled)

class _Policy:
    def __init__(self, enabled):
        self.config = _Config(enabled)
    def predict_action_chunk(self, batch, **kwargs):
        return torch.tensor([
            1.0 if torch.is_grad_enabled() else 0.0,
            1.0 if getattr(torch, "is_inference_mode_enabled", lambda: False)() else 0.0,
        ], dtype=torch.float32)

runtime_off = TorchChunkPolicyRuntime(_Policy(False), device=torch.device("cpu"), use_amp=False)
assert runtime_off.predict_action_chunk({}).tolist() == [0.0, 1.0]

runtime_on = TorchChunkPolicyRuntime(_Policy(True), device=torch.device("cpu"), use_amp=False)
assert runtime_on.predict_action_chunk({}).tolist() == [1.0, 0.0]

try:
    with torch.inference_mode():
        runtime_on.predict_action_chunk({})
except RuntimeError as exc:
    assert "torch.inference_mode()" in str(exc), str(exc)
else:
    raise AssertionError("Expected RTC runtime to fail fast under outer inference_mode().")

print("self-check ok")
PY
```

结果：

- `py_compile` 通过。
- 轻量 stub 输出 `self-check ok`。
- 已验证三条关键语义：
  - `RTC off -> grad disabled + inference_mode enabled`
  - `RTC on -> grad enabled + inference_mode disabled`
  - `RTC on + outer inference_mode -> fail-fast`

## 风险

- 本次只修 launcher 包裹边界，没有改 policy 内部实现。如果后续有人在更外层重新加了全局 `torch.inference_mode()`，RTC 路径现在会显式报错；这比静默梯度失效更安全，但仍需要调用方接受这个新失败模式。
- 未做真机/相机/机械臂联调，本轮证据只覆盖导入级与纯 stub 上下文行为。
- `torch.is_inference_mode_enabled()` 的检测依赖当前 PyTorch 提供该 API。代码里做了 `getattr(...)` 保护；如果遇到更旧的 PyTorch 版本而该 API 缺失，则不会做这层 fail-fast 检查，但 `RTC off/on` 的上下文切换逻辑仍然存在。

## 建议上机命令

先做无硬件导入检查：

```bash
python scripts/run_pi05_torch_infer_so101.py --dry-run --policy-device cuda
```

再做无机械臂动作的 preflight 检查：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --policy-device cuda \
  --policy-use-amp \
  --preflight-only
```

最后做 RTC 开启的小时长实跑验证：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --policy-device cuda \
  --policy-use-amp \
  --rtc-enable \
  --rtc-execution-horizon 8 \
  --run-time-s 20 \
  --joint-delta-limit 0.15 \
  --robot-max-relative-target 0.15
```

建议重点观察：

- 启动日志中的 `Resolved RTC config`
- 运行期是否出现新的 `RuntimeError`，提示外层 `torch.inference_mode()` 仍在 RTC 路径上
- RTC 开启时是否还能稳定生成 chunk，而不是在 denoising 阶段报 autograd 相关错误
