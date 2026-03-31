# PI0.5 Policy RTC Grad Fix

## 范围

- 工作目录：`/data/tfj/lerobot_tfj`
- 代码所有权约束：只修改 `src/lerobot/policies/pi05/modeling_pi05.py`
- 本次目的：修复 policy 封装侧 RTC 梯度屏蔽问题，使 RTC on 时保留 `RTCProcessor.autograd.grad(...)` 所需梯度，RTC off 时尽量保持原推理语义
- 记录时间：`2026-03-15 22:16:28 CST`

## 改动摘要

仅修改了 `src/lerobot/policies/pi05/modeling_pi05.py`，没有改其他代码文件。

### 1. `PI05Policy.predict_action_chunk(...)`

- 去掉了函数级 `@torch.no_grad()`
- 改成条件化推理上下文：
  - RTC on：`nullcontext()`
  - RTC off：`torch.no_grad()`

这样 RTC 打开时，policy wrapper 不再把整条 chunk 推理路径硬性包在 no-grad 里；RTC 关闭时仍保持原有 no-grad 推理语义。

### 2. `PI05Pytorch.sample_actions(...)`

- 去掉了函数级 `@torch.no_grad()`
- 保留 prefix/cache 这段在 `torch.no_grad()` 下执行，避免无关图构建
- 在 denoise closure 内增加条件化 grad gate：
  - 仅当 `self._rtc_enabled()` 且 `prev_chunk_left_over is not None` 时
  - 使用 `torch.enable_grad()`
  - 并对传入的 `input_x_t` 显式执行 `requires_grad_(True)`

这一步是本次修复的关键。RTCProcessor 在内部会回调 policy 提供的 denoise closure 并调用 `autograd.grad(...)`；如果 closure 没有让 `x_t` 参与图构建，那么 guidance 会退化成无 Jacobian 的路径。

### 3. 保持不变

- `select_action(...)` 仍保留 `@torch.no_grad()`
- 该路径本身已有断言：RTC 不支持 `select_action`，必须走 `predict_action_chunk`

## 关键位置

- `src/lerobot/policies/pi05/modeling_pi05.py:750`
  - `sample_actions(...)` 不再是函数级 `@torch.no_grad()`
- `src/lerobot/policies/pi05/modeling_pi05.py:776-790`
  - prefix/cache 继续走 `torch.no_grad()`
- `src/lerobot/policies/pi05/modeling_pi05.py:801-813`
  - RTC guidance active 时对 denoise closure 启用 `torch.enable_grad()` 与 `requires_grad_(True)`
- `src/lerobot/policies/pi05/modeling_pi05.py:1237-1254`
  - `predict_action_chunk(...)` 不再是函数级 `@torch.no_grad()`
  - RTC on/off 走条件化上下文

## 执行命令与结果

### 1. 语法检查

```bash
python -m py_compile src/lerobot/policies/pi05/modeling_pi05.py
```

- 结果：通过
- 退出码：`0`

### 2. 源码核对

```bash
rg -n "@torch\\.no_grad\\(|def predict_action_chunk|nullcontext\\(|torch\\.enable_grad\\(|requires_grad_\\(|with torch\\.no_grad\\(" src/lerobot/policies/pi05/modeling_pi05.py
```

- 结果确认：
  - `predict_action_chunk` 已不再有 `@torch.no_grad()`
  - `predict_action_chunk` 使用 `nullcontext() if self._rtc_enabled() else torch.no_grad()`
  - `sample_actions` 内存在 `torch.enable_grad()`
  - denoise closure 内存在 `input_x_t.requires_grad_(True)`
  - prefix/cache 仍在 `with torch.no_grad():` 下

### 3. 本地 RTC/autograd 最小验证

```bash
python - <<'PY'
import torch
from types import SimpleNamespace
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

cfg = SimpleNamespace(
    enabled=True,
    execution_horizon=2,
    max_guidance_weight=10.0,
    prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
    debug=False,
    debug_maxlen=32,
)
rtc = RTCProcessor(cfg)

x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
prev = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
time = torch.tensor(0.5)

@torch.no_grad()
def run_without_grad_gate():
    def closure(inp):
        return inp * inp
    return rtc.denoise_step(
        x_t=x,
        prev_chunk_left_over=prev,
        inference_delay=1,
        time=time,
        original_denoise_step_partial=closure,
        execution_horizon=1,
    )

@torch.no_grad()
def run_with_grad_gate():
    def closure(inp):
        inp = inp.requires_grad_(True)
        return inp * inp
    return rtc.denoise_step(
        x_t=x,
        prev_chunk_left_over=prev,
        inference_delay=1,
        time=time,
        original_denoise_step_partial=closure,
        execution_horizon=1,
    )

print('without_grad_gate', run_without_grad_gate())
print('with_grad_gate', run_with_grad_gate())
PY
```

- 结果：
  - `without_grad_gate tensor([[1., 1.]])`
  - `with_grad_gate tensor([[1., 7.]])`

解释：

- 在外层 `@torch.no_grad()` 仍存在的情况下，只要 RTC 回调的 closure 内显式给 `x_t` 打开 grad，并在 grad-enabled 上下文里执行 denoise，RTCProcessor 的 `autograd.grad(...)` 就能走到非退化路径
- 这正是本次在 PI0.5 policy 封装侧补上的链路

### 4. 现成 PI0.5 RTC 测试尝试

先尝试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/policies/pi0_pi05/test_pi05_rtc.py -q -k test_pi05_rtc_inference_with_prev_chunk
```

- 结果：当前 `lerobot` 环境无 `pytest`
- 错误：`No module named pytest`

改为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/policies/pi0_pi05/test_pi05_rtc.py -q -k test_pi05_rtc_inference_with_prev_chunk
```

- 结果：测试未通过，但失败原因不是本次代码断言，而是环境无法联网下载模型
- 退出码：`1`
- 关键错误：
  - 请求 `https://huggingface.co/google/paligemma-3b-pt-224/...`
  - `Network is unreachable`
  - `We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`

结论：

- 该端到端测试当前被远端模型依赖阻塞，不能作为本次改动正确性或错误性的直接证据

## 结论

- 修复已落在 policy 封装侧，且满足“最小且正确”的目标：
  - RTC on：`predict_action_chunk` 不再强制 no-grad，denoise closure 显式保留 `RTCProcessor.autograd.grad(...)` 所需梯度
  - RTC off：`predict_action_chunk` 和 `sample_actions` 仍尽量维持原先的 no-grad 推理语义
- 本地可验证结果包括：
  - 目标文件编译通过
  - 源码路径满足预期
  - RTC/autograd 最小复现实验证明 grad gate 的必要性和有效性
- 尚未完成的验证：
  - 真正的 PI0.5 RTC 端到端测试受 Hugging Face 模型下载阻塞，待本地缓存齐全或网络恢复后可重跑
