# Worker A2 Helper Round 3 Retry

## 改动摘要

- 仅修改 `scripts/pi05_chunk_runtime.py`。
- 收紧 `build_chunk_predict_kwargs(...)`：
  - 当 RTC 最终未启用时，如果调用方显式提供了 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 中的任一 RTC-only 输入，不再静默返回 `{}`，而是直接抛出 `ValueError`。
  - 保留当前 launcher 主路径兼容性：`predicted_delay_steps` 仍可在 RTC-off 路径上传入并被忽略，因为 TRT/ONNX launcher 目前都会在 RTC 关闭时传这个参数。
- 收紧 `merge_chunk_prediction_result(...)`：
  - 不再在无法可靠确定 delay 时默默回落成 `0`。
  - 现在只接受三类可靠来源：
    - 显式 `real_delay`
    - `action_index_before_inference` + `action_index_after_inference` / `action_queue.get_action_index()`
    - 已写入 `prediction.real_delay`
  - 如果三类来源都不可用，会抛出带上下文细节的 `ValueError`。
- 增加更明确的异常信息，方便 future caller 在 shared helper 层就发现契约不完整的问题，而不是在运行中静默退化。

## 自检命令

已执行：

```bash
python -m py_compile scripts/pi05_chunk_runtime.py scripts/run_pi05_trt_infer_so101.py scripts/run_pi05_onnx_infer_so101.py
python - <<'PY'
from types import SimpleNamespace
import torch

from scripts.pi05_chunk_runtime import (
    ChunkPredictionResult,
    build_chunk_predict_kwargs,
    merge_chunk_prediction_result,
)


class DummyQueue:
    def __init__(self, *, enabled: bool, execution_horizon: int, left_over=None, action_index: int = 0):
        self.cfg = SimpleNamespace(enabled=enabled, execution_horizon=execution_horizon)
        self._left_over = left_over
        self._action_index = action_index
        self.merge_calls = []

    def get_left_over(self):
        return self._left_over

    def get_action_index(self):
        return self._action_index

    def merge(self, original_actions, processed_actions, *, real_delay, action_index_before_inference=None):
        self.merge_calls.append(
            {
                "real_delay": int(real_delay),
                "action_index_before_inference": action_index_before_inference,
                "original_shape": tuple(original_actions.shape),
                "processed_shape": tuple(processed_actions.shape),
            }
        )


def make_prediction(*, before_idx, real_delay=None):
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    processed = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    processed_tensor = torch.stack(processed, dim=0)
    return ChunkPredictionResult(
        original_actions=original,
        processed_actions=processed,
        preprocess_time_s=0.01,
        inference_time_s=0.02,
        postprocess_time_s=0.03,
        processed_actions_tensor=processed_tensor,
        action_index_before_inference=before_idx,
        real_delay=real_delay,
    )


rtc_runtime_off = SimpleNamespace(config=SimpleNamespace(enabled=False, execution_horizon=7))
queue_off = DummyQueue(enabled=False, execution_horizon=7, left_over=torch.ones((2, 2)), action_index=0)
assert build_chunk_predict_kwargs(
    rtc_runtime=rtc_runtime_off,
    action_queue=queue_off,
    predicted_delay_steps=0,
) == {}

rtc_runtime_on = SimpleNamespace(config=SimpleNamespace(enabled=True, execution_horizon=9))
queue_on = DummyQueue(
    enabled=True,
    execution_horizon=9,
    left_over=torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32),
    action_index=5,
)
kwargs_on = build_chunk_predict_kwargs(
    rtc_runtime=rtc_runtime_on,
    action_queue=queue_on,
    predicted_delay_steps=2,
)
assert kwargs_on["inference_delay"] == 2
assert kwargs_on["execution_horizon"] == 9
assert torch.equal(kwargs_on["prev_chunk_left_over"], queue_on.get_left_over())
assert kwargs_on["prev_chunk_left_over"] is not queue_on.get_left_over()

async_prediction = make_prediction(before_idx=3)
async_delay = merge_chunk_prediction_result(queue_on, async_prediction)
assert async_delay == 2
assert queue_on.merge_calls[-1]["real_delay"] == 2

sync_prediction = make_prediction(before_idx=5)
queue_on._action_index = 99
sync_delay = merge_chunk_prediction_result(queue_on, sync_prediction, real_delay=0)
assert sync_delay == 0
assert queue_on.merge_calls[-1]["real_delay"] == 0

try:
    build_chunk_predict_kwargs(rtc_enabled=False, inference_delay=1)
except ValueError as exc:
    assert "RTC predict kwargs" in str(exc)
else:
    raise AssertionError("expected build_chunk_predict_kwargs to fail fast")


class MergeOnlyQueue:
    def merge(self, original_actions, processed_actions, *, real_delay, action_index_before_inference=None):
        return None


try:
    merge_chunk_prediction_result(MergeOnlyQueue(), make_prediction(before_idx=None))
except ValueError as exc:
    assert "reliable `real_delay`" in str(exc)
else:
    raise AssertionError("expected merge_chunk_prediction_result to require explicit delay context")

print("helper_launcher_style_smoke_ok")
PY
```

结果：

- `py_compile` 通过。
- 轻量 smoke 输出 `helper_launcher_style_smoke_ok`。
- 本轮未做真机、相机、ONNX Runtime 或 TensorRT engine 的硬件执行。

## 契约变化

### `build_chunk_predict_kwargs(...)`

- 旧行为：
  - 只要 `_resolve_rtc_enabled(...)` 最终为 `False`，helper 直接返回 `{}`。
  - 即使调用方已经显式给了 RTC 相关输入，也会被静默吞掉。
- 新行为：
  - RTC 未启用且显式提供了 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 之一时，立即报错。
  - RTC 未启用但仅传 launcher 现有惯例中的 `predicted_delay_steps` 时，仍返回 `{}`，避免打断 TRT/ONNX 当前 RTC-off 主路径。

### `merge_chunk_prediction_result(...)`

- 旧行为：
  - 如果拿不到 `action_index_after_inference`，`action_queue` 也没有 `get_action_index()`，同时 `prediction.real_delay` 为空，helper 最终会把 delay 静默记成 `0`。
- 新行为：
  - 必须能从以下任一来源可靠拿到 delay：
    - 显式 `real_delay`
    - `action_index_before_inference` 与 `action_index_after_inference` / `action_queue.get_action_index()`
    - `prediction.real_delay`
  - 否则抛出清晰异常，不再伪造 `real_delay=0`。

## 与当前 TRT/ONNX launcher 的兼容性说明

- TRT launcher 当前主路径兼容：
  - `build_chunk_predict_kwargs(...)` 的 RTC-off 调用只传 `predicted_delay_steps`，不会触发新报错。
  - initial sync、async collect、async wait、sync refill 这几条 merge 路径都保留了 `prediction.action_index_before_inference`，并且 `ActionQueue` 提供 `get_action_index()`，所以 shared helper 仍能计算真实 delay。
- ONNX launcher 当前主路径兼容：
  - RTC-off 同样只依赖 `predicted_delay_steps`，不会被新契约挡住。
  - async collect / async wait 仍可通过 action index 推断 delay。
  - initial sync / sync refill 本来就显式传 `real_delay=0`，因此不会受“禁止默认 0 回落”影响。
- 轻量 smoke 已按上述 launcher 风格分别覆盖：
  - RTC-off build
  - RTC-on build
  - async merge 由 index 推断 `real_delay`
  - sync merge 显式 `real_delay=0`

## 剩余风险

- 本轮没有把这些契约收紧补成 repo 里的正式自动化测试，只做了本地 smoke；如果后续 launcher 再改调用形态，仍需补对应测试。
- 由于现有函数签名里 `inference_delay` 和 `execution_horizon` 的默认值本身就是 `None`，helper 无法区分“调用方没传”和“调用方显式传了 `None`”；要彻底识别这两种情况，需要额外引入 sentinel，属于更明显的接口变化，本轮未做。
- 为兼容当前 launcher RTC-off 主路径，`predicted_delay_steps` 在 RTC 未启用时仍会被忽略；这属于保留兼容的明确例外，而不是新的静默退化点。
- 未做真机时序和真实模型联调，因此这里只能证明 shared helper 契约收紧后不会打断当前 TRT/ONNX launcher 的主路径参数形态，不能证明硬件时延表现。
