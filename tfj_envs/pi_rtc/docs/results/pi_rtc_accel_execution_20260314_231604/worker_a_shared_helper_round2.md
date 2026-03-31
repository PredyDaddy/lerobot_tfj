# Worker A Shared Helper Round 2

## 改动摘要

- 在 `scripts/pi05_chunk_runtime.py` 内新增通用 RTC glue：`build_chunk_predict_kwargs(...)`。
- 该 helper 会在 RTC 关闭时直接返回空字典，在 RTC 开启时统一产出 `inference_delay`、`prev_chunk_left_over`、`execution_horizon` 三个 `predict_action_chunk(...)` kwargs。
- `build_chunk_predict_kwargs(...)` 不依赖 TRT launcher 私有 dataclass。它既兼容当前 TRT 风格的 `rtc_runtime + action_queue + predicted_delay_steps`，也支持显式传入 `rtc_enabled`、`inference_delay`、`execution_horizon`、`prev_chunk_left_over` 或 `left_over_provider`。
- 新增轻量 queue merge convenience helper：`merge_chunk_prediction_result(...)`。该函数只要求外部对象提供 `merge(...)`，可选提供 `get_action_index()`，不把 `ActionQueue` 类型搬进共享层。
- 保持现有共享 schema 和 runtime 能力不变：`ChunkPredictionResult`、`predict_processed_action_chunk(...)`、`AsyncChunkPrefetcher` 的既有职责和对外形状未被破坏。

## 自检命令

```bash
python -m py_compile scripts/pi05_chunk_runtime.py
python - <<'PY'
from scripts.pi05_chunk_runtime import ChunkPredictionResult, build_chunk_predict_kwargs, merge_chunk_prediction_result
import torch

class RtcConfigStub:
    def __init__(self, enabled=True, execution_horizon=7):
        self.enabled = enabled
        self.execution_horizon = execution_horizon

class RuntimeStub:
    def __init__(self):
        self.config = RtcConfigStub(enabled=True, execution_horizon=9)

class QueueStub:
    def __init__(self):
        self.cfg = RtcConfigStub(enabled=True, execution_horizon=5)
        self._left_over = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self._action_index = 4
        self.merged = None

    def get_left_over(self):
        return self._left_over

    def get_action_index(self):
        return self._action_index

    def merge(self, original_actions, processed_actions, real_delay, action_index_before_inference=0):
        self.merged = {
            "original_actions": original_actions.clone(),
            "processed_actions": processed_actions.clone(),
            "real_delay": int(real_delay),
            "action_index_before_inference": action_index_before_inference,
        }

queue = QueueStub()
kwargs = build_chunk_predict_kwargs(
    rtc_runtime=RuntimeStub(),
    action_queue=queue,
    predicted_delay_steps=2,
)
assert kwargs["inference_delay"] == 2
assert kwargs["execution_horizon"] == 9
assert torch.equal(kwargs["prev_chunk_left_over"], queue._left_over)
assert kwargs["prev_chunk_left_over"] is not queue._left_over

prediction = ChunkPredictionResult(
    original_actions=torch.arange(12, dtype=torch.float32).reshape(4, 3),
    processed_actions=[torch.arange(3, dtype=torch.float32) for _ in range(4)],
    preprocess_time_s=0.0,
    inference_time_s=0.0,
    postprocess_time_s=0.0,
    processed_actions_tensor=torch.arange(12, dtype=torch.float32).reshape(4, 3),
    action_index_before_inference=1,
)
real_delay = merge_chunk_prediction_result(queue, prediction)
assert real_delay == 3
assert queue.merged is not None
assert queue.merged["real_delay"] == 3
assert queue.merged["action_index_before_inference"] == 1
print("self-check ok")
PY
```

自检结果：

- `python -m py_compile scripts/pi05_chunk_runtime.py` 通过。
- 轻量 stub 脚本输出 `self-check ok`，说明新增 RTC glue 和 merge helper 至少在导入级、参数归一化级、通用 queue 协议级可用。

## 导出的 helper API 清单

当前 `scripts/pi05_chunk_runtime.py` 的 `__all__` 包含：

- `AsyncChunkPrefetcher`
- `ChunkPredictionResult`
- `build_chunk_predict_kwargs`
- `compute_real_delay`
- `estimate_prefetch_threshold`
- `merge_chunk_prediction_result`
- `postprocess_action_chunk`
- `predict_processed_action_chunk`
- `prepare_policy_observation`

其中本轮新增、供 launcher 复用的重点 API：

- `build_chunk_predict_kwargs(...)`
- `merge_chunk_prediction_result(...)`

## 兼容性说明

- 对 TRT launcher 兼容：
  - 共享 `build_chunk_predict_kwargs(...)` 保留了 `rtc_runtime`、`action_queue`、`predicted_delay_steps` 这组三元输入，迁移时不需要先引入 launcher 私有 wrapper。
  - `ChunkPredictionResult.action_queue_payload()` 仍可直接给 `ActionQueue.merge(...)` 使用；新增 `merge_chunk_prediction_result(...)` 只是减少样板代码，不强制替换现有 merge 路径。
- 对 ONNX launcher 兼容：
  - 共享 `build_chunk_predict_kwargs(...)` 支持显式传值模式：`rtc_enabled`、`inference_delay`、`execution_horizon`、`prev_chunk_left_over`，不要求先造 TRT 那套 `ResolvedRTCRuntimeConfig`。
  - `left_over_provider` 支持 callable / `get_left_over()` provider / 直接 tensor，便于 ONNX 侧后续按自己节奏接入 queue。
- 对现有共享运行时兼容：
  - `predict_processed_action_chunk(...)` 和 `AsyncChunkPrefetcher` 的公开签名未被破坏，已有调用路径仍可继续使用。
  - `ChunkPredictionResult` 仍同时支持 list-based rollout 和 tensor-based queue merge，未收窄 schema。

## 剩余风险

- 本轮只改共享 helper，没有修改 TRT launcher / ONNX launcher；最终复用效果仍取决于其他 worker 后续是否切到共享导入。
- `merge_chunk_prediction_result(...)` 依赖外部 queue 对象提供 `merge(...)`。如果调用方不显式传 `real_delay`，最好还能提供 `get_action_index()`，否则 helper 会回落到 `prediction.real_delay`，再没有则回落到 `0`。
- `build_chunk_predict_kwargs(...)` 在 RTC 开启时会返回三项键，即使其中某些值是 `None`。当前 PI05 / ONNX / TRT adapter 的 kwargs 解析逻辑可接受这一点；如果后续某个 wrapper 改为严格拒绝 `None`，调用方需要在边界层过滤。
