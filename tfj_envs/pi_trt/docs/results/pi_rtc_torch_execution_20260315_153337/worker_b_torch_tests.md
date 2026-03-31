# Worker B Torch RTC Tests

## 新增文件

- `tests/test_worker_b_torch_rtc_contracts.py`
- `docs/results/pi_rtc_torch_execution_20260315_153337/worker_b_torch_tests.md`

## 执行命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_b_torch_rtc_contracts.py -q
```

## 通过情况

- 结果：通过
- 摘要：`4 passed in 2.07s`

本次新增测试覆盖了以下最小无硬件契约：

- torch launcher 的 `--rtc-*` 参数可通过 parser 解析
- torch launcher 源码不再静态依赖 `lerobot.utils.control_utils.predict_action`
- shared helper `build_chunk_predict_kwargs(...)` 的 RTC-off / RTC-on 构造路径
- `merge_chunk_prediction_result(...)` 会把计算出的 `real_delay` 传给假 `ActionQueue`

## 未覆盖点

- 未跑真实机器人、相机、串口、标定目录、policy checkpoint 的端到端路径
- 未覆盖真实 `ActionQueue` / `RTCProcessor` 对接，只锁了 shared helper 和最小假对象语义
- 未覆盖 async prefetch 主循环的时序分支，例如 hold/refill/timeout 的实时间行为
- 未覆盖 CUDA AMP、engine、ONNX、网络或任何硬件依赖
