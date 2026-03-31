# Worker Tests Validation

## 变更范围

本轮只新增测试与结果文档，未改动业务实现文件。

新增测试文件：

- `tests/test_worker_c_torch_rtc_grad_context.py`

## 新增测试覆盖

新增用例聚焦 RTC on/off 的梯度上下文契约，全部为 CPU 轻量测试，不依赖真实硬件、相机、串口或真实 PI0.5 checkpoint。

覆盖点：

1. `TorchChunkPolicyRuntime` 在 RTC off 时必须把 policy 调用包在 `torch.inference_mode()` 中
2. `TorchChunkPolicyRuntime` 在 RTC on 时必须避免 `torch.inference_mode()`，否则 RTC 内部 `torch.autograd.grad(...)` 无法工作
3. 通过 stub policy 的 `@torch.no_grad()` 入口，验证当前边界语义：
   - policy 入口本身仍然是 `no_grad`
   - 但 RTCProcessor 内部可以用 `torch.enable_grad()` 重新打开 autograd
4. 直接验证 RTCProcessor 在 `torch.inference_mode()` 下仍会失败，防止 launcher/runtime 回退到错误上下文

实现策略：

- 使用 stub policy 模拟 launcher 到 policy 的边界
- 使用真实 `RTCProcessor` 验证 RTC 内部 autograd 语义
- 不触碰真实模型参数、真实 robot runtime、真实数据流

## 执行命令

```bash
python -m py_compile tests/test_worker_c_torch_rtc_grad_context.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_c_torch_rtc_grad_context.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n base python -m pytest tests/test_worker_b_torch_rtc_contracts.py tests/test_worker_c_torch_rtc_grad_context.py -q
```

## 结果

`py_compile`：

- 通过

新测试文件：

- `3 passed in 2.19s`

相关 RTC 合同测试合并执行：

- `7 passed in 2.19s`

新增测试名称：

- `test_torch_chunk_policy_runtime_keeps_rtc_off_path_in_inference_mode`
- `test_torch_chunk_policy_runtime_keeps_rtc_on_path_out_of_inference_mode`
- `test_rtc_processor_grad_probe_still_fails_if_forced_under_inference_mode`

## 结论

这轮最小验证已经把 RTC on/off 的 grad context 契约锁在测试里：

- RTC off：launcher/runtime 仍走 `inference_mode`
- RTC on：launcher/runtime 不再强制 `inference_mode`
- policy 边界：即使 policy 入口是 `@torch.no_grad()`，RTCProcessor 仍可在内部重新开启 autograd
- 回归保护：如果后续有人把 RTC-on 路径重新包回 `inference_mode`，新测试会直接报错

这说明当前至少在“上下文边界”这一层已经有了可复跑的无硬件保护，不再完全依赖人工阅读代码判断。

## 局限

1. 这些测试是 stub + 真 `RTCProcessor` 的边界验证，不是完整 PI0.5 模型集成测试
2. 未覆盖真实 `PI05Policy.predict_action_chunk(...)` 的整条数据预处理和模型采样路径
3. 未覆盖 CUDA、AMP、线程化 prefetch、相机输入、串口发送或真实机器人节拍
4. 未覆盖 `ActionQueue` / refill / hold / sync refill 的主循环状态机
5. 未验证 RTC-on 相比 RTC-off 的输出质量或控制效果，只验证上下文契约

## 建议

1. 下一步补一个无硬件的主循环状态机测试，覆盖 `async_collect`、`async_wait`、`hold_pending_async`、`sync_refill`
2. 在条件允许时补一个真实 checkpoint 的 CPU 或单卡本地 smoke，验证 `predict_action_chunk(..., prev_chunk_left_over=..., inference_delay=...)` 的真实调用边界
3. 继续保持 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免 ROS/外部 pytest 插件污染当前测试环境
