# PI0.5 后端测速记录（2026-03-31）

## 1. 文档目的

这份文档汇总本次对 PI0.5 模型在三种推理后端上的真实测速结果：

- `safetensors / PyTorch`
- `ONNX Runtime`
- `TensorRT`

目标不是只看某一个局部子图快不快，而是同时回答两个问题：

1. 一次完整 `action chunk` 生成要多久
2. 在纯推理场景下，均摊到每一步的吞吐有多高

## 2. 测试对象

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- ONNX / TRT 工件来源：
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
- TRT 工件状态：
  - `Stage 4 = pass`
  - `Stage 5 = pass`
  - `requested_precision = fp32`

## 3. 测试环境

- date: `2026-03-31`
- conda env: `lerobot`
- python: `3.10.19`
- torch: `2.7.1+cu126`
- onnxruntime: `1.23.2`
- tensorrt: `10.13.0.35`
- GPU: `NVIDIA GeForce RTX 4090`
- git commit: `3ad38e91128cd1bbe0e8925acdefb19f1a7f2d21`

## 4. 测试口径

### 4.1 `pipeline_chunk` benchmark

含义：

- 测一次完整的 action chunk 生成时延
- 会覆盖：
  - `vision_encoder`
  - `prefix_cache`
  - `denoise_step`
  - `policy.predict_action_chunk(...)` 对应的完整 chunk 路径

不包含：

- 相机采集
- MJPG 解码
- 机器人 observation
- 串口通信
- `send_action`
- 控频 sleep
- 真机闭环控制

### 4.2 `1000-step pure inference` benchmark

含义：

- 只测纯推理，不接机器人
- 走 `select_action()` 的 chunk queue 刷新与复用逻辑
- 更适合看均摊吞吐，而不是单次 chunk 刷新峰值

## 5. 执行命令

### 5.1 完整 chunk 推理时延

```bash
conda run -n lerobot python scripts/benchmark_pi_inference.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_recheck_20260331_191647
```

结果目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_recheck_20260331_191647`

### 5.2 1000-step pure inference

原始命令是：

```bash
conda run -n lerobot python scripts/benchmark_pi_select_action.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_recheck_20260331_191647
```

但这条脚本当前存在一个 ONNX safety report 接口兼容问题：

- `benchmark_pi_select_action.py` 期望 `resolve_onnx_artifacts(...)` 返回 `(artifacts, stage2_policy_dir)`
- 但当前运行时返回的是 `(artifacts, OnnxArtifactSafetyReport)`

因此，本次为了拿到真实测速结果，使用了一次性运行时兼容包装完成测试，没有保留永久代码修改。

最终结果目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_recheck_20260331_191647_patched`

## 6. 测试结果

### 6.1 `pipeline_chunk`：完整 chunk 推理时延

| Backend | pipeline_chunk mean_ms | p50_ms | p95_ms | 说明 |
| --- | ---: | ---: | ---: | --- |
| PyTorch | 93.971 | 94.335 | 94.993 | 当前完整 chunk 最快 |
| ONNX Runtime CUDA | 155.415 | 155.326 | 156.570 | 最慢 |
| TensorRT FP32 | 122.893 | 122.755 | 126.917 | 比 ONNX 快，但仍慢于 PyTorch |

进一步拆开看关键 stage：

| Backend | vision_encoder_pair | prefix_cache | denoise_step |
| --- | ---: | ---: | ---: |
| PyTorch | 8.134 ms | 25.147 ms | 6.300 ms |
| ONNX Runtime CUDA | 15.630 ms | 63.168 ms | 7.443 ms |
| TensorRT FP32 | 12.778 ms | 63.147 ms | 4.620 ms |

### 6.2 `1000-step pure inference`：均摊吞吐

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 2831.235 | 2.831 | 353.203 |
| pytorch_amp_bf16 | 2924.659 | 2.925 | 341.920 |
| onnx_cuda_runtime | 3105.749 | 3.106 | 321.984 |
| tensorrt_fp32 | 2457.798 | 2.458 | 406.868 |

## 7. 结果解读

### 7.1 如果看完整一次 chunk 生成

排序是：

1. `PyTorch`
2. `TensorRT FP32`
3. `ONNX Runtime`

也就是说，当前这套安全 `FP32` 工件下，`PyTorch` 的完整 chunk latency 仍然最好。

### 7.2 如果看 1000-step 均摊纯推理吞吐

排序是：

1. `TensorRT FP32`
2. `PyTorch`
3. `ONNX Runtime`

也就是说，`TensorRT` 在均摊吞吐上是有优势的。

### 7.3 为什么会出现“TRT 吞吐更高，但 chunk latency 仍输给 PyTorch”

关键原因在于不同 stage 的收益并不均匀：

- `TensorRT` 在 `denoise_step` 上确实更快
  - TRT: `4.620 ms`
  - PyTorch: `6.300 ms`
- 但 `prefix_cache` 仍然很慢
  - TRT: `63.147 ms`
  - PyTorch: `25.147 ms`

所以当前最准确的说法不是“TRT 没有价值”，而是：

- TRT 提升了均摊吞吐
- 但完整 `pipeline_chunk` 仍然被 `prefix_cache` 这类前缀主路径拖住

## 8. 当前结论

### 8.1 safetensors / PyTorch

- 优点：
  - 当前完整 chunk latency 最好
  - 路径最直接
  - 现阶段更适合作为 chunk latency 基线
- 缺点：
  - 长期均摊吞吐不如 TRT

### 8.2 ONNX Runtime

- 优点：
  - 作为中间验证边界很有价值
  - 适合验证 Torch -> ONNX 导出一致性
- 缺点：
  - 当前测速结果里速度最差
  - 不适合作为最终性能目标

### 8.3 TensorRT FP32

- 优点：
  - 1000-step 纯推理吞吐最高
  - `denoise_step` 子图加速明显
- 缺点：
  - 完整 chunk latency 仍落后于 PyTorch
  - 当前前缀主路径仍是明显瓶颈

## 9. 工程意义

这次结果说明了一个很重要的工程事实：

- “某个子图更快” 不等于 “完整 chunk 更快”
- “完整 chunk 更快” 也不等于 “真机控制一定更稳”

如果后续目标是继续优化端到端控制效果，那么关注点应该优先放在：

1. `prefix_cache` 这类真正拖慢完整 chunk 的路径
2. 真机闭环中的 queue、delay、hold step、sync refill
3. 不能只盯着 `denoise_step` 的局部 benchmark

## 10. 结果文件

### 10.1 本次完整 chunk benchmark

- JSON:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_recheck_20260331_191647/benchmark_report.json`
- Markdown:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_recheck_20260331_191647/benchmark_report.md`

### 10.2 本次 1000-step pure inference benchmark

- JSON:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_recheck_20260331_191647_patched/report.json`
- Markdown:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_recheck_20260331_191647_patched/report.md`

## 11. 一句话总结

如果你要一个最直接的结论：

- 看完整 chunk latency：`PyTorch` 当前最好
- 看 1000-step 均摊吞吐：`TensorRT FP32` 当前最好
- `ONNX Runtime` 这次主要还是验证边界的角色，不是性能赢家

