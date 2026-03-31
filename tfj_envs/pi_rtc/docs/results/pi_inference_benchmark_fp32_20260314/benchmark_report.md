# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/engines`
- measured_at_utc: `2026-03-13T16:31:23.492408+00:00`

## 2. 测试设置

- warmup_iterations: `10`
- measured_iterations: `30`
- num_inference_steps: `10`
- n_action_steps: `50`
- chunk_size: `50`
- torch_device: `cuda:0`
- torch_use_amp: `False`
- onnx_provider: `cuda`
- trt_device: `cuda:0`

## 3. 环境

- git_commit: `24acade992bb2b8b5e7cd0407bc2698d5d725b5c`
- python: `3.10.19`
- torch: `2.7.1+cu126`
- onnxruntime: `1.23.2`
- tensorrt: `10.13.0.35`
- gpu: `NVIDIA GeForce RTX 4090, 570.133.20, 49140 MiB`

## 4. 结果总表

| Backend | Stage | mean_ms | p50_ms | p95_ms | min_ms | max_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pytorch | vision_encoder_single | 4.103 | 4.098 | 4.127 | 4.085 | 4.165 |
| pytorch | vision_encoder_pair | 8.110 | 8.111 | 8.126 | 8.086 | 8.135 |
| pytorch | prefix_cache | 25.028 | 24.835 | 25.603 | 24.777 | 25.614 |
| pytorch | denoise_step | 6.286 | 6.278 | 6.380 | 6.170 | 6.413 |
| pytorch | pipeline_staged | 95.453 | 95.349 | 95.997 | 95.066 | 96.118 |
| pytorch | pipeline_chunk | 94.934 | 94.882 | 95.801 | 94.087 | 95.969 |
| onnx | vision_encoder_single | 7.645 | 7.640 | 7.654 | 7.624 | 7.815 |
| onnx | vision_encoder_pair | 15.720 | 15.631 | 16.064 | 15.548 | 17.640 |
| onnx | prefix_cache | 63.220 | 63.171 | 63.471 | 63.001 | 64.019 |
| onnx | denoise_step | 7.540 | 7.514 | 7.715 | 7.406 | 7.894 |
| onnx | pipeline_chunk | 155.994 | 155.989 | 156.619 | 155.163 | 156.711 |
| tensorrt | vision_encoder_single | 6.378 | 6.351 | 6.518 | 6.323 | 6.672 |
| tensorrt | vision_encoder_pair | 12.797 | 12.771 | 12.838 | 12.732 | 13.458 |
| tensorrt | prefix_cache | 63.059 | 63.023 | 63.471 | 62.159 | 63.967 |
| tensorrt | denoise_step | 4.650 | 4.625 | 4.736 | 4.562 | 4.750 |
| tensorrt | pipeline_chunk | 123.053 | 122.230 | 127.458 | 118.879 | 129.289 |

## 5. Chunk 推理的实际意义

- pytorch: chunk_mean=94.934 ms, amortized_per_action_step=1.899 ms, estimated_denoise_loop_total=62.858 ms
- onnx: chunk_mean=155.994 ms, amortized_per_action_step=3.120 ms, estimated_denoise_loop_total=75.403 ms
- tensorrt: chunk_mean=123.053 ms, amortized_per_action_step=2.461 ms, estimated_denoise_loop_total=46.500 ms

## 6. 说明

- 这是离线纯推理 benchmark，不等价于机器人闭环控制 latency。
- 这次只测模型推理链，不包含相机采集、MJPG 解码、robot observation、send_action、安全限幅、sleep 控频。
- 输入是 `build_runtime_context()` 生成的 deterministic baseline batch，适合做固定口径对比，不代表真实任务数据分布。
- `vision_encoder_single` 表示单相机一次调用；`vision_encoder_pair` 表示 top+wrist 两次调用总和。
- `denoise_step` 是单次 denoise 迭代，不是完整 chunk；`pipeline_chunk` 才是完整一次 action chunk 生成。
- `amortized_per_action_step_ms = pipeline_chunk / n_action_steps`，这是均摊值，不是每个 control loop 的最坏时延。
- ONNX Runtime 结果反映的是当前工程实现路径，runner 使用常规 `session.run(...)` 和 `numpy <-> torch` 边界，不是纯 GPU kernel benchmark。
- PyTorch 子图拆分时延来自 export wrapper 路径，用来和 ONNX/TRT 子图做对应比较。
- PyTorch 的 `pipeline_chunk` 额外给了真实 `policy.predict_action_chunk(...)` 路径，不是 wrapper 拼接模拟。
- TensorRT 结论只对当前已验证通过的 static-shape、batch=1、固定 token length engine 成立。
