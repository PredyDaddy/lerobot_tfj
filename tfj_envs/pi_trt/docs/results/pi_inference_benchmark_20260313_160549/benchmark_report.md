# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/engines`
- measured_at_utc: `2026-03-13T16:06:52.226280+00:00`

## 2. 测试设置

- warmup_iterations: `10`
- measured_iterations: `30`
- num_inference_steps: `10`
- n_action_steps: `50`
- chunk_size: `50`
- torch_device: `cuda:0`
- torch_use_amp: `True`
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
| pytorch | vision_encoder_single | 4.779 | 4.779 | 4.790 | 4.765 | 4.795 |
| pytorch | vision_encoder_pair | 9.457 | 9.438 | 9.509 | 9.422 | 9.948 |
| pytorch | prefix_cache | 25.014 | 24.843 | 25.648 | 24.795 | 26.030 |
| pytorch | denoise_step | 6.780 | 6.772 | 6.839 | 6.712 | 6.853 |
| pytorch | pipeline_staged | 101.467 | 101.394 | 101.805 | 101.186 | 102.392 |
| pytorch | pipeline_chunk | 98.855 | 98.713 | 99.562 | 98.378 | 99.665 |
| onnx | vision_encoder_single | 7.644 | 7.644 | 7.664 | 7.626 | 7.676 |
| onnx | vision_encoder_pair | 15.645 | 15.625 | 15.699 | 15.578 | 16.103 |
| onnx | prefix_cache | 63.138 | 63.129 | 63.633 | 62.891 | 63.775 |
| onnx | denoise_step | 7.698 | 7.704 | 7.985 | 7.509 | 8.005 |
| onnx | pipeline_chunk | 156.713 | 156.638 | 158.199 | 154.884 | 159.193 |
| tensorrt | vision_encoder_single | 6.427 | 6.392 | 6.621 | 6.368 | 6.624 |
| tensorrt | vision_encoder_pair | 12.806 | 12.802 | 12.843 | 12.760 | 12.851 |
| tensorrt | prefix_cache | 63.071 | 63.035 | 63.624 | 62.253 | 64.006 |
| tensorrt | denoise_step | 4.614 | 4.582 | 4.748 | 4.563 | 4.767 |
| tensorrt | pipeline_chunk | 122.870 | 123.095 | 126.002 | 118.342 | 127.782 |

## 5. Chunk 推理的实际意义

- pytorch: chunk_mean=98.855 ms, amortized_per_action_step=1.977 ms, estimated_denoise_loop_total=67.802 ms
- onnx: chunk_mean=156.713 ms, amortized_per_action_step=3.134 ms, estimated_denoise_loop_total=76.977 ms
- tensorrt: chunk_mean=122.870 ms, amortized_per_action_step=2.457 ms, estimated_denoise_loop_total=46.145 ms

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
