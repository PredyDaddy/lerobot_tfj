# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/engines`
- measured_at_utc: `2026-03-13T15:57:11.922873+00:00`

## 2. 测试设置

- warmup_iterations: `1`
- measured_iterations: `1`
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
| pytorch | vision_encoder_single | 4.871 | 4.871 | 4.871 | 4.871 | 4.871 |
| pytorch | vision_encoder_pair | 9.518 | 9.518 | 9.518 | 9.518 | 9.518 |
| pytorch | prefix_cache | 24.626 | 24.626 | 24.626 | 24.626 | 24.626 |
| pytorch | denoise_step | 6.978 | 6.978 | 6.978 | 6.978 | 6.978 |
| pytorch | pipeline_staged | 101.983 | 101.983 | 101.983 | 101.983 | 101.983 |
| pytorch | pipeline_chunk | 99.206 | 99.206 | 99.206 | 99.206 | 99.206 |
| onnx | vision_encoder_single | 7.675 | 7.675 | 7.675 | 7.675 | 7.675 |
| onnx | vision_encoder_pair | 16.010 | 16.010 | 16.010 | 16.010 | 16.010 |
| onnx | prefix_cache | 63.153 | 63.153 | 63.153 | 63.153 | 63.153 |
| onnx | denoise_step | 8.099 | 8.099 | 8.099 | 8.099 | 8.099 |
| onnx | pipeline_chunk | 155.375 | 155.375 | 155.375 | 155.375 | 155.375 |
| tensorrt | vision_encoder_single | 7.392 | 7.392 | 7.392 | 7.392 | 7.392 |
| tensorrt | vision_encoder_pair | 14.773 | 14.773 | 14.773 | 14.773 | 14.773 |
| tensorrt | prefix_cache | 64.039 | 64.039 | 64.039 | 64.039 | 64.039 |
| tensorrt | denoise_step | 5.066 | 5.066 | 5.066 | 5.066 | 5.066 |
| tensorrt | pipeline_chunk | 120.087 | 120.087 | 120.087 | 120.087 | 120.087 |

## 5. Chunk 推理的实际意义

- pytorch: chunk_mean=99.206 ms, amortized_per_action_step=1.984 ms, estimated_denoise_loop_total=69.781 ms
- onnx: chunk_mean=155.375 ms, amortized_per_action_step=3.107 ms, estimated_denoise_loop_total=80.991 ms
- tensorrt: chunk_mean=120.087 ms, amortized_per_action_step=2.402 ms, estimated_denoise_loop_total=50.659 ms

## 6. 说明

- 这是离线纯推理 benchmark，不包含相机采集、机械臂通信、preprocessor、postprocessor。
- ONNX Runtime 结果反映的是当前工程实现路径，当前 runner 使用常规 `session.run(...)`，包含主机侧输入输出搬运成本。
- PyTorch 子图拆分时延来自 export wrapper 路径，用来和 ONNX/TRT 子图做对应比较。
- PyTorch 的 `pipeline_chunk` 额外给了真实 `policy.predict_action_chunk(...)` 路径，不是 wrapper 拼接模拟。
