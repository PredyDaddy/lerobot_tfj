# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/engines`
- measured_at_utc: `2026-03-14T09:43:35.473404+00:00`

## 2. 测试设置

- warmup_iterations: `10`
- measured_iterations: `30`
- num_inference_steps: `10`
- n_action_steps: `50`
- chunk_size: `50`
- torch_device: `cuda:0`
- torch_use_amp: `False`
- torch_amp_mode: `None`
- onnx_provider: `cuda`
- trt_device: `cuda:0`

## 3. TRT Provenance

- variant: `pi05`
- requested_precision: `fp32`
- metadata_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json`
- checkpoint_dir: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage4_build_engines.json`
- stage4_report_status: `pass`
- stage5_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.json`
- stage5_report_status: `pass`
- allow_unsafe_trt_artifacts: `False`

## 4. 环境

- git_commit: `24acade992bb2b8b5e7cd0407bc2698d5d725b5c`
- python: `3.10.19`
- torch: `2.7.1+cu126`
- onnxruntime: `1.23.2`
- tensorrt: `10.13.0.35`
- gpu: `NVIDIA GeForce RTX 4090, 570.133.20, 49140 MiB`

## 5. 结果总表

| Backend | Stage | mean_ms | p50_ms | p95_ms | min_ms | max_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pytorch | vision_encoder_single | 4.108 | 4.101 | 4.109 | 4.083 | 4.382 |
| pytorch | vision_encoder_pair | 8.135 | 8.114 | 8.173 | 8.091 | 8.646 |
| pytorch | prefix_cache | 25.114 | 24.984 | 25.647 | 24.890 | 25.702 |
| pytorch | denoise_step | 6.269 | 6.266 | 6.336 | 6.193 | 6.350 |
| pytorch | pipeline_staged | 96.383 | 96.101 | 98.328 | 94.941 | 98.528 |
| pytorch | pipeline_chunk | 95.468 | 95.486 | 96.108 | 94.550 | 96.369 |
| onnx | vision_encoder_single | 7.627 | 7.621 | 7.667 | 7.611 | 7.669 |
| onnx | vision_encoder_pair | 15.712 | 15.630 | 16.201 | 15.566 | 16.652 |
| onnx | prefix_cache | 63.256 | 63.188 | 63.564 | 63.003 | 63.818 |
| onnx | denoise_step | 7.621 | 7.600 | 7.787 | 7.503 | 7.955 |
| onnx | pipeline_chunk | 157.021 | 156.610 | 159.954 | 155.645 | 160.814 |
| tensorrt | vision_encoder_single | 6.358 | 6.343 | 6.443 | 6.336 | 6.518 |
| tensorrt | vision_encoder_pair | 12.760 | 12.736 | 12.789 | 12.693 | 13.431 |
| tensorrt | prefix_cache | 63.335 | 63.360 | 63.563 | 62.846 | 64.233 |
| tensorrt | denoise_step | 4.670 | 4.626 | 4.976 | 4.594 | 4.979 |
| tensorrt | pipeline_chunk | 123.501 | 122.640 | 130.163 | 118.380 | 131.164 |

## 6. Chunk 推理的实际意义

- pytorch: chunk_mean=95.468 ms, amortized_per_action_step=1.909 ms, estimated_denoise_loop_total=62.694 ms
- onnx: chunk_mean=157.021 ms, amortized_per_action_step=3.140 ms, estimated_denoise_loop_total=76.208 ms
- tensorrt: chunk_mean=123.501 ms, amortized_per_action_step=2.470 ms, estimated_denoise_loop_total=46.698 ms

## 7. 说明

- 这是离线纯推理 benchmark，不等价于机器人闭环控制 latency。
- 这次只测模型推理链，不包含相机采集、MJPG 解码、robot observation、send_action、安全限幅、sleep 控频。
- 输入是 `build_runtime_context()` 生成的 deterministic baseline batch，适合做固定口径对比，不代表真实任务数据分布。
- 当 `torch_use_amp=true` 时，这里的 `PyTorch AMP` 明确表示 `CUDA BF16 autocast`，不是 `Torch FP16`。
- `vision_encoder_single` 表示单相机一次调用；`vision_encoder_pair` 表示 top+wrist 两次调用总和。
- `denoise_step` 是单次 denoise 迭代，不是完整 chunk；`pipeline_chunk` 才是完整一次 action chunk 生成。
- `amortized_per_action_step_ms = pipeline_chunk / n_action_steps`，这是均摊值，不是每个 control loop 的最坏时延。
- ONNX Runtime 结果反映的是当前工程实现路径，runner 使用常规 `session.run(...)` 和 `numpy <-> torch` 边界，不是纯 GPU kernel benchmark。
- PyTorch 子图拆分时延来自 export wrapper 路径，用来和 ONNX/TRT 子图做对应比较。
- PyTorch 的 `pipeline_chunk` 额外给了真实 `policy.predict_action_chunk(...)` 路径，不是 wrapper 拼接模拟。
- TensorRT 结论只对当前已验证通过的 static-shape、batch=1、固定 token length engine 成立。
