# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/artifacts/engines`
- measured_at_utc: `2026-03-14T09:45:54.859803+00:00`

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
- requested_precision: `fp16`
- metadata_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/pi_trt_metadata.json`
- checkpoint_dir: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json`
- stage4_report_status: `pass`
- stage5_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json`
- stage5_report_status: `fail`
- allow_unsafe_trt_artifacts: `True`

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
| pytorch | vision_encoder_single | 4.124 | 4.118 | 4.151 | 4.104 | 4.245 |
| pytorch | vision_encoder_pair | 8.211 | 8.185 | 8.350 | 8.140 | 8.401 |
| pytorch | prefix_cache | 25.139 | 25.026 | 25.720 | 24.931 | 25.753 |
| pytorch | denoise_step | 6.276 | 6.266 | 6.390 | 6.172 | 6.492 |
| pytorch | pipeline_staged | 95.275 | 95.125 | 95.894 | 94.654 | 96.830 |
| pytorch | pipeline_chunk | 94.973 | 94.893 | 95.592 | 94.171 | 97.865 |
| onnx | vision_encoder_single | 7.692 | 7.681 | 7.751 | 7.671 | 7.774 |
| onnx | vision_encoder_pair | 15.812 | 15.789 | 15.964 | 15.711 | 16.110 |
| onnx | prefix_cache | 63.329 | 63.288 | 63.632 | 63.142 | 63.705 |
| onnx | denoise_step | 7.491 | 7.493 | 7.557 | 7.359 | 7.790 |
| onnx | pipeline_chunk | 156.026 | 155.659 | 158.416 | 154.552 | 159.984 |
| tensorrt | vision_encoder_single | 2.285 | 2.283 | 2.294 | 2.277 | 2.325 |
| tensorrt | vision_encoder_pair | 4.585 | 4.582 | 4.608 | 4.571 | 4.619 |
| tensorrt | prefix_cache | 13.348 | 13.322 | 13.385 | 13.253 | 14.083 |
| tensorrt | denoise_step | 3.239 | 3.205 | 3.287 | 3.185 | 3.570 |
| tensorrt | pipeline_chunk | 50.665 | 50.624 | 50.914 | 50.438 | 50.931 |

## 6. Chunk 推理的实际意义

- pytorch: chunk_mean=94.973 ms, amortized_per_action_step=1.899 ms, estimated_denoise_loop_total=62.765 ms
- onnx: chunk_mean=156.026 ms, amortized_per_action_step=3.121 ms, estimated_denoise_loop_total=74.907 ms
- tensorrt: chunk_mean=50.665 ms, amortized_per_action_step=1.013 ms, estimated_denoise_loop_total=32.391 ms

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
- 本次 TensorRT benchmark 显式允许了 `unsafe` 工件，因此这些数字只能用于诊断，不可直接当作已通过正确性 gate 的部署结论。
