# PI0.5 推理时延 Benchmark

## 1. 测试对象

- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- trt_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/engines`
- measured_at_utc: `2026-03-14T09:25:46.292797+00:00`

## 2. 测试设置

- warmup_iterations: `0`
- measured_iterations: `1`
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
| pytorch | vision_encoder_single | 4.159 | 4.159 | 4.159 | 4.159 | 4.159 |
| pytorch | vision_encoder_pair | 8.238 | 8.238 | 8.238 | 8.238 | 8.238 |
| pytorch | prefix_cache | 24.596 | 24.596 | 24.596 | 24.596 | 24.596 |
| pytorch | denoise_step | 6.704 | 6.704 | 6.704 | 6.704 | 6.704 |
| pytorch | pipeline_staged | 97.504 | 97.504 | 97.504 | 97.504 | 97.504 |
| pytorch | pipeline_chunk | 245.307 | 245.307 | 245.307 | 245.307 | 245.307 |
| onnx | vision_encoder_single | 7.905 | 7.905 | 7.905 | 7.905 | 7.905 |
| onnx | vision_encoder_pair | 15.809 | 15.809 | 15.809 | 15.809 | 15.809 |
| onnx | prefix_cache | 62.924 | 62.924 | 62.924 | 62.924 | 62.924 |
| onnx | denoise_step | 13.477 | 13.477 | 13.477 | 13.477 | 13.477 |
| onnx | pipeline_chunk | 155.786 | 155.786 | 155.786 | 155.786 | 155.786 |
| tensorrt | vision_encoder_single | 7.493 | 7.493 | 7.493 | 7.493 | 7.493 |
| tensorrt | vision_encoder_pair | 14.848 | 14.848 | 14.848 | 14.848 | 14.848 |
| tensorrt | prefix_cache | 61.921 | 61.921 | 61.921 | 61.921 | 61.921 |
| tensorrt | denoise_step | 7.214 | 7.214 | 7.214 | 7.214 | 7.214 |
| tensorrt | pipeline_chunk | 124.342 | 124.342 | 124.342 | 124.342 | 124.342 |

## 6. Chunk 推理的实际意义

- pytorch: chunk_mean=245.307 ms, amortized_per_action_step=4.906 ms, estimated_denoise_loop_total=67.038 ms
- onnx: chunk_mean=155.786 ms, amortized_per_action_step=3.116 ms, estimated_denoise_loop_total=134.770 ms
- tensorrt: chunk_mean=124.342 ms, amortized_per_action_step=2.487 ms, estimated_denoise_loop_total=72.142 ms

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
