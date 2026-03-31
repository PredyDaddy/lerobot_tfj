# PI0.5 1000-step Pure Inference Compare

- measured_at_utc: `2026-03-14T09:46:52.515683+00:00`
- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- steps: `1000`
- warmup_steps: `100`
- n_action_steps: `50`
- expected_chunk_refreshes: `20`
- num_inference_steps: `10`

## 1. TRT Provenance

- variant: `pi05`
- requested_precision: `fp16`
- metadata_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/pi_trt_metadata.json`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json`
- stage5_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json`
- allow_unsafe_trt_artifacts: `True`

## 2. Results

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 2885.647 | 2.886 | 346.543 |
| pytorch_amp_bf16 | 2974.741 | 2.975 | 336.164 |
| onnx_cuda_runtime | 3123.643 | 3.124 | 320.139 |
| tensorrt_fp16 | 1015.324 | 1.015 | 984.907 |

## 3. Notes

- 这是纯 `select_action()` 推理 benchmark，不接机器人、不读串口、不下发动作。
- 计时包含 chunk queue 的刷新与复用，因此反映的是均摊后的纯推理吞吐，而不是单次 chunk 刷新时延。
- `PyTorch AMP` 在本报告中明确表示 `CUDA BF16 autocast`，不是 `Torch FP16`。
- TensorRT 结果只对当前已验证通过的 static-shape、batch=1、固定 token length 工件成立。
- 本次 TensorRT pure benchmark 显式允许了 `unsafe` 工件，因此这些数字只能用于诊断，不可直接当作已通过正确性 gate 的部署结论。
