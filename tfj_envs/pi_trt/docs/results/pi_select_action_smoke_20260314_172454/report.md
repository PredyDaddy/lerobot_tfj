# PI0.5 1000-step Pure Inference Compare

- measured_at_utc: `2026-03-14T09:27:39.726301+00:00`
- policy_path: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- steps: `20`
- warmup_steps: `0`
- n_action_steps: `50`
- expected_chunk_refreshes: `1`
- num_inference_steps: `10`

## 1. TRT Provenance

- variant: `pi05`
- requested_precision: `fp32`
- metadata_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage4_build_engines.json`
- stage5_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.json`

## 2. Results

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 262.654 | 13.133 | 76.146 |
| pytorch_amp_bf16 | 122.612 | 6.131 | 163.116 |
| onnx_cuda_runtime | 282.172 | 14.109 | 70.879 |
| tensorrt_fp32 | 143.899 | 7.195 | 138.986 |

## 3. Notes

- 这是纯 `select_action()` 推理 benchmark，不接机器人、不读串口、不下发动作。
- 计时包含 chunk queue 的刷新与复用，因此反映的是均摊后的纯推理吞吐，而不是单次 chunk 刷新时延。
- `PyTorch AMP` 在本报告中明确表示 `CUDA BF16 autocast`，不是 `Torch FP16`。
- TensorRT 结果只对当前已验证通过的 static-shape、batch=1、固定 token length 工件成立。
