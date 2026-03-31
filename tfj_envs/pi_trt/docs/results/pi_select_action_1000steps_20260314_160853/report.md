# PI0.5 1000-step Pure Inference Compare

- steps: `1000`
- warmup_steps: `100`
- n_action_steps: `50`
- expected_chunk_refreshes: `20`
- num_inference_steps: `10`

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 2854.845 | 2.855 | 350.282 |
| pytorch_amp | 2904.835 | 2.905 | 344.254 |
| onnx_cuda_runtime | 3100.257 | 3.100 | 322.554 |
| tensorrt_fp32 | 2456.692 | 2.457 | 407.051 |
