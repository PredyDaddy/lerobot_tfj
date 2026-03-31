# ACT Safetensors vs TRT Comparison

## Primary Comparison

The main comparison for my own exported TensorRT engine is:

- checkpoint safetensors: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- my exported TRT engine: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`
- metadata: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json`
- JSON report: [compare_my_exported_016000_trt_vs_safetensors.json](/data/tfj/lerobot_tfj/tfj_envs/compare_my_exported_016000_trt_vs_safetensors.json)
- Markdown summary: [ACT_SAFETENSORS_VS_TRT_016000_MY_EXPORTED.md](/data/tfj/lerobot_tfj/tfj_envs/ACT_SAFETENSORS_VS_TRT_016000_MY_EXPORTED.md)

This is the comparison that should be used when the question is:

> Does the `016000` checkpoint restored from `model.safetensors` match the TRT engine that I exported?

Short answer:

- yes
- this pair matches

Key numbers from the `016000` report:

- `num_cases = 6`
- `all_within_threshold = true`
- `threshold_max_abs_diff = 1e-4`
- `max_abs_diff = 4.470348358154297e-06`
- `max_mean_abs_diff = 1.7280876818404067e-06`
- `min_cosine_similarity = 0.9999999403953552`

Interpretation:

- this is normal floating-point-level drift
- this is not a material output mismatch
- the `016000 safetensors -> my exported 016000 TRT` path is numerically consistent

## Legacy Debug Material

The following files are not the main comparison for my exported TRT:

- [ACT_SAFETENSORS_VS_TRT_018000_FP32.json](/data/tfj/lerobot_tfj/tfj_envs/ACT_SAFETENSORS_VS_TRT_018000_FP32.json)
- [ACT_SAFETENSORS_VS_ONNX_018000.json](/data/tfj/lerobot_tfj/tfj_envs/ACT_SAFETENSORS_VS_ONNX_018000.json)
- [ACT_ONNX_VS_TRT_018000_FP32.json](/data/tfj/lerobot_tfj/tfj_envs/ACT_ONNX_VS_TRT_018000_FP32.json)
- [compare_current_runtime_trt_vs_safetensors.json](/data/tfj/lerobot_tfj/tfj_envs/compare_current_runtime_trt_vs_safetensors.json)

Those files were produced only to diagnose the older runtime engine under:

- checkpoint: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/018000/pretrained_model`
- engine: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/018000/pretrained_model/act_core_b1_fp32.plan`

That `018000` engine is not my exported validated engine.

Its result was:

- `all_within_threshold = false`
- `max_abs_diff = 0.07780838012695312`

Interpretation:

- that old runtime engine diverges
- it should be treated as a separate legacy debug case
- it should not be confused with my `016000` export result
