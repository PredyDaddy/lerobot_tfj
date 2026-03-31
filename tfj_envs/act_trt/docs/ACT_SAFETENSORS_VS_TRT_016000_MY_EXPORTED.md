# ACT Safetensors vs TRT: 016000 My Export

## Artifacts

- checkpoint safetensors: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- TRT engine: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`
- metadata: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json`
- report: [compare_my_exported_016000_trt_vs_safetensors.json](/data/tfj/lerobot_tfj/tfj_envs/compare_my_exported_016000_trt_vs_safetensors.json)

## Result

This comparison uses:

- `016000` checkpoint restored through the normal PyTorch `model.safetensors` path
- my own exported TensorRT engine for the same `016000` checkpoint

Summary:

- `num_cases = 6`
- `all_within_threshold = true`
- `threshold_max_abs_diff = 1e-4`
- `max_abs_diff = 4.470348358154297e-06`
- `max_mean_abs_diff = 1.7280876818404067e-06`
- `max_rel_diff = 0.013286828994750977`
- `min_cosine_similarity = 0.9999999403953552`

Worst case:

- case: `ones`
- `max_abs_diff = 4.470348358154297e-06`
- `mean_abs_diff = 1.7280876818404067e-06`

## Conclusion

The `016000` checkpoint safetensors and my exported `016000` TRT engine are numerically aligned.

This is the correct comparison result to use for:

- my TRT export quality
- the pure TRT inference script in `tfj_envs`
- any statement about whether my exported engine matches the source checkpoint
