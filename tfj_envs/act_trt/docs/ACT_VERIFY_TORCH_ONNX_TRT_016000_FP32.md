# ACT Verify Torch ONNX TRT 016000 FP32

## Artifacts

- checkpoint: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- onnx: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single.onnx`
- engine: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`
- report: [ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json)

## Summary

- passed: `true`
- threshold_max_abs_diff: `0.0001`

### Torch vs ONNX

- `max_abs_diff = 5.4836273193359375e-06`
- `mean_abs_diff = 2.0535787825792795e-06`
- `max_rel_diff = 0.017065271735191345`
- `min_cosine_similarity = 1.0`

### Torch vs TRT

- `max_abs_diff = 4.470348358154297e-06`
- `mean_abs_diff = 1.7280876818404067e-06`
- `max_rel_diff = 0.013286828994750977`
- `min_cosine_similarity = 0.9999999403953552`

### ONNX vs TRT

- `max_abs_diff = 1.6614794731140137e-06`
- `mean_abs_diff = 4.0187811123360007e-07`
- `max_rel_diff = 0.0037150438874959946`
- `min_cosine_similarity = 0.9999999403953552`

## Conclusion

For the latest fully local `tfj_envs` re-export, the ONNX and TensorRT outputs both stay within the configured `1e-4` threshold.

The local fix was to rebuild the FP32 TensorRT engine with TF32 disabled.
