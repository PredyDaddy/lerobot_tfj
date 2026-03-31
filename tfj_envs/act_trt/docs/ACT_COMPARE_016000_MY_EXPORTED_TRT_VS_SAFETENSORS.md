# ACT Compare: 016000 Safetensors vs My Exported TRT

## Artifacts

- checkpoint: `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- engine: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`
- metadata: `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json`
- trt_device: `cuda:0`

## Summary

- num_cases: `6`
- all_within_threshold: `True`
- threshold_max_abs_diff: `0.0001`
- max_abs_diff: `4.470348358154297e-06`
- max_mean_abs_diff: `1.7280876818404067e-06`
- max_rel_diff: `0.013286828994750977`
- min_cosine_similarity: `0.9999999403953552`

## Engine IO

- state_input_name: `obs_state_norm`
- camera_input_names: `{'observation.images.top': 'img0_norm', 'observation.images.wrist': 'img1_norm'}`
- output_name: `actions_norm`

## Worst Case

- case: `ones`
- max_abs_diff: `4.470348358154297e-06`
- mean_abs_diff: `1.7280876818404067e-06`
- cosine_similarity: `1.0000001192092896`

## Conclusion

The 016000 checkpoint restored from `model.safetensors` is numerically aligned with my exported 016000 TensorRT engine.

