# PI05 Stage 2 ONNX Export Fix

## Scope

- Checkpoint: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- Result root: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500`
- Environment: `conda run --no-capture-output -n lerobot ...`

## Export Routes

- `vision_encoder`: `legacy`
- `prefix_cache`: `legacy`
- `denoise_step`: `dynamo`

`denoise_step` was switched to `dynamo` because the legacy exporter can drop `timestep` from the ONNX graph input contract. In this run, `timestep` is preserved as a real ONNX input.

## Contract Check

- `denoise_step_timestep_is_graph_input`: `true`
- `denoise_step` ONNX inputs:
  - `x_t`
  - `timestep`
  - `prefix_pad_masks`
  - `past_key_values.layer_00.key/value` through `layer_17.key/value`

## Immediate Compare

Primary fidelity metric:
- `export_reference_vs_onnx`

Why this is the primary metric:
- Current runtime-like Torch reference uses `cuda + autocast(bfloat16)` when CUDA is available.
- Export reference uses `float32 + no autocast`.
- The dominant drift is therefore `runtime-reference` vs `export-reference`, not ONNX conversion itself.

Observed summaries:

- `runtime_vs_export_reference`
  - `vision_encoder`: `max_abs_diff=0.0438898`, `mean_abs_diff=3.40309e-4`, `min_cos=0.999966`
  - `prefix_cache`: `max_abs_diff=2.17097`, `mean_abs_diff=7.83762e-2`, `min_cos=0.999532`
  - `denoise_step`: `max_abs_diff=0.0242019`, `mean_abs_diff=6.01017e-3`, `min_cos=0.999921`

- `export_reference_vs_onnx`
  - `vision_encoder`: `max_abs_diff=1.09673e-5`, `mean_abs_diff=8.2841e-8`, `min_cos=0.99999994`
  - `prefix_cache`: `max_abs_diff=5.05447e-5`, `mean_abs_diff=4.73302e-6`, `min_cos=0.99999976`
  - `denoise_step`: `max_abs_diff=1.54972e-6`, `mean_abs_diff=2.90621e-7`, `min_cos=1.0`

- `runtime_reference_vs_onnx`
  - `vision_encoder`: `max_abs_diff=0.0438809`, `mean_abs_diff=3.40314e-4`, `min_cos=0.999966`
  - `prefix_cache`: `max_abs_diff=2.17098`, `mean_abs_diff=7.83761e-2`, `min_cos=0.999532`
  - `denoise_step`: `max_abs_diff=0.0242023`, `mean_abs_diff=6.01015e-3`, `min_cos=0.999921`

- `pipeline_compare`
  - `runtime_reference_vs_onnx`: `max_abs_diff=0.0242019`, `mean_abs_diff=6.01014e-3`, `min_cos=0.999921`
  - `export_reference_vs_onnx`: `max_abs_diff=1.85985e-6`, `mean_abs_diff=3.18965e-7`, `min_cos=1.0`

## Conclusion

- ONNX conversion itself is now well aligned with the export boundary.
- The main unresolved gap remains `runtime Torch (autocast/bfloat16)` vs `export Torch / ONNX (float32, no autocast)`.
- Stage 2 should therefore be interpreted with two separate references:
  - runtime reference: useful for deployment realism
  - export reference: useful for exporter fidelity

## Artifacts

- JSON report: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/stage2_export_onnx.json`
- ONNX directory: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/onnx`
