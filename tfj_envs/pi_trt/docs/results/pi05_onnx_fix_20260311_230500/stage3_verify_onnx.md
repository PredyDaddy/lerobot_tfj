# PI05 Stage 3 ONNX Verification

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_export_20260311_214232/run_main`
- onnx_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_export_20260311_214232/run_main/artifacts/onnx`
- overall_status: `warn`
- report_contract: `runtime_reference_vs_onnx` and `export_reference_vs_onnx` are reported separately.
- compare_scopes: `local_subgraph_compare` means Torch intermediates are used as ONNX inputs; `chained_compare` means ONNX vision -> ONNX prefix -> ONNX denoise.

## Coverage

- compared_pairs: `82`
- missing_pairs: `0`
- missing_pair_list: `none`

## Local Subgraph Compare

- runtime_reference_vs_onnx: `warn`
- runtime vision summary: `max_abs=0.0414982, mean_abs=0.00034767, min_cos=0.999964`
- runtime prefix summary: `max_abs=0.904474, mean_abs=0.0673935, min_cos=0.999721`
- runtime denoise summary: `max_abs=0.025332, mean_abs=0.00551668, min_cos=0.999934`
- export_reference_vs_onnx: `pass`
- export vision summary: `max_abs=0.00625014, mean_abs=3.94293e-05, min_cos=0.999999`
- export prefix summary: `max_abs=0.0212836, mean_abs=0.00253183, min_cos=0.999998`
- export denoise summary: `max_abs=0.00113094, mean_abs=0.000249435, min_cos=1`

## Chained Compare

- runtime_reference_vs_onnx: `warn`
- runtime pipeline summary: `max_abs=0.0248366, mean_abs=0.00602709, min_cos=0.999921`
- export_reference_vs_onnx: `pass`
- export pipeline summary: `max_abs=0.000972457, mean_abs=0.000250706, min_cos=1`

## Denoise Contract

- local runtime denoise session_input_names: `['x_t', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- local runtime denoise dropped_inputs: `['timestep']`
- local export denoise session_input_names: `['x_t', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- local export denoise dropped_inputs: `['timestep']`
- chained pipeline denoise session_input_names: `['x_t', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- chained pipeline denoise dropped_inputs: `['timestep']`
