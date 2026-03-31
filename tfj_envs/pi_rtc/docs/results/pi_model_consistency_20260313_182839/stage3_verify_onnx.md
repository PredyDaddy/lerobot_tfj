# PI05 Stage 3 ONNX Verification

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
- onnx_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/artifacts/onnx`
- overall_status: `warn`
- stage3_acceptance: `pass`
- report_goal: `export-fidelity compare` and `runtime-oriented compare` are separated so CPU ORT export fidelity is not conflated with CUDA ORT runtime drift.
- compare_scopes: `local_subgraph_compare` means Torch intermediates are used as ONNX inputs; `chained_compare` means ONNX vision -> ONNX prefix -> ONNX denoise.

## Acceptance

- local_export_fidelity_compare: `pass`
  - message: `Local export-fidelity ONNX compare passed.`
- chained_export_fidelity_compare: `pass`
  - message: `Chained export-fidelity ONNX compare passed.`
- denoise_timestep_live_input: `pass`
  - message: `The denoise timestep remained a live ONNX session input for every Stage 3 execution path.`

## Coverage

- compared_pairs: `82`
- missing_pairs: `0`
- missing_pair_list: `none`

## Compare Profiles

- Export Fidelity Compare (`export_reference_vs_onnx`):
  - torch_mode: `policy.cpu().float() with use_autocast=False`
  - scope: `compare ONNX outputs against export-mode Torch reference, aligned to the Stage 2 export boundary instead of the CUDA runtime Torch path`
  - note: `Stage 2 immediate compare keeps CPU ORT for vision_encoder/prefix_cache and allows denoise_step to fall back to CUDAExecutionProvider if CPUExecutionProvider cannot execute it.`
  - onnx_execution_plan:
    - `vision_encoder` providers=[['CPUExecutionProvider']] optimization_order=['all']
    - `prefix_cache` providers=[['CPUExecutionProvider']] optimization_order=['disable']
    - `denoise_step` providers=[['CPUExecutionProvider'], ['CUDAExecutionProvider', 'CPUExecutionProvider']] optimization_order=['disable', 'basic', 'all']
- Runtime-Oriented Compare (`runtime_reference_vs_onnx`):
  - torch_mode: `policy on runtime device with use_autocast=True`
  - scope: `compare ONNX outputs against runtime Torch reference using a CUDA-preferred, optimized ORT path that reflects deployment-oriented execution more closely`
  - note: `Runtime-oriented compare keeps the CUDA-preferred ORT path and allows optimization fallbacks if the preferred runtime configuration does not execute.`
  - onnx_execution_plan:
    - `vision_encoder` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
    - `prefix_cache` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
    - `denoise_step` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']

## Export Fidelity Compare

- local_subgraph_compare: `pass`
- local vision summary: `max_abs=0.000495911, mean_abs=3.74889e-06, min_cos=1`
- local prefix summary: `max_abs=0.000406265, mean_abs=7.47351e-06, min_cos=1`
- local denoise summary: `max_abs=1.54972e-06, mean_abs=1.12942e-07, min_cos=1`
- chained_compare: `pass`
- chained pipeline summary: `max_abs=7.39098e-06, mean_abs=3.51248e-07, min_cos=1`
- local_subgraph_execution:
  - `vision_encoder.top` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable
  - `denoise_step` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable
- chained_execution:
  - `vision_encoder.top` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable
  - `denoise_step` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable

## Runtime-Oriented Compare

- local_subgraph_compare: `warn`
- local vision summary: `max_abs=2.07516, mean_abs=0.023193, min_cos=0.999939`
- local prefix summary: `max_abs=3.85945, mean_abs=0.0992535, min_cos=0.999305`
- local denoise summary: `max_abs=0.0170748, mean_abs=0.00230241, min_cos=0.999972`
- chained_compare: `pass`
- chained pipeline summary: `max_abs=0.0259067, mean_abs=0.00297993, min_cos=0.999947`
- local_subgraph_execution:
  - `vision_encoder.top` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `denoise_step` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
- chained_execution:
  - `vision_encoder.top` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `denoise_step` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all

## Denoise Timestep Live Input

- local export-fidelity consumed: `True`
- local export-fidelity session_input_names: `['x_t', 'timestep', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- local export-fidelity dropped_inputs: `[]`
- local runtime-oriented consumed: `True`
- local runtime-oriented session_input_names: `['x_t', 'timestep', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- local runtime-oriented dropped_inputs: `[]`
- chained export-fidelity consumed: `True`
- chained export-fidelity session_input_names: `['x_t', 'timestep', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- chained export-fidelity dropped_inputs: `[]`
- chained runtime-oriented consumed: `True`
- chained runtime-oriented session_input_names: `['x_t', 'timestep', 'prefix_pad_masks', 'past_key_values.layer_00.key', 'past_key_values.layer_00.value', 'past_key_values.layer_01.key', 'past_key_values.layer_01.value', 'past_key_values.layer_02.key', 'past_key_values.layer_02.value', 'past_key_values.layer_03.key', 'past_key_values.layer_03.value', 'past_key_values.layer_04.key', 'past_key_values.layer_04.value', 'past_key_values.layer_05.key', 'past_key_values.layer_05.value', 'past_key_values.layer_06.key', 'past_key_values.layer_06.value', 'past_key_values.layer_07.key', 'past_key_values.layer_07.value', 'past_key_values.layer_08.key', 'past_key_values.layer_08.value', 'past_key_values.layer_09.key', 'past_key_values.layer_09.value', 'past_key_values.layer_10.key', 'past_key_values.layer_10.value', 'past_key_values.layer_11.key', 'past_key_values.layer_11.value', 'past_key_values.layer_12.key', 'past_key_values.layer_12.value', 'past_key_values.layer_13.key', 'past_key_values.layer_13.value', 'past_key_values.layer_14.key', 'past_key_values.layer_14.value', 'past_key_values.layer_15.key', 'past_key_values.layer_15.value', 'past_key_values.layer_16.key', 'past_key_values.layer_16.value', 'past_key_values.layer_17.key', 'past_key_values.layer_17.value']`
- chained runtime-oriented dropped_inputs: `[]`
