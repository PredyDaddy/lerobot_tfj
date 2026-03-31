# PI05 Stage 3 ONNX Verification

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807`
- onnx_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807/artifacts/onnx`
- overall_status: `fail`
- stage3_acceptance: `fail`
- report_goal: `export-fidelity compare` and `runtime-oriented compare` are separated so CPU ORT export fidelity is not conflated with CUDA ORT runtime drift.
- compare_scopes: `local_subgraph_compare` means Torch intermediates are used as ONNX inputs; `chained_compare` means ONNX vision -> ONNX prefix -> ONNX denoise.

## Acceptance

- local_export_fidelity_compare: `fail`
  - message: `Local export-fidelity ONNX compare did not meet acceptance.`
- chained_export_fidelity_compare: `fail`
  - message: `Chained export-fidelity ONNX compare did not meet acceptance.`
- denoise_timestep_live_input: `fail`
  - message: `The denoise timestep contract regressed in one or more Stage 3 execution paths.`

## Coverage

- compared_pairs: `78`
- missing_pairs: `4`
- missing_pair_list:
  - `local_subgraph_compare.runtime_reference_vs_onnx.denoise_step.v_t`
  - `local_subgraph_compare.export_reference_vs_onnx.denoise_step.v_t`
  - `chained_compare.runtime_reference_vs_onnx.pipeline.v_t`
  - `chained_compare.export_reference_vs_onnx.pipeline.v_t`

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

- local_subgraph_compare: `error`
- local vision summary: `max_abs=0.000495911, mean_abs=3.74889e-06, min_cos=1`
- local prefix summary: `max_abs=0.000406265, mean_abs=7.47351e-06, min_cos=1`
- local denoise summary: `not_available`
- chained_compare: `error`
- chained pipeline summary: `not_available`
- local_subgraph_execution:
  - `vision_encoder.top` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable
  - `denoise_step` status=error
- chained_execution:
  - `vision_encoder.top` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CPUExecutionProvider'], graph_optimization_level=disable
  - `denoise_step` status=error

## Runtime-Oriented Compare

- local_subgraph_compare: `error`
- local vision summary: `max_abs=2.07516, mean_abs=0.023193, min_cos=0.999939`
- local prefix summary: `max_abs=3.85945, mean_abs=0.0992535, min_cos=0.999305`
- local denoise summary: `not_available`
- chained_compare: `error`
- chained pipeline summary: `not_available`
- local_subgraph_execution:
  - `vision_encoder.top` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `denoise_step` status=error
- chained_execution:
  - `vision_encoder.top` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `vision_encoder.wrist` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `prefix_cache` active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], graph_optimization_level=all
  - `denoise_step` status=error

## Denoise Timestep Live Input

- local export-fidelity consumed: `False`
- local export-fidelity session_input_names: `[]`
- local export-fidelity dropped_inputs: `[]`
- local runtime-oriented consumed: `False`
- local runtime-oriented session_input_names: `[]`
- local runtime-oriented dropped_inputs: `[]`
- chained export-fidelity consumed: `False`
- chained export-fidelity session_input_names: `[]`
- chained export-fidelity dropped_inputs: `[]`
- chained runtime-oriented consumed: `False`
- chained runtime-oriented session_input_names: `[]`
- chained runtime-oriented dropped_inputs: `[]`
