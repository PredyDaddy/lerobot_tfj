# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/run_stage5_fp32`
- overall_status: `warn`

## ONNX Compare Profiles

- primary_onnx_compare_profile: `export_fidelity`
- Export Fidelity ONNX (`export_fidelity`)
  - purpose: `Primary ONNX baseline for Stage 5 metrics. Align to Stage 2 CPU ORT export fidelity so Torch/ONNX/TRT comparisons are not conflated with CUDA ORT runtime drift.`
  - `vision_encoder` providers=[['CPUExecutionProvider']] optimization_order=['all']
  - `prefix_cache` providers=[['CPUExecutionProvider']] optimization_order=['disable']
  - `denoise_step` providers=[['CPUExecutionProvider']] optimization_order=['disable']
- Runtime-Oriented ONNX (`runtime_oriented`)
  - purpose: `Optional runtime-oriented ONNX execution profile. Keep CUDA-preferred optimized ORT separate from the primary export-fidelity metric.`
  - `vision_encoder` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
  - `prefix_cache` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
  - `denoise_step` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']

## Subgraphs

- vision_encoder: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.04388093948364258, 'mean_abs_diff': 0.0003403137670829892, 'max_rel_diff': 18290.703125, 'min_cosine_similarity': 0.9999664425849915}
  - torch_vs_trt: {'max_abs_diff': 0.04388236999511719, 'mean_abs_diff': 0.00034031341783702374, 'max_rel_diff': 18288.859375, 'min_cosine_similarity': 0.9999664425849915}
  - onnx_vs_trt: {'max_abs_diff': 1.1444091796875e-05, 'mean_abs_diff': 7.340729268889845e-08, 'max_rel_diff': 9.285477638244629, 'min_cosine_similarity': 0.9999998211860657}
  - onnx_runtime[top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
- prefix_cache: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.9015045166015625, 'mean_abs_diff': 0.06723145395517349, 'max_rel_diff': 22089148.0, 'min_cosine_similarity': 0.9997216463088989}
  - torch_vs_trt: {'max_abs_diff': 0.9015483856201172, 'mean_abs_diff': 0.06723199784755707, 'max_rel_diff': 22086906.0, 'min_cosine_similarity': 0.9997217059135437}
  - onnx_vs_trt: {'max_abs_diff': 0.000133514404296875, 'mean_abs_diff': 1.001203327177791e-05, 'max_rel_diff': 5.8788862228393555, 'min_cosine_similarity': 0.9999998211860657}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.025364339351654053, 'mean_abs_diff': 0.005503620952367783, 'max_rel_diff': 43.91902160644531, 'min_cosine_similarity': 0.9999344348907471}
  - torch_vs_trt: {'max_abs_diff': 0.0253639817237854, 'mean_abs_diff': 0.005503613967448473, 'max_rel_diff': 43.92070770263672, 'min_cosine_similarity': 0.9999344348907471}
  - onnx_vs_trt: {'max_abs_diff': 1.6689300537109375e-06, 'mean_abs_diff': 3.675970958738617e-07, 'max_rel_diff': 0.0007507806876674294, 'min_cosine_similarity': 1.0}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- pipeline: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.02420186996459961, 'mean_abs_diff': 0.006010141223669052, 'max_rel_diff': 61.458274841308594, 'min_cosine_similarity': 0.9999213218688965}
  - torch_vs_trt: {'max_abs_diff': 0.024201631546020508, 'mean_abs_diff': 0.006010174751281738, 'max_rel_diff': 61.456642150878906, 'min_cosine_similarity': 0.9999213218688965}
  - onnx_vs_trt: {'max_abs_diff': 1.6689300537109375e-06, 'mean_abs_diff': 4.2590372117956576e-07, 'max_rel_diff': 0.023350227624177933, 'min_cosine_similarity': 1.0}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - onnx_runtime[denoise_step]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
