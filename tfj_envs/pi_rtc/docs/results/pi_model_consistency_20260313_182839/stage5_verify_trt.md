# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
- overall_status: `pass`
- torch_reference_mode: `export_reference_torch`

## ONNX Compare Profiles

- primary_onnx_compare_profile: `export_fidelity`
- Export Fidelity ONNX (`export_fidelity`)
  - purpose: `Primary ONNX baseline for Stage 5 metrics. Align to the Stage 2 export boundary without conflating Torch/ONNX/TRT comparisons with CUDA runtime Torch drift.`
  - `vision_encoder` providers=[['CPUExecutionProvider']] optimization_order=['all']
  - `prefix_cache` providers=[['CPUExecutionProvider']] optimization_order=['disable']
  - `denoise_step` providers=[['CPUExecutionProvider'], ['CUDAExecutionProvider', 'CPUExecutionProvider']] optimization_order=['disable', 'basic', 'all']
- Runtime-Oriented ONNX (`runtime_oriented`)
  - purpose: `Optional runtime-oriented ONNX execution profile. Keep CUDA-preferred optimized ORT separate from the primary export-fidelity metric.`
  - `vision_encoder` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
  - `prefix_cache` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']
  - `denoise_step` providers=[['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']] optimization_order=['all', 'basic', 'disable']

## Subgraphs

- vision_encoder: `pass`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.00049591064453125, 'mean_abs_diff': 3.7488850921363337e-06, 'max_rel_diff': 2.2103426456451416, 'min_cosine_similarity': 0.9999999403953552}
  - torch_vs_trt: {'max_abs_diff': 0.000396728515625, 'mean_abs_diff': 3.0867395253153518e-06, 'max_rel_diff': 6.401015281677246, 'min_cosine_similarity': 0.9999998211860657}
  - onnx_vs_trt: {'max_abs_diff': 0.00051116943359375, 'mean_abs_diff': 3.3218570933968294e-06, 'max_rel_diff': 70.5, 'min_cosine_similarity': 0.9999998211860657}
  - onnx_runtime[top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
- prefix_cache: `pass`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.0004062652587890625, 'mean_abs_diff': 7.473509867850225e-06, 'max_rel_diff': 11.333333015441895, 'min_cosine_similarity': 0.9999997615814209}
  - torch_vs_trt: {'max_abs_diff': 0.0007801055908203125, 'mean_abs_diff': 1.658303699514363e-05, 'max_rel_diff': 17.280054092407227, 'min_cosine_similarity': 0.9999995827674866}
  - onnx_vs_trt: {'max_abs_diff': 0.0005645751953125, 'mean_abs_diff': 1.7323412976111285e-05, 'max_rel_diff': 9.619119644165039, 'min_cosine_similarity': 0.999999463558197}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `pass`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 1.5497207641601562e-06, 'mean_abs_diff': 1.1294230972680452e-07, 'max_rel_diff': 0.0004069552815053612, 'min_cosine_similarity': 1.0}
  - torch_vs_trt: {'max_abs_diff': 1.7881393432617188e-06, 'mean_abs_diff': 1.362143109417957e-07, 'max_rel_diff': 0.00024128549557644874, 'min_cosine_similarity': 0.9999998807907104}
  - onnx_vs_trt: {'max_abs_diff': 1.1324882507324219e-06, 'mean_abs_diff': 1.2247706138168724e-07, 'max_rel_diff': 0.00020263063197489828, 'min_cosine_similarity': 0.9999998807907104}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- pipeline: `pass`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 7.3909759521484375e-06, 'mean_abs_diff': 3.5124793384966324e-07, 'max_rel_diff': 0.00041719962609931827, 'min_cosine_similarity': 1.0}
  - torch_vs_trt: {'max_abs_diff': 5.304813385009766e-06, 'mean_abs_diff': 2.3295369544484856e-07, 'max_rel_diff': 0.00017628175555728376, 'min_cosine_similarity': 1.0}
  - onnx_vs_trt: {'max_abs_diff': 7.3909759521484375e-06, 'mean_abs_diff': 4.4638494500759407e-07, 'max_rel_diff': 0.0004984175902791321, 'min_cosine_similarity': 1.0}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - onnx_runtime[denoise_step]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
