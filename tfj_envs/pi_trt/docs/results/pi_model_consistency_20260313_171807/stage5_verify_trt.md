# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807`
- overall_status: `error`
- torch_reference_mode: `export_reference_torch`

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
  - torch_vs_trt: {'max_abs_diff': 0.0009136199951171875, 'mean_abs_diff': 1.211302605952369e-05, 'max_rel_diff': 19.41794204711914, 'min_cosine_similarity': 0.9999996423721313}
  - onnx_vs_trt: {'max_abs_diff': 0.000507354736328125, 'mean_abs_diff': 1.281352433579741e-05, 'max_rel_diff': 4.980867385864258, 'min_cosine_similarity': 0.9999997615814209}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `error`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_trt: {'max_abs_diff': 3.814697265625e-06, 'mean_abs_diff': 2.9584592198261817e-07, 'max_rel_diff': 0.0004197658854536712, 'min_cosine_similarity': 0.9999998807907104}
  - errors: ["ONNX denoise_step failed: RuntimeError: Unable to execute ONNX model /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807/artifacts/onnx/pi05_denoise_step.onnx with any provider candidate. attempts=[{'providers': ['CPUExecutionProvider'], 'optimization_level': 'disable', 'stage': 'load', 'status': 'error', 'error': 'Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807/artifacts/onnx/pi05_denoise_step.onnx failed:Type Error: Type parameter (T) of Optype (Mul) bound to different types (tensor(float) and tensor(double) in node (node_Mul_106).'}]"]
- pipeline: `error`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_trt: {'max_abs_diff': 4.0531158447265625e-06, 'mean_abs_diff': 3.463273685611057e-07, 'max_rel_diff': 0.00046266464050859213, 'min_cosine_similarity': 0.9999998807907104}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - errors: ["ONNX pipeline failed: RuntimeError: Unable to execute ONNX model /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807/artifacts/onnx/pi05_denoise_step.onnx with any provider candidate. attempts=[{'providers': ['CPUExecutionProvider'], 'optimization_level': 'disable', 'stage': 'load', 'status': 'error', 'error': 'Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_171807/artifacts/onnx/pi05_denoise_step.onnx failed:Type Error: Type parameter (T) of Optype (Mul) bound to different types (tensor(float) and tensor(double) in node (node_Mul_106).'}]"]
