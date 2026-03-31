# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/run_stage5_bf16_safe`
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
  - torch_vs_trt: {'max_abs_diff': 0.109375, 'mean_abs_diff': 0.0006172290886752307, 'max_rel_diff': 27084.572265625, 'min_cosine_similarity': 0.9998968839645386}
  - onnx_vs_trt: {'max_abs_diff': 0.06751227378845215, 'mean_abs_diff': 0.0005274850991554558, 'max_rel_diff': 56811.21875, 'min_cosine_similarity': 0.9999324679374695}
  - onnx_runtime[top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
- prefix_cache: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.9015045166015625, 'mean_abs_diff': 0.06723145395517349, 'max_rel_diff': 22089148.0, 'min_cosine_similarity': 0.9997216463088989}
  - torch_vs_trt: {'max_abs_diff': 26.375, 'mean_abs_diff': 0.47371625900268555, 'max_rel_diff': 768750016.0, 'min_cosine_similarity': 0.9434547424316406}
  - onnx_vs_trt: {'max_abs_diff': 26.418460845947266, 'mean_abs_diff': 0.47394442558288574, 'max_rel_diff': 466945.0, 'min_cosine_similarity': 0.9434659481048584}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.025364339351654053, 'mean_abs_diff': 0.005503620952367783, 'max_rel_diff': 43.91902160644531, 'min_cosine_similarity': 0.9999344348907471}
  - torch_vs_trt: {'max_abs_diff': 0.15234375, 'mean_abs_diff': 0.0296552125364542, 'max_rel_diff': 368.058837890625, 'min_cosine_similarity': 0.9980466365814209}
  - onnx_vs_trt: {'max_abs_diff': 0.14656466245651245, 'mean_abs_diff': 0.028947001323103905, 'max_rel_diff': 47.3579216003418, 'min_cosine_similarity': 0.9981362223625183}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- pipeline: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.02420186996459961, 'mean_abs_diff': 0.006010141223669052, 'max_rel_diff': 61.458274841308594, 'min_cosine_similarity': 0.9999213218688965}
  - torch_vs_trt: {'max_abs_diff': 0.173828125, 'mean_abs_diff': 0.0360330194234848, 'max_rel_diff': 479.1176452636719, 'min_cosine_similarity': 0.9971083998680115}
  - onnx_vs_trt: {'max_abs_diff': 0.16530191898345947, 'mean_abs_diff': 0.03596564754843712, 'max_rel_diff': 1195.401123046875, 'min_cosine_similarity': 0.9971283078193665}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - onnx_runtime[denoise_step]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
