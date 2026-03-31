# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/run_stage5_bf16_mmfp32`
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
  - torch_vs_trt: {'max_abs_diff': 0.109375, 'mean_abs_diff': 0.0006469333311542869, 'max_rel_diff': 7858.6328125, 'min_cosine_similarity': 0.9998803734779358}
  - onnx_vs_trt: {'max_abs_diff': 0.07250165939331055, 'mean_abs_diff': 0.0005475422367453575, 'max_rel_diff': 42775.26171875, 'min_cosine_similarity': 0.9999209046363831}
  - onnx_runtime[top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
- prefix_cache: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.9015045166015625, 'mean_abs_diff': 0.06723145395517349, 'max_rel_diff': 22089148.0, 'min_cosine_similarity': 0.9997216463088989}
  - torch_vs_trt: {'max_abs_diff': 26.375, 'mean_abs_diff': 0.4739958643913269, 'max_rel_diff': 771875008.0, 'min_cosine_similarity': 0.9434547424316406}
  - onnx_vs_trt: {'max_abs_diff': 26.418460845947266, 'mean_abs_diff': 0.47437313199043274, 'max_rel_diff': 458753.0, 'min_cosine_similarity': 0.9434660077095032}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.025364339351654053, 'mean_abs_diff': 0.005503620952367783, 'max_rel_diff': 43.91902160644531, 'min_cosine_similarity': 0.9999344348907471}
  - torch_vs_trt: {'max_abs_diff': 0.140625, 'mean_abs_diff': 0.025684844702482224, 'max_rel_diff': 179.8235321044922, 'min_cosine_similarity': 0.9985114336013794}
  - onnx_vs_trt: {'max_abs_diff': 0.14731422066688538, 'mean_abs_diff': 0.025180919095873833, 'max_rel_diff': 41.94583511352539, 'min_cosine_similarity': 0.9985612034797668}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- pipeline: `warn`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.02420186996459961, 'mean_abs_diff': 0.006010141223669052, 'max_rel_diff': 61.458274841308594, 'min_cosine_similarity': 0.9999213218688965}
  - torch_vs_trt: {'max_abs_diff': 0.19921875, 'mean_abs_diff': 0.035335473716259, 'max_rel_diff': 96.19999694824219, 'min_cosine_similarity': 0.9971752166748047}
  - onnx_vs_trt: {'max_abs_diff': 0.18630468845367432, 'mean_abs_diff': 0.03528498485684395, 'max_rel_diff': 2892.621337890625, 'min_cosine_similarity': 0.997198760509491}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - onnx_runtime[denoise_step]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
