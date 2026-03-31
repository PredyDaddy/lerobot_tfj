# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_prefix_probe_mm_softmax_20260314_180013`
- overall_status: `fail`
- variant: `pi05`
- checkpoint_dir: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- requested_precision: `fp16`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_prefix_probe_mm_softmax_20260314_180013/stage4_build_engines.json`
- torch_reference_mode: `export_reference_torch`
- stage5_scope: `export-boundary single-step correctness gate`

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

- prefix_cache: `fail`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.0004062652587890625, 'mean_abs_diff': 7.473509867850225e-06, 'max_rel_diff': 11.333333015441895, 'min_cosine_similarity': 0.9999997615814209}
  - torch_vs_trt: {'max_abs_diff': 12.206457138061523, 'mean_abs_diff': 0.32431307435035706, 'max_rel_diff': 74835.6953125, 'min_cosine_similarity': 0.5297124981880188}
  - onnx_vs_trt: {'max_abs_diff': 12.206454277038574, 'mean_abs_diff': 0.32431310415267944, 'max_rel_diff': 46079.0, 'min_cosine_similarity': 0.5297127366065979}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
