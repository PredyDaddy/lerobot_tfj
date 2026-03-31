# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`
- overall_status: `fail`
- variant: `pi05`
- checkpoint_dir: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- requested_precision: `fp16`
- stage4_report_path: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json`
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

- vision_encoder: `fail`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.00049591064453125, 'mean_abs_diff': 3.7488850921363337e-06, 'max_rel_diff': 2.2103426456451416, 'min_cosine_similarity': 0.9999999403953552}
  - torch_vs_trt: {'max_abs_diff': 0.9991912841796875, 'mean_abs_diff': 0.008834916166961193, 'max_rel_diff': 16616.73828125, 'min_cosine_similarity': 0.9999884963035583}
  - onnx_vs_trt: {'max_abs_diff': 0.9992446899414062, 'mean_abs_diff': 0.008835150860249996, 'max_rel_diff': 125185.0, 'min_cosine_similarity': 0.9999882578849792}
  - onnx_runtime[top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
- prefix_cache: `fail`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 0.0004062652587890625, 'mean_abs_diff': 7.473509867850225e-06, 'max_rel_diff': 11.333333015441895, 'min_cosine_similarity': 0.9999997615814209}
  - torch_vs_trt: {'max_abs_diff': 12.222936630249023, 'mean_abs_diff': 0.34146174788475037, 'max_rel_diff': 144375.3125, 'min_cosine_similarity': 0.5300003886222839}
  - onnx_vs_trt: {'max_abs_diff': 12.222933769226074, 'mean_abs_diff': 0.3414613604545593, 'max_rel_diff': 88897.3671875, 'min_cosine_similarity': 0.5300004482269287}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- denoise_step: `fail`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 1.5497207641601562e-06, 'mean_abs_diff': 1.1294230972680452e-07, 'max_rel_diff': 0.0004069552815053612, 'min_cosine_similarity': 1.0}
  - torch_vs_trt: {'max_abs_diff': 0.025226354598999023, 'mean_abs_diff': 0.0008215145207941532, 'max_rel_diff': 0.7931674122810364, 'min_cosine_similarity': 0.999991774559021}
  - onnx_vs_trt: {'max_abs_diff': 0.02522575855255127, 'mean_abs_diff': 0.0008215119014494121, 'max_rel_diff': 0.7932515144348145, 'min_cosine_similarity': 0.999991774559021}
  - onnx_runtime: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
- pipeline: `fail`
  - onnx_compare_profile: `export_fidelity`
  - torch_vs_onnx: {'max_abs_diff': 7.3909759521484375e-06, 'mean_abs_diff': 3.5124793384966324e-07, 'max_rel_diff': 0.00041719962609931827, 'min_cosine_similarity': 1.0}
  - torch_vs_trt: {'max_abs_diff': 0.0631941556930542, 'mean_abs_diff': 0.0020231143571436405, 'max_rel_diff': 2.1415579319000244, 'min_cosine_similarity': 0.9999227523803711}
  - onnx_vs_trt: {'max_abs_diff': 0.06318771839141846, 'mean_abs_diff': 0.0020229255314916372, 'max_rel_diff': 2.14151930809021, 'min_cosine_similarity': 0.9999228119850159}
  - onnx_runtime[vision_top]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[vision_wrist]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=all`
  - onnx_runtime[prefix_cache]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
  - onnx_runtime[denoise_step]: `active_providers=['CPUExecutionProvider'], graph_optimization_level=disable`
