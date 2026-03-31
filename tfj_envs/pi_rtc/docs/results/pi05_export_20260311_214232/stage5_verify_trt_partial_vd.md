# PI05 Torch / ONNX / TensorRT Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_export_20260311_214232/run_main`
- overall_status: `warn`

## Subgraphs

- vision_encoder: `warn`
  - torch_vs_onnx: {'max_abs_diff': 0.04149818420410156, 'mean_abs_diff': 0.00034767002216540277, 'max_rel_diff': 17233.85546875, 'min_cosine_similarity': 0.9999642372131348}
  - torch_vs_trt: {'max_abs_diff': 0.04380226135253906, 'mean_abs_diff': 0.00034404188045300543, 'max_rel_diff': 13943.0341796875, 'min_cosine_similarity': 0.9999661445617676}
  - onnx_vs_trt: {'max_abs_diff': 0.005472898483276367, 'mean_abs_diff': 5.970504935248755e-05, 'max_rel_diff': 4704.1748046875, 'min_cosine_similarity': 0.9999986290931702}
- denoise_step: `warn`
  - torch_vs_onnx: {'max_abs_diff': 0.02533203363418579, 'mean_abs_diff': 0.005516675766557455, 'max_rel_diff': 40.97930145263672, 'min_cosine_similarity': 0.9999340176582336}
  - torch_vs_trt: {'max_abs_diff': 0.025572121143341064, 'mean_abs_diff': 0.005480028223246336, 'max_rel_diff': 61.476417541503906, 'min_cosine_similarity': 0.9999357461929321}
  - onnx_vs_trt: {'max_abs_diff': 0.003560197539627552, 'mean_abs_diff': 0.0009598922915756702, 'max_rel_diff': 1.7056461572647095, 'min_cosine_similarity': 0.9999980330467224}
