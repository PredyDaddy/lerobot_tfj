# PI05 Torch vs ONNX Report

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- run_dir: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_export_20260311_214232/run_main`
- passed: `False`

## Execution

- vision_encoder_execution: {'status': 'ok', 'providers': ['CPUExecutionProvider']}
- prefix_cache_execution: {'status': 'ok', 'providers': ['CPUExecutionProvider']}
- denoise_step_execution: {'status': 'error', 'error_type': 'InvalidArgument', 'error': '[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid input name: timestep', 'providers': []}

## Summary

- vision_encoder: {'max_abs_diff': 0.04388093948364258, 'mean_abs_diff': 0.0003403137670829892, 'max_rel_diff': 18290.703125, 'min_cosine_similarity': 0.9999664425849915}
- vision_encoder_check: {'thresholds': {'max_abs_diff': 0.05, 'mean_abs_diff': 0.005, 'min_cosine_similarity': 0.999}, 'passed': True, 'failures': []}
- prefix_cache: {'max_abs_diff': 0.9015045166015625, 'mean_abs_diff': 0.06723145395517349, 'max_rel_diff': 22089148.0, 'min_cosine_similarity': 0.9997216463088989}
- prefix_cache_check: {'thresholds': {'max_abs_diff': 0.05, 'mean_abs_diff': 0.005, 'min_cosine_similarity': 0.999}, 'passed': False, 'failures': ['max_abs_diff 0.901505 > 0.05', 'mean_abs_diff 0.0672315 > 0.005']}
- denoise_step: None
- denoise_step_check: None
