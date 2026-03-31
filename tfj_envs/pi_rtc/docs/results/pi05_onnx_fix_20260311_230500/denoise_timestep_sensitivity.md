# PI05 Denoise Timestep Sensitivity

## Scope

- policy: `/data/tfj/lerobot_tfj/pi_model/pretrained_model`
- onnx: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/onnx/pi05_denoise_step.onnx`
- engine: `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/engines_fp32/pi05_denoise_step.engine`
- reference mode: `export-style torch` (`policy.cpu().float() + use_autocast=False`)

## Why This Check Exists

Stage 4 build logs warned that `timestep` might be unused or compile-time-only in the TensorRT `denoise_step` engine.
This check varies `timestep` at the same `x_t` and the same prefix-cache tensors, then compares whether Torch / ONNX / TensorRT outputs actually change.

## Tested Timesteps

- `1.0`
- `0.5`
- `0.1`

## Same-Timestep Framework Agreement

- `t = 1.0`
  - `torch_vs_onnx`: `max_abs_diff=1.43051e-06`, `mean_abs_diff=2.70022e-07`
  - `torch_vs_trt`: `max_abs_diff=1.54972e-06`, `mean_abs_diff=3.66267e-07`
  - `onnx_vs_trt`: `max_abs_diff=1.84774e-06`, `mean_abs_diff=3.57350e-07`
- `t = 0.5`
  - `torch_vs_onnx`: `max_abs_diff=1.43051e-06`, `mean_abs_diff=2.70022e-07`
  - `torch_vs_trt`: `max_abs_diff=1.54972e-06`, `mean_abs_diff=3.66267e-07`
  - `onnx_vs_trt`: `max_abs_diff=1.84774e-06`, `mean_abs_diff=3.57350e-07`
- `t = 0.1`
  - `torch_vs_onnx`: `max_abs_diff=1.43051e-06`, `mean_abs_diff=2.70022e-07`
  - `torch_vs_trt`: `max_abs_diff=1.54972e-06`, `mean_abs_diff=3.66267e-07`
  - `onnx_vs_trt`: `max_abs_diff=1.84774e-06`, `mean_abs_diff=3.57350e-07`

## Cross-Timestep Sensitivity

- `t=1.0` vs `t=0.5`
  - Torch: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - ONNX: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - TensorRT: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
- `t=1.0` vs `t=0.1`
  - Torch: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - ONNX: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - TensorRT: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
- `t=0.5` vs `t=0.1`
  - Torch: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - ONNX: `max_abs_diff=0.0`, `mean_abs_diff=0.0`
  - TensorRT: `max_abs_diff=0.0`, `mean_abs_diff=0.0`

## Conclusion

- For this checkpoint and this export boundary, `denoise_step` output is currently invariant to the tested `timestep` values in Torch, ONNX, and TensorRT.
- The TensorRT warning about `timestep` being unused or compile-time-only is therefore consistent with observed behavior, but it is not a TensorRT-only issue.
- At the current boundary, TensorRT still reproduces ONNX tightly. The bigger open question is whether the PI0.5 checkpoint itself is expected to respond to `timestep` at this one-step denoise boundary.
