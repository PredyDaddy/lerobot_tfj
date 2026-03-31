# Stage3 Runner Report

## Scope

- Focus: `Torch vs ONNX` only for `PI0.5` Stage 3 verification.
- TensorRT was not modified or debugged in this task.
- Conda environment: `lerobot`

## Execution Commands

```bash
conda run -n lerobot python -V
conda run -n lerobot python -c "import torch, onnxruntime as ort; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ort', ort.__version__); print('providers', ort.get_available_providers())"
conda run -n lerobot python tfj_envs/pi_trt/scripts/step3_verify_onnx.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --run-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_onnx_debug_20260311_235500/run_main \
  --onnx-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_export_20260311_214232/run_main/artifacts/onnx \
  --report-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_onnx_debug_20260311_235500/reports/stage3_verify_onnx.json \
  --markdown-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_onnx_debug_20260311_235500/reports/stage3_verify_onnx.md
```

## Whether It Ran Through

- Script execution: `yes`
- Process exit code: `0`
- Generated reports:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_onnx_debug_20260311_235500/reports/stage3_verify_onnx.json`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/tmp_pi05_onnx_debug_20260311_235500/reports/stage3_verify_onnx.md`
- Report overall status: `warn`
- Pair coverage: `82 compared`, `0 missing`

## Key Numerical Results

- ONNX Runtime environment:
  - `torch=2.7.1+cu126`
  - `cuda=True`
  - `onnxruntime=1.23.2`
  - available providers: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- Local subgraph compare, `export_reference_vs_onnx`: `pass`
  - `vision_encoder`: `max_abs_diff=0.00625014`, `mean_abs_diff=3.94293e-05`, `min_cosine_similarity=0.999999`
  - `prefix_cache`: `max_abs_diff=0.0212836`, `mean_abs_diff=0.00253183`, `min_cosine_similarity=0.999998`
  - `denoise_step`: `max_abs_diff=0.00113094`, `mean_abs_diff=0.000249435`, `min_cosine_similarity=1.0`
- Chained compare, `export_reference_vs_onnx`: `pass`
  - `pipeline`: `max_abs_diff=0.000972457`, `mean_abs_diff=0.000250706`, `min_cosine_similarity=1.0`
- Local subgraph compare, `runtime_reference_vs_onnx`: `warn`
  - `vision_encoder`: `max_abs_diff=0.0414982`, `mean_abs_diff=0.00034767`, `min_cosine_similarity=0.999964`
  - `prefix_cache`: `max_abs_diff=0.904474`, `mean_abs_diff=0.0673935`, `min_cosine_similarity=0.999721`
  - `denoise_step`: `max_abs_diff=0.025332`, `mean_abs_diff=0.00551668`, `min_cosine_similarity=0.999934`
- Chained compare, `runtime_reference_vs_onnx`: `warn`
  - `pipeline`: `max_abs_diff=0.0248366`, `mean_abs_diff=0.00602709`, `min_cosine_similarity=0.999921`
- Denoise contract note:
  - ONNX denoise session input list does not consume `timestep`; report records `dropped_inputs=['timestep']` for local runtime, local export, and chained pipeline execution.

## Modified Files

- No source files were modified.
- Specifically, no edits were made to:
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/step3_verify_onnx.py`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/export_subgraphs.py`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/pi_compare_common.py`

## Remaining Risks

- `export_reference_vs_onnx` is clean, but `runtime_reference_vs_onnx` still warns. The current Stage 3 report explicitly defines those as different comparison bases:
  - runtime reference: policy on runtime device with `use_autocast=True`
  - export reference: `policy.cpu().float()` with `use_autocast=False`
- The largest remaining drift is in `runtime_reference_vs_onnx.prefix_cache`, so any downstream consumer that expects runtime-autocast parity, not export-boundary parity, still needs separate alignment work.
- `denoise_step` dropping `timestep` is not blocking this Stage 3 run, but it is still a contract detail worth tracking if later tooling assumes `timestep` must remain an ONNX input.
- This run used the explicit existing ONNX artifact directory from `tmp_pi05_export_20260311_214232/run_main/artifacts/onnx`; it did not regenerate ONNX files.
