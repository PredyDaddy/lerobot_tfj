# ACT TRT Export And Deploy Guide

## 1. Directory Layout

This workspace is organized as:

- [README.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/README.md)
- [scripts/export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)
- [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)
- [scripts/run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py)
- [scripts/run_act_onnx_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_onnx_so101_eval.py)
- [scripts/compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)
- [scripts/trt_act_policy.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/trt_act_policy.py)

Use this document as the single entry point for:

- exporting an ACT checkpoint to ONNX and TensorRT
- verifying whether the exported TensorRT engine matches the checkpoint
- running the exported engine on the SO101 real robot
- knowing exactly which file to edit when paths or parameters change

## 2. Environment Split

Use two environments:

- `lerobot_flex`
  - for export, TensorRT build, and Torch/ONNX/TRT verification
- `lerobot`
  - for real-robot runtime
  - this env must already be able to `import tensorrt`

Recommended checks:

```bash
conda run -n lerobot_flex python -c "import torch; print(torch.__version__)"
conda run -n lerobot_flex python -c "import tensorrt; print(tensorrt.__version__)"
conda run -n lerobot python -c "import tensorrt; print(tensorrt.__version__)"
```

## 3. Current Local Export Status

The latest fully local `tfj_envs` re-export uses:

- checkpoint:
  - `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- my exported TRT engine:
  - `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`
- export metadata:
  - `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json`

Current comparison outputs:

- [ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json)
- [ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md)
- [ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json)
- [ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.md)

Current status:

- `Torch vs ONNX` is good
- `Torch vs TRT` is good
- `ONNX vs TRT` is good

Interpretation:

- the latest fully local export path is numerically aligned end-to-end
- the key local fix was rebuilding the FP32 TensorRT engine with TF32 disabled

## 4. Export Flow

### 4.1 One-command export

Use [scripts/export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py).

Example:

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0
```

This wrapper will:

1. export `act_single.onnx`
2. write `export_metadata.json`
3. build `act_single_fp32.plan`
4. run Torch/ONNX/TRT verification

Important local builder note:

- the local FP32 TensorRT builder disables `TF32` by default
- this is required to keep the rebuilt engine numerically aligned with checkpoint and ONNX outputs

Default output directory rule:

- `outputs/deploy/act_trt/<run_name>/<checkpoint_step>/`

For the current checkpoint this becomes:

- `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/`

### 4.2 If you want to reuse an existing ONNX

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --trt-device cuda:0 \
  --skip-export
```

## 5. Comparison Flow

### 5.1 Compare `016000 safetensors` vs my exported `016000 TRT`

Use [scripts/compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py).

Command:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py
```

Default outputs:

- `/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json`
- `/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md`

### 5.2 How to read the result

Main fields:

- `all_within_threshold`
  - `true` means the TRT engine matches the checkpoint under the configured threshold
- `max_abs_diff`
  - the biggest element-wise absolute difference
- `engine_io_mapping`
  - confirms which TensorRT input/output names are actually bound

For the current fully local re-export:

- `all_within_threshold = true`
- `max_abs_diff = 4.470348358154297e-06`

## 6. Real-Robot Runtime

There are two runtime modes:

- pure inference, no dataset recording
  - [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)
- eval + recording dataset
  - [scripts/run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py)

### 6.1 Pure TRT inference command

This is the recommended minimal runtime command for the engine I exported:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py \
  --robot-id=my_so101 \
  --robot-port=/dev/ttyACM0 \
  --robot-calibration-dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index=4 \
  --wrist-cam-index=6 \
  --camera-width=640 \
  --camera-height=480 \
  --camera-fps=30 \
  --task="grasp block in bin" \
  --policy-path=/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --trt-path=/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --trt-metadata-path=/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json \
  --trt-device=cuda:0 \
  --run-time-s=300
```

### 6.2 Pure TRT preflight command

Run this first if you only want to check that camera and engine load succeed:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py \
  --preflight-only
```

If the cameras are temporarily unavailable but you still want to verify the TensorRT engine:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py \
  --preflight-only \
  --skip-camera-preflight
```

### 6.3 TRT eval + recording command

If you need episode recording as well:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py \
  --robot-id=my_so101 \
  --robot-port=/dev/ttyACM0 \
  --robot-calibration-dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index=4 \
  --wrist-cam-index=6 \
  --camera-width=640 \
  --camera-height=480 \
  --camera-fps=30 \
  --dataset-repo-id=admin123/eval_grasp_block_in_bin1 \
  --dataset-root=/home/cqy/.cache/huggingface/lerobot/admin123/eval_grasp_block_in_bin2 \
  --dataset-push-to-hub=false \
  --dataset-num-episodes=5 \
  --dataset-episode-time-s=300 \
  --dataset-reset-time-s=10 \
  --dataset-single-task="grasp block in bin" \
  --policy-path=/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --trt-path=/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --trt-metadata-path=/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json \
  --trt-device=cuda:0 \
  --display-data=false \
  --play-sounds=false
```

## 7. Where To Modify Things

Prefer passing CLI arguments first. Only edit script defaults if you want to permanently change the local baseline.

### 7.1 Change the default checkpoint / engine / metadata

Edit these constants:

- [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)
  - `DEFAULT_POLICY_PATH`
  - `DEFAULT_TRT_PATH`
  - `DEFAULT_TRT_METADATA_PATH`
- [scripts/run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py)
  - `DEFAULT_POLICY_PATH`
  - `DEFAULT_TRT_PATH`
  - `DEFAULT_TRT_METADATA_PATH`
- [scripts/compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)
  - `DEFAULT_CHECKPOINT`
  - `DEFAULT_ENGINE`
  - `DEFAULT_METADATA`

### 7.2 Change the robot hardware parameters

Edit or override:

- `--robot-port`
- `--top-cam-index`
- `--wrist-cam-index`
- `--camera-width`
- `--camera-height`
- `--camera-fps`
- `--robot-calibration-dir`

The default values live in:

- [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)
- [scripts/run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_so101_eval.py)
- [scripts/run_act_onnx_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_onnx_so101_eval.py)

### 7.3 Change ACT deployment behavior

If you want less aggressive chunk caching during deployment, modify:

- `--policy-n-action-steps`

This is handled in:

- [scripts/run_act_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py)

Rule:

- it must be within `[1, chunk_size]`
- if you set `--policy-temporal-ensemble-coeff`, then `--policy-n-action-steps` must be `1`

### 7.4 Change TensorRT input/output name mapping

If a future engine uses different tensor names, edit:

- [scripts/trt_act_policy.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/trt_act_policy.py)

Specifically:

- `resolve_trt_io_mapping(...)`

This is the only place that should decide how:

- `observation.state`
- `observation.images.top`
- `observation.images.wrist`

map into TensorRT tensor names.

### 7.5 Change report output locations

Edit:

- [scripts/compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)
  - `DEFAULT_REPORT_JSON`
  - `DEFAULT_REPORT_MD`
- [scripts/compare_act_safetensors_trt.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py)
  - `resolve_default_report_path(...)`

## 8. Common Problems

### 8.1 `conda run` looks stuck

Use `--live-stream`:

```bash
conda run --live-stream -n lerobot ...
```

### 8.2 TensorRT can import in `lerobot_flex` but not in `lerobot`

Install it into `lerobot`, because real-robot runtime uses that env.

### 8.3 The script exits before robot connect

Run the staged checks first:

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py --dry-run
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/run_act_trt_infer_so101.py --preflight-only --skip-camera-preflight
```

### 8.4 Old `018000` debug material is still in `docs/`

Treat these as legacy debugging artifacts only:

- `ACT_SAFETENSORS_VS_TRT_018000_FP32.json`
- `ACT_SAFETENSORS_VS_ONNX_018000.json`
- `ACT_ONNX_VS_TRT_018000_FP32.json`
- `compare_current_runtime_trt_vs_safetensors.json`

They are not the main result for my exported `016000` engine.
