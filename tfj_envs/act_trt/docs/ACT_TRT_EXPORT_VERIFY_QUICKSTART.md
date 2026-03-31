# ACT TRT Export Verify Quickstart

新手如果想先看“直接抄命令”的版本，先读：

- [ACT_TRT_PYTHON_SCRIPT_USAGE.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_TRT_PYTHON_SCRIPT_USAGE.md)

## 1. Current Status

当前这套本地 `tfj_envs` 流程已经验证通过。

对应对象：

- checkpoint:
  - `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model`
- onnx:
  - `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single.onnx`
- engine:
  - `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan`

验证结果：

- [ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json)
- [ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.md)
- [ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json)
- [ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md](/data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md)

关键数值：

- `Torch vs ONNX max_abs_diff = 5.4836273193359375e-06`
- `Torch vs TRT max_abs_diff = 4.470348358154297e-06`
- `ONNX vs TRT max_abs_diff = 1.6614794731140137e-06`
- `passed = true`
- `all_within_threshold = true`

## 2. Important Rule

本地 FP32 TensorRT builder 必须禁用 `TF32`。

原因：

- 如果 FP32 build 允许 `TF32`，本地测试里会出现大约 `6.28e-4` 的系统性漂移
- 禁用 `TF32` 后，误差回到 `1e-6` 量级

对应代码位置：

- [build_act_trt_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/build_act_trt_engine.py#L19)
- [build_act_trt_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/build_act_trt_engine.py#L58)

## 3. Export Environment

推荐环境划分：

- `lerobot_flex`
  - 用于导出 ONNX、构建 TensorRT engine
- `lerobot`
  - 用于验证、对比、真实机器人运行

环境检查：

```bash
conda run -n lerobot_flex python -c "import tensorrt; print(tensorrt.__version__)"
conda run -n lerobot python -c "import tensorrt; print(tensorrt.__version__)"
conda run -n lerobot python -c "import onnxruntime as ort; print(ort.__version__)"
```

## 4. How To Export

使用总入口脚本：

- [export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)

### 4.1 从 checkpoint 重新导出 ONNX + engine

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0
```

这条命令会做四件事：

1. 导出 `act_single.onnx`
2. 写 `export_metadata.json`
3. 构建 `act_single_fp32.plan`
4. 跑 Torch/ONNX/TRT 验证

### 4.2 只重编 engine，不重导 ONNX

当你已经有 `act_single.onnx` 和 `export_metadata.json` 时：

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --trt-device cuda:0 \
  --skip-export
```

### 4.3 只导出，不做验证

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0 \
  --no-verify-export \
  --no-verify-engine
```

## 5. Export Outputs

默认输出目录规则：

- `outputs/deploy/act_trt/<run_name>/<checkpoint_step>/`

当前例子里是：

- `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/`

主要产物：

- `act_single.onnx`
- `export_metadata.json`
- `act_single_fp32.plan`
- `trt_build_summary_fp32.json`

## 6. How To Verify

### 6.1 Torch vs ONNX vs TRT 三方验证

用这个脚本：

- [verify_act_torch_onnx_trt.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/verify_act_torch_onnx_trt.py)

命令：

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/verify_act_torch_onnx_trt.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --onnx /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single.onnx \
  --engine /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --device cuda:0 \
  --report /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json
```

通过标准：

- `passed = true`
- 三组 `max_abs_diff` 都要小于阈值 `1e-4`

### 6.2 Safetensors vs TRT 直接对比

用这个脚本：

- [compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)

命令：

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py \
  --report-json /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json \
  --report-md /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md
```

通过标准：

- `all_within_threshold = true`

## 7. How To Read The Results

### 7.1 如果 `Torch vs ONNX` 好，但 `Torch vs TRT` / `ONNX vs TRT` 不好

说明：

- `safetensors -> ONNX` 这段是对的
- `ONNX -> TensorRT` 这段有问题

优先检查：

1. `build_act_trt_engine.py` 是否关闭了 `TF32`
2. 是否真的重新编了 engine，而不是复用了旧 plan
3. 是否误用了别的 engine 路径

### 7.2 如果三方都好

说明：

- checkpoint、onnx、engine 这条链路是通的
- 可以继续做上机推理或录制评测

## 8. Where To Modify

### 8.1 改默认导出路径或模型路径

改：

- [export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)

主要看：

- `--checkpoint`
- `resolve_output_dir(...)`

### 8.2 改本地 ONNX 导出逻辑

改：

- [export_act_onnx.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_onnx.py)

### 8.3 改 TensorRT build 行为

改：

- [build_act_trt_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/build_act_trt_engine.py)

重点参数：

- `--precision`
- `--allow-tf32`
- `--opt-level`
- `--workspace-gb`

### 8.4 改验证逻辑

改：

- [verify_act_torch_onnx_trt.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/verify_act_torch_onnx_trt.py)
- [compare_act_016000_my_exported_trt_vs_safetensors.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py)

## 9. Recommended Minimal Workflow

每次要重新确认本地导出链条时，按这个顺序：

1. 重新导出

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0
```

2. 再跑三方验证

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/verify_act_torch_onnx_trt.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --onnx /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single.onnx \
  --engine /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --device cuda:0 \
  --report /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_VERIFY_TORCH_ONNX_TRT_016000_FP32.json
```

3. 再跑 safetensors vs TRT 对比

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_016000_my_exported_trt_vs_safetensors.py \
  --report-json /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.json \
  --report-md /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_MY_EXPORTED_TRT_VS_SAFETENSORS.md
```

只要这两份验证都通过，你再去上机。
