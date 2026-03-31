# GROOT TensorRT 自运行手册

这份文档只做一件事：

- 让你在当前仓库里，按最短路径自己完成导出、数值测试、以及本地 TRT 运行

## 1. 先确认你理解的是当前真实结构

当前 GROOT TRT 不是一个单体 engine，而是：

- 7 个 TensorRT 子图
- PyTorch glue
- checkpoint processors
- LeRobot runtime

固定 7 个子图：

- backbone:
  - `vit_fp16`
  - `llm_fp16`
- action head:
  - `vlln_vl_self_attention`
  - `state_encoder`
  - `action_encoder`
  - `DiT_fp16`
  - `action_decoder`

运行时硬依赖仍然包括：

- `pretrained_model/`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

不要误解成：

- “只要有 engine 文件就能单独推理”
- “可以直接走 `run_groot_infer.py --backend=tensorrt`”

当前 checkout 里：

- 数值测试 Stage 3 和 Stage 5 可以跑
- 本地 hybrid TRT 运行可以跑
- Stage 6 mock compare 还没有重新实现

## 2. 已验证路径

仓库根目录：

- `/data/tfj/lerobot_tfj`

已验证 checkpoint：

- `/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000`

推荐你直接把下面两个变量设好：

```bash
cd /data/tfj/lerobot_tfj

export POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000
export RUN_DIR=/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_$(date +%Y%m%d_%H%M%S)
```

## 3. 环境准备

默认 conda 环境：

```bash
export CONDA_ENV=lerobot_flex
```

如果当前 conda 环境里不能直接 `import tensorrt`，再额外设置：

```bash
export TENSORRT_PY_DIR=/your/tensorrt/python/path
```

如果你不想用系统默认临时目录，可以设置：

```bash
export TMPDIR=/your/tmp/path
```

如果你不设置 `TMPDIR`，当前脚本会自动使用：

- `<RUN_DIR>/.tmp`

建议你先做两个最小检查：

```bash
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.cuda.is_available())"
```

```bash
conda run -n "${CONDA_ENV}" python -c "import tensorrt as trt; print(trt.__version__)"
```

如果第二条失败，再去设置 `TENSORRT_PY_DIR`。

## 4. 第一步：导出 ONNX 并构建 engine

这是最推荐的入口：

```bash
bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

默认行为：

- `VIDEO_VIEWS=2`
- 导出 seq len 默认跟随 2-view
- TRT profile 默认是 `80 / 568 / 600`

如果你只想显式声明 2-view：

```bash
VIDEO_VIEWS=2 bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

导出成功后，你应该能看到这些关键产物：

- `${RUN_DIR}/stage1_safetensors_to_torch.json`
- `${RUN_DIR}/stage2_export_onnx.json`
- `${RUN_DIR}/stage4_build_engines.json`
- `${RUN_DIR}/gr00t_onnx/`
- `${RUN_DIR}/gr00t_engine_api_trt1013/`

## 5. 第二步：做数值比较

导出完成后，直接跑统一 compare 入口：

```bash
bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

成功后你应该看到：

- `${RUN_DIR}/stage3_verify_onnx.json`
- `${RUN_DIR}/stage5_verify_trt.json`
- `${RUN_DIR}/compare_safetensor_onnx_trt.json`

这一步的语义是：

- `safetensors -> PyTorch policy`
- `ONNX`
- `TensorRT`

三者通过同一套输入边界做数值比较。

底层比较脚本是：

- `scripts/compare_torch_onnx_local.py`
- `scripts/compare_torch_trt_local.py`
- `scripts/compare_safetensor_onnx_trt.py`

当前这条链路里，重点看 `denoising` 相关 cosine 是否保持很高。

已经验证过的一组健康参考值：

- ONNX denoising cosine
  - 1-view: `0.999998525`
  - 2-view: `0.999996353`
- TRT denoising cosine
  - 1-view: `0.999997003`
  - 2-view: `0.999990810`

这些值不是死阈值，但如果你明显低很多，就需要继续排查。

## 6. 第三步：做本地 TRT 运行

当前能跑的是本地 hybrid runtime，不是缺失的历史 `backend=tensorrt` 路径。

最小 smoke：

```bash
NUM_STEPS=1 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh "${POLICY_PATH}" "${RUN_DIR}"
```

如果你想多跑几步：

```bash
NUM_STEPS=8 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh "${POLICY_PATH}" "${RUN_DIR}"
```

成功后关键产物通常在：

- `${RUN_DIR}/local_run_*/run_report.json`
- `${RUN_DIR}/local_run_*/actions_raw.npy`
- `${RUN_DIR}/local_run_*/actions_postprocessed.npy`

## 7. 第四步：SO101 真机 GROOT TRT 推理

真机入口不是 ACT 风格的单 `--trt-path`，而是：

- `--run-dir`
- `--engine-dir`
- `--policy-path`

因为当前真实运行时仍然是：

- 7 个 TRT engines
- PyTorch glue
- checkpoint processors

Python 入口：

- [run_groot_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py)

shell 入口：

- [one_click_run_groot_trt_so101.sh](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh)

建议先只做非硬件 TRT preflight：

```bash
cd /data/tfj/lerobot_tfj

conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --preflight-only \
  --skip-camera-preflight
```

当前这条 preflight 已经在本仓库成功跑通，能够加载：

- GROOT policy checkpoint
- 7 个 TensorRT engines
- `TrtGrootPolicyAdapter`

如果你要上真机，先把下面这几个参数改成你的现场值：

- `--robot-port`
- `--robot-calibration-dir`
- `--top-cam-index`
- `--wrist-cam-index`
- `--task`

最小真机命令示例：

```bash
cd /data/tfj/lerobot_tfj

conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin"
```

如果你更喜欢 shell 包装入口：

```bash
cd /data/tfj/lerobot_tfj

bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin"
```

建议首次上机先加这些安全参数之一：

- `--preflight-only`
- `--dry-run`
- `--run-time-s 10`

不要把这个入口误解为“纯 engine 单独跑”。

它仍然依赖：

- `pretrained_model/`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- GROOT PyTorch glue

## 8. 如果你想复用已有导出

如果 `RUN_DIR` 里已经有导出和 engine，可以跳过重复构建：

```bash
SKIP_EXISTING=1 bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

```bash
SKIP_EXISTING=1 bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

## 9. 运行中看到这些现象，不要先慌

`strict=True` 探测时当前 checkpoint 会出现一个 missing key：

- `_groot_model.backbone.eagle_model.language_model.model.embed_tokens.weight`

这条信息目前是已知现象。当前导出/构建链路实际使用的是 repo 默认的 `strict=False` 加载。

另外，TensorRT logger 的 warning 以及 image processor fast/slow fallback 也可能出现，它们不等于导出失败。

## 10. 明确不要做的事

- 不要默认走 `run_groot_infer.py --backend=tensorrt`
- 不要把 engine 文件当成完整运行时
- 不要把不存在的 mock compare 当成当前已完成能力

## 11. 你只要记住的最短命令集

```bash
cd /data/tfj/lerobot_tfj

export POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000
export RUN_DIR=/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_$(date +%Y%m%d_%H%M%S)
export CONDA_ENV=lerobot_flex

bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
NUM_STEPS=1 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh "${POLICY_PATH}" "${RUN_DIR}"
```

如果这三条都过了，说明：

- 7 个 ONNX 子图导出通过
- 7 个 TensorRT engine 构建通过
- Torch vs ONNX 数值测试通过
- Torch vs TRT 数值测试通过
- 当前本地 hybrid TRT runtime 可以实际执行
