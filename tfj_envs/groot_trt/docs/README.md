# GROOT TRT README

这份 README 是当前 `tfj_envs/groot_trt` 目录的总览手册。

目标不是讲理想集成，而是讲当前这个 checkout 里实际可用、可运行、可验证的脚本。

如果你现在最关心的是：

- GROOT engine 导出到了哪里
- 真机 SO101 上机脚本在哪里
- 端口号、相机号、标定目录怎么传

先读：

- [GROOT_TRT_SO101_README.md](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/docs/GROOT_TRT_SO101_README.md)

## 1. 先记住当前真实部署边界

当前 GROOT TensorRT 不是单体 engine。

当前真实运行时是：

- 7 个 TensorRT 子图
- PyTorch glue
- checkpoint processors
- LeRobot runtime

固定 7 个子图：

- backbone: `vit_fp16`
- backbone: `llm_fp16`
- action head: `vlln_vl_self_attention`
- action head: `state_encoder`
- action head: `action_encoder`
- action head: `DiT_fp16`
- action head: `action_decoder`

当前运行时硬依赖：

- `pretrained_model/`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

不要误解成下面这些错误说法：

- “engine 文件齐了就等于完整推理系统”
- “GROOT TRT 跟 ACT 一样是单 `--trt-path` 路径”
- “`run_groot_infer.py --backend=tensorrt` 在当前 checkout 里已经可用”

## 2. 当前目录里有什么

脚本目录：

- `tfj_envs/groot_trt/scripts/`

文档目录：

- `tfj_envs/groot_trt/docs/`

当前保留的脚本分三层：

- 用户直接运行的入口脚本
- Stage 脚本
- 底层实现脚本和辅助模块

## 3. 推荐你怎么用

如果你只想完成常规工作流，优先用这 4 个入口：

- `scripts/one_click_export_groot_trt.sh`
- `scripts/one_click_compare_groot_trt.sh`
- `scripts/one_click_run_groot_trt_local.sh`
- `scripts/one_click_run_groot_trt_so101.sh`

推荐工作流顺序：

1. `step1_safetensors_to_torch.py`
2. `step2_export_onnx.py`
3. `step3_verify_onnx.py`
4. `step4_build_engines.py`
5. `step5_verify_trt.py`
6. `run_groot_infer_trt_local.py`
7. `run_groot_trt_infer_so101.py`

one-click 包装脚本和 Stage 的对应关系：

- `one_click_export_groot_trt.sh` = Stage 1 + Stage 2 + Stage 4
- `one_click_compare_groot_trt.sh` = Stage 3 + Stage 5 + 汇总
- `one_click_run_groot_trt_local.sh` = 本地随机观测 smoke run
- `one_click_run_groot_trt_so101.sh` = 真机 SO101 TRT 推理入口

## 4. 环境与路径约定

仓库根目录：

```bash
cd /data/tfj/lerobot_tfj
```

默认 conda 环境：

```bash
export CONDA_ENV=lerobot_flex
```

推荐先设的变量：

```bash
export POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000
export RUN_DIR=/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_$(date +%Y%m%d_%H%M%S)
```

如果 `lerobot_flex` 里不能直接 `import tensorrt`，再设置：

```bash
export TENSORRT_PY_DIR=/your/tensorrt/python/path
```

如果不设置 `TMPDIR`，wrapper 会默认使用：

```bash
${RUN_DIR}/.tmp
```

最小环境检查：

```bash
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.cuda.is_available())"
```

```bash
conda run -n "${CONDA_ENV}" python -c "import tensorrt as trt; print(trt.__version__)"
```

## 5. 当前已验证过的关键路径

已验证 checkpoint：

- `/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000/pretrained_model`

已验证 compare run：

- `/data/tfj/lerobot_tfj/outputs/trt/groot_export_verify_20260311_152547`

已验证真机脚本 TRT preflight run：

- `/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210`

一组已经跑过的 compare 参考值：

- 1-view ONNX denoising cosine: `0.999998525`
- 1-view TRT denoising cosine: `0.999997003`
- 2-view ONNX denoising cosine: `0.999996353`
- 2-view TRT denoising cosine: `0.999990810`

这些是 sanity reference，不是死阈值。

## 6. 用户直接运行的入口脚本

### `scripts/one_click_export_groot_trt.sh`

作用：

- 从 safetensors 检查点出发
- 做 Stage 1 加载检查
- 导出 7 个 ONNX 子图
- 构建 7 个 TensorRT engines

什么时候用：

- 第一次导出 GROOT TRT 产物
- 想在新 `RUN_DIR` 里完整产出 ONNX 和 engine

输入：

- `POLICY_PATH`
- 可选 `RUN_DIR`

常用环境变量：

- `VIDEO_VIEWS`
- `SEQ_LEN`
- `MIN_SEQ_LEN`
- `OPT_SEQ_LEN`
- `MAX_SEQ_LEN`
- `MAX_BATCH`
- `VIT_OPT_BATCH`
- `OPT_BATCH`
- `WORKSPACE_GB`
- `SKIP_EXISTING`

最小命令：

```bash
bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

2-view 常用命令：

```bash
VIDEO_VIEWS=2 bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

主要产物：

- `${RUN_DIR}/stage1_safetensors_to_torch.json`
- `${RUN_DIR}/stage2_export_onnx.json`
- `${RUN_DIR}/stage4_build_engines.json`
- `${RUN_DIR}/gr00t_onnx/`
- `${RUN_DIR}/gr00t_engine_api_trt1013/`

### `scripts/one_click_compare_groot_trt.sh`

作用：

- 对已有 `RUN_DIR` 里的 ONNX 和 engine 做统一数值比较
- 比较链路是 `safetensors-loaded PyTorch vs ONNX vs TensorRT`

什么时候用：

- 导出完成后想确认数值是否健康
- 你已经有 `gr00t_onnx/` 和 `gr00t_engine_api_trt1013/`

输入：

- `POLICY_PATH`
- `RUN_DIR`

常用环境变量：

- `SEED`
- `SKIP_EXISTING`
- `MIN_LLM_FROM_VIT_COSINE`
- `MIN_DENOISING_COSINE`

最小命令：

```bash
bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

主要产物：

- `${RUN_DIR}/stage3_verify_onnx.json`
- `${RUN_DIR}/stage5_verify_trt.json`
- `${RUN_DIR}/compare_safetensor_onnx_trt.json`

### `scripts/one_click_run_groot_trt_local.sh`

作用：

- 不接机器人
- 用随机观测跑本地 hybrid TRT runtime
- 检查 `select_action()`、action queue、postprocess 是否能走通

什么时候用：

- engine 构建完后做本地 smoke test
- 想先确认 runtime 不会一上来在真机上炸

输入：

- `POLICY_PATH`
- `RUN_DIR`
- 可选 `OUT_DIR`

常用环境变量：

- `NUM_STEPS`
- `SEED`
- `TASK`
- `ROBOT_TYPE`
- `REFRESH_OBS_PER_STEP`

最小命令：

```bash
NUM_STEPS=1 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh "${POLICY_PATH}" "${RUN_DIR}"
```

主要产物：

- `${OUT_DIR}/run_report.json`
- `${OUT_DIR}/actions_raw.npy`
- `${OUT_DIR}/actions_postprocessed.npy`

### `scripts/one_click_run_groot_trt_so101.sh`

作用：

- 真机 SO101 TRT 推理 shell 入口
- 自动用 `conda run`
- 自动处理 `TMPDIR`
- 自动把 `policy-path / run-dir / engine-dir` 填给 Python 真机脚本

什么时候用：

- 你已经有 checkpoint 和 7 个 engine
- 你要在 SO101 上跑 GROOT TRT

输入：

- `POLICY_PATH`
- `RUN_DIR`
- `ENGINE_DIR`
- 其余参数原样透传给 `run_groot_trt_infer_so101.py`

先做 preflight：

```bash
bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --preflight-only \
  --skip-camera-preflight
```

真机短时运行：

```bash
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
  --task "grasp block in bin" \
  --run-time-s 10
```

## 7. Stage 脚本

### `scripts/step1_safetensors_to_torch.py`

作用：

- 解析 `policy-path`
- 找到真实的 `pretrained_model/`
- 加载 safetensors 到当前 checkout 的 PyTorch policy
- 校验 preprocessor 和 postprocessor 是否存在
- 记录 CUDA、TensorRT import、strict probe 等环境信息

什么时候用：

- 所有后续工作前的第一步
- 你想确认当前 checkpoint 是不是能被当前源码正确加载

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step1_safetensors_to_torch.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}"
```

严格探测命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step1_safetensors_to_torch.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}" \
  --strict
```

主要产物：

- `${RUN_DIR}/stage1_safetensors_to_torch.json`

### `scripts/step2_export_onnx.py`

作用：

- 调用 backbone 和 action head 的本地导出脚本
- 导出固定 7 个 ONNX 子图
- 校验 ONNX 文件合同是否符合预期

什么时候用：

- Stage 1 通过后导出 ONNX

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step2_export_onnx.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}" \
  --seq-len 568 \
  --video-views 2
```

主要产物：

- `${RUN_DIR}/stage2_export_onnx.json`
- `${RUN_DIR}/gr00t_onnx/backbone/vit_fp16.onnx`
- `${RUN_DIR}/gr00t_onnx/backbone/llm_fp16.onnx`
- `${RUN_DIR}/gr00t_onnx/action_head/vlln_vl_self_attention.onnx`
- `${RUN_DIR}/gr00t_onnx/action_head/state_encoder.onnx`
- `${RUN_DIR}/gr00t_onnx/action_head/action_encoder.onnx`
- `${RUN_DIR}/gr00t_onnx/action_head/DiT_fp16.onnx`
- `${RUN_DIR}/gr00t_onnx/action_head/action_decoder.onnx`

### `scripts/step3_verify_onnx.py`

作用：

- 对 1-view 和 2-view synthetic batch 运行 Torch vs ONNX compare
- 汇总 ONNX 数值一致性

什么时候用：

- Stage 2 完成后
- build engine 前先确认 ONNX 边界没有明显问题

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step3_verify_onnx.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}"
```

可选阈值命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step3_verify_onnx.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}" \
  --min-denoising-cosine 0.9999
```

主要产物：

- `${RUN_DIR}/stage3_verify_onnx.json`
- `${RUN_DIR}/logs/compare_onnx_1view.log`
- `${RUN_DIR}/logs/compare_onnx_2view.log`

### `scripts/step4_build_engines.py`

作用：

- 从 ONNX 构建 7 个 TensorRT engines
- 根据 `video-views` 和 seq len 选择 profile
- 记录 build report

什么时候用：

- Stage 3 没问题以后

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step4_build_engines.py \
  --run-dir "${RUN_DIR}" \
  --video-views 2 \
  --min-seq-len 80 \
  --opt-seq-len 568 \
  --max-seq-len 600
```

主要产物：

- `${RUN_DIR}/stage4_build_engines.json`
- `${RUN_DIR}/gr00t_engine_api_trt1013/`

### `scripts/step5_verify_trt.py`

作用：

- 对 1-view 和 2-view synthetic batch 运行 Torch vs TRT compare
- 汇总 TensorRT 数值一致性

什么时候用：

- engine 构建完成之后
- 真机前的最后一道数值检查

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/step5_verify_trt.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}"
```

主要产物：

- `${RUN_DIR}/stage5_verify_trt.json`
- `${RUN_DIR}/logs/compare_trt_1view.log`
- `${RUN_DIR}/logs/compare_trt_2view.log`

## 8. 低层可直接运行脚本

这些脚本是 Stage 脚本内部真正调用的实现层。你可以单独运行，但一般只在调试某一层时才需要。

### `scripts/export_backbone_onnx_local.py`

作用：

- 导出 backbone 的 2 个 ONNX 子图
- `vit_fp16.onnx`
- `llm_fp16.onnx`

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/export_backbone_onnx_local.py \
  --policy-path "${POLICY_PATH}" \
  --onnx-out-dir "${RUN_DIR}/gr00t_onnx" \
  --seq-len 568 \
  --video-views 2
```

### `scripts/export_action_head_onnx_local.py`

作用：

- 导出 action head 的 5 个 ONNX 子图

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/export_action_head_onnx_local.py \
  --policy-path "${POLICY_PATH}" \
  --onnx-out-dir "${RUN_DIR}/gr00t_onnx" \
  --seq-len 568 \
  --state-horizon 1
```

### `scripts/build_groot_engines_local.py`

作用：

- 直接从 `gr00t_onnx/` 构建 7 个 TensorRT engines
- 是 Stage 4 的底层 builder

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/build_groot_engines_local.py \
  --onnx-dir "${RUN_DIR}/gr00t_onnx" \
  --engine-out-dir "${RUN_DIR}/gr00t_engine_api_trt1013" \
  --max-batch 2 \
  --vit-opt-batch 2 \
  --opt-batch 1 \
  --min-seq-len 80 \
  --opt-seq-len 568 \
  --max-seq-len 600
```

### `scripts/compare_torch_onnx_local.py`

作用：

- 对某一个固定 `(seq_len, video_views)` 组合
- 直接比较 Torch pipeline 和 ONNX pipeline

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/compare_torch_onnx_local.py \
  --policy-path "${POLICY_PATH}" \
  --onnx-dir "${RUN_DIR}/gr00t_onnx" \
  --seq-len 568 \
  --video-views 2 \
  --json-out "${RUN_DIR}/compare_onnx_2view.json"
```

### `scripts/compare_torch_trt_local.py`

作用：

- 对某一个固定 `(seq_len, video_views)` 组合
- 直接比较 Torch pipeline 和 TRT pipeline

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/compare_torch_trt_local.py \
  --policy-path "${POLICY_PATH}" \
  --engine-dir "${RUN_DIR}/gr00t_engine_api_trt1013" \
  --seq-len 568 \
  --video-views 2 \
  --json-out "${RUN_DIR}/compare_trt_2view.json"
```

### `scripts/compare_safetensor_onnx_trt.py`

作用：

- 统一驱动 Stage 3 和 Stage 5
- 对一个已有 `RUN_DIR` 输出总 compare 报告

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/compare_safetensor_onnx_trt.py \
  --policy-path "${POLICY_PATH}" \
  --run-dir "${RUN_DIR}"
```

输出：

- `${RUN_DIR}/compare_safetensor_onnx_trt.json`

### `scripts/run_groot_infer_trt_local.py`

作用：

- 本地 TRT runtime smoke test
- 用随机观测驱动 preprocessor、TRT adapter、postprocessor
- 记录 action queue 行为

最小命令：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_infer_trt_local.py \
  --policy-path "${POLICY_PATH}" \
  --engine-dir "${RUN_DIR}/gr00t_engine_api_trt1013" \
  --out-dir "${RUN_DIR}/local_run_debug" \
  --num-steps 4 \
  --task "Perform the task."
```

### `scripts/run_groot_trt_infer_so101.py`

作用：

- 真机 SO101 GROOT TRT 推理主脚本
- 接机器人
- 接 top / wrist camera
- 用 `predict_action()` 和 GROOT TRT adapter 做控制回路

常用参数：

- `--robot-id`
- `--robot-port`
- `--robot-calibration-dir`
- `--top-cam-index`
- `--wrist-cam-index`
- `--camera-width`
- `--camera-height`
- `--camera-fps`
- `--policy-path`
- `--run-dir`
- `--engine-dir`
- `--trt-device`
- `--task`
- `--run-time-s`
- `--skip-camera-preflight`
- `--skip-trt-preflight`
- `--preflight-only`
- `--dry-run`

先只做 dry run：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --dry-run
```

先只做 TRT preflight：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --preflight-only \
  --skip-camera-preflight
```

短时真机运行：

```bash
conda run -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --task "grasp block in bin" \
  --run-time-s 10
```

## 9. 只给上层脚本 import 的辅助模块

这些脚本主要是库模块，不建议你手动直接运行。

### `scripts/common.py`

作用：

- 提供通用路径解析
- 统一 `pretrained_model/` 解析
- 统一 `TMPDIR` / `TENSORRT_PY_DIR` 处理
- 统一 JSON report 写出
- 统一调用 conda 子进程

什么时候看它：

- 你在改路径逻辑
- 你在排查 `RUN_DIR`、`TMPDIR`、`TENSORRT_PY_DIR`

### `scripts/one_click_common.sh`

作用：

- shell wrapper 的公共逻辑
- 统一 conda 调用
- 统一 `TMPDIR` fallback
- 统一 1-view / 2-view profile 默认值

什么时候看它：

- 你在改 shell wrapper 的默认环境变量

### `scripts/groot_compare_common_local.py`

作用：

- compare 相关共用逻辑
- synthetic 输入构造
- Torch pipeline runner
- cosine / diff 统计

什么时候看它：

- 你在排查 compare 数值差异
- 你想理解 compare 到底比较了哪些张量

### `scripts/groot_trt_adapter_local.py`

作用：

- GROOT 的 TRT adapter
- 把 7 个 engines 串成完整的 GROOT runtime
- 处理 ViT 后处理、LLM embed 组装、denoising loop、action queue

什么时候看它：

- 你在排查 runtime 行为
- 你在排查 `select_action()` 的 chunk、queue、denoising
- 你在排查 engine 虽然能跑但最终 action 不对

### `scripts/trt_runtime_local.py`

作用：

- TensorRT Python runtime 的最薄封装
- 管理 engine 载入、binding、CUDA tensor 输入输出

什么时候看它：

- 你在排查 TensorRT engine 执行层
- 你在排查 dtype、contiguous、shape、binding 问题

## 10. 常见工作流示例

### 示例 A：从 checkpoint 开始完整导出 + compare

```bash
cd /data/tfj/lerobot_tfj

export POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000
export RUN_DIR=/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_$(date +%Y%m%d_%H%M%S)
export CONDA_ENV=lerobot_flex

VIDEO_VIEWS=2 bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
```

### 示例 B：只做本地 smoke run

```bash
cd /data/tfj/lerobot_tfj

NUM_STEPS=4 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210
```

### 示例 C：真机前只做 TRT preflight

```bash
cd /data/tfj/lerobot_tfj

bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --preflight-only \
  --skip-camera-preflight
```

## 11. 当前已知限制

- 当前 README 讲的是 `tfj_envs/groot_trt` 这套本地 source-of-truth，不依赖 `my_devs`
- 当前 direct runtime 仍然要求 7 个 engine 全齐
- 当前 LLM 边界是 `inputs_embeds + attention_mask`，不是 token ids
- 当前 ViT 的 batch 维在这个 flow 里代表 camera views
- 当前没有重新暴露 Stage 6 mock compare 作为推荐入口
- 不要默认把 `run_groot_infer.py --backend=tensorrt` 当成可用路径

## 12. 出问题时先查什么

先查这几个文件：

- `stage1_safetensors_to_torch.json`
- `stage2_export_onnx.json`
- `stage3_verify_onnx.json`
- `stage4_build_engines.json`
- `stage5_verify_trt.json`
- `compare_safetensor_onnx_trt.json`
- `logs/*.log`

最常见的几个问题方向：

- checkpoint 不是实际的 `pretrained_model/`
- `tensorrt` 没有在当前 conda 环境里可导入
- `TMPDIR` 不可写
- engine profile 和运行时 seq len / view count 不匹配
- 7 个 engine 没有齐
- 真机参数里相机 index、串口、标定目录填错

## 13. 如果你只想记最短命令

```bash
cd /data/tfj/lerobot_tfj

export POLICY_PATH=/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000
export RUN_DIR=/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_$(date +%Y%m%d_%H%M%S)
export CONDA_ENV=lerobot_flex

VIDEO_VIEWS=2 bash tfj_envs/groot_trt/scripts/one_click_export_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
bash tfj_envs/groot_trt/scripts/one_click_compare_groot_trt.sh "${POLICY_PATH}" "${RUN_DIR}"
NUM_STEPS=1 bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_local.sh "${POLICY_PATH}" "${RUN_DIR}"
```

如果你要上 SO101，先跑：

```bash
bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --preflight-only \
  --skip-camera-preflight
```
