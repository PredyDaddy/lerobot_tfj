# PI0.5 TensorRT 导出、验证与上机说明

本文档记录以下内容：

1. 最常用的导出、验证、上机指令
2. 当前已经验证通过的 TensorRT 工件目录
3. 这次为跑通 `pi_model` 做了哪些代码改动
4. 如何判断每个阶段是否真正成功
5. 这次实际踩过的坑，以及为什么之前会失败

本文档对应的模型路径：

- `policy.path=/data/tfj/lerobot_tfj/pi_model/pretrained_model`

本文档对应的当前已验证通过的 TRT 工件目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

本文档对应的当前最新 FP16 诊断 run：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`

这套 FP16 run 的当前状态是：

- `stage2_export_onnx = pass`
- `stage3_verify_onnx = pass`
- `stage4_build_engines = pass`
- `stage5_verify_trt = fail`
- Stage 4 当前使用的保守 escape hatch：
  - `force_fp32_layer_types = REDUCE ELEMENTWISE UNARY`

因此它现在只能作为“离线诊断 benchmark 工件”，不能直接当作默认上机工件。

## 1. 指令总览

### 1.1 进入环境

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt
```

### 1.2 从 `pi_model` 重新导出并做全量一致性验证

建议每次新跑都生成一个新的 `RUN_DIR`，不要覆盖旧结果。

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

RUN_DIR=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_$(date +%Y%m%d_%H%M%S)

python scripts/step2_export_onnx.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --run-dir "$RUN_DIR"

python scripts/step3_verify_onnx.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --run-dir "$RUN_DIR"

python scripts/step4_build_engines.py \
  --run-dir "$RUN_DIR" \
  --precision fp32 \
  --device cuda:0

python scripts/step5_verify_trt.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --run-dir "$RUN_DIR" \
  --device cuda:0
```

如果这四步都走通，则会在该 `RUN_DIR` 下得到：

- `stage2_export_onnx.json`
- `stage3_verify_onnx.json`
- `stage4_build_engines.json`
- `stage5_verify_trt.json`
- `pi_trt_metadata.json`
- `artifacts/onnx/*.onnx`
- `artifacts/engines/*.engine`

### 1.3 当前已经验证通过的已知可用 TRT 工件

这套目录已经完整通过 `Stage 2 -> Stage 5`：

```bash
TRT_RUN=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839
```

里面的关键产物是：

```bash
$TRT_RUN/artifacts/onnx/pi_shared_vision_encoder.onnx
$TRT_RUN/artifacts/onnx/pi_shared_prefix_cache.onnx
$TRT_RUN/artifacts/onnx/pi05_denoise_step.onnx

$TRT_RUN/artifacts/engines/pi_shared_vision_encoder.engine
$TRT_RUN/artifacts/engines/pi_shared_prefix_cache.engine
$TRT_RUN/artifacts/engines/pi05_denoise_step.engine
```

### 1.4 真机启动前先做预检

下面这条命令只做环境校验、相机预检、TRT 预热预检，不真正开始持续控制机械臂。

这条命令是按当前 `safetensors` 上机配置等价转换出来的 TRT 版本，并显式保留了：

- 串口 `/dev/ttyACM0`
- 顶部相机 `4`
- 手腕相机 `6`
- 分辨率 `640x480`
- 帧率 `30`
- `MJPG` fourcc
- 任务 `"Clean the desk"`

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/run_pi05_trt_infer_so101.py \
  --robot-id so101_follower \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --top-cam-fourcc MJPG \
  --wrist-cam-fourcc MJPG \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --task "Clean the desk" \
  --preflight-only
```

### 1.5 真机正式运行

建议第一次不要直接无限时运行，先跑 30 到 60 秒确认动作平稳。

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/run_pi05_trt_infer_so101.py \
  --robot-id so101_follower \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --top-cam-fourcc MJPG \
  --wrist-cam-fourcc MJPG \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --task "Clean the desk" \
  --run-time-s 60
```

确认稳定后，如果要持续运行，可以去掉 `--run-time-s 60`：

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/run_pi05_trt_infer_so101.py \
  --robot-id so101_follower \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --top-cam-fourcc MJPG \
  --wrist-cam-fourcc MJPG \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --task "Clean the desk"
```

### 1.6 仅做启动链路检查但不访问硬件

如果只是想检查工件路径、metadata、policy 加载、参数解析是否正常，可以用：

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/run_pi05_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --dry-run
```

### 1.7 统一离线 benchmark 指令

现在有两类 benchmark 要分开看：

1. `pipeline_chunk` benchmark
2. `1000-step pure inference select_action` benchmark

推荐先跑安全基线：

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/benchmark_pi_inference.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_$(date +%Y%m%d_%H%M%S)

python scripts/benchmark_pi_select_action.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839 \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp32_recheck_$(date +%Y%m%d_%H%M%S)
```

这次最新安全基线结果目录是：

```bash
/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_20260314_174221
/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp32_recheck_20260314_174221
```

如果你只是想诊断当前 `unsafe fp16` 工件的速度，而不是把它当作可上机结论，可以显式加 `--allow-unsafe-trt-artifacts`：

```bash
conda activate lerobot
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt

python scripts/benchmark_pi_inference.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759 \
  --allow-unsafe-trt-artifacts \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_unsafe_$(date +%Y%m%d_%H%M%S)

python scripts/benchmark_pi_select_action.py \
  --policy-path /data/tfj/lerobot_tfj/pi_model \
  --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759 \
  --trt-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759 \
  --allow-unsafe-trt-artifacts \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_unsafe_$(date +%Y%m%d_%H%M%S)
```

这次最新 `unsafe fp16` 诊断结果目录是：

```bash
/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_unsafe_20260314_174221
/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_unsafe_20260314_174221
```

更详细的解读见：

- [INFERENCE_COMPARISON.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/INFERENCE_COMPARISON.md)

## 2. 与 `lerobot-record` 命令的对应关系

你之前的 `safetensors` 上机命令是：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30,fourcc: "MJPG"}}" \
  --robot.id=so101_follower \
  --display_data=false \
  --dataset.repo_id=local/eval_pi05_so101_debug \
  --dataset.single_task="Clean the desk" \
  --policy.path=/data/tfj/lerobot_tfj/pi_model/pretrained_model
```

对应到 TRT 启动脚本时，参数映射关系是：

- `--robot.port` 对应 `--robot-port`
- `--robot.id` 对应 `--robot-id`
- `top camera index=4` 对应 `--top-cam-index 4`
- `wrist camera index=6` 对应 `--wrist-cam-index 6`
- `width/height/fps` 对应 `--camera-width/--camera-height/--camera-fps`
- `fourcc=MJPG` 对应 `--top-cam-fourcc MJPG --wrist-cam-fourcc MJPG`
- `--dataset.single_task` 对应 `--task`
- `--policy.path` 对应 `--policy-path`

注意：

- `run_pi05_trt_infer_so101.py` 是真机推理脚本，不是录制脚本。
- 它不会自动生成 `dataset.repo_id` 这一类录制数据集参数。
- 如果要录制评估数据，需要另外串联录制工具，不要把这个 TRT 推理脚本和 `lerobot-record` 混为一条命令。

## 3. 这次最终状态

当前成功的验证目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

最终 metadata 状态：

- `stage2_export_onnx = pass`
- `stage3_verify_onnx = pass`
- `stage4_build_engines = pass`
- `stage5_verify_trt = pass`
- `last_completed_stage = stage5_verify_trt`

对应文件：

- `docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json`

最终 Stage 5 关键结果：

- `vision_encoder = pass`
- `prefix_cache = pass`
- `denoise_step = pass`
- `pipeline = pass`

对应文件：

- `docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.json`
- `docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.md`

关键数值结果摘要：

- `vision_encoder`
  - `torch_vs_trt max_abs_diff = 3.967e-04`
  - `torch_vs_trt mean_abs_diff = 3.087e-06`
  - `torch_vs_trt min_cosine_similarity = 0.99999982`
- `prefix_cache`
  - `torch_vs_trt max_abs_diff = 7.801e-04`
  - `torch_vs_trt mean_abs_diff = 1.658e-05`
  - `torch_vs_trt min_cosine_similarity = 0.99999958`
- `denoise_step`
  - `torch_vs_onnx max_abs_diff = 1.550e-06`
  - `torch_vs_trt max_abs_diff = 1.788e-06`
  - `onnx_vs_trt max_abs_diff = 1.132e-06`
- `pipeline`
  - `torch_vs_onnx max_abs_diff = 7.391e-06`
  - `torch_vs_trt max_abs_diff = 5.305e-06`
  - `onnx_vs_trt max_abs_diff = 7.391e-06`

### 3.1 当前可上机默认工件

当前唯一满足“默认上机”条件的仍然是这套 `FP32 verified run`：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

原因很简单：

- 它已经完整通过 `Stage 2 -> Stage 5`
- launcher 默认的 provenance 检查会接受它
- 它对应的安全 benchmark 也已经重新复测

### 3.2 当前 FP16 重建的真实状态

这次最新的 FP16 run 是：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`

它的真实状态不是“完全成功”，而是：

- `Stage 2 = pass`
- `Stage 3 = pass`
- `Stage 4 = pass`
- `Stage 5 = fail`

而且当前通过 Stage 4 的版本不是“干净纯 FP16”，而是带保守 escape hatch 的版本：

- `force_fp32_layer_types = REDUCE ELEMENTWISE UNARY`

这次 Stage 5 失败的关键事实是：

- `vision_encoder` 漂移已经显著缩小，但仍高于当前严格阈值
- `denoise_step` 和 `pipeline` 已经接近可接受区间
- 真正还过不去的是 `prefix_cache` 的 raw KV 对比

所以当前结论必须写死：

- 这套 FP16 工件可以做离线诊断 benchmark
- 这套 FP16 工件现在不能当作默认上机工件

## 4. 这次改了什么

这次不是只改了一个点，而是把导出、验证、构建、上机四段都收紧了。

### 4.1 导出与对比基础设施

- `scripts/pi_compare_common.py`
  - 加了 `GemmaRMSNorm` 的 `repr` 兼容 shim，修复 `torch.onnx.export` 在导出前 `repr(model)` 时崩溃的问题。
  - 本地 tokenizer 路径发现逻辑统一走 `common.discover_local_tokenizer_path(...)`。

- `scripts/export_wrappers.py`
  - 在 denoise 导出路径上，针对 CPU 导出场景临时强制时间嵌入的正余弦分支使用 `float32`，避免生成 `float64` 时间嵌入图。
  - 这是这次跑通 `denoise_step.onnx` 的关键修复。

- `scripts/export_subgraphs.py`
  - Stage 2 强化了 `denoise_step.timestep` 的契约检查：
    - `timestep` 必须在 ONNX graph inputs 中
    - `timestep` 必须作为 live session input 被真正消费
  - 即时导出对比里，`denoise_step` 的 ONNX 执行从纯 CPU 限制改成了更合理的 provider fallback 逻辑。
  - 保留了更严格的 Stage 2 acceptance gate。

### 4.2 阶段 gate 和 metadata 语义

- `scripts/step2_export_onnx.py`
  - Stage 2 只有在 acceptance 真正通过时才写 `last_completed_stage`。
  - 失败时返回非零。

- `scripts/step3_verify_onnx.py`
  - Stage 3 的 acceptance 明确区分：
    - `local_export_fidelity_compare`
    - `chained_export_fidelity_compare`
    - `denoise_timestep_live_input`
  - 只有 Stage 3 acceptance 通过时才推进 metadata gate。

- `scripts/step4_build_engines.py`
  - 必须在 Stage 2 和 Stage 3 gate 都是 `pass` 时才允许构建 TRT engine。
  - 不再允许“上游没通过但继续 build”。

- `scripts/step5_verify_trt.py`
  - Torch 基线改成 export-fidelity 风格：
    - `policy.cpu().float()`
    - `use_autocast=False`
  - Stage 5 退出码与 `overall_status` 对齐。
  - 只有 Stage 5 真 `pass` 时才写 `last_completed_stage=stage5_verify_trt`。

### 4.3 TRT 运行时与真机入口

- `scripts/build_pi_trt_engine.py`
  - 构建时显式在所选 CUDA device 下进行。
  - 精度约束优先使用 `OBEY_PRECISION_CONSTRAINTS`。

- `scripts/trt_runtime.py`
  - 强化了 TRT 输入检查：
    - 拒绝多余输入
    - 拒绝静态 shape 不匹配
    - 拒绝超 profile bounds
    - 拒绝未解析 shape

- `scripts/trt_pi_adapter.py`
  - `denoise_step` 现在把 `timestep` 当成硬性契约。
  - 预测循环里无条件喂 `timestep`，不再保留“可选传参”的松口。

- `scripts/run_pi05_trt_infer_so101.py`
  - 加强了工件 provenance 检查，默认 fail-close。
  - 旧的 warning artifact 不会被默认拿去上真机。
  - 启动摘要现在会直接打印：
    - resolved `variant`
    - requested `precision`
    - `stage4_report_path`
    - `stage5_report_path`
  - 增加了：
    - `--dry-run`
    - `--preflight-only`
    - camera preflight
    - TRT preflight
    - 运动平滑与增量保护参数
  - 另外补充了：
    - `--top-cam-fourcc`
    - `--wrist-cam-fourcc`
  - 这样现在可以无损复现你原来 `MJPG` 的相机设置。

## 5. 如何判断“真的成功”

不要只看脚本有没有退出，也不要只看某个阶段有没有产出文件。

### 5.1 Stage 2 成功判据

必须同时满足：

- `stage2_export_onnx.json` 中 `overall_status = pass`
- `stage2_acceptance.status = pass`
- `pi_trt_metadata.json` 中 `validation_gates.stage2_export_onnx.status = pass`

### 5.2 Stage 3 成功判据

这里最容易误判。

本项目里 `stage3_verify_onnx.json` 的 `overall_status` 可能是 `warn`，但只要以下三个 acceptance 都通过，Stage 3 gate 仍然算成功：

- `local_export_fidelity_compare = pass`
- `chained_export_fidelity_compare = pass`
- `denoise_timestep_live_input = pass`

真正看 gate 时，以 `pi_trt_metadata.json` 为准：

- `validation_gates.stage3_verify_onnx.status = pass`

当前成功 run 就是这种情况：

- `stage3_verify_onnx.md` 里 `overall_status = warn`
- 但 `stage3_acceptance = pass`
- metadata 里的 Stage 3 gate 也是 `pass`

这是预期行为，不是失败。

### 5.3 Stage 4 成功判据

必须同时满足：

- `stage4_build_engines.json` 中 `overall_status = pass`
- `all_succeeded = true`
- 三个 engine 文件实际存在

### 5.4 Stage 5 成功判据

必须同时满足：

- `stage5_verify_trt.json` 中 `overall_status = pass`
- `vision_encoder = pass`
- `prefix_cache = pass`
- `denoise_step = pass`
- `pipeline = pass`
- metadata 里 `validation_gates.stage5_verify_trt.status = pass`

## 6. 这次踩过的坑

这一节是这次真正踩过、而且已经花时间验证过的坑，不是泛泛总结。

### 6.1 `torch.onnx.export` 在导出前就崩了

现象：

- 导出时直接报：
  - `AttributeError: 'GemmaRMSNorm' object has no attribute 'weight'`

根因：

- 新的导出路径会在内部 `repr(model)`。
- 当前环境里的 `transformers.models.gemma.modeling_gemma.GemmaRMSNorm.extra_repr()` 对 adaptive 分支仍然直接访问 `self.weight`。
- 但 adaptive RMSNorm 分支只有 `dense`，没有 `weight`。

处理：

- 在 `scripts/pi_compare_common.py` 里加了 `GemmaRMSNorm` 的 `repr` shim。

### 6.2 `denoise_step.onnx` 导出来了，但 Stage 2/3 一直失败

现象：

- ONNX 文件存在，但：
  - `denoise_step` 执行失败
  - `timestep` live input 检查失败
  - Stage 2 / Stage 3 gate 失败

根因分两层：

- 第一层：Stage 2 旧逻辑把 `denoise_step` 硬绑 CPU ORT，而当前环境 CPU provider 对某些算子缺 kernel。
- 第二层：更本质的问题是 CPU 导出路径的时间嵌入分支走了 `float64`，把 denoise 图污染成了带双精度时间分支的 ONNX。

处理：

- `export_wrappers.py` 中把 denoise 导出路径的 CPU sin-cos 时间嵌入固定成 `float32`
- `export_subgraphs.py` 和 `step3_verify_onnx.py` 中把 `denoise_step` 的 ONNX provider 策略改成更合理的 fallback 逻辑

### 6.3 `denoise_step.onnx` 可以用，但 TRT build 失败

现象：

- Stage 4 构建时只剩 `denoise_step` 失败
- TRT 报：
  - `Failed to import initialzer`
  - 外部权重尺寸异常

根因：

- 当时导出的 denoise 图里时间嵌入分支仍带 `float64` 常量和类型信息。
- TRT parser 对这类图非常敏感。

处理：

- 最终不是靠“硬改成品 ONNX”解决，而是回到导出源头，把时间嵌入导出成 `float32`，重新生成干净的 `denoise_step.onnx`。

### 6.4 Stage 5 一开始会给人一种“TRT 不一致”的错觉

现象：

- `vision_encoder`、`prefix_cache` 已经 pass
- `torch_vs_trt` 数值也很好
- 但 Stage 5 整体仍然 error

根因：

- 不是 TRT engine 本身有问题，而是 Stage 5 当时还用旧的 export-fidelity ONNX 配置去加载一个对 CPU ORT 不友好的 denoise 图。
- 也就是说，错在基线执行策略，不在 TRT 结果。

处理：

- 把 Stage 5 的 ONNX profile 和真实导出/验证逻辑对齐。
- 最终重新导出后，Stage 5 才真正全 pass。

### 6.5 上机脚本原来没有 `MJPG fourcc`

现象：

- 你原来的 `lerobot-record` 命令显式使用了 `MJPG`
- TRT 真机脚本只支持 camera index / width / height / fps，不支持 fourcc

风险：

- 有些相机如果不设 `MJPG`，帧率、分辨率或打开方式会和 `lerobot-record` 不一致

处理：

- 在 `scripts/run_pi05_trt_infer_so101.py` 里新增：
  - `--top-cam-fourcc`
  - `--wrist-cam-fourcc`
- 同时 camera preflight 也会按这个 fourcc 开相机

### 6.6 `FP16 build pass` 不等于 `FP16 可以上机`

现象：

- 这次 `FP16` run 可以做到 `Stage 4 = pass`
- 而且在显式允许 `unsafe` 的前提下，离线 benchmark 数字非常亮眼
- 但 `Stage 5` 仍然 fail

根因：

- `TensorRT FP16 build path` 和 “当前严格的 export-boundary correctness gate” 不是一回事
- clean `fp16-enabled` build 在当前工程里会出现非常大的数值漂移
- 加上 `REDUCE/ELEMENTWISE/UNARY -> FP32` 的保守 escape hatch 以后，`denoise_step` 和 `pipeline` 已经大幅收敛，但 `prefix_cache` 的 raw KV 对比仍然过不了当前 gate

处理：

- benchmark 侧新增了 `--allow-unsafe-trt-artifacts`
- 默认仍然 fail-fast，不允许把这套工件伪装成“已验证通过”
- 只有在明确做诊断时，才允许单独测它的速度

这件事的工程含义非常重要：

- 现在不能说“FP16 已经成功替换 FP32”
- 只能说“当前这条 `unsafe fp16` 诊断路线展现出了很强的性能潜力，但还没有通过正确性 gate”

- `scripts/benchmark_pi_inference.py`
  - benchmark 输出现在会显式写出：
    - `variant`
    - requested `precision`
    - `metadata_path`
    - `stage4_report_path`
    - `stage5_report_path`
  - 默认会对 TRT 工件做 provenance fail-fast。
  - 只有显式传 `--allow-unsafe-trt-artifacts` 时，才允许对未通过 Stage 5 的工件做诊断性 benchmark。

- `scripts/benchmark_pi_select_action.py`
  - 新增了纯 `select_action()` benchmark 入口。
  - 不接机器人、不下发动作，只统计 1000-step 纯推理吞吐。
  - 报告里明确把 `PyTorch AMP` 标成 `CUDA BF16 autocast`，避免和 `Torch FP16` 混淆。

## 7. 最后建议

建议实际使用时遵循以下顺序：

1. 先用 `--dry-run` 检查路径和工件
2. 再用 `--preflight-only` 检查相机、TRT adapter、robot config
3. 再用 `--run-time-s 30` 或 `--run-time-s 60` 短时上机
4. 动作稳定后再去掉 `--run-time-s`

当前推荐直接使用这套已验证通过的工件目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

不要再回头使用之前失败或半成功的老 run 目录。
