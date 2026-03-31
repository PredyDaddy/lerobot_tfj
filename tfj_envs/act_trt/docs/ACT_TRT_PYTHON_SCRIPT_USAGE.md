# ACT TRT Python 脚本怎么用

这篇文档专门给第一次接触这套脚本的人看。

如果你只想知道：

- 这个脚本是干什么的
- 运行时要输入什么命令
- 参数分别填什么
- 跑完以后看哪里算成功

那就直接看这篇，不用先读别的文档。

## 1. 先记住两个最常用脚本

在 `tfj_envs/act_trt/scripts/` 下面，最常用的是这两个：

1. `export_act_checkpoint_engine.py`
   作用：把一个 ACT checkpoint 导出成 `ONNX + TensorRT engine`

2. `compare_act_safetensors_trt.py`
   作用：检查导出的 TensorRT engine 和原始 checkpoint 的输出是不是一致

你可以把它们理解成：

- 第一个脚本负责“导出”
- 第二个脚本负责“验货”

## 2. 运行前要用哪个环境

推荐这样分：

- `lerobot_flex`
  用来导出 ONNX、构建 TensorRT
- `lerobot`
  用来做对比验证、真实机器人运行

为什么这样分：

- 导出 ONNX 需要 `onnx`
- 构建 TensorRT 需要 `tensorrt`
- 你现在这台机器上，导出通常放在 `lerobot_flex` 更稳

## 3. 导出脚本怎么用

脚本路径：

- [export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)

### 3.1 这个脚本是干什么的

它会帮你自动做下面几件事：

1. 从 checkpoint 恢复 PyTorch 模型
2. 导出 `act_single.onnx`
3. 写 `export_metadata.json`
4. 构建 TensorRT engine
5. 生成验证报告

也就是说，你平时导出 ACT TRT，优先跑这一个就够了。

### 3.2 最常用命令

下面这条命令就是一个可以直接照抄的例子：

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0
```

### 3.3 这条命令里的参数是什么意思

- `--checkpoint`
  你要导出的模型目录。
  这里必须传 `pretrained_model` 这个目录，不是只传 `016000`。

- `--precision fp32`
  表示导出 FP32 的 TensorRT engine。
  如果你没有特殊需求，先固定用 `fp32`。

- `--device cpu`
  表示 ONNX 导出阶段在 CPU 上做。
  这是当前最稳的默认方式。

- `--trt-device cuda:0`
  表示 TensorRT 构建和检查时使用第 0 张 GPU。

### 3.4 跑完以后会得到什么

默认会输出到：

- `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/<任务名>/<checkpoint步数>/`

对于上面的例子，会输出到：

- `/data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/`

你会看到这些文件：

- `act_single.onnx`
- `export_metadata.json`
- `act_single_fp32.plan`
- `trt_build_summary_fp32.json`
- `consistency_report_act_single_fp32.json`

### 3.5 怎么判断导出成功了

终端里通常会打印这些东西：

```text
[ARTIFACT] onnx=...
[ARTIFACT] metadata=...
[ARTIFACT] engine=...
[ARTIFACT] build_report=...
[ARTIFACT] verify_report=...
```

只要没有报错，并且这些文件都生成出来，一般就说明导出已经完成了。

更严格一点的话，再打开这个文件看：

- `consistency_report_act_single_fp32.json`

重点看里面：

- `passed = true`

如果这里是 `true`，说明 `Torch / ONNX / TensorRT` 三方数值是一致的。

## 4. 对比脚本怎么用

脚本路径：

- [compare_act_safetensors_trt.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py)

### 4.1 这个脚本是干什么的

它会把：

- checkpoint 里的 safetensors 模型输出

和

- 你导出来的 TensorRT engine 输出

做逐元素对比，然后告诉你两边差多少。

你可以把它理解成：

- PyTorch 原模型是“标准答案”
- TensorRT engine 是“压缩后的部署版本”
- 这个脚本就是看它们两边算出来的结果有没有跑偏

### 4.2 最常用命令

下面这条命令也是可以直接照抄的例子：

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --engine /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --metadata /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json \
  --trt-device cuda:0 \
  --threshold-max-abs-diff 1e-4 \
  --report-json /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_SAFETENSORS_VS_TRT.json
```

### 4.3 这些参数是什么意思

- `--checkpoint`
  原始 ACT checkpoint 路径。
  这个路径决定“标准答案”是哪一个模型。

- `--engine`
  你要拿来对比的 TensorRT engine 文件路径。
  这里传 `.plan` 或 `.engine` 都可以，本质上都是 TensorRT engine。

- `--metadata`
  导出 ONNX 时生成的 `export_metadata.json`。
  它的作用不是保存结果，而是告诉脚本：
  这份 engine 的输入输出名字是什么、相机顺序是什么。

- `--trt-device cuda:0`
  表示 TensorRT 推理跑在第 0 张 GPU 上。

- `--threshold-max-abs-diff 1e-4`
  表示允许的最大绝对误差阈值。
  如果 TensorRT 和 PyTorch 的最大误差没有超过这个值，就算通过。

- `--report-json`
  表示把这次对比结果保存到哪个 JSON 文件里。
  这个文件是“结果报告”，方便你以后回看。

### 4.4 跑完以后看什么

跑完后终端会打印一段 summary，例如：

```json
{
  "num_cases": 6,
  "all_exact_equal": false,
  "all_within_threshold": true,
  "max_abs_diff": 4.678964614868164e-06,
  "max_mean_abs_diff": 1.807535682019079e-06,
  "max_rel_diff": 0.01511376816779375,
  "min_cosine_similarity": 0.9999998807907104
}
```

重点只看这几个：

- `all_within_threshold`
  如果是 `true`，说明通过

- `max_abs_diff`
  最大绝对误差，通常 `1e-6` 到 `1e-5` 就很不错

- `min_cosine_similarity`
  越接近 `1` 越好

`all_exact_equal = false` 一般不用担心，因为 PyTorch 和 TensorRT 不是同一个后端，浮点结果不逐位相等很正常。

## 5. 小白最推荐的完整流程

如果你今天只想从 0 到 1 跑通一次，就按这个顺序：

### 第一步：导出

```bash
conda run --live-stream -n lerobot_flex python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --precision fp32 \
  --device cpu \
  --trt-device cuda:0
```

### 第二步：做 safetensors 和 TRT 的对比

```bash
conda run --live-stream -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py \
  --checkpoint /data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/016000/pretrained_model \
  --engine /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/act_single_fp32.plan \
  --metadata /data/tfj/lerobot_tfj/outputs/deploy/act_trt/act_grasp_block_in_bin1/016000/export_metadata.json \
  --trt-device cuda:0 \
  --threshold-max-abs-diff 1e-4 \
  --report-json /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_COMPARE_016000_SAFETENSORS_VS_TRT.json
```

### 第三步：判断通过没有

看对比输出里：

- `all_within_threshold = true`

如果是 `true`，就说明这份 TensorRT engine 和原始 checkpoint 对齐是正常的。

## 6. 最容易犯的错

### 6.1 `--checkpoint` 传错

错误示例：

- 只传到 `016000`

正确示例：

- 传到 `016000/pretrained_model`

### 6.2 在错误的 conda 环境里导出

如果你在 `lerobot` 环境里导出，可能会碰到：

- `ModuleNotFoundError: No module named 'onnx'`

这时候不要怀疑模型，先换到：

- `lerobot_flex`

### 6.3 `--metadata` 不传

有时候不传也能跑，但不稳。

因为脚本需要靠 metadata 判断：

- 相机顺序
- TensorRT 输入名
- TensorRT 输出名

所以建议一直传。

### 6.4 看到 `.plan` 以为不是 engine

不是问题。

- `.plan` 是 TensorRT engine
- `.engine` 也是 TensorRT engine

很多时候只是文件后缀命名习惯不同。

## 7. 如果你只想记一句话

导出就跑：

- [export_act_checkpoint_engine.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/export_act_checkpoint_engine.py)

验货就跑：

- [compare_act_safetensors_trt.py](/data/tfj/lerobot_tfj/tfj_envs/act_trt/scripts/compare_act_safetensors_trt.py)

不会写参数时，就照抄这篇文档里的例子，把路径改成你自己的 checkpoint 路径。*** Update File: /data/tfj/lerobot_tfj/tfj_envs/act_trt/docs/ACT_TRT_EXPORT_VERIFY_QUICKSTART.md
