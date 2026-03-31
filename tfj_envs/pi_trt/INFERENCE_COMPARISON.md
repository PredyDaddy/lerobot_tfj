# PI0.5 PyTorch / ONNX / TensorRT 推理对比

本文档只讨论推理链路对比，不重复展开导出和上机流程细节。  
完整操作流程请参考：

- [README.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/README.md)

本文档对应的当前已验证通过 run 目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

本文档对应的当前最新实测 benchmark 报告：

- [FP32 chunk recheck](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_20260314_174221/benchmark_report.md)
- [FP32 pure select_action recheck](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp32_recheck_20260314_174221/report.md)
- [FP16 unsafe chunk diagnostic](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_unsafe_20260314_174221/benchmark_report.md)
- [FP16 unsafe pure select_action diagnostic](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_unsafe_20260314_174221/report.md)

## 1. 结论先行

如果只看当前这套工程和 2026-03-14 这轮最新实测结果，可以直接用下面这条判断：

- 开发和排错阶段，优先用 PyTorch
- 导出边界核对和跨后端一致性核对，优先用 ONNX
- 真机部署和长期运行，当前仍然优先用“已通过 Stage 5 的 TensorRT FP32”

还要补一句当前最重要的新结论：

- `unsafe fp16` 诊断 run 的速度非常亮眼
- 但它还没有通过 `Stage 5`
- 所以现在不能把“它更快”直接翻译成“它已经可以替换当前 FP32 默认上机链路”

原因不是一句“TRT 更快”这么简单，而是三条链路在目标、复杂度和风险上本来就不一样：

- PyTorch 是源实现，最容易看懂、最容易调试、最适合作为真值基线
- ONNX 是中间层，最适合检查导出边界、输入输出契约、provider 行为和跨运行时一致性
- TensorRT 是部署层，最适合 NVIDIA GPU 上的最终推理执行，但前提是 ONNX 和 engine 工件已经被严格验证过

## 2. 三条链路分别是什么

### 2.1 PyTorch 推理

对应脚本：

- [run_pi05_torch_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py)

输入工件：

- `pretrained_model/config.json`
- `pretrained_model/model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

运行方式：

- 直接 `from_pretrained(...)` 加载 PI0.5 模型
- 使用原始 PyTorch 图执行推理
- 可以选择 AMP

工程定位：

- 源实现
- 真值基线
- 最适合开发调试和行为核对

### 2.2 ONNX 推理

对应脚本：

- [run_pi05_onnx_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py)

输入工件：

- `artifacts/onnx/pi_shared_vision_encoder.onnx`
- `artifacts/onnx/pi_shared_prefix_cache.onnx`
- `artifacts/onnx/pi05_denoise_step.onnx`

运行方式：

- 当前工程不是把整个 PI0.5 导成一个单体 ONNX
- 而是拆成 3 个子图：
  - `vision_encoder`
  - `prefix_cache`
  - `denoise_step`
- 再由 ONNX Runtime 串起来跑

工程定位：

- 导出边界验证层
- 中间表示层
- 最适合做 Torch 和部署后端之间的“桥梁”

### 2.3 TensorRT 推理

对应脚本：

- [run_pi05_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_trt_infer_so101.py)

输入工件：

- `artifacts/engines/pi_shared_vision_encoder.engine`
- `artifacts/engines/pi_shared_prefix_cache.engine`
- `artifacts/engines/pi05_denoise_step.engine`
- `pi_trt_metadata.json`
- `stage5_verify_trt.json`

运行方式：

- 和 ONNX 一样，也是 3 段子图/子 engine 运行
- 不是单 engine 全图推理
- 由 TRT adapter 做输入契约检查、预热、前缀缓存衔接和 denoise 循环

工程定位：

- 最终部署层
- 真机运行层
- 当前工程里唯一适合作为最终上机默认链路的后端

## 3. 一张表看三条链路

| 维度 | PyTorch | ONNX Runtime | TensorRT |
| --- | --- | --- | --- |
| 源工件 | `model.safetensors` | `*.onnx` | `*.engine` |
| 对应脚本 | `run_pi05_torch_infer_so101.py` | `run_pi05_onnx_infer_so101.py` | `run_pi05_trt_infer_so101.py` |
| 运行后端 | PyTorch | ONNX Runtime | TensorRT |
| 当前工程形态 | 原始模型 | 3 个 ONNX 子图 | 3 个 TRT engine |
| 真值地位 | 最高 | 中间基线 | 最终部署结果 |
| 可调试性 | 最高 | 中等 | 最低 |
| 导出/构建复杂度 | 最低 | 中等 | 最高 |
| 对 NVIDIA GPU 依赖 | 可选 | 可选 | 必须 |
| 跨平台性 | 一般 | 最好 | 最差 |
| 当前推荐场景 | 开发/调试 | 导出验证/一致性验证 | 真机部署 |

## 4. 当前仓库里的实际验证结果

### 4.1 这次真正通过的是哪一套工件

当前已经完整通过 `Stage 2 -> Stage 5` 的目录是：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

对应最终 metadata：

- [pi_trt_metadata.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json)

最终状态：

- `stage2_export_onnx = pass`
- `stage3_verify_onnx = pass`
- `stage4_build_engines = pass`
- `stage5_verify_trt = pass`

### 4.2 数值一致性结论

最终 Stage 5 已经证明：

- `vision_encoder = pass`
- `prefix_cache = pass`
- `denoise_step = pass`
- `pipeline = pass`

对应文件：

- [stage5_verify_trt.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.json)
- [stage5_verify_trt.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.md)

这里要特别强调一个口径问题：

- 这里的 `pipeline = pass` 是当前一致性验证里的 chained compare gate
- 它说明当前验证口径下的链式输出没有被导出/部署后端搞坏
- 它不等于“完整 action chunk iterative sampling 的时延已经验证完了”
- 真正的完整 chunk 时延，要看下面新增的实测 benchmark

### 4.3 当前已经验证到的数值误差

下面这些不是“预期值”，而是当前这次成功 run 的实测结果。

#### `vision_encoder`

- `torch_vs_onnx`
  - `max_abs_diff = 4.959e-04`
  - `mean_abs_diff = 3.749e-06`
  - `min_cosine_similarity = 0.99999994`
- `torch_vs_trt`
  - `max_abs_diff = 3.967e-04`
  - `mean_abs_diff = 3.087e-06`
  - `min_cosine_similarity = 0.99999982`
- `onnx_vs_trt`
  - `max_abs_diff = 5.112e-04`
  - `mean_abs_diff = 3.322e-06`
  - `min_cosine_similarity = 0.99999982`

#### `prefix_cache`

- `torch_vs_onnx`
  - `max_abs_diff = 4.063e-04`
  - `mean_abs_diff = 7.474e-06`
  - `min_cosine_similarity = 0.99999976`
- `torch_vs_trt`
  - `max_abs_diff = 7.801e-04`
  - `mean_abs_diff = 1.658e-05`
  - `min_cosine_similarity = 0.99999958`
- `onnx_vs_trt`
  - `max_abs_diff = 5.646e-04`
  - `mean_abs_diff = 1.732e-05`
  - `min_cosine_similarity = 0.99999946`

#### `denoise_step`

- `torch_vs_onnx`
  - `max_abs_diff = 1.550e-06`
  - `mean_abs_diff = 1.129e-07`
  - `min_cosine_similarity = 1.0`
- `torch_vs_trt`
  - `max_abs_diff = 1.788e-06`
  - `mean_abs_diff = 1.362e-07`
  - `min_cosine_similarity = 0.99999988`
- `onnx_vs_trt`
  - `max_abs_diff = 1.132e-06`
  - `mean_abs_diff = 1.225e-07`
  - `min_cosine_similarity = 0.99999988`

#### `pipeline`

- `torch_vs_onnx`
  - `max_abs_diff = 7.391e-06`
  - `mean_abs_diff = 3.512e-07`
  - `min_cosine_similarity = 1.0`
- `torch_vs_trt`
  - `max_abs_diff = 5.305e-06`
  - `mean_abs_diff = 2.330e-07`
  - `min_cosine_similarity = 1.0`
- `onnx_vs_trt`
  - `max_abs_diff = 7.391e-06`
  - `mean_abs_diff = 4.464e-07`
  - `min_cosine_similarity = 1.0`

### 4.4 这些数值说明了什么

这里最值得关注的是两点：

- `denoise_step` 和整条 `pipeline` 的误差已经非常小，说明最终部署层没有把关键 denoise 行为搞坏
- `vision_encoder` 和 `prefix_cache` 的误差虽然比 denoise 大，但仍然在当前 gate 允许范围内，并且最终 `pipeline` 误差依然很低

从工程角度看，这说明：

- ONNX 现在已经足够接近 PyTorch，可以作为可靠中间基线
- TRT 现在已经足够接近 ONNX / PyTorch，可以作为当前默认部署后端

### 4.5 真实时延 benchmark

这一节不是工程判断，而是我刚刚在当前机器上直接实测出来的结果。  
完整原始报告见：

- [benchmark_report.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_20260314_174221/benchmark_report.md)
- [benchmark_report.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_20260314_174221/benchmark_report.json)

本次 benchmark 固定条件如下：

- GPU：`NVIDIA GeForce RTX 4090`
- 环境：`torch 2.7.1+cu126`、`onnxruntime 1.23.2`、`TensorRT 10.13.0.35`
- 输入：`build_runtime_context()` 生成的同一份 deterministic baseline batch
- `warmup_iterations = 10`
- `measured_iterations = 30`
- `num_inference_steps = 10`
- `torch_device = cuda:0`
- `PyTorch` 使用的是 `FP32` 基线
- `ONNX Runtime` 使用的是当前工程实际 runtime 路径，也就是 `session.run(...) + numpy/torch 边界`
- `TensorRT` 使用的是当前这套已经通过 `Stage 2 -> Stage 5` 的 static-shape、batch=1 engine

这里的几个名字也要先说清楚：

- `vision_encoder_single` 是单相机一次调用
- `vision_encoder_pair` 是 `top + wrist` 两次 vision 调用总和，这个更接近真实运行口径
- `denoise_step` 是单次 denoise 迭代，不是完整 chunk
- `pipeline_chunk` 是生成一个完整 action chunk 的模型推理时间
- `amortized_per_action_step` 是 `pipeline_chunk / n_action_steps`，它只是均摊值，不是每个 control loop 的最坏时延

实测总表如下：

| Backend | Stage | mean_ms | p50_ms | p95_ms |
| --- | --- | ---: | ---: | ---: |
| PyTorch FP32 | `vision_encoder_single` | `4.108` | `4.101` | `4.109` |
| PyTorch FP32 | `vision_encoder_pair` | `8.135` | `8.114` | `8.173` |
| PyTorch FP32 | `prefix_cache` | `25.114` | `24.984` | `25.647` |
| PyTorch FP32 | `denoise_step` | `6.269` | `6.266` | `6.336` |
| PyTorch FP32 | `pipeline_chunk` | `95.468` | `95.486` | `96.108` |
| ONNX Runtime | `vision_encoder_single` | `7.627` | `7.621` | `7.667` |
| ONNX Runtime | `vision_encoder_pair` | `15.712` | `15.630` | `16.201` |
| ONNX Runtime | `prefix_cache` | `63.256` | `63.188` | `63.564` |
| ONNX Runtime | `denoise_step` | `7.621` | `7.600` | `7.787` |
| ONNX Runtime | `pipeline_chunk` | `157.021` | `156.610` | `159.954` |
| TensorRT FP32 | `vision_encoder_single` | `6.358` | `6.343` | `6.443` |
| TensorRT FP32 | `vision_encoder_pair` | `12.760` | `12.736` | `12.789` |
| TensorRT FP32 | `prefix_cache` | `63.335` | `63.360` | `63.563` |
| TensorRT FP32 | `denoise_step` | `4.670` | `4.626` | `4.976` |
| TensorRT FP32 | `pipeline_chunk` | `123.501` | `122.640` | `130.163` |

如果把 chunk 结果继续换算成“每个 action 的均摊推理成本”，当前这次实测是：

- PyTorch FP32：`95.468 / 50 = 1.909 ms`
- ONNX Runtime：`157.021 / 50 = 3.140 ms`
- TensorRT FP32：`123.501 / 50 = 2.470 ms`

这组数字最值得注意的地方有三点：

- 按当前仓库这条真实实现链路，`PyTorch FP32` 的完整 `pipeline_chunk` 仍然比当前安全 `TensorRT FP32` 更快
- `TensorRT FP32` 的优势主要体现在 `denoise_step`，但当前 `prefix_cache` 几乎和 `ONNX Runtime` 一样慢，导致最终 chunk 没有把总时延拉到最低
- `ONNX Runtime` 现在的数值更像“当前工程真实 runtime 成本”，而不是“ORT 纯 GPU kernel 极限性能”，因为它包含了当前 runner 的 `numpy <-> torch` 边界搬运

所以这里必须给出一个更严谨的结论：

- 如果你问“当前这套仓库现在真实跑出来的离线模型推理时间是多少”，那就以上面这张表为准
- 如果你问“为什么明明用了 TRT，完整 chunk 还没有快过当前这组 PyTorch FP32 基线”，答案不是模型不对，而是当前 runtime 组合里 `prefix_cache` 很重，且 ONNX/TRT 链路还有额外的工程边界成本
- 如果你问“那为什么上机还是推荐 TRT”，原因依然是部署稳定性、工件收敛、输入契约严格和后续真机运行一致性，而不是一句简单的“这次 chunk benchmark 里 TRT 最快”

### 4.6 2026-03-14 的 FP16 重建诊断结果

这一轮我实际又跑了一次 `FP16 build path` 重建，目录是：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`

这套 run 的真实状态要分开看：

1. 正确性 gate
   - `Stage 2 = pass`
   - `Stage 3 = pass`
   - `Stage 4 = pass`
   - `Stage 5 = fail`

2. Stage 4 当前不是“干净纯 FP16”
   - 为了抑制 layernorm 相关的 FP16 漂移，当前 build 最终使用了：
   - `force_fp32_layer_types = REDUCE ELEMENTWISE UNARY`
   - 这意味着它更准确的名字是“带保守 escape hatch 的 fp16-enabled build”

3. Stage 5 为什么还 fail
   - clean `fp16-enabled` build 在当前严格阈值下会出现非常大的数值偏移
   - 加了上述 escape hatch 以后，`denoise_step` 和 `pipeline` 已经显著收敛
   - 但 `prefix_cache` 的 raw KV 对比仍然过不了当前 export-boundary gate

最关键的是，这套 `unsafe fp16` 工件虽然还没通过 Stage 5，但速度提升非常明显。下面是我这次实测的对照：

#### chunk benchmark

安全基线 `TRT FP32`：

- `pipeline_chunk = 123.501 ms`
- `vision_encoder_pair = 12.760 ms`
- `prefix_cache = 63.335 ms`
- `denoise_step = 4.670 ms`

诊断性 `unsafe TRT FP16`：

- `pipeline_chunk = 50.665 ms`
- `vision_encoder_pair = 4.585 ms`
- `prefix_cache = 13.348 ms`
- `denoise_step = 3.239 ms`

这组数字的含义非常直接：

- 如果只看速度，当前这套 `unsafe fp16` 的 chunk 路径已经大幅快过 `TRT FP32`
- 改善最大的是 `prefix_cache`
- 这也说明“FP16 完全没有价值”这个结论是不成立的

#### 1000-step pure inference select_action

安全基线 `TRT FP32`：

- `mean_per_step = 2.491 ms`

诊断性 `unsafe TRT FP16`：

- `mean_per_step = 1.015 ms`

这里同样说明：

- 如果只看长期均摊吞吐，当前这套 `unsafe fp16` 的纯推理速度也明显更快

但这个 subsection 必须用一句硬结论收住：

- 这些 `FP16` 数字现在只说明“性能潜力很强”
- 它们不说明“当前 FP16 工件已经通过正确性 gate”
- 所以它们只能当作诊断 benchmark，不能直接当作真机默认部署结论

## 5. 工件形式对比

### 5.1 当前 run 的 ONNX 文件大小

来自 [stage2_export_onnx.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage2_export_onnx.json)：

- `vision_encoder.onnx = 1,659,622,704 bytes`
- `prefix_cache.onnx = 397,963 bytes`
- `denoise_step.onnx = 3,250,285 bytes`

### 5.2 当前 run 的 TRT engine 大小

来自 [stage4_build_engines.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage4_build_engines.json)：

- `vision_encoder.engine = 1,662,862,340 bytes`
- `prefix_cache.engine = 9,600,472,172 bytes`
- `denoise_step.engine = 1,722,120,188 bytes`

### 5.3 这对实际部署意味着什么

这组大小直接说明了三个现实问题：

- TRT 不是“更小的模型格式”，尤其不是在这个拆分方案下
- `prefix_cache.engine` 非常大，部署时必须考虑磁盘、显存和构建时间
- ONNX 更适合做中间验证工件，TRT 更适合做最终执行工件

## 6. 从推理角度如何理解三条链路

### 6.1 PyTorch 的优势和短板

优势：

- 最容易调试
- 最接近原始模型实现
- 不需要导出和构建
- 最适合做行为基线

短板：

- 对部署环境依赖更重
- 图级优化能力不如 TRT
- 真机长期跑时，作为最终后端的工程收益不如 TRT

适合：

- 先把模型行为跑对
- 看单步输出、hook 中间张量
- 确认“模型本身有没有问题”

### 6.2 ONNX 的优势和短板

优势：

- 很适合做导出边界核对
- 比 TRT 更容易看输入输出契约
- provider 可切 CPU/CUDA，适合做跨后端比较
- 是 PyTorch 和 TRT 之间最重要的中间层

短板：

- 不同 provider、不同优化级别会影响行为和算子支持
- 很容易出现“不是模型错，而是 provider 不支持/类型不兼容”的假失败
- 在当前工程里并不是最终推荐上机后端

适合：

- 检查 `timestep` 是否还是 live input
- 检查拆图后的子图 I/O 契约
- 检查 PyTorch 与部署后端之间到底是哪里开始偏

### 6.3 TensorRT 的优势和短板

优势：

- 当前工程中最终部署最合适
- 对 NVIDIA GPU 最友好
- 在当前验证结果中已经实现了与 PyTorch/ONNX 的高一致性
- 运行时输入检查和工件 provenance 检查已经做得比较严

短板：

- 导出和构建链路最复杂
- 最难调试
- 对图、dtype、外部权重、shape 契约都很敏感
- 强依赖 NVIDIA GPU 和 TensorRT 环境

适合：

- 最终真机部署
- 固定工件、固定环境、重复运行
- 已经通过 Stage 2 到 Stage 5 验证的模型

## 7. 当前工程里应该怎么选

如果目标是：

### 7.1 我要看模型行为是不是对的

用 PyTorch。

原因：

- 它是源实现
- 最方便看 `policy` 自身行为
- 所有后端最终都要回到 PyTorch 做真值对比

### 7.2 我要看导出有没有把模型搞坏

用 ONNX。

原因：

- 它正好卡在 PyTorch 和 TensorRT 中间
- 最容易定位是导出问题、provider 问题，还是下游构建问题

### 7.3 我要真的上机跑

用 TensorRT。

前提：

- Stage 2 到 Stage 5 都已经 pass
- `pi_trt_metadata.json` 里的 `stage5_verify_trt = pass`
- 不要拿 warning / fail 工件直接上真机

## 8. 这次对比里最容易误判的点

### 8.1 `stage3 overall_status = warn` 不等于 Stage 3 失败

这点必须单独强调。

当前成功 run 的 Stage 3 是：

- `overall_status = warn`
- 但 `stage3_acceptance = pass`
- metadata gate 也是 `pass`

这说明：

- runtime-oriented compare 仍然可以出现 `warn`
- 但只要 export-fidelity compare 和 `timestep` 契约都通过，Stage 3 gate 仍然是成功的

### 8.2 不能把 provider 问题误判成模型问题

这次真实踩过的坑就说明了：

- ONNX Runtime 的 CPU provider 行为
- ONNX 图里的 `float64` 分支
- TensorRT 的 parser 限制

都可能导致“模型本身没错，但某个中间层失败”。

所以比较三条链路时，不应该只看一句“哪个后端失败了”，而是要分清楚：

- 是模型数值不一致
- 还是导出边界不一致
- 还是 provider / backend 不支持

## 9. 本次 benchmark 的边界

虽然现在已经补上了统一 benchmark，但这组数字仍然只能按下面这个口径理解：

- 这是离线模型推理 benchmark，不等价于真机闭环控制 latency
- 这里没有测相机采集、MJPG 解码、`robot.get_observation()`、`send_action()`、smoothing、delta clamp、`precise_sleep`
- 当前 `pipeline_chunk` 表示“生成一个 action chunk 的时间”，不是“每个 control loop 的最坏时延”
- 当前 `amortized_per_action_step` 只是 `chunk / n_action_steps` 的均摊值，不能替代真实闭环 wall-clock
- 当前 ONNX 结果必须理解成“当前仓库 ONNX runtime 实现路径的时间”，不能直接写成“纯 ORT CUDA kernel 对比”
- 当前 TensorRT 结论只对这套 `static-shape + batch=1 + fixed token length` engine 成立，不能自动外推到别的 batch size、别的 prompt 长度、别的分辨率
- 当前这组主表里的 PyTorch 结果是 `FP32` 口径；如果你后面要继续比较 `PyTorch AMP / BF16 autocast`，需要单独再测一组
- 当前 deterministic baseline batch 更适合做固定口径对比，不代表真实业务输入分布

如果后面还要继续补 benchmark，我建议下一轮单独再加四类数据：

- `cold start latency`
- `真实 control loop wall-clock`
- `显存占用`
- `PyTorch FP32` 与 `PyTorch AMP` 对照组

## 10. 最终建议

当前仓库状态下，建议固定以下策略：

- PyTorch 作为源行为基线
- ONNX 作为导出和中间一致性基线
- TensorRT 作为默认真机部署后端

当前直接推荐上机使用的 TRT 工件目录：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

当前不建议再使用此前失败或半成功的旧 run 目录。
