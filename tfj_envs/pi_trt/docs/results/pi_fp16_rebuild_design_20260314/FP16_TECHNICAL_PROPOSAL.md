# PI0.5 FP16 TensorRT Rebuild Technical Proposal

## 1. 文档目的

本文档用于收敛当前 `PI0.5` 的 `FP16 TensorRT` 重建方案。它回答四个问题：

1. 当前为什么需要做 `FP16` 重建
2. 这次重建到底要解决什么问题，不解决什么问题
3. 方案的技术边界、验证方式和验收标准是什么
4. 如果第一轮 `FP16` 结果不理想，下一步该怎么推进

本方案基于以下三份头脑风暴报告收敛而来：

- [brainstorm_architecture.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/brainstorm_architecture.md)
- [brainstorm_precision_build.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/brainstorm_precision_build.md)
- [brainstorm_critic.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/brainstorm_critic.md)

## 2. 当前问题定义

### 2.1 当前已经验证通过什么

当前 `FP32 TensorRT` 工件已经通过：

- `Stage 2`: ONNX 导出通过
- `Stage 3`: ONNX 一致性验收通过
- `Stage 4`: TensorRT engine 构建通过
- `Stage 5`: Torch / ONNX / TRT 一致性验收通过

当前已验证 run：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`

这说明当前实现首先是“对的”，不是“跑不通”。

### 2.2 当前真正不满意的是什么

当前不满意的是性能结构，而不是正确性。

根据已经实测的 chunk benchmark：

- `PyTorch FP32 pipeline_chunk = 94.934 ms`
- `TensorRT FP32 pipeline_chunk = 123.053 ms`

根据已经实测的 1000-step 纯推理 benchmark：

- `PyTorch FP32 mean_per_step = 2.855 ms`
- `TensorRT FP32 mean_per_step = 2.457 ms`

所以当前状态不是简单一句“TRT 更慢”或者“TRT 更快”，而是：

- 看 `chunk` 生成时间，当前 `TRT FP32` 还没有赢
- 看长时间 `select_action` 均摊时间，当前 `TRT FP32` 已经能赢

这说明当前问题集中在“chunk 刷新路径”的固定成本，而不是所有推理路径都慢。

## 3. 当前架构和原理映射

当前 TensorRT 不是单 engine 全图推理，而是三段子图：

1. `vision_encoder`
   - 对应 `PaliGemma` 视觉侧
   - 负责把 `top / wrist` 图像变成 `image_embs`

2. `prefix_cache`
   - 对应 `PaliGemma language model` 的 prefix prefill
   - 负责把视觉 embedding 和语言 token 组成条件前缀
   - 产出 `past_key_values`

3. `denoise_step`
   - 对应 `action expert` 的单步 denoise
   - 输入 `x_t + timestep + prefix KV`
   - 输出单步 `v_t`

当前真实 runtime 流程是：

1. vision 跑两次
2. prefix 跑一次
3. denoise 跑 `num_inference_steps` 次

因此总性能不是只由 `action expert` 决定，而是由下面这条式子决定：

`chunk_latency = 2 * vision + prefix_cache + N * denoise_step + engine boundary overhead`

## 4. 根因判断

### 4.1 当前已经被 TRT 加速的部分

`denoise_step` 这一段已经被 TRT 加速了。

当前实测：

- `PyTorch FP32 denoise_step = 6.286 ms`
- `TensorRT FP32 denoise_step = 4.650 ms`

也就是说，后面的 `action expert` 并不是没吃到 TRT 红利。

### 4.2 当前真正拖后腿的部分

在当前工程实现路径下，`prefix_cache` 是最主要的性能瓶颈，但不是唯一回退项。

当前实测：

- `PyTorch FP32 prefix_cache = 25.028 ms`
- `TensorRT FP32 prefix_cache = 63.059 ms`

这段差距远大于 `denoise_step` 带来的收益，所以完整 `chunk` 反而输给了 Torch。

这里需要强调两点：

1. 这个结论描述的是“当前 `static-shape + batch=1 + fixed token length`、当前 runner 路径、当前 benchmark 口径下”的阶段总成本。
2. 这不等价于“`prefix_cache` 内部模型结构本身就是唯一根因”，因为当前阶段耗时里还混有 cache 物化、engine boundary 和同步开销。

### 4.3 当前 FP32 TRT 为什么没把 chunk 拉下来

当前 `TRT FP32` 没赢 `chunk`，按现有证据更稳妥的主次排序是：

1. `prefix_cache` 阶段总成本回退是主因
2. `vision_encoder` 也有回退，但量级次于 `prefix_cache`
3. 当前 runtime 每段 `infer()` 都有显式边界和同步成本，但从现有分段计时看更像次级项
4. 当前 build 还是 `FP32`，`FP16` 仍然是尚未验证的优化空间，而不是已被证明的根因

因此，下一步最合理的第一实验，不是马上改图边界，而是先做一轮“在当前图边界下启用 `TensorRT FP16 build path`”的重建实验。

## 5. 本次 FP16 重建的目标

### 5.1 主目标

本次 `FP16` 重建的主目标是：

1. 保持当前三段子图架构不变
2. 将三段 engine 全部重建为新的 `fp16-enabled` 版本
3. 保证正确性 gate 不被破坏
4. 观察 `prefix_cache` 是否得到足够改善
5. 观察完整 `pipeline_chunk` 是否明显优于当前 `TRT FP32`

### 5.2 非目标

这轮不做下面这些事：

1. 不做图边界重构
2. 不做单 engine 融合
3. 不同时改真机控制逻辑
4. 不在第一轮里引入多种结构性优化，避免归因失真

## 6. 技术路线

### 6.1 Phase A: `FP16 build path` 重建实验

第一阶段只做一件事：

- 在当前三段子图职责不变、当前 runtime 合同和当前验证口径不变的前提下，重建一套新的 `fp16-enabled TensorRT` 工件集合

这一步的核心原则是：

- 保持 architecture 不变，只引入 `TensorRT FP16 build path` 这个新变量

这里需要明确一个证据边界：

1. 这不是严格意义上的“端到端纯 FP16 管线实验”。
2. 原因是本轮仍然需要在新 run-dir 中重新执行 `Stage 2 -> Stage 5`，而当前 ONNX 子图接口本身仍以现有导出合同为准。
3. 因此，`Phase A` 更准确的定义是：“在现有子图边界和现有导出合同下，测试 `TensorRT FP16 build path` 是否对当前工程实现路径产生正向收益。”

这样做的好处是：

- 如果性能变好了，可以初步支持“当前 build precision 选择对现有架构有正向贡献”
- 如果性能没有明显改善，只能说明“仅靠 `FP16 build path` 不足以解决当前主瓶颈”，不能直接跳结论说问题只在 `prefix_cache` 结构本身

### 6.2 Phase A 之后的决策规则

如果 `FP16` 重建后满足：

1. `Stage 5` 仍然通过
2. `pipeline_chunk` 明显优于当前 `TRT FP32`
3. `1000-step pure inference` 不回退

那么可以继续考虑把它作为新的 `TensorRT fp16-enabled` 部署候选。

这里的前提还需要补一句：

- `Stage 5` 只是 export-boundary correctness gate，不单独代表完整 runtime correctness 或真机闭环 correctness。
- 因此只有在 `Stage 5 + chunk benchmark + 1000-step pure inference + 运行时工件 provenance 自检` 同时成立时，才讨论默认候选。

如果 `FP16` 重建后：

1. `denoise_step` 变快
2. 但 `prefix_cache` 还是过重
3. `pipeline_chunk` 仍然没有实质改善

那么下一阶段重点不应继续只盯 `action expert`，但也不能直接断言问题已经被定位为 `prefix_cache` 内部结构本身。

更合理的下一步是先补一层中间诊断：

1. 核对 Stage 4/单 engine build report 里关于 precision 的证据是否充分
2. 核对各阶段 benchmark 是否真的显示 `vision/prefix/denoise` 都没有从 `FP16 build path` 获益
3. 再决定问题更偏向 engine 内部精度采用不足、I/O 边界与同步成本，还是 `prefix_cache` 结构本身

### 6.3 Phase B: 只在必要时启动的结构性优化

只有在 `Phase A` 完成且结果仍不理想时，才考虑：

1. 降低 `prefix_cache` 的 engine boundary 开销
2. 重新审视 prefix KV 的物化方式
3. 评估是否需要改变当前三段拆图方式

这一步不是这轮的主任务。

## 7. 必须保持不变的合同

为了让 `FP16` 实验可解释，这几条合同必须保持：

1. 三段子图名字和职责不变
   - `vision_encoder`
   - `prefix_cache`
   - `denoise_step`

2. cache tensor naming 不变
   - `past_key_values.layer_XX.key`
   - `past_key_values.layer_XX.value`

3. `denoise_step` 必须继续保留 live `timestep`

4. benchmark 输入口径不变
   - 同一 deterministic baseline batch
   - 同一 `num_inference_steps`
   - 同一 `n_action_steps`
   - 同一 GPU

5. artifact provenance 必须更严格，而不是更松

6. 文档中必须把“requested precision”和“effective precision evidence”分开

## 8. 验证与验收方案

### 8.1 正确性 gate

`fp16-enabled` rebuild 不能只看速度，必须先过正确性。

必跑链路：

1. `Stage 4` 重建新的 `fp16-enabled` engine 集合
2. `Stage 5` 验证新的 engine 集合在 export-boundary 单步链路上的一致性

要求：

- 不允许用“先放宽阈值再说”的方式偷过 gate
- 如果确实需要对 FP16 漂移做政策性调整，必须显式记录原因和幅度

同时必须明确：

1. `Stage 5` 当前验证的是 export-fidelity 的子图与单步 pipeline 一致性。
2. `Stage 5` 不覆盖完整 `predict_action_chunk()` 的多步累积漂移。
3. `Stage 5` 也不单独覆盖当前 runtime/autocast 路径与真机闭环行为。
4. 因此，`Stage 5 pass` 是必要条件，但不是“可上机”或“默认部署候选”的充分条件。

### 8.2 必跑 benchmark

至少必须跑下面两类 benchmark：

1. `chunk` benchmark
   - `vision_encoder_pair`
   - `prefix_cache`
   - `denoise_step`
   - `pipeline_chunk`

2. `1000-step pure inference select_action`
   - 用于观察长时间均摊口径

### 8.3 必比对照组

至少比较下面四组：

1. `TRT FP32` vs `TRT FP16`
2. `PyTorch FP32` vs `PyTorch AMP (CUDA BF16 autocast)`
3. `TRT FP16` vs `PyTorch FP32`
4. `TRT FP16` vs `PyTorch AMP (CUDA BF16 autocast)`

### 8.4 建议验收标准

第一轮工程验收建议先采用下面三档“工程政策阈值”：

1. 最低可接受
   - `pipeline_chunk` 相比当前 `TRT FP32` 改善至少 `10%`
   - `1000-step pure inference` 不回退
   - `Stage 5` 通过

2. 比较理想
   - `prefix_cache` 改善至少 `20%`
   - `denoise_step` 改善至少 `15%`
   - `pipeline_chunk` 改善至少 `15%`

3. 强成功
   - `pipeline_chunk` 接近或超过当前 `PyTorch FP32`

这些阈值的含义是：

1. 它们用于第一轮工程验收，不是由当前样本量直接推导出的统计显著性阈值。
2. 如果结果落在阈值边缘，需要结合重复测量、stage-level 数据和 provenance 证据一起解释。

## 9. 文档和方法学限制

无论最后结果如何，文档里都必须提前写清楚这些限制：

1. 当前 `FP16` 结果只对现有 `static-shape + batch=1 + fixed token length` engine 成立
2. 离线 benchmark 不等于机器人闭环 wall-clock
3. `pipeline_chunk` 和 `1000-step select_action` 是两个不同指标
4. 当前 deterministic baseline batch 适合固定口径对比，不代表真实业务输入分布
5. 不能用 “某一段更快” 代替 “完整 chunk 更快”
6. 当前文档中的 “TRT FP16 / FP16 工件” 默认表示“requested precision=fp16 的 build 工件集合”，不自动等价于“关键路径逐层已经有效 FP16 执行”

## 10. 最终收敛结论

本次主方案确定如下：

1. 先做一轮 `TensorRT FP16 build path` 重建
2. 第一轮不改图边界，不做结构性重构
3. 把 precision provenance 做严，避免 `FP32 / FP16` artifact 混用
4. 先过 `Stage 5` 这个 export-boundary gate，再跑 `chunk` 和 `1000-step` 两类 benchmark
5. 只有当 `Phase A` 无法显著改善 `pipeline_chunk`，且中间归因步骤也不能证明收益被 precision 证据不足或边界成本掩盖时，才进入 `prefix_cache` 结构优化议题

这是一条可解释、可审计、可回退的技术路线。
