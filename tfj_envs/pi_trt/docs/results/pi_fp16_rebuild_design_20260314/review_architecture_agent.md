# FP16 技术方案独立审稿报告

## 结论

未发现阻塞性问题。

当前 proposal 对现有 PI0.5 TensorRT 三段式架构、当前 FP32 结果的总体方向判断、以及先做 `Phase A` 再决定是否进入 `Phase B` 的工程顺序，整体上是成立的。尤其是“三段子图不是全图单 engine”“`denoise_step` 已经受益于 TRT”“`prefix_cache` 是当前最值得优先关注的瓶颈”这三点，和现有代码、验证链路、benchmark 结果基本一致。

需要修订的主要不是路线本身，而是归因强度。按当前实现，`Phase A` 实际验证的是“在现有 FP32 图边界和 runtime 合同下启用 TensorRT `FP16 build path` 的收益”，而不是“端到端 half-precision PI0.5 TRT 管线的收益”。因此，proposal 里几处“只改 precision、可明确归因”“如果没变快就说明问题在 prefix_cache 结构本身”的说法偏强，建议收紧。

## 主要发现

- proposal 对当前 TRT 架构的描述基本准确。现有 runtime 的确是 `vision_encoder` 跑两次、`prefix_cache` 跑一次、`denoise_step` 按 `num_inference_steps` 循环多次，见 `scripts/trt_pi_adapter.py:461-504`；`select_action` 也是基于 chunk queue 的刷新逻辑，见 `scripts/trt_pi_adapter.py:507-512`。`Stage 5` 的 pipeline 验证也确实是单步 `vision -> prefix -> denoise` 链路，而不是单 engine 全图，见 `scripts/step5_verify_trt.py:746-845`。

- proposal 对当前性能大势的判断也基本准确。`docs/results/pi_inference_benchmark_fp32_20260314/benchmark_report.md:35-56` 显示，TRT 相比 PyTorch 在 `denoise_step` 上有收益，但在 `prefix_cache` 上明显回退，最终导致 `pipeline_chunk` 更慢。按现有数字估算，TRT 相对 PyTorch 的 chunk 差额里，`prefix_cache` 约贡献 `+38.0 ms`，`vision_encoder_pair` 约贡献 `+4.7 ms`，`denoise_step` 则回收约 `-16.4 ms`，因此“prefix_cache 是主矛盾”成立，但“只有 prefix_cache 在拖后腿”不够严谨，`vision_encoder` 也在回退。

- proposal 对“engine boundary overhead” 的定位偏重。用现有 TRT 子图计时近似展开，`2 * vision + prefix + 10 * denoise = 12.797 + 63.059 + 46.500 = 122.356 ms`，而实测 `pipeline_chunk = 123.053 ms`，差额只有约 `0.697 ms`。这说明边界/同步成本确实存在，但按当前 benchmark 证据，它更像次级项，不足以和 `prefix_cache` 自身回退并列成同等级主因。proposal 在 `FP16_TECHNICAL_PROPOSAL.md:110-117` 这部分建议改成“结构性假设”或“次级假设”，而不是已经被充分证明的主因。

- `Phase A` 目前不是严格意义上的“纯 FP16、只改 precision”实验。现有导出 wrapper 明确把关键张量又转回了 `float32`，见 `scripts/export_wrappers.py:94-97`、`scripts/export_wrappers.py:153-158`、`scripts/export_wrappers.py:216-218`。也就是说，当前 ONNX 子图边界本身就是 FP32 合同，尤其 `prefix_cache` 会把 KV 明确以 `float32` 输出。与此同时，Stage 4 build 只是对 TensorRT builder 设置 `BuilderFlag.FP16`，见 `scripts/build_pi_trt_engine.py:224-289`，并没有改变导出边界。换言之，`Phase A` 更准确的定义应是“对现有 FP32-interface 子图做 FP16 build-path 实验”，而不是“端到端 FP16 管线实验”。

- 当前 runtime 仍保留显式同步与 host-side 循环开销，这会限制归因纯度。`TensorRTRunner.infer()` 每次都会 `self.stream.synchronize()`，见 `scripts/trt_runtime.py:217-255`；`predict_action_chunk()` 也在 Python 侧构造 timestep、循环更新 `x_t`，见 `scripts/trt_pi_adapter.py:478-504`。因此，哪怕 FP16 engine 内部更快，端到端收益也可能被同步、KV 物化、FP32 子图边界和 Python-side 循环稀释。proposal 现在把负结果直接归到 `prefix_cache` 结构本身，证据还不够闭合。

- proposal 正确地区分了 `pipeline_chunk` 和 `1000-step select_action` 是两个指标，但需要更明确说明这二者不能互相代换。现有文档已经写明二者不同，见 `docs/results/pi_inference_benchmark_fp32_20260314/benchmark_report.md:64-69` 与 `docs/results/pi_select_action_1000steps_20260314_160853/report.md:3-14`。这一点建议保留并进一步强化。

- compare matrix 里 `PyTorch AMP` 的表述存在潜在误导。当前 benchmark 里的 autocast 明确使用的是 `bfloat16`，见 `scripts/benchmark_pi_inference.py:70-73`，export wrapper 里的 autocast 也是 `bfloat16`，见 `scripts/export_wrappers.py:17-20`。因此 proposal 中的 `PyTorch AMP` 不能被读者自然理解成 “PyTorch FP16”。如果后续报告写成 “TRT FP16 vs PyTorch AMP” 而不注明 `AMP = BF16`，会削弱结论的可解释性。

- `Stage 5` 作为正确性 gate 是必要的，但它当前验证的是 export-fidelity 的单步子图与单步 pipeline，不是完整 `10-step denoise` 累积后的 chunk 数值稳定性。见 `scripts/step5_verify_trt.py:830-891`。因此，proposal 把 “先过 `Stage 5`，再考虑默认部署” 作为主要 gate 是合理的，但若要把 FP16 设为默认候选，仍建议增加一个 deterministic chunk-level 数值 smoke check，避免多步累积漂移被漏检。

## 需要修改的点

- 建议把 `FP16_TECHNICAL_PROPOSAL.md:142-156` 的“纯 FP16 重建”“只改 precision，不改 architecture”改成更技术准确的说法，例如：“在现有三段子图与现有 FP32 graph interface 不变前提下，测试 TensorRT FP16 build path 的收益”。当前写法容易让读者误以为跨 engine 边界也已经 half precision。

- 建议把 `FP16_TECHNICAL_PROPOSAL.md:154-156` 的强归因改弱。现阶段最多只能说：“如果性能变好，说明当前 build precision 选择对现有架构有正向贡献；如果性能不明显改善，说明仅靠 build precision 不足以解决问题，需要继续拆分 engine 内部精度采用情况、子图边界开销和 prefix 结构成本。”不建议直接写成“没变好就说明问题在 prefix_cache 结构本身，而不是 build flag”。

- 建议在 `Phase A` 和 `Phase B` 之间增加一个显式的中间诊断步骤，哪怕文档里不单独命名为 `Phase A.5`，也应至少写入决策规则：
  1. 先看 `FP16` 的分段 benchmark 是否真的改善了 `vision/prefix/denoise`。
  2. 再看 Stage 4/单 engine build report 是否记录了足够的“effective precision”证据。
  3. 再决定问题是 engine 内部精度采用不足，还是 prefix 结构/边界本身需要进入 `Phase B`。
  否则 `Phase B` 启动条件会过早，把“builder 没真正吃到 FP16 红利”和“prefix 架构有问题”混为一谈。

- 建议把当前“FP32 TRT 没赢 chunk 的四个原因”重写成带主次关系的表述。按现有证据，更稳妥的排序应是：
  1. `prefix_cache` 回退是主因。
  2. `vision_encoder` 也有回退，但量级次于 prefix。
  3. 同步/边界开销存在，但当前 benchmark 证据下更像次级项。
  4. `FP32` 而非 `FP16` 是尚未验证的优化空间，不宜直接写成既定根因。

- 建议在 compare matrix、benchmark 文档、以及未来 `benchmark_pi_select_action.py` 输出中，把 `PyTorch AMP` 明确写成 `PyTorch AMP (BF16)` 或等价表述。否则 `TRT FP16 vs PyTorch AMP` 看起来像精确的精度对照，实际不是。

- 建议在 implementation plan 的 provenance 设计里，把“requested precision”与“effective precision evidence”分开。当前 `Commit A` 的方向是对的，但不要只重复记录已有的 `precision=fp16/fp32` 文本；更有价值的是增加：
  1. engine I/O dtype 摘要。
  2. 是否存在 forced-fp32 layer 约束。
  3. 若可行，增加 engine inspector 或等价的 layer precision 摘要。
  否则 `FP16 provenance` 依然只能证明“请求了 FP16 build”，不能证明“engine 实际大范围采用了 FP16”。

- 建议在“可以把 FP16 作为默认部署候选”的判断里，补上一条 chunk-level 数值稳定性说明。当前 `Stage 5` 只覆盖单步 pipeline，不足以完全替代完整 chunk 的多步累积检查。

## 可接受保留项

- 可以保留“三段子图名字和职责不变、cache tensor naming 不变、`denoise_step` 保留 live `timestep`”这组合同。它们和当前实现是一致的，也有利于 Phase A 保持可审计性。

- 可以保留“先做 `Phase A`，不在第一轮同时做边界重构/单 engine 融合/真机控制改动”的工程策略。当前代码已经有稳定的 `Stage 2 -> Stage 5` 基线，先做小范围变量实验是合理的。

- 可以保留“`chunk benchmark` 与 `1000-step pure inference` 必须分开看”的方法学要求。这一点当前 benchmark 文档已经支撑，继续强调是正确的。

- 可以保留 implementation plan 中“新 `run-dir` 内自洽地重跑 `Stage 2 -> Stage 5`，并强化 provenance”的思路。对于存在 `fp32/fp16` 并存工件的阶段，这个设计是必要的，不是形式主义。

- 可以保留“如果 `Phase A` 不理想，`Phase B` 优先关注 `prefix_cache`”这个方向判断，但前提应是先补完上文提到的中间归因步骤，而不是直接跳结论。

## 残余风险

- 最大残余风险是：即便 `FP16 build` 生效，当前子图之间的 FP32 合同和 KV 物化仍可能成为性能上限，导致 `FP16` 收益被低估。相反，如果 `FP16` 收益很小，也不必然说明 prefix 结构本身就是唯一问题。

- 当前 TRT runtime 每次 `infer()` 都显式同步，且 denoise 循环在 Python 侧推进；这会让 benchmark 结果混入一部分 runtime orchestration 成本，而不只是 engine kernel 成本。

- `Stage 5` 不覆盖完整 multi-step chunk 的累积误差，因此 FP16 若出现“单步可过、十步累积漂移扩大”的情形，现有 gate 可能较晚才暴露。

- `1000-step pure inference` 结果当前已有落盘结果目录，但 repo 内脚本尚未固化进版本库。implementation plan 已经识别到这个问题，建议按计划补齐，否则其可审计性仍弱于 chunk benchmark。

- 现有全部结论仍只对当前 `static-shape + batch=1 + fixed token length` 的已验证工件成立，不应外推到更一般的输入分布或真机闭环 wall-clock。
