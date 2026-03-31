# PI0.5 FP16 Rebuild 方案挑刺审稿报告

本报告只基于当前工作区可见证据下结论，主要核对了以下材料：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/FP16_TECHNICAL_PROPOSAL.md`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/FP16_IMPLEMENTATION_PLAN.md`
- `scripts/step4_build_engines.py`
- `scripts/build_pi_trt_engine.py`
- `scripts/step5_verify_trt.py`
- `scripts/benchmark_pi_inference.py`
- `scripts/run_pi05_trt_infer_so101.py`
- 现有结果目录：
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_20260314`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_20260314_160853`
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500`

结论先行：当前文档已经比最初草稿严谨很多，尤其补上了“空 run-dir 不能直接跑 Stage 4”的实现约束。但如果按“审稿人专门挑刺”的标准看，仍有几处会直接影响结论成立性的阻塞项，以及几处很容易把 FP16 rebuild 归因做错的高风险项。

## 阻塞性问题

1. `“纯 precision rebuild / 只改 precision”` 这个核心叙事目前并没有被文档中的执行方案真正保护住。

证据：

- Proposal 在 `FP16_TECHNICAL_PROPOSAL.md:146-155` 明确把 Phase A 定义成“只改 precision，不改 architecture”，并进一步给出“如果性能变好了，能明确归因给 precision rebuild”。
- 但实施计划在 `FP16_IMPLEMENTATION_PLAN.md:13` 和 `FP16_IMPLEMENTATION_PLAN.md:159-164` 又要求在全新的 `FP16_RUN` 里重新执行 `Stage 2 -> Stage 5`，也就是重新导出 ONNX，再重做验证和 build。
- 当前元数据已经明确表明导出路径本身就是实质性变量，不只是格式搬运：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json:713-718` 写明了 `denoise_step` 的 exporter route、runtime/export mode split、以及 post-export sanitization 都会影响验证结果。
- 历史结果也证明过这一点：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/stage3_verify_onnx.md:40-41` 记录过 chained pipeline 的 denoise ONNX 不消费 `timestep`，而 `stage3_verify_onnx_dynamo.md:40-41` 才恢复为 live `timestep`。

为什么这是阻塞项：

- 一旦 `FP16_RUN` 里的 Stage 2 产物和当前 verified FP32 run 的 ONNX 语义不完全一致，那么最终观测到的性能或数值变化就不再是“纯 precision 变量”，而是“重新导出 + 重新构图 + 重新 build”的混合结果。
- 在这种前提下，proposal 里“性能变好就能归因给 precision rebuild；性能没变好就说明核心问题在 prefix_cache 结构本身”的判断链条就站不住。

建议：

- 要么把叙事降级成“同架构下的 fp16-enabled rebuild”，明确承认它不是严格隔离的单变量实验。
- 要么把“新 FP16 ONNX 与基线 FP32 ONNX 的等价性证明”写成硬性 gate，例如至少要求：
  - 子图级路径、输入输出名、动态轴/静态形状一致。
  - ONNX 文件哈希不同也必须解释原因。
  - `stage2_export_onnx.json` 中显式对照 baseline run 的 export route / sanitization / output contract。

2. 当前文档把 `Stage 5 pass` 放在了过高的位置，但现有 `Stage 5` 并不能证明 FP16 运行时边界正确，更不能单独支撑“真机候选”。

证据：

- Proposal 在 `FP16_TECHNICAL_PROPOSAL.md:161-165` 把 `Stage 5` 通过当成进入默认候选判断的第一前提。
- 实施计划在 `FP16_IMPLEMENTATION_PLAN.md:22`、`FP16_IMPLEMENTATION_PLAN.md:376-377`、`FP16_IMPLEMENTATION_PLAN.md:536-537` 都把 `Stage 5 pass` 放到了接近“上机前主 gate”的位置。
- 但现有 `Stage 5` 的定义并不是“运行时真实性能/正确性对齐”，而是“export-fidelity compare”：
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/stage5_verify_trt.json:21-31` 明确写的是 `export_reference_torch`，`policy_device=cpu`，`policy_dtype=float32`，`use_autocast=false`。
  - 同文件 `:75-120` 明确 primary profile 是 `export_fidelity`，vision/prefix 甚至优先走 CPUExecutionProvider。
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json:46-76` 同时又明确记录当前 runtime reference contract 在 CUDA 下是 `autocast/bfloat16`。
  - 同文件 `:713-723` 进一步直说：`runtime_reference_vs_onnx` 被单独汇报，原因就是 runtime path 使用 `autocast/bfloat16`，而 export reference 是 `float32/no-autocast`。
- 另外，`Stage 5` 的 `pipeline` 也不是完整 chunk：
  - `scripts/step5_verify_trt.py:120` 明确写的是 “`pipeline` means the one-step vision->prefix->denoise chain.”
  - `scripts/step5_verify_trt.py:739-845` 也确实只比较单次 `vision -> prefix -> denoise -> v_t`，并没有比较完整 `predict_action_chunk()` 的 10-step denoise rollout，更没有比较 `select_action()` 的长期缓存行为。

为什么这是阻塞项：

- 这意味着“`Stage 5 pass`”最多只能说明：当前 engine 集合与 export-fidelity 边界在单步链路上数值接近。
- 它不能单独证明：
  - 完整 chunk 的 10 次 denoise 迭代累计后仍然稳定。
  - 当前 CUDA runtime/autocast 路径不会出现额外漂移。
  - `select_action()` 的 chunk cache 刷新/复用路径在 FP16 下没有额外行为差异。
- 如果文档继续把 `Stage 5 pass` 表述成 FP16 真机候选的主要 gate，就很容易把“导出边界正确”误解为“运行时边界正确”。

建议：

- 保留 `Stage 5`，但把它重新命名或描述成“导出边界正确性 gate”，不要单独承包“真机候选”的含义。
- 在文档里追加至少一个独立 gate：
  - `predict_action_chunk()` 级别的数值 smoke。
  - 或 `select_action()` 纯推理路径的行为正确性检查。
  - 或一个明确声明“runtime/autocast correctness is not covered by Stage 5”的红字限制。

3. `“FP16 工件/FP16 engine”` 这个命名现在过满，但计划里没有定义“什么算可验证的 FP16”。

证据：

- `scripts/build_pi_trt_engine.py:224-241` 对 `fp16` 做的关键动作是 `config.set_flag(trt.BuilderFlag.FP16)`，以及可选的 `force_fp32_layer_types`；这里没有任何“逐层有效精度”回读逻辑。
- 同文件 `:275-289` 输出的 build report 也主要是：
  - `precision` 字段
  - `precision_constraints`
  - `builder_capabilities`
  - `network_tensors`
  - `engine_summary`
  这些都不足以证明关键 kernel 确实以 FP16 在跑。
- 现有 BF16 build 结果已经说明“请求低精度 build”不等于“工件对外表现成低精度 I/O”，更不等于“关键层一定低精度执行”：
  - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/stage4_build_engines_bf16.json:12`、`:47`、`:159`、`:1372` 都写了 `precision=bf16`。
  - 但同文件 `:63-76`、`:175-216`、`:874-900`、`:1388-1418` 又显示 vision/prefix/denoise 的大量 I/O 和 cache tensor dtype 依然是 `DataType.FLOAT`。

为什么这是阻塞项：

- 如果文档继续把结果直接命名成“FP16 版本”“FP16 engine 成功/失败”，团队很容易误以为这已经证明了关键路径真的吃到了 FP16，而不是：
  - 仅仅打开了 FP16 builder flag。
  - 或者得到的是“混合精度 + 若干 FP32 fallback”的 engine。
- 这会直接污染后面的性能归因和失败归因。

建议：

- 在文档里把术语收紧成 `fp16-enabled build` 或“请求 precision=fp16 的 engine 集合”。
- 如果想继续用“FP16 工件”，那就必须同时补一句：当前报告证明的是“builder 请求与工件 provenance”，不是“逐层有效 FP16 执行率”。
- 对 `--force-fp32-layer-types` 命中的情况，必须单独分支命名，不能和“干净 FP16 rebuild”混称。

## 高风险问题

1. `prefix_cache 是当前性能黑洞` 这个结论方向大体对，但表述仍然容易把问题归因错到“prefix_cache 内部算子”，而忽略当前实现路径的边界成本。

证据：

- 基线 benchmark 确实显示 `prefix_cache` 最慢：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_20260314/benchmark_report.md:43-50` 中 `prefix_cache` 为 `63.059 ms`，明显高于 `denoise_step` 的 `4.650 ms`。
- 但 proposal 自己在 `FP16_TECHNICAL_PROPOSAL.md:82` 已经把公式写成 `2 * vision + prefix_cache + N * denoise_step + engine boundary overhead`，又在 `:114-115` 说 runtime 每段 `infer()` 都有显式边界和同步成本。

风险：

- 现有 `prefix_cache` 时间里混着至少三类东西：
  - 子图内部计算本身。
  - KV/pad mask 的大批量张量物化与搬运。
  - 子图 runner 边界与同步成本。
- 如果文档后续直接把它简称成“prefix_cache 结构太差”，容易把后续优化路线错误地收缩到模型结构，而忽略 runtime/IO contract 才可能是大头。

建议：

- 这句话可以保留，但要改成“当前工程实现路径下，`prefix_cache` 阶段总成本最高”。
- 不要默认等价成“prefix LM compute 本身就是唯一根因”。

2. 单一 deterministic batch + 固定 token length 的 gate 太弱，文档虽然写了限制，但还没有把它提升成“遗漏的失败场景”。

证据：

- Proposal 在 `FP16_TECHNICAL_PROPOSAL.md:200-204` 固定 benchmark 口径，在 `:266-271` 也承认当前结果只对 `static-shape + batch=1 + fixed token length` 和 deterministic baseline batch 成立。
- 当前 benchmark 报告也明确这样写了：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_20260314/benchmark_report.md:60-69`。
- 现有 `Stage 5` 也是围绕单个 runtime context 构造输入，而不是覆盖更广的输入分布。

风险：

- FP16 在当前单 batch 下通过，不代表它在以下场景不会出问题：
  - 语言 token 分布更极端。
  - 接近最大 token length 的 prompt。
  - 图像内容导致激活范围异常。
  - prefix KV 更接近数值极限时的 NaN/Inf。
- 这件事尤其和当前 hotspot `prefix_cache` 直接相关，因为 prefix 路径是最重、张量也最大的部分。

建议：

- 不一定要把这轮扩成大规模测试，但至少要在风险清单里补一句：当前 gate 不覆盖输入分布鲁棒性，只覆盖固定 baseline batch。

3. `PyTorch AMP` 这个对照组当前实际是 `CUDA bfloat16 autocast`，不是 `PyTorch FP16`，如果不写清楚很容易被误读成同精度对比。

证据：

- `scripts/benchmark_pi_inference.py:70-73` 里 `torch_use_amp` 实际开启的是 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`。
- 基线元数据也反复记录 runtime reference contract 是 `autocast_dtype=bfloat16`：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839/pi_trt_metadata.json:51-54`、`:73-76`、`:142-145`。
- 当前 1000-step 报告把 backend 名字写成 `pytorch_amp`：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_20260314_160853/report.md:11-14`，但没有在表格标题里说明 AMP 在这里具体是 BF16。

风险：

- 读者会自然把 `TRT FP16 vs PyTorch AMP` 理解成“FP16 对 FP16”，但这里其实是“TRT fp16-enabled build 对 Torch BF16 autocast runtime”。
- 这个误读会直接污染横向结论，尤其在文档把 AMP 当成关键对照组时。

建议：

- 后续所有文档和报告里，`PyTorch AMP` 至少应改写成 `PyTorch AMP (CUDA BF16 autocast)`。

4. 现有验收阈值更像工程政策，而不是由当前数据直接支持的技术阈值。

证据：

- Proposal 在 `FP16_TECHNICAL_PROPOSAL.md:248-260` 给了 `10% / 15% / 20%` 的多档阈值。
- 但当前 chunk benchmark 只是 `30` 次迭代：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_20260314/benchmark_report.md:12-19`。
- 1000-step 纯推理报告当前也只是单次总时长汇总：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_20260314_160853/report.md:3-14`。

风险：

- 这些阈值如果被写成“技术上合理的硬标准”，会让后续结论带上并不存在的客观性。
- 现在能支持的最多是“工程上先暂定的 acceptance target”，而不是“已经被统计稳定性证明过的阈值”。

建议：

- 阈值可以保留，但要明确写成“第一轮工程政策阈值”，不是“由当前实验显著性直接推出的科学阈值”。

5. `--force-fp32-layer-types` 被保留为逃生口，但文档还没有把它视为一条需要单独解释的分叉结果。

证据：

- `scripts/build_pi_trt_engine.py:211-241` 支持按 layer type 强制 FP32。
- 实施计划在 `FP16_IMPLEMENTATION_PLAN.md:531-532` 只要求“如果必须使用，就在 build report 和文档中说明”。

风险：

- 一旦实际 FP16 rebuild 只能靠若干 FP32 escape hatch 才通过，最终得到的就不是 proposal 叙事里的“纯 FP16 rebuild”。
- 如果这类 run 仍然和“干净 FP16 run”共用同一套验收叙事，后续所有性能归因都会变脏。

建议：

- 需要在文档里提前定义：
  - `clean fp16-enabled run`
  - `fp16-with-fp32-escapes run`
  这两类结果必须分开命名、分开总结、分开比较。

## 建议保留但加注释的表述

1. `“当前真正的性能黑洞是 prefix_cache。”`

建议保留，但后面要补一句限制：

- 这是“当前 static-shape、batch=1、固定 token length、当前 runner 路径下”的阶段总耗时判断。
- 它不自动等价于“prefix_cache 内部模型结构就是唯一根因”，因为 proposal 自己已经承认还存在 `engine boundary overhead`。

2. `“如果性能变好了，能明确归因给 precision rebuild；如果没变好，说明核心问题还在 prefix_cache 结构本身。”`

建议保留方向，但改弱语气：

- 可以写成“如果在保持 ONNX/export contract 不变的前提下性能变好，初步支持 precision rebuild 带来收益”。
- 以及“如果没有显著改善，只能说明在当前 build/runtime path 下，fp16-enabled rebuild 没解决主要瓶颈；不能直接排除 tactic 选择、FP32 fallback、I/O 边界成本等因素”。

3. `“TRT FP16 vs PyTorch AMP”`

建议保留这个对照组，但必须在首次出现时加注释：

- 这里的 `PyTorch AMP` 在当前仓库脚本里是 `CUDA BF16 autocast`，不是 `Torch FP16`。
- 所以这个对照组回答的是“TRT fp16-enabled build vs 当前 Torch 混合精度运行时”，不是“同精度 apples-to-apples”。

4. `“Stage 5 通过后，FP16 可以作为新的默认 TensorRT 部署候选。”`

建议保留这个工程目标，但要在同一句里补充分层条件：

- `Stage 5` 只覆盖 export-fidelity 边界。
- 若要谈“默认部署候选”，还应同时满足 chunk benchmark、1000-step 纯推理、以及至少一个 runtime-oriented smoke 不异常。

5. `“强成功：pipeline_chunk 接近或超过当前 PyTorch FP32。”`

建议保留为业务导向目标，但加一句：

- 这只是阶段性工程目标，不代表 TensorRT 路线已经达到最优，也不代表 `prefix_cache` 结构问题被根治。

## 总体判断

如果只看当前文档质量，最大的进步是：已经不再把 `FP16 rebuild` 写成“直接对空 run-dir 跑 Stage 4”的不可执行方案，而且对 provenance 的重视是对的。

但如果以“真实 FP16 rebuild 最危险的技术风险”来排序，我认为优先级最高的仍然是下面三件事：

1. 不要把“重新导出 ONNX 再 build”的实验，包装成未经证明的“纯 precision 单变量实验”。
2. 不要把 `Stage 5 pass` 误用成“运行时正确性已经被证明”的替代物。
3. 不要把“请求 precision=fp16 的 build”直接写成“已经证明关键路径有效 FP16 执行”的事实。

这三点如果不先在文档层面说清楚，后面即使跑出了一个看起来不错的 FP16 数字，结论依然很容易被质疑，而且这种质疑是合理的。
