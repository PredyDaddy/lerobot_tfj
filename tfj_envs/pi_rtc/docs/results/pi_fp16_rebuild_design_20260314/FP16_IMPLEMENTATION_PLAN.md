# PI0.5 TensorRT FP16 重建实施计划

基线方案：`/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_rebuild_design_20260314/FP16_TECHNICAL_PROPOSAL.md`

本文档只把主技术方案落成实施计划，不重新做架构决策。所有步骤都以“当前三段子图架构不变，先做第一轮 `TensorRT FP16 build path` 重建实验”为前提。

## 1. 目标与范围

### 1.1 目标

本次实施的唯一目标是把当前 PI0.5 的 TensorRT 路径重建出一套可验证、可追溯、可复测的 `fp16-enabled` 工件集合，并用统一口径完成以下闭环：

1. 新 run-dir 内重新完成 `Stage 2 -> Stage 5`，保证 `fp16-enabled` 工件集合是自洽的，不混入旧 run 的报告。
2. 保持当前三段子图边界不变：
   - `vision_encoder`
   - `prefix_cache`
   - `denoise_step`
3. 加强 build / verify / benchmark / runtime 的 provenance，避免 `FP32` 与 `FP16` 工件被静默混用。
4. 用两类实测 benchmark 评价 FP16：
   - `pipeline_chunk` benchmark
   - `1000-step pure inference select_action` benchmark
5. 只在 `Stage 5` 通过后，才允许进入后续 runtime benchmark 与候选评估；`Stage 5` 本身不是完整 runtime correctness 或真机 correctness 的充分条件。

### 1.2 范围内

本计划范围内允许改动的内容：

1. `Stage 4` build 报告与 metadata 字段补强
2. `Stage 5` verify 报告与 metadata 字段补强
3. benchmark 输出补强与纯推理 benchmark 固化
4. real-robot TRT launcher 的工件自检与启动摘要
5. `README.md` 与 `INFERENCE_COMPARISON.md` 的结果补充
6. 新 FP16 run 的导出、验证、benchmark 结果落盘
7. 文档和报告中显式区分 `requested precision` 与 `effective precision evidence`

### 1.3 范围外

以下内容不在本轮实施内，禁止混入同一 patch：

1. 三段子图重新拆分或合并
2. `prefix_cache` 结构重写
3. 单 engine 全图实验
4. 真机控制回路重构
5. 任何把数值阈值先放宽再说的 shortcut

## 2. 代码改动清单

下面列的是实施阶段允许改动的代码文件、目的、改动边界，以及必须分开提交的原因。

| 提交批次 | 文件 | 必改内容 | 不允许顺手做的事 | 提交边界要求 |
| --- | --- | --- | --- | --- |
| Commit A | `scripts/build_pi_trt_engine.py` | 补齐单 engine build report 中的 precision/provenance 字段。至少明确记录：请求 precision、实际 builder flags、force-fp32 layer 约束、timing cache 路径、variant，以及可用的 effective precision evidence 摘要。 | 不改网络边界，不改 TensorRT runtime 行为。 | 必须单独提交。它定义了后续所有消费端的字段基线。 |
| Commit A | `scripts/step4_build_engines.py` | 统一 Stage 4 顶层 report 与 metadata 的 precision/variant/run-dir/checkpoint_dir/engine_dir 表达；保证每个子图 report 和总 report 都能明确说明是 `fp16` 还是 `fp32`。 | 不把 benchmark、runtime launcher、README 修改混进来。 | 必须和 `build_pi_trt_engine.py` 一起提交，不得和后续 benchmark 消费端混提。 |
| Commit A | `scripts/step5_verify_trt.py` | 在 Stage 5 report 与 metadata 中显式落盘“验证的是哪一套 engine、来自哪个 run-dir、precision 是什么、Stage 4 report 在哪里”；并明确 `Stage 5` 只是 export-boundary correctness gate。默认阈值先保持不变。 | 不提前修改阈值，不把 benchmark 逻辑塞进 verify。 | 必须和 Commit A 同批提交。 |
| Commit B | `scripts/benchmark_pi_inference.py` | 读取并输出 TRT 工件 provenance，报告里显式写出 `variant`、`precision`、`metadata_path`、`stage4_report_path`、`stage5_report_path`、`checkpoint_dir`；如果发现 run-dir 混工件，直接 fail fast。 | 不改变计时边界，不改 benchmark 的 warmup/iterations 语义。 | 必须与 Commit A 分开提交，因为它依赖 A 中新增字段。 |
| Commit B | `scripts/run_pi05_trt_infer_so101.py` | 启动摘要中打印 resolved TRT precision/variant/run-dir；对 metadata、Stage 5、engine 目录不一致场景继续拦截或显式警告。 | 不改真机控制策略，不加新的运动逻辑。 | 必须与 Commit A 分开提交；否则无法区分 schema 问题和 runtime 问题。 |
| Commit B | `scripts/benchmark_pi_select_action.py` | 将现有“1000-step pure inference select_action only”实验固化为 repo 脚本。输出 `report.json` 与 `report.md`，并在结果中明确区分 `pytorch_fp32`、`pytorch_amp_bf16`、`onnx_cuda_runtime`、`tensorrt_fp32/fp16`。 | 不把机器人执行逻辑掺进来；该脚本只能做纯推理，不允许驱动机械臂。 | 必须与 Commit A 分开提交；如果新增文件，放在 Commit B。 |
| Commit C | `README.md` | 补充最终可执行的 FP16 导出/验证/benchmark/真机命令，并注明 run-dir 约束与上机前 gate。 | 不在没有实测结果前先写结论。 | 必须在所有数字跑完后单独提交。 |
| Commit C | `INFERENCE_COMPARISON.md` | 追加本轮实测的 FP16 vs FP32 对比，保留原有说明，同时新增基于真实测量的 chunk 与 1000-step 结果。 | 不用估算数据，不抄旧报告数字冒充新测量。 | 必须在所有数字跑完后单独提交。 |

### 2.1 必须分开提交的最小切分

至少分成以下 3 个提交：

1. `Commit A: Stage 4 / Stage 5 / 单 engine build report 的 provenance 改动`
2. `Commit B: benchmark / 纯推理 benchmark / runtime launcher 的消费端改动`
3. `Commit C: 实测结果与文档更新`

不要把 `Commit A` 和 `Commit B` 合并。原因很直接：如果一次性混提，后面出现“字段不对”“benchmark 误读”“launcher 误判”时，无法快速判断问题在 schema 生产端还是消费端。

## 3. 分阶段实施步骤

### 阶段 0：锁定基线与命名约定

目的：

1. 锁定本轮对照基线，避免 FP16 跑完后拿历史不同口径结果做比较。
2. 约定本轮结果目录命名，避免覆盖已验证的 FP32 run。

固定基线：

1. 现有 verified TRT FP32 run：
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
2. 现有 chunk benchmark 参考：
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_20260314`
3. 现有 1000-step pure inference 参考：
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_20260314_160853`

本轮新目录命名：

1. `FP16_RUN=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_YYYYMMDD_HHMMSS`
2. `FP32_RECHECK_CHUNK=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_YYYYMMDD_HHMMSS`
3. `FP16_CHUNK=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_YYYYMMDD_HHMMSS`
4. `FP32_RECHECK_SELECT=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp32_recheck_YYYYMMDD_HHMMSS`
5. `FP16_SELECT=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_YYYYMMDD_HHMMSS`

注意：

1. 不覆盖 `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839`
2. 不在老 run-dir 里直接重写 `stage4_build_engines.json` 或 `stage5_verify_trt.json`

### 阶段 1：提交 Commit A，补强 Stage 4/5 的 provenance

实施顺序：

1. 修改 `scripts/build_pi_trt_engine.py`
2. 修改 `scripts/step4_build_engines.py`
3. 修改 `scripts/step5_verify_trt.py`
4. 先做静态校验，再提交 `Commit A`

本阶段必须完成的具体改动：

1. 单 engine build report 要能独立说明“这个 engine 是什么、怎么 build 的、precision 是什么”。
2. Stage 4 总 report 要能从顶层直接看见 precision 与 artifact path，而不需要读每个子图 report 才知道。
3. Stage 5 report 要能明确绑定：
   - 当前验证 run-dir
   - 当前 engine_dir
   - 当前 checkpoint_dir
   - 当前 precision/variant
   - 对应 Stage 4 report 路径
4. metadata 必须能让 benchmark 与 runtime launcher 可靠读取同一套信息。
5. Stage 4 / Stage 5 的文案必须区分“requested precision”与“effective precision evidence”。

### 阶段 2：提交 Commit B，补强 benchmark 与 runtime 消费端

实施顺序：

1. 修改 `scripts/benchmark_pi_inference.py`
2. 修改 `scripts/run_pi05_trt_infer_so101.py`
3. 固化 `scripts/benchmark_pi_select_action.py`
4. 对现有 FP32 工件做轻量 smoke，确认新字段能被正确消费
5. 提交 `Commit B`

本阶段必须完成的具体改动：

1. `benchmark_pi_inference.py` 输出里必须能直接看到 TRT 工件来源和 precision。
2. `run_pi05_trt_infer_so101.py` 启动前必须打印 resolved artifact summary，避免操作员把 FP32/FP16 混看。
3. `benchmark_pi_select_action.py` 必须是纯推理脚本，不接机器人、不读串口、不下发动作。
4. benchmark 消费端必须在工件 provenance 不一致时 fail fast，而不是继续给出“看上去正常”的数字。
5. 所有 benchmark 文档必须把 `PyTorch AMP` 明确写成 `PyTorch AMP (CUDA BF16 autocast)`。

### 阶段 3：用新代码先重跑一遍 FP32 基线 benchmark

目的：

1. 建立与新 schema 完全一致的对照组。
2. 避免直接把 FP16 新数字和旧脚本时代的 FP32 数字混比。

实施顺序：

1. 保持现有 verified FP32 TRT run 不变
2. 用新 benchmark 脚本重跑 chunk benchmark
3. 用新 pure inference 脚本重跑 1000-step select_action benchmark
4. 保存新一轮 `fp32_recheck` 结果目录

本阶段不新增 engine，不改旧工件，只新增 benchmark 结果目录。

### 阶段 4：在全新 run-dir 内重建自洽的 FP16 工件

这是本轮最关键的阶段。这里必须注意一个实现约束：

`step4_build_engines.py` 会检查 `Stage 2/3` gate，所以不能对一个空 run-dir 直接执行 `Stage 4`。正确顺序必须是：

1. 在新的 `FP16_RUN` 中执行 `Stage 2`
2. 在同一个 `FP16_RUN` 中执行 `Stage 3`
3. 在同一个 `FP16_RUN` 中执行 `Stage 4 --precision fp16`
4. 在同一个 `FP16_RUN` 中执行 `Stage 5`

这样可以保证：

1. `Stage 2/3/4/5` 的报告全部落在同一个 FP16 run-dir
2. metadata 与 validation gate 自洽
3. 后续 benchmark 与 launcher 不会误把旧 FP32 报告认成新 FP16 工件

### 阶段 5：跑 FP16 benchmark，并与 FP32 recheck 对照

实施顺序：

1. 跑 `FP16 chunk benchmark`
2. 跑 `FP16 1000-step pure inference`
3. 与 `阶段 3` 的 `FP32 recheck` 对照
4. 如果 Stage 5 未通过，禁止进入本阶段
5. 即使 Stage 5 通过，也必须再经过 chunk benchmark、1000-step pure inference 和 runtime provenance 自检，才讨论“真机候选”

输出要求：

1. chunk benchmark 单独出目录
2. 1000-step pure inference 单独出目录
3. 文档只引用本轮实际生成的结果目录，不引用手工摘抄的数字

### 阶段 6：提交 Commit C，更新 README 与对比文档

实施顺序：

1. 先整理本轮实测数字
2. 再修改 `README.md`
3. 再修改 `INFERENCE_COMPARISON.md`
4. 最后提交 `Commit C`

本阶段的文档必须明确区分：

1. `pipeline_chunk` 反映 chunk 刷新路径成本
2. `1000-step select_action` 反映均摊纯推理吞吐
3. 离线 benchmark 不等于真机闭环 wall-clock
4. `Stage 5` 只代表 export-boundary correctness，不代表完整 runtime correctness

## 4. 每阶段验证命令与预期产物

所有命令都在 `conda activate lerobot` 后执行，工作目录固定为：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/pi_trt
```

建议先固定公共环境变量：

```bash
export POLICY=/data/tfj/lerobot_tfj/pi_model/pretrained_model
export FP32_RUN=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_consistency_20260313_182839
export FP16_RUN=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_$(date +%Y%m%d_%H%M%S)
export FP32_RECHECK_CHUNK=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp32_recheck_$(date +%Y%m%d_%H%M%S)
export FP16_CHUNK=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_$(date +%Y%m%d_%H%M%S)
export FP32_RECHECK_SELECT=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp32_recheck_$(date +%Y%m%d_%H%M%S)
export FP16_SELECT=/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_$(date +%Y%m%d_%H%M%S)
```

### 阶段 1 验证

验证命令：

```bash
python -m py_compile \
  scripts/build_pi_trt_engine.py \
  scripts/step4_build_engines.py \
  scripts/step5_verify_trt.py

python scripts/step4_build_engines.py --help
python scripts/step5_verify_trt.py --help
```

预期结果：

1. 无语法错误
2. CLI 参数正常显示
3. 代码 review 能确认新增字段没有破坏原有 `Stage 4/5` 主流程

落盘产物：

1. 无 GPU 实测工件要求
2. Git 提交产物为 `Commit A`

### 阶段 2 验证

验证命令：

```bash
python -m py_compile \
  scripts/benchmark_pi_inference.py \
  scripts/run_pi05_trt_infer_so101.py \
  scripts/benchmark_pi_select_action.py

python scripts/benchmark_pi_inference.py --help
python scripts/run_pi05_trt_infer_so101.py --help
python scripts/benchmark_pi_select_action.py --help
```

推荐轻量 smoke：

```bash
python scripts/benchmark_pi_inference.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP32_RUN" \
  --trt-path "$FP32_RUN" \
  --warmup-iterations 0 \
  --iterations 1 \
  --output-dir "$FP32_RECHECK_CHUNK"
```

预期结果：

1. `benchmark_report.json` 和 `benchmark_report.md` 成功生成
2. report 中能直接看到 TRT 的 precision/variant/run-dir 来源字段
3. 若故意传入不一致工件，脚本应 fail fast 或打印明确阻断信息
4. 报告中的 `PyTorch AMP` 对照组明确标注为 `CUDA BF16 autocast`

落盘产物：

1. `FP32_RECHECK_CHUNK/benchmark_report.json`
2. `FP32_RECHECK_CHUNK/benchmark_report.md`
3. Git 提交产物为 `Commit B`

### 阶段 3 验证

chunk benchmark 命令：

```bash
python scripts/benchmark_pi_inference.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP32_RUN" \
  --trt-path "$FP32_RUN" \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir "$FP32_RECHECK_CHUNK"
```

1000-step pure inference 命令：

```bash
python scripts/benchmark_pi_select_action.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP32_RUN" \
  --trt-path "$FP32_RUN" \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir "$FP32_RECHECK_SELECT"
```

预期结果：

1. 生成新的 FP32 recheck 结果，而不是覆盖旧目录
2. `FP32_RECHECK_CHUNK` 下包含：
   - `benchmark_report.json`
   - `benchmark_report.md`
3. `FP32_RECHECK_SELECT` 下包含：
   - `report.json`
   - `report.md`
4. 两类 benchmark 的设置项与输出字段都带 provenance
5. pure inference 报告中不得把 `pytorch_amp` 误表述成 `torch fp16`

落盘产物：

1. `FP32_RECHECK_CHUNK`
2. `FP32_RECHECK_SELECT`

### 阶段 4 验证

Stage 2：

```bash
python scripts/step2_export_onnx.py \
  --policy-path "$POLICY" \
  --run-dir "$FP16_RUN"
```

Stage 3：

```bash
python scripts/step3_verify_onnx.py \
  --policy-path "$POLICY" \
  --run-dir "$FP16_RUN"
```

Stage 4：

```bash
python scripts/step4_build_engines.py \
  --run-dir "$FP16_RUN" \
  --precision fp16 \
  --device cuda:0
```

Stage 5：

```bash
python scripts/step5_verify_trt.py \
  --policy-path "$POLICY" \
  --run-dir "$FP16_RUN" \
  --device cuda:0
```

预期结果：

1. `FP16_RUN/stage2_export_onnx.json` 存在且 `overall_status=pass`
2. `FP16_RUN/stage3_verify_onnx.json` 与 `FP16_RUN/stage3_verify_onnx.md` 存在且通过
3. `FP16_RUN/stage4_build_engines.json` 存在，`build_settings.precision=fp16`
4. `FP16_RUN/artifacts/engines/` 下存在三段 engine 及其 build report：
   - `pi_shared_vision_encoder.engine`
   - `pi_shared_prefix_cache.engine`
   - `pi05_denoise_step.engine`
   - `vision_encoder_build_report.json`
   - `prefix_cache_build_report.json`
   - `denoise_step_build_report.json`
5. `FP16_RUN/stage5_verify_trt.json` 与 `FP16_RUN/stage5_verify_trt.md` 存在且 `overall_status=pass`
6. `FP16_RUN/pi_trt_metadata.json` 中的 `stage_status`、`validation_gates`、`engine_build_settings` 与 run-dir 内容一致

落盘产物：

1. `FP16_RUN/pi_trt_metadata.json`
2. `FP16_RUN/stage2_export_onnx.json`
3. `FP16_RUN/stage3_verify_onnx.json`
4. `FP16_RUN/stage3_verify_onnx.md`
5. `FP16_RUN/stage4_build_engines.json`
6. `FP16_RUN/stage5_verify_trt.json`
7. `FP16_RUN/stage5_verify_trt.md`
8. `FP16_RUN/artifacts/onnx/*`
9. `FP16_RUN/artifacts/engines/*`

### 阶段 5 验证

FP16 chunk benchmark：

```bash
python scripts/benchmark_pi_inference.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP16_RUN" \
  --trt-path "$FP16_RUN" \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir "$FP16_CHUNK"
```

FP16 1000-step pure inference：

```bash
python scripts/benchmark_pi_select_action.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP16_RUN" \
  --trt-path "$FP16_RUN" \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir "$FP16_SELECT"
```

预期结果：

1. `FP16_CHUNK/benchmark_report.json` 与 `benchmark_report.md` 成功生成
2. `FP16_SELECT/report.json` 与 `report.md` 成功生成
3. `FP16_CHUNK` 中的 TRT 结果明确标注为 `fp16`
4. `FP16_SELECT` 中的 TensorRT backend 名称明确标注为 `tensorrt_fp16`
5. 与 `FP32_RECHECK_CHUNK`、`FP32_RECHECK_SELECT` 的对照可以直接进行，不需要手工猜测 precision

落盘产物：

1. `FP16_CHUNK/benchmark_report.json`
2. `FP16_CHUNK/benchmark_report.md`
3. `FP16_SELECT/report.json`
4. `FP16_SELECT/report.md`

### 阶段 6 验证

文档更新后，至少检查：

```bash
rg -n "fp16|pipeline_chunk|1000-step|select_action|prefix_cache" README.md INFERENCE_COMPARISON.md
```

预期结果：

1. `README.md` 有完整的 FP16 导出、验证、benchmark、上机命令
2. `INFERENCE_COMPARISON.md` 同时保留原说明和本轮新增实测数字
3. 文档中明确说明：
   - `TRT FP16` 是否优于 `TRT FP32`
   - `prefix_cache` 是否改善
   - `1000-step pure inference` 是否回退

落盘产物：

1. 文档修改对应的 `Commit C`

## 5. 回退策略

回退只针对本轮新增改动，按提交边界倒序回退，不碰其他人的历史改动。

### 5.1 Commit A 回退条件

满足任一条件就回退 `Commit A`：

1. `Stage 4/5` 因 schema 变更直接失效
2. metadata 与 report 字段冲突，导致 benchmark 或 launcher 无法解析
3. 无法在不改架构的前提下修正字段设计

回退方式：

1. 只回退 `Commit A`
2. 保留工作目录中其他非本轮改动
3. 不删除已有 FP32 verified run

### 5.2 Commit B 回退条件

满足任一条件就回退 `Commit B`，保留 `Commit A`：

1. benchmark 消费端无法正确解析新增 provenance
2. launcher 误判安全工件为不安全，或反过来放过了混工件
3. 纯推理 benchmark 脚本行为不稳定，无法稳定复现已有 1000-step 指标

回退方式：

1. 只回退 `Commit B`
2. 不回退 `Commit A`
3. 先恢复为“schema 可写、消费端保守”的状态，再单独修复 benchmark/launcher

### 5.3 FP16 run 回退条件

满足任一条件，停止把 FP16 作为候选默认工件：

1. `Stage 5` 失败
2. `pipeline_chunk` 相比当前 `TRT FP32` 没有可接受改善
3. `1000-step pure inference` 出现明显回退
4. provenance 检查不能稳定证明 FP16 工件自洽

处置方式：

1. 保留 `FP16_RUN` 结果目录用于复盘，不删除失败证据
2. 真机默认继续使用当前 verified FP32 工件
3. 下一轮工作转入主方案定义的 `Phase B`，也就是围绕 `prefix_cache` 的结构性优化，而不是继续在本轮 patch 内堆改动

### 5.4 文档回退条件

如果最终数字未完成或结论不成立：

1. 不提交 `Commit C`
2. `README.md` 和 `INFERENCE_COMPARISON.md` 不提前写入未验证结论

## 6. 风险检查清单

开始实施前和每个阶段结束后都要复查以下清单。

### 6.1 run-dir 与工件一致性

1. `Stage 4` 不能对空 run-dir 直接执行
2. `FP16_RUN` 必须是本轮新目录，不能复用旧 verified FP32 run
3. 同一个 benchmark 只能消费同一 run-dir 下的一套工件
4. `pi_trt_metadata.json`、`stage4_build_engines.json`、`stage5_verify_trt.json` 三者的 precision/run-dir/checkpoint_dir 必须一致

### 6.2 benchmark 口径一致性

1. FP32 recheck 与 FP16 必须使用同一代码版本
2. 两边必须使用同一 GPU
3. 两边必须使用同一 `warmup_iterations` / `iterations`
4. 两边必须使用同一 `steps` / `warmup_steps`
5. 两边必须使用同一 `num_inference_steps`
6. `pipeline_chunk` 和 `1000-step select_action` 不允许混为一个指标解释

### 6.3 数值与验收风险

1. 第一轮先保持 `Stage 5` 默认阈值不变
2. 若 FP16 数值漂移超阈值，先记录失败，不要先调阈值掩盖问题
3. 若必须使用 `--force-fp32-layer-types`，必须在 build report 中明确落盘并在文档中说明

### 6.4 操作风险

1. 纯推理 benchmark 脚本禁止驱动机器人
2. 真机 launcher 只有在 `Stage 5 pass + provenance 自检通过` 后才允许使用 FP16 工件
3. `README.md` 中的上机命令必须使用经过 Stage 5 验证的 run-dir

### 6.5 文档风险

1. 不能把旧 benchmark 目录里的数字冒充本轮结果
2. 文档必须同时写出“哪项变快了”和“哪项没有变快”
3. 文档必须明确说明当前瓶颈判断仍然是 `prefix_cache`，除非本轮新测量推翻这一点

## 7. 完成判定

只有同时满足以下条件，本实施计划才算完成：

1. `Commit A`、`Commit B`、`Commit C` 都已按边界落地
2. 新的 `FP32 recheck` 与 `FP16` benchmark 结果都已落盘
3. `FP16_RUN` 已完成 `Stage 2 -> Stage 5`
4. `Stage 5` 对 FP16 工件判定为 `pass`
5. `README.md` 与 `INFERENCE_COMPARISON.md` 已写入真实实测数字和解释边界
6. 真机使用说明引用的是新的、已验证的工件路径，而不是历史默认路径
