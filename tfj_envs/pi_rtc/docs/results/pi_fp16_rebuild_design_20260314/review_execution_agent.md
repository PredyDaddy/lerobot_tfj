# FP16 实施计划“执行可落地性”审稿报告

## 1. 审查范围与结论

本次只核对以下两类内容：

1. `FP16_IMPLEMENTATION_PLAN.md` 里的命令、阶段顺序、预期产物，是否与当前代码实现一致。
2. 以下脚本在“run-dir / 上游 gate / precision provenance / Stage 4/5 可执行性”上的现状：
   - `scripts/step4_build_engines.py`
   - `scripts/build_pi_trt_engine.py`
   - `scripts/step5_verify_trt.py`
   - `scripts/run_pi05_trt_infer_so101.py`
   - `scripts/benchmark_pi_inference.py`

审查结论：

1. `阶段 4` 的最终命令链已经和当前代码对齐，前提是严格按 `Stage 2 -> Stage 3 -> Stage 4 -> Stage 5` 顺序在同一个新 `run-dir` 内执行。这里 `plan` 是对的，代码也确实这么要求。
2. `Stage 4` 不能对空 `run-dir` 直接执行，这一点是代码硬门禁，不是建议项。`plan` 在 `阶段 4` 的说明里已经写对了。
3. `Stage 4/5` 的 CLI 参数本身都存在，`benchmark_pi_inference.py` 也确实支持把整个 `run-dir` 作为 `--onnx-path` / `--trt-path` 输入。
4. 但 `precision provenance` 相关改动目前只完成了一部分。`step4_build_engines.py` 和 `build_pi_trt_engine.py` 已经有基础 precision 信息；`step5_verify_trt.py`、`benchmark_pi_inference.py`、`run_pi05_trt_infer_so101.py` 还没有达到 `plan` 承诺的“显式、统一、可消费”的程度。
5. `阶段 2 / 阶段 3 / 阶段 5` 里所有依赖 `scripts/benchmark_pi_select_action.py` 的命令当前都会直接失败，因为这个脚本在仓库里不存在。仓库里只有历史结果目录 `docs/results/pi_select_action_1000steps_20260314_160853/`，没有对应脚本入口。

换句话说：

1. `Stage 4/5` 主链路具备落地条件。
2. `1000-step pure inference` 这条链路当前还不具备落地条件。
3. `plan` 里关于 precision provenance 的验收口径，大部分还没有被当前代码真正实现。

## 2. 与当前代码一致、可以执行的部分

### 2.1 新 `run-dir` 方案是可行的

`plan` 在 `FP16_IMPLEMENTATION_PLAN.md:155-170` 明确写了：

1. 新 `FP16_RUN` 内先跑 `Stage 2`
2. 再跑 `Stage 3`
3. 再跑 `Stage 4 --precision fp16`
4. 再跑 `Stage 5`

这和当前代码一致：

1. `scripts/step4_build_engines.py:165-258` 会先检查 `stage2_export_onnx` 和 `stage3_verify_onnx` gate。
2. gate 不通过时，`Stage 4` 会直接写失败 report 并退出 `1`，不会进入 build。
3. `scripts/common.py:73-84` 的 `prepare_run_layout()` 支持绝对路径 `run-dir`，所以 `docs/results/pi_model_fp16_YYYYMMDD_HHMMSS` 这种命名方式没有问题。

结论：`plan` 现在写的 `Stage 2 -> 5` 顺序是正确且必要的，不能再退回“对空目录直接跑 Stage 4”的写法。

### 2.2 `Stage 4` 命令本身与当前 CLI 一致

`plan` 的 `阶段 4` 命令：

```bash
python scripts/step4_build_engines.py \
  --run-dir "$FP16_RUN" \
  --precision fp16 \
  --device cuda:0
```

和当前 CLI 一致：

1. `scripts/step4_build_engines.py:30-64` 定义了 `--run-dir`、`--precision`、`--device`。
2. `python scripts/step4_build_engines.py --help` 当前可正常显示。
3. `scripts/build_pi_trt_engine.py:224-227` 明确支持 `precision == "fp16"`，并在 `platform_has_fast_fp16 == False` 时硬失败。
4. `scripts/build_pi_trt_engine.py:185-189` 明确要求 CUDA 设备，`--device cuda:0` 是正确用法。

结论：只要同一 `run-dir` 里已经有 `Stage 2/3` 产物，这条 `Stage 4` 命令是可执行的。

### 2.3 `Stage 5` 命令本身与当前 CLI 一致

`plan` 的 `阶段 4` 后续命令：

```bash
python scripts/step5_verify_trt.py \
  --policy-path "$POLICY" \
  --run-dir "$FP16_RUN" \
  --device cuda:0
```

和当前 CLI 一致：

1. `scripts/step5_verify_trt.py:93-121` 定义了 `--policy-path`、`--run-dir`、`--device`。
2. `python scripts/step5_verify_trt.py --help` 当前可正常显示。
3. `scripts/step5_verify_trt.py:390-417` 会从 `run-dir` 默认解析 `artifacts/onnx` 和 `artifacts/engines`。
4. `scripts/step5_verify_trt.py:410` 对 `--policy-path` 做了 `resolve_checkpoint_dir()`，因此 `pretrained_model` 路径能被正确归一化。

结论：在 `Stage 4` 已经产出 engine 的前提下，这条 `Stage 5` 命令是可执行的。

### 2.4 chunk benchmark 命令本身可执行

`plan` 的 chunk benchmark 命令：

```bash
python scripts/benchmark_pi_inference.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP16_RUN" \
  --trt-path "$FP16_RUN" \
  --warmup-iterations 10 \
  --iterations 30 \
  --output-dir "$FP16_CHUNK"
```

和当前实现是匹配的：

1. `scripts/benchmark_pi_inference.py:826-842` 定义了这些参数。
2. `scripts/run_pi05_onnx_infer_so101.py:303-350` 的 `resolve_onnx_artifacts()` 支持把整个 `run-dir` 作为 `--onnx-path`。
3. `scripts/run_pi05_trt_infer_so101.py:456-563` 的 `resolve_trt_artifacts()` 支持把整个 `run-dir` 作为 `--trt-path`。

结论：这条 chunk benchmark 命令从“命令是否能跑起来”的角度是成立的。

## 3. 阻塞项

### 阻塞项 1：`scripts/benchmark_pi_select_action.py` 缺失，导致 1000-step 命令链全部不可执行

这是当前最直接的阻塞。

仓库现状：

1. `scripts/benchmark_pi_select_action.py` 文件不存在。
2. `test -f scripts/benchmark_pi_select_action.py` 当前结果为 `missing`。
3. 仓库里只有历史结果目录 `docs/results/pi_select_action_1000steps_20260314_160853/`，没有对应脚本源码。

因此，`plan` 里以下命令按当前代码都会失败：

1. `FP16_IMPLEMENTATION_PLAN.md:252-259`

```bash
python -m py_compile \
  scripts/benchmark_pi_inference.py \
  scripts/run_pi05_trt_infer_so101.py \
  scripts/benchmark_pi_select_action.py
```

失败原因：

1. `scripts/benchmark_pi_select_action.py` 不存在，`py_compile` 会直接报文件不存在。

2. `FP16_IMPLEMENTATION_PLAN.md:257-259`

```bash
python scripts/benchmark_pi_select_action.py --help
```

失败原因：

1. 脚本不存在，Python 无法打开该文件。

3. `FP16_IMPLEMENTATION_PLAN.md:303-309`
4. `FP16_IMPLEMENTATION_PLAN.md:408-414`

```bash
python scripts/benchmark_pi_select_action.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP32_RUN" \
  --trt-path "$FP32_RUN" \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir "$FP32_RECHECK_SELECT"
```

以及：

```bash
python scripts/benchmark_pi_select_action.py \
  --policy-path "$POLICY" \
  --onnx-path "$FP16_RUN" \
  --trt-path "$FP16_RUN" \
  --steps 1000 \
  --warmup-steps 100 \
  --output-dir "$FP16_SELECT"
```

失败原因相同：

1. 脚本文件缺失。

结论：

1. `plan` 中所有与 `1000-step pure inference` 直接相关的“验证命令”和“执行命令”目前都不可落地。
2. 这不是文档措辞问题，而是执行入口缺失。

### 阻塞项 2：`Stage 5` 的 precision provenance 还没有落到 `plan` 要求的程度

`plan` 对 `scripts/step5_verify_trt.py` 的要求，在 `FP16_IMPLEMENTATION_PLAN.md:51-55`、`112-120` 写得很明确，至少要显式绑定：

1. 当前验证的 engine set
2. run-dir
3. checkpoint_dir
4. precision / variant
5. 对应 Stage 4 report 路径

当前代码现状：

1. `scripts/step5_verify_trt.py:871-909` 的 report 顶层已经有：
   - `policy_dir`
   - `run_dir`
   - `onnx_dir`
   - `engine_dir`
   - `artifact_paths`
2. 但同一段代码里没有：
   - 显式 `precision`
   - 显式 `variant` / `trt_variant`
   - 显式 `stage4_report_path`
3. `scripts/step5_verify_trt.py:913-935` 写 metadata 时，也没有把“这次验证对应的 precision / Stage 4 report 路径”写成稳定字段。

这意味着：

1. `Stage 5` 命令能跑。
2. 但 `Stage 5` 的输出还不能满足 `plan` 所要求的“显式区分 FP16 / FP32 验证”的标准。
3. 后续 benchmark 和 runtime 想直接消费 `Stage 5` 的 precision provenance，当前字段不够。

结论：

1. 这不会阻止 `Stage 5` 执行。
2. 但会阻止 `plan` 在 provenance 维度完成验收。

### 阻塞项 3：`benchmark_pi_inference.py` 目前不能满足 `plan` 对 provenance 和 fail-fast 的要求

`plan` 对 `scripts/benchmark_pi_inference.py` 的要求在 `FP16_IMPLEMENTATION_PLAN.md:54-55`、`134-137`，要求 benchmark 输出直接写出：

1. `variant`
2. `precision`
3. `metadata_path`
4. `stage4_report_path`
5. `stage5_report_path`
6. `checkpoint_dir`
7. 工件不一致时 fail fast

当前代码现状：

1. `scripts/benchmark_pi_inference.py:811-821` 的 TRT artifact summary 只写：
   - `engine_dir`
   - 三个 engine 路径
   - `metadata_path`
2. `scripts/benchmark_pi_inference.py:930-950` 的最终 report 只把这份 `trt_artifacts` 原样放进去。
3. 当前没有把以下字段显式写进 benchmark report：
   - `precision`
   - `variant`
   - `stage4_report_path`
   - `stage5_report_path`
   - `checkpoint_dir`
4. 当前 benchmark 只调用 `resolve_trt_artifacts()`（`scripts/benchmark_pi_inference.py:708`），不会调用 `run_pi05_trt_infer_so101.py` 里的 `assess_trt_artifact_safety()` 那套更强的 provenance/Stage 5 安全检查。

这带来两个问题：

1. benchmark 输出里不能直接看出自己测的是 `fp32` 还是 `fp16`。
2. benchmark 不会像 real-robot launcher 那样检查：
   - `stage5_verify_trt == pass`
   - build report precision 是否一致
   - build report / stage5 report / metadata 是否来自同一套工件

结论：

1. chunk benchmark 命令可以跑。
2. 但 `plan` 预期的 benchmark provenance 输出和 fail-fast 行为，目前并没有被实现。
3. `FP16_IMPLEMENTATION_PLAN.md:274-278` 里“report 中能直接看到 TRT 的 precision/variant/run-dir 来源字段”这一预期，按当前代码不成立。

### 阻塞项 4：`run_pi05_trt_infer_so101.py` 的启动摘要和默认工件选择还不够 precision-aware

`plan` 对 real-robot launcher 的要求在 `FP16_IMPLEMENTATION_PLAN.md:55-56`、`134-137`，核心是：

1. 启动摘要必须直接打印 resolved TRT precision / variant / run-dir
2. 多 precision 共存时，不能让操作员混淆

当前代码现状：

1. `scripts/run_pi05_trt_infer_so101.py:565-784` 的 `assess_trt_artifact_safety()` 已经做了很多正确的事：
   - 检查 metadata
   - 检查 checkpoint_dir
   - 检查 Stage 5 status
   - 检查 build report 状态
   - 检查 build report precision 是否彼此一致
2. 但是这段逻辑只把“是否存在 precision 不一致”当成拦截条件，没有把“resolved precision 是什么”作为稳定字段返回。
3. `scripts/run_pi05_trt_infer_so101.py:954-988` 的启动摘要会打印：
   - metadata path
   - metadata stage_status
   - build report status
   - stage5 report path
   - stage5 overall_status
   但不会直接打印：
   - `resolved_precision=fp16/fp32`
   - `resolved_variant=...`
4. `scripts/run_pi05_trt_infer_so101.py:132-156` 的默认工件发现逻辑只按：
   - `variant == "pi05"`
   - `checkpoint_dir` 存在
   - `stage5_verify_trt == pass`
   - `created_at` 最新
   来选默认工件，并没有 precision 维度。
5. 同文件 `:163` 的 CLI 描述字符串仍然写着 `Run PI0.5 FP32 TensorRT inference...`。

这意味着：

1. 一旦同时存在 `verified FP32 run` 和 `verified FP16 run`，默认自动选择会变得不透明。
2. 即使脚本没有放过不安全工件，操作员仍然不能从启动摘要里一眼看出当前用的是 `fp16` 还是 `fp32`。

结论：

1. 这不阻塞脚本启动。
2. 但它阻塞了 `plan` 在“避免 FP32/FP16 混看”和“真机摘要清晰可审计”这两个目标上的完成度。

## 4. 非阻塞改进项

### 4.1 `Stage 4` 的部分 provenance 其实已经有了，`plan` 可以写得更精确

当前代码并不是“完全没有 precision 信息”：

1. `scripts/build_pi_trt_engine.py:269-289` 的单 engine build report 已经有：
   - `precision`
   - `precision_constraints`
   - `allow_tf32`
   - `device`
   - `timing_cache`
   - `builder_capabilities`
2. `scripts/step4_build_engines.py:307-333` 的 Stage 4 总 report 已经有：
   - `run_dir`
   - `onnx_dir`
   - `engine_dir`
   - `build_settings.precision`
3. `scripts/step4_build_engines.py:343-353` 也已经把 `engine_build_settings` 和 `validation_gates.stage4_build_engines` 写进 metadata。

所以：

1. `Commit A` 不需要从零开始补 precision 字段。
2. 更准确的说法应该是：继续补齐“variant / Stage 4 到 Stage 5 的显式链接 / benchmark 可消费字段”。

### 4.2 benchmark 命令还依赖本地 tokenizer 条件，`plan` 没有明说

`scripts/benchmark_pi_inference.py:855-860` 会先跑 `ensure_pi_runtime_compatibility(require_local_tokenizer=True)`。

而 `scripts/common.py:285-325` 明确要求：

1. 本地 offline tokenizer 能被发现
2. 否则 benchmark 会在正式测时延前直接报错退出

建议：

1. `plan` 的命令区至少补一句前置条件，说明需要本地 tokenizer 可用。

### 4.3 `Stage 5` 不检查 `Stage 4 pass`，它只是直接消费 artifacts

当前只有 `Stage 4` 对 `Stage 2/3` 做了硬 gate。

`scripts/step5_verify_trt.py` 当前行为是：

1. 直接解析 ONNX / engine 路径
2. 如果文件缺失或执行失败，就把对应 subgraph 标成 `error`
3. 最终 `overall_status != pass` 时退出 `1`

这不算 bug，但要说清楚：

1. `Stage 5` 没有像 `Stage 4` 那样的上游 gate。
2. 它更像一个“执行并验收”的消费者，而不是“先检查 stage4 状态再决定是否跑”。

## 5. 建议的修订内容

### 5.1 对 `plan` 文本的修订建议

建议把下面几处写得更准确：

1. 在 `阶段 2`、`阶段 3`、`阶段 5` 的命令块前明确写出：`scripts/benchmark_pi_select_action.py` 需要先落地，否则这些命令不可执行。
2. 在 `阶段 2` 的“预期结果”里，不要先写“report 中能直接看到 TRT 的 precision/variant/run-dir 来源字段”，因为当前代码还做不到；应改成“这是 Commit B 完成后的预期，不是当前仓库现状”。
3. 在 benchmark 相关章节补一条前置条件：本地 tokenizer 必须可用，否则 `benchmark_pi_inference.py` 会在兼容性检查阶段直接失败。
4. 在 real-robot 相关说明里，在 precision-aware 摘要真正落地之前，文档应要求显式传 `--trt-path`，不要依赖默认自动发现。

### 5.2 对代码改动顺序的修订建议

建议把 `Commit B` 的最小落地顺序写成下面这样：

1. 先新增 `scripts/benchmark_pi_select_action.py`
2. 再补 `benchmark_pi_inference.py` 的 provenance 输出和 fail-fast
3. 再补 `run_pi05_trt_infer_so101.py` 的 precision-aware 摘要
4. 最后再跑 `FP32 recheck` 和 `FP16` benchmark

原因很简单：

1. 当前真正阻塞执行的是缺脚本，不是字段命名。
2. 如果先讨论 benchmark 文档而不先补执行入口，`plan` 的阶段 3 / 5 仍然无法落地。

### 5.3 对字段设计的修订建议

建议新增并统一消费下面这组字段，避免继续靠“猜路径名”识别 precision：

1. `resolved_precision`
2. `resolved_variant`
3. `checkpoint_dir`
4. `stage4_report_path`
5. `stage5_report_path`
6. `source_run_dir`

建议来源：

1. `Stage 4` 产出这些字段
2. `Stage 5` 复制并确认这些字段
3. benchmark 与 runtime 只读这一套，不自己猜

### 5.4 对 benchmark / launcher 的共享校验建议

当前最强的 provenance 校验逻辑其实在 `run_pi05_trt_infer_so101.py` 里。

建议：

1. 把 `resolve_trt_artifacts()` + `assess_trt_artifact_safety()` 抽成 benchmark/launcher 共用的校验入口。
2. `benchmark_pi_inference.py` 不要只做“能解析到 engine 就跑”，而要至少复用 launcher 那套：
   - build report 一致性检查
   - Stage 5 pass 检查
   - metadata / build report / stage5 report 同源检查

## 6. 最终判断

按当前仓库状态，我对 `FP16_IMPLEMENTATION_PLAN.md` 的“执行可落地性”判断如下：

1. `Stage 4/5` 的主链路可以落地，前提是严格保持新 `run-dir` 内的 `Stage 2 -> 5` 顺序。
2. `run-dir` 设计和 `Stage 4` 上游 gate 设计是对齐的，这部分没有原则性问题。
3. `precision provenance` 还没有落地到 `plan` 期望的程度，尤其是 `Stage 5`、benchmark、launcher 三个消费端。
4. `1000-step pure inference` 这条链路当前不可执行，因为 `scripts/benchmark_pi_select_action.py` 缺失。

因此，本计划当前状态更准确的表述应该是：

1. `Stage 4/5` 可执行
2. chunk benchmark 可执行
3. 1000-step benchmark 不可执行
4. provenance 验收暂时不可通过

如果只允许先处理最关键的阻塞，我建议优先级是：

1. 先补 `scripts/benchmark_pi_select_action.py`
2. 再补 `step5_verify_trt.py` / `benchmark_pi_inference.py` / `run_pi05_trt_infer_so101.py` 的 precision provenance
3. 最后再跑 `FP32 recheck`、`FP16` rebuild 和对比文档
