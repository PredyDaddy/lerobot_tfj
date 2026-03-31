# FP16 不能部署原因挑刺审查报告

## 审查范围

本报告只基于本地仓库现有代码、报告和 benchmark 结果，不使用任何外部资料。

本次专门挑三类最容易犯的错：

1. 把 `Stage 5 gate` 和真机 `deployability` 混为一谈
2. 把问题简单归因成“只有 `prefix_cache` 的 raw KV 对比有问题”，却没有检查最终 action 级证据
3. 在要宣称“当前 FP16 不能部署”时，没有给出足够硬的证据链

## 核心结论

1. 现在可以下的最硬结论，不是“FP16 一定不能在机器人上工作”，而是“当前这套 FP16 工件没有资格作为本仓库默认的安全部署工件”。
2. 原因不是单一一句“`prefix_cache` 坏了”。本地证据显示 `prefix_cache` 是最严重的数值漂移源，但 `vision_encoder`、`denoise_step`、`pipeline` 也都没有通过当前 `Stage 5` 阈值。
3. 反过来，现有本地仓库也没有给出“最终 action 已经足够可用”的硬证据，因为当前没有任何一个报告真正比较了 `Torch` 与 `TRT FP16` 的完整 `predict_action_chunk()` 或 `select_action()` 数值误差。
4. 如果要写“不能部署”，最严谨的说法应该是：
   - “当前 `FP16` 工件未通过本仓库的 `Stage 5` 正确性 gate，因此按现有 fail-closed policy，不应作为默认真机部署工件。”
   - 不能直接升级成：
   - “已经证明它的最终动作一定错误，所以绝对不能上机。”

## 1. `Stage 5 gate` 和真机 deployability 是否被混为一谈

### 1.1 结论

有这个风险，而且这是当前最容易过度表述的地方。

### 1.2 本地证据

`Stage 5` 自己已经把作用范围写得很清楚：

- `scripts/step5_verify_trt.py` 明确写了注释性说明：
  - `Stage 5 is an export-boundary single-step correctness gate. It does not by itself prove full runtime correctness or multi-step chunk stability.`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.md` 也明确写了：
  - `stage5_scope: export-boundary single-step correctness gate`

这说明 `Stage 5` 的定义是：

- 单步
- export-boundary
- correctness gate

它没有直接覆盖下面这些更强命题：

- 多步 iterative denoise 后的完整 `action chunk` 是否仍然稳定
- `select_action()` 长时调用后是否仍然稳定
- 真机相机采集、串口、限幅、sleep、控制回路是否稳定

### 1.3 但为什么当前代码又会拦截真机

因为仓库把 `Stage 5 pass` 用作了“默认 live-robot 使用”的 fail-closed policy。

本地代码证据：

- `scripts/run_pi05_trt_infer_so101.py` 中，若 `artifact_safety.is_safe == False` 且未显式传 `--allow-unsafe-trt-artifacts`，会直接拒绝真机使用。
- 同一文件里，`stage5_verify_trt overall_status must be 'pass' for live robot use` 被写成了阻断条件。

所以当前正确表述应该是：

- `Stage 5 fail` 足以推出“按本仓库默认安全策略，不允许真机部署”
- `Stage 5 fail` 不足以单独推出“已经证明这套 FP16 最终动作一定不可用”

### 1.4 最容易写错的话

下面这种说法太强：

- “因为 `Stage 5 fail`，所以这套 FP16 已经被证明不能部署。”

更严谨的版本应该是：

- “因为 `Stage 5 fail`，所以这套 FP16 在当前仓库的 fail-closed policy 下不是默认可部署工件。”

## 2. 有没有证据表明问题只在 `prefix_cache` raw KV，而最终 action 可能已经够用

### 2.1 结论

当前本地证据既不支持“问题只在 `prefix_cache`”，也不支持“最终 action 已经证明够用”。

当前最安全的结论只能是：

- `prefix_cache` 是最严重的单步数值漂移源
- 但不是唯一 failing 点
- 最终 action 是否足够可用，仓库里现在没有硬证据

### 2.2 为什么不能说“问题只在 prefix_cache”

`FP16` 这套 run 的 `Stage 5` 失败并不只发生在 `prefix_cache`。

文件：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json`
- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.md`

当前阈值：

- `max_abs_diff <= 0.001`
- `mean_abs_diff <= 0.0001`
- `min_cosine_similarity >= 0.999`

本地实测：

- `vision_encoder` 失败
  - `torch_vs_trt max_abs_diff = 0.999191`
  - `mean_abs_diff = 0.00883492`
- `prefix_cache` 失败
  - `torch_vs_trt max_abs_diff = 12.2229`
  - `mean_abs_diff = 0.341462`
  - `min_cosine_similarity = 0.530000`
- `denoise_step` 失败
  - `torch_vs_trt max_abs_diff = 0.0252264`
  - `mean_abs_diff = 0.000821515`
- `pipeline` 失败
  - `torch_vs_trt max_abs_diff = 0.0631942`
  - `mean_abs_diff = 0.00202311`
  - `min_cosine_similarity = 0.999923`

所以严格说法应该是：

- `prefix_cache` 是最坏的一段
- 但 `Stage 5` 的失败不是“只剩 prefix_cache 一项没过”

### 2.3 为什么也不能说“最终 action 可能已经够用，所以 prefix_cache 不重要”

这个结论同样证据不足。

本地仓库当前缺的正是 action 级比较：

- `scripts/benchmark_pi_select_action.py` 的结果只记录时间和 `last_action_shape`
- 它没有记录 `Torch` vs `TRT` 的 action 数值误差
- `scripts/benchmark_pi_inference.py` 的 `pipeline_chunk` benchmark 也只给各后端自己的输出摘要，不做跨后端 action diff
- `scripts/trt_pi_adapter.py` 的 `run_preflight()` 只检查：
  - 输出 shape 对不对
  - 输出是否 finite
  - `timestep` 是否活着
- 它不检查 `Torch` 与 `TRT` 的 action 等价性

尤其要注意：

- `Stage 5 pipeline` 比较的是单步 `v_t`
- 它不是完整多步采样后的 `predict_action_chunk()`
- 更不是闭环 `select_action()` 的长期 rollout

所以现在没有本地证据能证明：

- “虽然 raw KV 漂了，但最终 action 已经足够接近 Torch”

### 2.4 但为什么很多人会误以为 action 已经没问题

因为当前有两类数字很容易让人误判：

第一类是高 cosine：

- `pipeline torch_vs_trt min_cosine_similarity = 0.999923`

这会让人产生“看起来已经很像了”的感觉。

第二类是 benchmark 输出摘要相近：

- `unsafe FP16` 的 `pipeline_chunk` 输出摘要里，`PyTorch` 和 `TensorRT` 的 `max_abs` / `mean_abs` 看起来很接近

但这都不是 action diff。

它们不能回答：

- 每一个 action 维度到底偏了多少
- 多步采样后误差会不会累积
- 经过 postprocess、smoothing、delta clamp 以后会不会跨过安全边界

### 2.5 当前能成立的中间结论

当前本地证据支持下面这句，但只支持到这句为止：

- `prefix_cache` raw KV drift 是当前 `FP16` 最显著的单步 export-boundary mismatch，而且它和当前性能收益高度相关

当前本地证据不支持下面这句：

- “所以最终 action 一定已经够用”

也不支持下面这句：

- “所以最终 action 一定不够用”

## 3. 如果要宣称“不能部署”，最硬的证据链应该是什么

### 3.1 先区分两种不同强度的结论

#### A. 工程发布结论

“当前不能作为默认安全部署工件。”

这个结论，仓库里已经有足够硬的证据链。

#### B. 行为正确性结论

“当前最终动作已经被证明不可信，因此实际上不能上机。”

这个结论，仓库里现在还没有足够硬的证据链。

### 3.2 对于 A，仓库里已经具备的硬证据链

要支持“不能作为默认安全部署工件”，当前本地证据已经足够：

1. `FP16` run 自洽存在
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`
   - `Stage 2 = pass`
   - `Stage 3 = pass`
   - `Stage 4 = pass`
   - `Stage 5 = fail`

2. benchmark 报告明确记录这套工件是 `unsafe`
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_inference_benchmark_fp16_unsafe_20260314_174221/benchmark_report.json`
   - `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_select_action_1000steps_fp16_unsafe_20260314_174221/report.json`
   - 两者都记录了 `stage5_report_status = fail`

3. 当前 live-robot launcher 默认会拦截它
   - `scripts/run_pi05_trt_infer_so101.py`
   - 若没有 `--allow-unsafe-trt-artifacts`，会拒绝使用这套工件

4. 即使 benchmark 允许测它，也必须显式加 `--allow-unsafe-trt-artifacts`
   - `scripts/benchmark_pi_inference.py`
   - `scripts/benchmark_pi_select_action.py`

这条证据链已经足够支撑下面这句：

- “当前 FP16 不是本仓库默认可发布、默认可上机的安全工件。”

### 3.3 对于 B，当前仓库还缺的硬证据链

如果要更强地宣称：

- “它的最终动作已经不可信，所以实际上不能部署”

那至少还缺下面几层证据，而这些在本地仓库里当前都没有现成结果：

1. 完整 `predict_action_chunk()` 数值对比
   - 同一 observation
   - 同一 noise
   - 同一 `num_inference_steps`
   - 直接比较 `Torch` vs `TRT FP16` 的 action chunk
   - 记录 `max_abs_diff`、`mean_abs_diff`、`cosine_similarity`

2. 完整 `select_action()` 数值对比
   - 不是只记 `last_action_shape`
   - 而是逐步记录 action drift

3. 多 batch 或离线 replay 证据
   - 不是只拿单个导出参考 batch
   - 而是在一批真实 observation 上看 action 误差分布

4. 控制量级证据
   - 把 action 漂移映射到最终控制量
   - 判断是否超过当前脚本里的 `joint_delta_limit`、`gripper_delta_limit` 或 smoothing 后的安全边界

5. 最好再加一层真机 shadow evidence
   - 即使不真正执行，也至少做同 observation 的 side-by-side action logging

没有这条更强证据链时，下面这种结论属于过度外推：

- “因为 `prefix_cache` raw KV 没过 gate，所以最终 action 一定不能用。”

## 4. 当前最容易犯的三种表述错误

### 错误一

- “`Stage 5 fail` = 已经证明真机不可部署。”

挑刺意见：

- 错。
- 它只严格推出“当前仓库默认安全策略下不可发布/不可默认上机”。

### 错误二

- “问题只在 `prefix_cache` raw KV，对最终 action 没影响。”

挑刺意见：

- 证据不足。
- 因为 `vision_encoder`、`denoise_step`、`pipeline` 也都 fail。
- 更关键的是，仓库里没有完整 action diff 报告。

### 错误三

- “既然 `unsafe FP16` 的 `pipeline_chunk` 和 `select_action` 都快很多，所以它应该已经可以部署。”

挑刺意见：

- 这是把性能结论偷换成正确性结论。
- 本地实测确实显示它很快：
  - `pipeline_chunk`: `123.501 ms -> 50.665 ms`
  - `prefix_cache`: `63.335 ms -> 13.348 ms`
  - `1000-step select_action`: `2.491 ms -> 1.015 ms`
- 但这些结果同时明确带着：
  - `stage5_report_status = fail`
- 所以它们只能证明“性能潜力强”，不能证明“当前可以安全部署”。

## 5. 建议采用的最终表述

如果要写到对外技术结论里，建议直接使用下面这种措辞：

- “当前这套 `fp16-enabled` TensorRT 工件已经表现出明显的性能潜力，但仍未通过本仓库定义的 `Stage 5` export-boundary single-step correctness gate，因此它目前不应作为默认真机部署工件。”
- “现有本地证据显示，`prefix_cache` raw KV drift 是最严重的单步数值异常，但仓库里尚无完整 action-chunk 或 offline replay 级别的对齐证据，因此还不能把‘Stage 5 fail’直接升级表述成‘最终动作一定不可用’。”

## 6. 审查员最终判词

一句话版本：

- 当前最硬、最不容易被反驳的结论不是“FP16 已被证明不能部署”，而是“当前 FP16 未通过本仓库默认部署 gate，所以不能作为默认安全部署工件；至于最终 action 是否已经不可信，仓库里还缺 action 级硬证据。”
