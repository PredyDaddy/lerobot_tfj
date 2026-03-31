# PI0.5 FP16 不能作为默认部署工件的原因

## 结论先行

当前这套 FP16 工件之所以不能作为默认真机部署工件，最硬的原因不是“TRT build 失败”，而是：

- `Stage 4` 只是 build 成功，不是数值正确性证明
- `Stage 5` 明确失败，而且失败主要集中在 `prefix_cache` 的 KV cache 生成
- 这类失败不是 ONNX 导出边界问题，而是 TensorRT FP16 / mixed-precision 执行路径把 KV cache 算坏了
- 当前 launcher 默认是 fail-close，`Stage 5 != pass` 的工件不会被当成安全工件上机

当前最准确的说法应该是：

- 这套 FP16 工件没有通过本仓库默认的安全部署 gate
- 因此现在不能作为默认真机部署工件

而不是更强但证据不足的说法：

- “已经证明它在任何情况下都绝对不能上机”

## 直接证据链

当前 FP16 run：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759`

状态：

- `Stage 2 = pass`
- `Stage 3 = pass`
- `Stage 4 = pass`
- `Stage 5 = fail`

关键报告：

- [stage4_build_engines.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage4_build_engines.json)
- [stage5_verify_trt.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/stage5_verify_trt.json)
- [pi_trt_metadata.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_20260314_172759/pi_trt_metadata.json)

最关键的数值对比：

- `prefix_cache torch_vs_onnx`
  - `max_abs_diff = 4.0626526e-04`
  - `mean_abs_diff = 7.4735099e-06`
  - `min_cosine_similarity = 0.99999976`
- `prefix_cache torch_vs_trt`
  - `max_abs_diff = 12.22293663`
  - `mean_abs_diff = 0.34146175`
  - `min_cosine_similarity = 0.53000039`
- `prefix_cache onnx_vs_trt`
  - 和 `torch_vs_trt` 几乎一样差

这说明：

- Torch 和 ONNX 本身是对得上的
- 漂移是在 TRT engine 里引入的

## 为什么主 blocker 是 `prefix_cache`

当前 `Stage 5` 不是只有 `prefix_cache` 失败，`vision_encoder`、`denoise_step`、`pipeline` 也都没过当前阈值。

但 `prefix_cache` 是最严重的失败点：

- `vision_encoder torch_vs_trt`
  - `max_abs_diff = 0.99919128`
  - `mean_abs_diff = 0.00883492`
- `denoise_step torch_vs_trt`
  - `max_abs_diff = 0.02522635`
  - `mean_abs_diff = 0.00082151`
- `pipeline torch_vs_trt`
  - `max_abs_diff = 0.06319416`
  - `mean_abs_diff = 0.00202311`
- `prefix_cache torch_vs_trt`
  - `max_abs_diff = 12.22293663`
  - `mean_abs_diff = 0.34146175`
  - `min_cosine_similarity = 0.53000039`

更具体地看 `prefix_cache`：

- `prefix_pad_masks` 是完全正确的
- 36 个 KV 输出里，只有 1 个非 KV 的 mask output 通过
- 36 个 KV 输出全部失败
- 18 层 transformer 的 key/value 都明显漂移，不是只有个别层出问题

也就是说：

- I/O 契约没坏
- 掩码没坏
- 真正坏的是 KV cache 浮点张量本身

## 为什么 `Stage 4 pass` 仍然会 `Stage 5 fail`

代码里已经把这个边界写得很清楚了。

在 [build_pi_trt_engine.py](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/build_pi_trt_engine.py) 里，`Stage 4` 能证明的是：

- ONNX 可以被 TRT parser 接受
- 静态 shape 合法
- builder flag 设置成功
- engine 能被序列化出来

它不能证明的是：

- 每个敏感层都真的按期望精度执行
- TensorRT 融合后的 kernel 内部累加精度是我们想要的
- build 前设置的 layer precision 约束在 build 后仍完整生效

build report 自己也明确写了：

- 只证明 requested precision、builder flags、forced-fp32 constraints、visible engine I/O dtypes
- 不保证 per-layer effective execution precision

## 当前 FP16 约束为什么不够

这次 FP16 run 的构建约束是：

- `force_fp32_layer_types = REDUCE ELEMENTWISE UNARY`

它确实命中了很多层：

- `vision_encoder = 475`
- `prefix_cache = 2057`
- `denoise_step = 1788`

但这并不代表真正最敏感的 attention 主路径已经被兜住。

`prefix_cache` ONNX 子图里最重要的算子族包括：

- `MatMul = 156`
- `Softmax = 17`
- `Mul = 351`
- `ReduceMean = 35`
- `Sqrt = 35`
- `Div = 35`

当前策略只强制了：

- `REDUCE`
- `ELEMENTWISE`
- `UNARY`

没有直接覆盖：

- `MATRIX_MULTIPLY`
- `SOFTMAX`
- 可能的融合 kernel 内部精度路径

## 我刚做的定向 probe 结果

我新建了一个本地 probe run：

- `/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_prefix_probe_mm_softmax_20260314_180013`

只重建 `prefix_cache`，把约束改成：

- `REDUCE ELEMENTWISE UNARY MATRIX_MULTIPLY SOFTMAX`

对应报告：

- [stage4_build_engines.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_prefix_probe_mm_softmax_20260314_180013/stage4_build_engines.json)
- [stage5_verify_trt.json](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_model_fp16_prefix_probe_mm_softmax_20260314_180013/stage5_verify_trt.json)

这个 probe 的结果非常关键：

- build 仍然 `pass`
- `prefix_cache` 仍然 `fail`
- 原始 FP16：
  - `max_abs_diff = 12.22293663`
  - `mean_abs_diff = 0.34146175`
  - `min_cosine_similarity = 0.53000039`
- probe 后：
  - `max_abs_diff = 12.20645714`
  - `mean_abs_diff = 0.32431307`
  - `min_cosine_similarity = 0.52971250`

这意味着：

- 单纯把 `MatMul/Softmax` 也拉回 FP32，只有非常有限的改善
- 最坏层基本没有本质收敛
- `prefix_cache` 仍然是 36/37 输出失败

这个结果反而把原因进一步坐实了：

- 问题不是“我们只是忘了加两个 layer type”
- 更像是 TRT 在 build 优化和 kernel 融合之后，真正执行的 attention/KV 路径并没有被当前这套 layer-type 级约束可靠控制住

另一个很重要的旁证是：

- 原始 `prefix_cache` probe engine `num_layers = 197`
- 加 `MATRIX_MULTIPLY/SOFTMAX` 后变成 `num_layers = 264`
- 但数值几乎没本质改善

这说明即使图被拆得更保守，当前 engine 里仍然存在主导误差的执行路径。

## 更准确的根因判断

结合本地报告、代码和 probe，当前更可信的根因是：

1. `prefix_cache` 的 ONNX 导出边界没有坏
2. 当前 FP16 TRT build 主要问题发生在 `prefix_cache` 的 transformer / attention / KV 生成数值路径
3. 现有的 `force_fp32_layer_types` 是 build 前、按解析后 `LayerType` 打标的约束
4. 这套约束不足以保证 build 后 fused kernel 的真实内部精度与累加精度
5. 因此会出现：
   - `Stage 4 pass`
   - 但 `Stage 5 prefix_cache` 仍然灾难性漂移

## 为什么现在不能默认部署

不是因为“FP16 一点价值都没有”，恰恰相反，性能很好：

- 安全 `TRT FP32 pipeline_chunk = 123.501 ms`
- `unsafe TRT FP16 pipeline_chunk = 50.665 ms`
- 安全 `TRT FP32 select_action = 2.490646 ms/step`
- `unsafe TRT FP16 select_action = 1.015324 ms/step`

但默认部署看的是“安全工件”，不是只看速度。

当前 launcher 的默认策略是：

- `Stage 5 pass` 才能作为安全 TRT 工件上机
- `Stage 5 fail` 的工件只能在显式 `allow-unsafe` 的语境下做诊断 benchmark

所以现在不能默认部署，不是因为它慢，而是因为它还没有通过 correctness gate。

## 下一步最有价值的方向

如果继续往“可部署 FP16”推进，优先级应该是：

1. 不要再泛泛调全链路，集中打 `prefix_cache`
2. 不要再假设“多加几个 layer type 就够了”，要开始验证 fused kernel / tactic 级别的问题
3. 优先尝试：
   - `NORMALIZATION`
   - `CUMULATIVE`
   - 更强的 anti-fusion / tactic 限制策略
   - 必要时退回 hybrid 路线：`prefix_cache` 保持 FP32，其它段继续探索 FP16
4. 如果要宣称“其实 action 已经够用”，必须补 action-level 数值比较，而不是只看吞吐 benchmark

## 相关分析文件

- [agent_prefix_cache_report.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_root_cause_20260314/agent_prefix_cache_report.md)
- [agent_build_precision_report.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_root_cause_20260314/agent_build_precision_report.md)
- [agent_critic_report.md](/data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi_fp16_root_cause_20260314/agent_critic_report.md)
