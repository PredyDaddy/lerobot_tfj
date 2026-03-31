# PI0.5 / TRT / RTC 项目复盘 PPT 生成提示词

## 使用方式

- 直接把下面的“完整版提示词”整体复制给支持生成 PPT 的 AI 工具。
- 如果目标工具上下文较短，再使用文末的“精简版提示词”。
- 这份提示词已经尽量把本轮项目里真正有价值的知识点、实测数据、踩坑、技术判断和落地命令都写全了。

---

## 完整版提示词

```text
你是一名资深技术内容策划 + 架构复盘作者，请基于以下素材，生成一份中文技术汇报 PPT。

任务目标：
- 生成一份 18 到 24 页的中文 PPT 内容方案
- 主题是：PI0.5 模型从 Safetensors 基线、ONNX / TensorRT 探索，到 RTC 真机落地的完整技术复盘
- 汇报对象同时包含：算法工程师、部署工程师、机器人现场联调人员、技术管理者
- 风格要求：专业、克制、结论清晰、时间线明确、强调“真实测量”和“工程判断”，不要写成营销稿

输出要求：
- 每一页都要给出：
  1. 页标题
  2. 页副标题
  3. 3 到 6 条核心 bullet
  4. 建议配图或图表形式
  5. 演讲者备注
- 最后额外给出：
  1. 一页“关键命令附录”
  2. 一页“经验教训与后续路线”
- 不要编造任何数据
- 如果某个结论没有被真实验证，请明确写“未验证”或“不能直接下结论”
- 必须准确区分：
  - 安全工件
  - 不安全但可用于诊断的工件
  - 离线纯推理 benchmark
  - 真机闭环运行

PPT 叙事主线建议：
1. 项目背景和目标
2. 参考文章给出的思路
3. 初始技术路线：Safetensors -> ONNX -> TensorRT
4. 一致性验证与 benchmark
5. 为什么 TensorRT 没有自然变快
6. 为什么 FP16 虽快但不能部署
7. 从 TRT/ONNX 方案转向 RTC
8. RTC 的原理和真正 blocker
9. 多智能体协作排查与修复
10. RTC 真机验证结果
11. 最终可执行命令
12. 经验总结与下一步

必须覆盖的核心知识点如下。

====================
一、项目背景
====================

这是一个围绕 LeRobot PI0.5 模型的真实工程复盘，场景是机器人真机部署与推理加速。核心工作路径经历了以下阶段：

1. 先使用 Safetensors / PyTorch 原生路径跑通模型与上机
2. 研究一篇“PI 系列模型的 TensorRT 推理加速”思路，尝试把 PI0.5 拆成子图做 ONNX / TRT
3. 做 Torch / ONNX / TensorRT 的一致性验证与速度对比
4. 发现 TensorRT 并没有自然得到更好的端到端效果，尤其安全 FP32 工件不一定比 PyTorch 快
5. 进一步尝试 FP16，但发现虽然速度很好，数值正确性不过关，不能作为默认部署工件
6. 最后把重心从 TRT/ONNX 转回 RTC（Real-Time Chunking）这条真正与现场控制更相关的方案
7. 修复 RTC 在 PI0.5 上机路径里的真实 blocker，并完成实机 smoke

一句话总结背景：
这个项目不是“单纯追求 benchmark 更快”，而是“在真实机器人上，让 PI0.5 既能跑、又稳定、又可解释地加速”。

====================
二、参考文章的核心思路
====================

可把参考文章概括为：

- 针对 PI 系列模型，不做整模型端到端黑盒转换，而是拆分关键子图
- 重点子图包括：
  - vision_encoder
  - prefix_cache
  - denoise_step
- 基本流程是：
  1. 检查 checkpoint
  2. 导出 ONNX
  3. 验证 Torch 与 ONNX 一致性
  4. 构建 TensorRT engine
  5. 验证 Torch 与 TRT 一致性
  6. 再接入运行时
- 这个思路的优点是边界清晰、每一步可验证
- 但难点在于：即便 engine build 成功，也不代表最终数值稳定，更不代表真机闭环一定更好

这一页要传达的重点：
“文章提供的是一条可操作的工程路线，不是自动成功的银弹。”

====================
三、PI0.5 推理链路的技术理解
====================

要解释清楚 PI0.5 这套系统到底在跑什么：

- PI0.5 是基于 flow-matching / denoising 的 chunk policy
- 一次生成的不是一个 action，而是一整个 action chunk
- 真实运行时会反复经历：
  - 图像编码
  - 文本 / 多模态 prefix 编码
  - prefix_cache / KV cache 生成
  - denoise_step 多次迭代
  - chunk queue 刷新与消费

建议强调两个容易混淆的概念：

1. `prefix_cache` 对应的是 PaliGemma 前缀部分的缓存生成
- 它本质上是图像 + 语言 prefix 进入 transformer 后形成的 KV cache
- 这些 prefix KV cache 会被后面的 denoise_step 反复复用
- 它不是“后处理”，而是 PI0.5 推理里的关键大头

2. `denoise_step` 只是单步迭代
- 不是完整 chunk
- 真正完整的 chunk latency 要看 `pipeline_chunk`

====================
四、早期基线路径
====================

早期基线是直接使用 Safetensors / PyTorch 跑通上机。

用户给出的基线路径命令如下，适合作为“项目起点”展示：

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

这页建议表达：

- 基线先保证“能跑”
- 后续所有加速路线都必须以这个可工作的基线为参考
- 工程上不能一开始就把真实部署路径打碎

====================
五、导出与验证链路
====================

导出 / 验证链路采用脚本化阶段式流程：

Stage 2：
```bash
python scripts/step2_export_onnx.py \
  --policy-path "$POLICY" \
  --run-dir "$RUN_DIR"
```

Stage 3：
```bash
python scripts/step3_verify_onnx.py \
  --policy-path "$POLICY" \
  --run-dir "$RUN_DIR"
```

Stage 4：
```bash
python scripts/step4_build_engines.py \
  --run-dir "$RUN_DIR" \
  --precision fp32 \
  --device cuda:0
```

Stage 5：
```bash
python scripts/step5_verify_trt.py \
  --policy-path "$POLICY" \
  --run-dir "$RUN_DIR" \
  --device cuda:0
```

必须讲清楚：

- Stage 4 pass 只说明 engine build 成功
- Stage 5 pass 才说明正确性 gate 基本通过
- “能 build” 和 “能安全部署” 是两回事

====================
六、模型一致性验证
====================

这一部分要说明：

- 已经对模型做过导出和一致性验证
- Torch / ONNX 在很多子图上是一致的
- 但 TensorRT 的 correctness 必须单独验证
- 不能把“导出成功”误当成“部署成功”

建议把一致性验证描述为：

- Torch vs ONNX：验证导出边界
- Torch vs TRT：验证 engine 执行正确性
- Pipeline level：验证整条推理链的误差能否接受

====================
七、真实速度对比：安全 FP32 工件
====================

必须使用下面这些真实测量值，不允许改写：

离线 `pipeline_chunk` 结果：

| Backend | pipeline_chunk mean_ms |
| --- | ---: |
| PyTorch FP32 | 95.468 |
| ONNX Runtime CUDA | 157.021 |
| TensorRT FP32 | 123.501 |

离线分阶段结果：

| Backend | vision_encoder_pair | prefix_cache | denoise_step |
| --- | ---: | ---: | ---: |
| PyTorch | 8.135 ms | 25.114 ms | 6.269 ms |
| ONNX | 15.712 ms | 63.256 ms | 7.621 ms |
| TensorRT FP32 | 12.760 ms | 63.335 ms | 4.670 ms |

1000-step pure inference 结果：

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 2894.332 | 2.894 | 345.503 |
| pytorch_amp_bf16 | 2971.220 | 2.971 | 336.562 |
| onnx_cuda_runtime | 3143.981 | 3.144 | 318.068 |
| tensorrt_fp32 | 2490.646 | 2.491 | 401.502 |

这里必须讲清楚一个非常关键的工程结论：

- TensorRT FP32 并不是“全链路更快”
- 在 `denoise_step` 上 TensorRT 确实更快
- 但 `prefix_cache` 在 TRT 路径上并没有比 PyTorch 快，反而接近 ONNX 的慢路径
- 所以最终的 `pipeline_chunk` 并没有赢过 PyTorch

要明确写出一句结论：
“局部子图加速，不等于端到端 chunk 推理更快，更不等于真实控制环更优。”

====================
八、为什么会出现“TensorRT 反而变慢”
====================

这一页必须回答一个直观疑问：为什么加速了反而没有变快？

建议答案如下：

1. TensorRT 真正加速最明显的是后面的 `denoise_step / action expert` 路径
2. 但 PI0.5 的前半段 `PaliGemma / prefix_cache` 仍然是大头
3. 安全 FP32 工件下，`prefix_cache` 在 TRT 上约 63 ms，而 PyTorch 只有约 25 ms
4. 因此即使后段变快，前段瓶颈没解决，最终 chunk 仍不占优
5. 这也是为什么不能只盯着单一子图 benchmark

请用一句容易理解的话总结：
“真正拖住端到端速度的，不是你已经加速的那一段，而是你还没有真正加速下来的前缀缓存主路径。”

====================
九、unsafe FP16：为什么看起来很快，但不能部署
====================

必须使用以下真实数据：

unsafe FP16 的 `pipeline_chunk`：

| Backend | pipeline_chunk mean_ms |
| --- | ---: |
| PyTorch FP32 | 94.973 |
| ONNX Runtime CUDA | 156.026 |
| TensorRT FP16 unsafe | 50.665 |

unsafe FP16 的 1000-step pure inference：

| Backend | total_time_ms | mean_per_step_ms | steps_per_s |
| --- | ---: | ---: | ---: |
| pytorch_fp32 | 2885.647 | 2.886 | 346.543 |
| pytorch_amp_bf16 | 2974.741 | 2.975 | 336.164 |
| onnx_cuda_runtime | 3123.643 | 3.124 | 320.139 |
| tensorrt_fp16 | 1015.324 | 1.015 | 984.907 |

但是必须立刻接上更重要的正确性结论：

- 这套 FP16 工件 Stage 4 pass，但 Stage 5 fail
- 真正失败集中在 `prefix_cache`
- Torch 和 ONNX 基本对得上，但 TRT FP16 明显漂移
- 所以它是“诊断用的快工件”，不是“可默认上机的安全工件”

请明确给出真实数值证据：

- `prefix_cache torch_vs_onnx`
  - `max_abs_diff = 4.0626526e-04`
  - `mean_abs_diff = 7.4735099e-06`
  - `min_cosine_similarity = 0.99999976`

- `prefix_cache torch_vs_trt`
  - `max_abs_diff = 12.22293663`
  - `mean_abs_diff = 0.34146175`
  - `min_cosine_similarity = 0.53000039`

这一页必须得出的结论：

- FP16 快是快，但没有通过 correctness gate
- 因此当前不能作为默认部署路线

====================
十、FP16 根因分析
====================

必须解释这几个判断：

1. 根因不是 ONNX 导出边界坏了
- 因为 Torch vs ONNX 对得上

2. 根因主要发生在 TensorRT FP16 的 `prefix_cache / KV cache` 数值路径

3. Stage 4 pass 不代表数值正确
- 它只说明 parser、shape、builder flags、engine serialization 成功

4. 即使额外把 `MATRIX_MULTIPLY` 和 `SOFTMAX` 强制拉回 FP32，问题也只得到非常有限改善

probe 后数据：
- 原始 FP16：
  - `max_abs_diff = 12.22293663`
  - `mean_abs_diff = 0.34146175`
  - `min_cosine_similarity = 0.53000039`
- probe 后：
  - `max_abs_diff = 12.20645714`
  - `mean_abs_diff = 0.32431307`
  - `min_cosine_similarity = 0.52971250`

所以更合理的工程判断是：

- 不是“忘记多加两个 layer type”这么简单
- 更像是 TRT build 优化和 kernel 融合后，真实执行精度路径没有被可靠约束

====================
十一、为什么后来转向 RTC
====================

这一页要说明“为什么不继续一条路走到黑，而是转向 RTC”。

核心原因：

- 现场真正关心的是控制循环是否稳定，而不只是子图 benchmark 是否更漂亮
- TRT/ONNX 路径在 PI0.5 上遇到的瓶颈是：
  - 安全 FP32 工件端到端不一定赢
  - unsafe FP16 虽快但 correctness 不过关
- RTC 是更贴近真实控制问题的方案，因为它直接作用于 action chunk 的实时生成与衔接

必须明确写一句：
“RTC 不是重新训练一个模型，而是推理时增强技术。”

====================
十二、RTC 原理介绍
====================

请用工程人员能听懂的话解释 RTC：

- RTC = Real-Time Chunking
- 它是 flow-matching policy 的 inference-time 技术
- 核心思想是：利用上一段 action chunk 的 leftover，对当前 chunk 的生成进行 guidance
- 目标不是让单次神经网络更快，而是让真实控制过程中 chunk 衔接更自然、更少卡顿、更少 queue 饿死

请明确说明：

- RTC 适用于 PI0.5 这类 chunk-based flow-matching policy
- RTC 依赖 `torch.autograd.grad(...)` 做 correction
- 因此它对推理上下文有特殊要求

====================
十三、RTC 路径上的真正 blocker
====================

必须把这个关键认知讲清楚：

最初真正导致 RTC 上机失败的，不是“没训练”，也不是“策略本身不支持”，而是：

- launcher 把 RTC 路径包进了 `torch.inference_mode()`
- 而 RTC guidance 在内部需要 `torch.autograd.grad(...)`
- 所以第一次真正带 prefix leftover 的 guided chunk 就会炸

报错核心是：
- `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

还要讲清楚一个更细的知识点：

- `torch.no_grad()` 不是最硬的 blocker，因为 RTCProcessor 内部可以重新 `torch.enable_grad()`
- `torch.inference_mode()` 才是硬 blocker，因为它不能被简单恢复成可求导路径

这个判断必须体现在 PPT 里，因为它是本轮排障最关键的技术洞察之一。

====================
十四、多智能体排查与修复策略
====================

请把多智能体协作过程写成一页“工程管理亮点”，但不要写成空话，要具体：

- 主代理负责架构拆分和任务分发
- 不同智能体分别负责：
  - launcher 侧 grad context 修复
  - policy 侧 grad / no_grad 边界修复
  - 轻量测试补齐
  - critic 审查与挑刺
  - 执行路径审查
- 所有智能体要求在工作目录内写报告
- 最后由主代理收敛文档、汇总结论、安排上机顺序

这里可以总结成一条方法论：
“复杂问题不是靠一个人硬改，而是靠架构拆分、责任边界、独立审查和结果收敛。”

====================
十五、RTC 代码修复要点
====================

请把修复内容讲清楚，但不要陷入太细碎的源码 diff。

修复点 1：launcher 侧
- `RTC off`：仍保持 `torch.inference_mode()`
- `RTC on`：改为 `torch.enable_grad()`
- 如果 RTC-on 路径被外层提前包进 `torch.inference_mode()`，直接 fail-fast

修复点 2：policy / runtime 边界
- 保证 `predict_action_chunk(...)` 的 RTC 路径不再被错误地压死
- `select_action(...)` 不作为 RTC 路径入口

修复点 3：测试
- 新增 grad-context 契约测试
- 核心结论被自动化锁住：
  - RTC off 仍走 inference_mode
  - RTC on 不能再走 inference_mode
  - 即使 policy 入口是 no_grad，RTCProcessor 仍可在内部重新打开 grad

测试结果：
- `7 passed`

====================
十六、RTC 真机验证结果
====================

必须使用以下真实结论：

1. `preflight-only` 成功
- 相机通过
- policy 加载通过
- RTC 配置解析通过

2. 2026-03-15 进行了 RTC 真机 5 秒 smoke
- 设备：
  - conda 环境：`lerobot`
  - robot port：`/dev/ttyACM0`
  - 相机：top=4, wrist=6
  - GPU：RTX 4090
  - policy path：`/data/tfj/lerobot_tfj/pi_model/pretrained_model`

3. 关键日志结果：
- 初始 chunk：`0.602 s`
  - preprocess：`0.019 s`
  - inference：`0.396 s`
  - postprocess：`0.188 s`
- 后续 async chunk 日志中出现过：
  - `0.464 s`
  - `0.355 s`
- 5 秒内：
  - `chunk_count` 增长到 `5`
  - `queue_underrun_count = 0`
  - `hold_step_count = 0`
  - `sync_refill_count = 0`
- 最终正常结束：
  - `Reached requested run_time_s. Exiting inference loop.`
  - `Inference finished.`

这一页的结论必须很明确：

- 这不是“理论上可以”
- 而是 RTC 的 PyTorch 上机路径已经完成一次真实实机 smoke
- 并且已经跨过了第一次 async guided chunk 这个此前会崩溃的关键节点

====================
十七、最终上机命令
====================

请把这页做成“可复制命令页”。

先做预检：

```bash
conda run -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/run_pi05_torch_infer_so101.py \
  --robot-id so101_follower \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --task "Clean the desk" \
  --rtc-enable \
  --preflight-only
```

再做 5 秒保守版 smoke：

```bash
conda run -n lerobot python /data/tfj/lerobot_tfj/tfj_envs/pi_rtc/scripts/run_pi05_torch_infer_so101.py \
  --robot-id so101_follower \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
  --task "Clean the desk" \
  --rtc-enable \
  --run-time-s 5 \
  --log-interval 1 \
  --joint-delta-limit 0.02 \
  --gripper-delta-limit 0.02 \
  --joint-action-alpha 0.2 \
  --gripper-action-alpha 0.2 \
  --robot-max-relative-target 0.05
```

要提醒的现场注意点：

- 第一轮不要把 `--run-time-s` 设成 `0`
- 当前机器上同时存在 `/dev/ttyACM0` 和 `/dev/ttyACM1`
- 如果插拔设备后串口变化，必须重新确认端口

====================
十八、踩坑总结
====================

请把以下内容做成“经验教训”页：

1. build 成功 != correctness 通过 != 真机可部署
2. 子图变快 != pipeline_chunk 变快 != 控制循环更稳
3. FP16 很诱人，但如果 Stage 5 不过，不能直接上机
4. RTC 是 inference-time 技术，不需要重新训练
5. `torch.inference_mode()` 和 `torch.no_grad()` 不是一回事
6. 真正需要看的不仅是推理时间，还有：
   - queue_underrun_count
   - hold_step_count
   - sync_refill_count
   - 真实 control loop 节拍
7. 没有真实硬件 smoke 的“理论接通”不算完成

====================
十九、后续路线建议
====================

请把后续路线写得务实一些：

1. 保留双路径
- RTC off
- RTC on
- 默认保持 RTC off，逐步扩大 RTC 的现场验证范围

2. 后续可继续做的方向
- 为 RTC 主循环补更强的无硬件状态机测试
- 增加“full loop but never send_action”的安全中间层
- 继续研究 prefix_cache 这类真正拖慢端到端速度的前缀主路径
- 如果继续研究 TensorRT，重点应该放在：
  - prefix_cache 的真实执行精度与性能
  - correctness gate 自动化
  - 区分 safe artifact 与 unsafe diagnostic artifact

3. 一句话总结未来路线
- 从“追求某个 backend 更快”转向“追求真实机器人系统更稳、更可控、更可解释”

====================
二十、PPT 风格要求
====================

请在最终 PPT 中使用以下风格：

- 语言：中文
- 风格：技术复盘风，不要营销腔
- 配色：白底或浅底，专业工业风
- 图表建议：
  - 时间线
  - 系统结构图
  - 表格对比
  - 关键日志截图风格版式
  - 风险矩阵
- 需要有“结论先行”页
- 需要有“真实数据页”
- 需要有“踩坑复盘页”
- 需要有“最终命令页”

最后，请直接输出完整的 PPT 页级内容，不要再反问我。
```

---

## 精简版提示词

```text
请基于以下主题，直接生成一份 18 到 22 页的中文技术复盘 PPT 内容：

主题：PI0.5 模型从 Safetensors 基线、ONNX/TensorRT 探索，到 RTC 真机落地的完整工程复盘

必须覆盖：
- 参考文章的思路：把 PI 系列模型拆成 vision_encoder、prefix_cache、denoise_step 做 ONNX/TRT
- 项目真实目标：不是只追 benchmark，而是要在真实机器人上稳定运行
- 基线路径：先用 Safetensors / PyTorch 跑通
- 导出链路：Stage2 导出 ONNX，Stage3 验证 ONNX，Stage4 build TRT，Stage5 验证 TRT
- 核心认知：Stage4 pass 不代表可部署，Stage5 才是 correctness gate
- 安全 FP32 benchmark：
  - PyTorch pipeline_chunk 95.468 ms
  - ONNX pipeline_chunk 157.021 ms
  - TRT FP32 pipeline_chunk 123.501 ms
  - 原因：TRT 的 denoise_step 变快了，但 prefix_cache 仍然很慢，端到端不赢
- 1000-step pure inference：
  - pytorch_fp32 2.894 ms/step
  - onnx 3.144 ms/step
  - tensorrt_fp32 2.491 ms/step
- unsafe FP16 虽快但不能部署：
  - TRT FP16 pipeline_chunk 50.665 ms
  - TRT FP16 pure inference 1.015 ms/step
  - 但 Stage5 fail，prefix_cache 严重漂移
  - torch_vs_onnx 对得上，torch_vs_trt 差很多
- prefix_cache 要解释清楚：它对应 PaliGemma 前缀编码后的 KV cache，是后续 denoise_step 反复复用的关键缓存
- RTC 是 inference-time 技术，不需要重新训练
- RTC 真正 blocker：
  - 不是“没训练”
  - 而是 RTC guidance 需要 autograd，而外层错误地用了 torch.inference_mode()
  - no_grad 不是最硬 blocker，inference_mode 才是
- 多智能体协作：架构师拆任务，编码、测试、挑刺、执行评审并行推进
- RTC 修复结论：
  - RTC off 继续走 inference_mode
  - RTC on 改为 enable_grad
  - 加了无硬件测试，7 passed
- RTC 真机 5 秒 smoke 已通过：
  - 初始 chunk 0.602 s
  - async chunk 约 0.355 到 0.464 s
  - chunk_count=5
  - queue_underrun_count=0
  - hold_step_count=0
  - sync_refill_count=0
- 最终上机命令必须放一页附录
- 最后一页要总结经验：build 成功不等于 correctness，通过 benchmark 不等于真机可用，真正要看控制环稳定性

输出要求：
- 每页给出：标题、副标题、3到6条 bullet、图表建议、演讲备注
- 风格：中文、专业、克制、技术复盘风
- 不允许编造数据
```

---

## 建议文件

- 如果需要把 AI 生成出的 PPT 文案再落成演讲稿，可以在这个文档基础上继续扩一版 `speaker notes`。
- 如果需要，我还可以继续给你补两份配套材料：
  - 一份“适合 Kimi / ChatGPT / Claude 的长提示词版本”
  - 一份“适合 Gamma / Beautiful.ai 的短提示词版本”

