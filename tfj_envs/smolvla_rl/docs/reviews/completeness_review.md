# 文档完整性补充审查报告

## 审查目标

检查以下两份文档是否覆盖了用户最关心的内容：

- `tfj_envs/smolvla_rl/docs/smolvla_rl_architecture_and_integration_20260315_zh.md`
- `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md`

重点检查项：

- 原理
- 接入方式
- 训练先后顺序
- 已验证的离线训练事实
- SO101 上机说明
- 限制与后续路线

## 覆盖充分之处

1. 原理部分覆盖充分。
   - 已解释 SmolVLA 原本的 flow-matching 动作生成机制；
   - 已解释为什么 actor 没有被完全替换；
   - 已解释 `value_head`、`prefix context`、`compute_fm_score` 的职责；
   - 已解释这套 RL 更像 advantage-weighted generative actor-critic，而不是 PPO / SAC。

2. 接入方式覆盖充分。
   - 已列出配置层、模型层、trainer、collector、buffer、losses 的关键代码位置；
   - 已讲清楚 RL 是通过 value head 与 online weighted flow loss 接进 SmolVLA 的；
   - 已说明 replay buffer 存的是 chunk transition，而不是单步 action transition。

3. 训练先后顺序覆盖充分。
   - 已明确区分“step 级同时训练”和“项目阶段级先离线后 hybrid”的两层语义；
   - 这部分表述是当前文档里最重要、也最容易被误解的一点，当前版本已经讲清楚。

4. 已验证的离线训练事实覆盖较充分。
   - 已写明 trimmed dataset 路径；
   - 已写明任务文本；
   - 已写明 AV1 / `pyav` / rename map 这些关键事实；
   - 已写明成功训练输出目录和 checkpoint 结果；
   - 已写明 `10000 steps` 与 `9.88 epoch` 的关系。

5. SO101 上机说明覆盖充分。
   - 已写明新的 `lerobot_record_so101_policy.py` 的定位；
   - 已写明离线环境变量的必要性；
   - 已写明默认机器人端口、相机索引、任务文本和录制脚本路径；
   - 已写明这个脚本属于“训练后执行 / 录制”而不是“在线 RL collector”。

6. 限制与后续路线覆盖充分。
   - 已明确说明当前实现不是完整的 SO101 真机在线 RL；
   - 已指出 collector、reward、安全壳、critic 设计和 actor-learner 架构方面的缺口；
   - 已提出先离线、再仿真 hybrid、最后再考虑真机闭环的推进顺序。

## 遗漏项

1. 没有严重遗漏项。

2. 如果一定要挑补充空间，主要是“进一步细化”，不是“缺关键块”。例如：
   - 可以再单独加一节“当前 hybrid trainer 的超参数含义速查表”；
   - 可以再单独加一节“如果把 collector 接到真实 SO101 需要新增哪些接口”。

## 建议追加项

1. 可追加一张“项目路线图式文档”。
   - 当前文档已经说明限制，但如果再单独做一张路线图，会更适合后续多人协作。

2. 可追加一张“脚本输入输出对照表”。
   - 哪个脚本吃什么输入；
   - 会产出什么日志、checkpoint、eval dataset；
   - 各自用于训练、监控还是上机。

3. 可追加一张“真实 SO101 在线 RL 改造 checklist”。
   - collector
   - reward
   - safety
   - rollout / replay
   - 人工接管
   - 恢复机制

## 总体结论

总体结论是：

- 当前两份主文档已经覆盖了用户最关心的核心问题；
- 原理、接入方式、训练先后顺序、离线训练事实、SO101 上机说明、限制与后续路线都已覆盖；
- 现阶段文档的主要任务已经完成，后续可以做的是“专项扩写”，而不是“补漏救火”。
