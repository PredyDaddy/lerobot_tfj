# GROOT Offline RL + SO101 Direct Inference 全量知识总纲

这份文档的目标不是简单复述某次 shell 操作，而是把本轮对话里真正有价值的知识、工程判断、代码改动、验证结论、部署风险和可讲述逻辑整理成一份统一素材，供后续直接制作 PPT、讲稿或项目汇报。

本文档强调五个原则：

1. 尽量把“概念层”和“工程层”同时讲清楚。
2. 尽量把“已经验证过的事实”和“仍然需要后续工作的事项”分开。
3. 尽量避免把 stage-1 监督训练、stage-2 offline RL、真实机器人推理混为一谈。
4. 尽量保留真实路径、真实文件、真实结论，方便后续追溯。
5. 尽量把本轮对话沉淀成一份可直接拆 PPT 的材料，而不是零散聊天记录。

---

## 1. 一句话总览

这轮工作的核心成果可以概括成一句话：

- 在保留 stage-1 GROOT 作为真实机器人默认安全部署权重的前提下，补齐了 stage-2 dataset-only offline RL 训练链路、修正了 checkpoint 落盘问题，并且把 SO101 的 GROOT 推理路径从“录制型 eval”改造成了真正的 direct no-save inference。

如果再拆成三个更具体的层面，就是：

1. 训练层：
   - 建立了 GROOT stage-2 dataset-only offline RL 训练入口、offline replay、checkpoint 落盘和 resume 路径。
2. 验证层：
   - 做了离线 replay 单测、trainer smoke、真实短程训练 smoke、resume 路径验证、真实机器人 stage1 / stage2 行为对比。
3. 部署层：
   - 修掉了原来 `run_groot_so101_infer.sh` 实际仍在建 dataset 的问题，改成不录制数据的 direct inference，并保持 stage-1 safe checkpoint 为默认部署权重。

---

## 2. 这轮工作里最重要的结论

先给出最关键的管理层 / 汇报层结论。

### 2.1 结论一：stage-2 RL 不是默认真实机器人部署权重

真实机器人上最重要的判断不是“能不能加载”，而是“动作尺度安不安全”。

本轮已经验证到：

- stage-2 RL checkpoint 在真实机器人上会产生异常大的动作请求，安全 guard 会立即拒绝。
- stage-1 checkpoint 动作尺度正常，能够稳定通过 guard 并持续执行。

因此：

- 真实机器人默认部署仍然必须用 stage-1 GROOT 权重。
- stage-2 RL 权重只能作为显式 opt-in 的实验路径，不能默认替换线上部署权重。

### 2.2 结论二：stage-2 RL 当前是 dataset-only offline RL，不是在线 RL

本轮准备好的 stage-2 路线是：

- 不接在线 env
- 不做真实 reward 采集
- 不做在线 rollout collector
- 只利用 demo 数据集构造 offline replay transitions
- 利用合成 reward / value target 做第二阶段微调

所以它更准确的描述是：

- dataset-only offline RL stage-2
- 或者更保守一点，叫“带 value / advantage 信号的离线第二阶段微调”

而不是标准意义上的：

- online RL
- real reward RL
- 边执行边更新的 on-robot RL

### 2.3 结论三：原来的 “infer” 路径其实是“录制型 eval”，现在已经改成直推 no-save

之前顶层 `run_groot_so101_infer.sh` 名字叫 infer，但真实行为仍然是：

- 进入 `lerobot_run_so101_pickplace.py`
- 再进入 `run_recording(...)`
- 再走 `base_record(...)`
- 然后创建 dataset 根目录

这意味着旧路径本质是：

- guarded eval recording

而不是纯粹的：

- direct inference

本轮已经把这个问题修正为：

- 真实 GROOT SO101 推理路径支持 direct no-save runtime
- 默认不创建 dataset 目录
- 默认也不记录 `events.jsonl`
- 只有显式指定 `EVENTS_JSONL_PATH` 时才保留事件日志

---

## 3. 这项工作的完整主题到底是什么

如果要做 PPT，必须先讲清楚这次工作覆盖的不是一个点，而是一个“从训练到部署”的完整链路。

具体来说，它覆盖了四个层面。

### 3.1 模型层

这里用的是 GROOT 策略，不是 ACT，不是 SmolVLA，不是 PI0.5。

它的输入包括：

- 顶视角图像
- 腕部图像
- 机器人状态
- 任务文本

它输出的是：

- action chunk

### 3.2 训练层

训练不是从零开始，也不是单阶段。

必须严格区分：

1. stage-1 supervised training
   - 这是已有的基座模型训练
   - 用户明确说明最近提供的模型权重已经训练好了，而且已经训练了 10 个 epoch
2. stage-2 offline RL
   - 这是在已有 stage-1 checkpoint 之上做的第二阶段
   - 不能误把它理解成重新把 stage-1 从头训 20 个 epoch

### 3.3 推理层

机器人执行时要区分两件事：

1. 使用哪个 checkpoint 推理
2. 推理过程中是否还在建 dataset / 录视频 / 写日志

本轮的核心部署改动就是把这条路径改成：

- 只推理
- 不录制数据
- 不默认写盘

### 3.4 安全层

真实机器人不是“只要模型能出动作就行”。

必须通过：

- action limit
- step delta limit
- halt / reject 逻辑

所以最终能不能上真实机器人，不是只看 loss，也不是只看是否 load 成功，而是看：

- 动作范围是否合理
- 是否能持续通过 guard

---

## 4. stage-1、stage-2、真实推理三者必须严格区分

这一节对 PPT 非常重要，因为这是最容易讲错的地方。

### 4.1 stage-1 是什么

stage-1 是已有的监督训练基座。

在这轮工作里，最重要的现实约束是：

- 用户提供的最近权重已经训练好了
- 并且已经训练了 10 个 epoch

所以后续所有 stage-2 工作都建立在这个前提上：

- stage-1 已完成
- stage-2 是续训 / 微调，不是重做 stage-1

### 4.2 stage-2 是什么

stage-2 在本轮里被定义为：

- dataset-only offline RL stage-2

它的输入是：

- 已有的 GROOT stage-1 checkpoint
- 已有 demo 数据集

它的核心做法是：

- 用 demo 轨迹构造 transition
- 用合成 reward / Monte Carlo value target
- 再做第二阶段优化

### 4.3 真实机器人推理是什么

真实机器人推理是：

- 把训练好的权重加载到真实 SO101
- 进行 policy-only inference
- 在控制循环里发动作

这里要强调：

- 真实机器人推理不等于在线 RL
- 真实机器人推理也不等于必须录制 dataset

本轮正是把这个行为显式改造成了：

- no-save direct inference

---

## 5. stage-2 offline RL 的核心设计

### 5.1 为什么叫 dataset-only

因为它不使用在线 env。

更具体地说，当前准备好的 stage-2 路线里：

- `env=None`
- collector 的在线分支被关掉
- replay buffer 的 transition 来自数据集，而不是来自线上 rollout

### 5.2 reward 是怎么来的

当前 reward 不是环境里真实采集到的 reward，而是由 demo 位置合成。

当前默认配置的核心语义是：

- `value_target_mode=monte_carlo`
- `terminal_reward=1.0`
- `step_reward=0.0`
- `success_value=true`

直观理解是：

- 到达 demo 终点给一个成功奖励
- 中间不额外给 dense step reward
- 用 Monte Carlo 的方式把回报回传到前面的 transition

### 5.3 这条路更像什么

从方法论上看，这条路更接近：

- 在成功 demo 上做带 value / advantage 信号的第二阶段行为优化

而不是最教科书式的：

- 带真实 reward logging 的标准 offline RL

因此在汇报时，建议用更准确的说法：

- dataset-only offline RL stage-2
- synthetic reward offline replay stage
- value / advantage guided offline fine-tuning

---

## 6. 这轮对训练链路做了哪些关键代码改动

### 6.1 增加了 GROOT hybrid train 配置和实现

相关核心文件：

- `/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py`
- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_train_groot_hybrid.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/trainer.py`

它们共同负责：

- 配置解析
- offline replay 构造
- hybrid trainer 的训练主循环
- checkpoint 管理

### 6.2 补上了 checkpoint 写盘逻辑

本轮发现了一个真实问题：

- GROOT hybrid 训练入口虽然能训练
- 但默认不会像标准 `lerobot-train` 那样写出可部署 checkpoint

这会直接导致：

- 训练虽然看似完成
- 但后续不能正常 resume
- 也无法稳定拿来部署

因此本轮补齐了：

- step checkpoint 落盘
- `checkpoints/last` 更新
- final checkpoint 可 resume / 可部署

### 6.3 增加了 stage-2 启动脚本

新增并整理的训练脚本包括：

- `/data/tfj/lerobot_tfj/scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`
- `/data/tfj/lerobot_tfj/scripts/train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`

这些脚本负责：

- 选择现有 stage-1 checkpoint
- 做数据集路径预检
- 做视频解码预检
- 关掉在线 env 相关逻辑
- 配置 offline replay 超参
- 启动 stage-2 dataset-only offline RL

---

## 7. 训练链路里踩到的工程坑和最后结论

### 7.1 pytest 被系统插件污染

一开始跑测试时出现的报错不是代码本身问题，而是环境问题：

- 系统里的 ROS pytest 插件污染了当前测试环境
- 出现了 `asyncio.coroutine` 相关旧插件错误

解决方式是：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

### 7.2 本地包导入还需要显式 PYTHONPATH

本地测试还需要：

- `PYTHONPATH=src`

否则会出现：

- 本地包解析不到

### 7.3 语法和单测最终通过

本轮与 stage-2 训练相关的关键验证结果包括：

- `python -m py_compile` 通过
- `tests/rl/groot_hybrid/test_offline_replay.py` 通过
- `tests/rl/groot_hybrid/test_trainer_smoke.py` 和 `tests/rl/groot_hybrid/test_buffer.py` 通过

这说明：

- offline replay 逻辑能正常构造
- hybrid trainer 主链路没有被新改动打坏

---

## 8. stage-2 训练 smoke 的真实验证结果

本轮不是只做了单测，还做了真实短程训练 smoke。

### 8.1 使用的输入

真实 smoke 使用的是：

- checkpoint:
  - `/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model`
- dataset:
  - `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`

### 8.2 输出目录

短程 smoke 输出目录是：

- `/data/tfj/lerobot_tfj/outputs/train/groot_offline_rl_stage2_smoke_20260318_1000`

### 8.3 核心结果

短程训练最终确认了以下事实：

1. 训练确实进入了真实训练 step，而不是只做初始化。
2. `checkpoints/000001` 和 `checkpoints/000002` 都被完整写出。
3. `checkpoints/last` 已更新到 `000002`。
4. `checkpoints/last/pretrained_model/train_config.json` 存在，可用于 resume。

### 8.4 这说明什么

它说明 stage-2 这条训练链路已经具备了最低可用性：

- 可以真实起训
- 可以写 checkpoint
- 可以 resume
- 可以作为后续部署和验证的基础

---

## 9. 训练过程里关于资源和耗时的现实观察

这部分很适合写到 PPT 的“工程代价 / 资源画像”页里。

### 9.1 GPU 环境

本轮确认的设备是：

- `NVIDIA GeForce RTX 4090`

### 9.2 训练显存占用

真实 smoke 过程中观测到：

- 训练进程显存占用大约 26GB

### 9.3 checkpoint 体积

短程 smoke 过程中观察到：

- `trainer.pt` 约 15.4GB
- `optimizer_state.safetensors` 约 8.6GB
- `model.safetensors` 约 7.6GB

### 9.4 意味着什么

这意味着：

1. 训练不是“轻量小实验”，而是明显的大模型训练链路。
2. checkpoint 保存会比较慢。
3. 看到 GPU 瓦数不高时，不一定代表“没在干活”，也可能是在：
   - 数据加载
   - checkpoint 写盘
   - CPU / IO 侧 bottleneck

---

## 10. 真实机器人推理路径里原来存在什么问题

### 10.1 表面叫 infer，实际还是 record

旧的 GROOT SO101 推理入口表面上叫：

- `run_groot_so101_infer.sh`

但它实际上走的是：

- `lerobot_run_so101_pickplace.py`
- `run_recording(...)`
- `base_record(...)`
- `LeRobotDataset.create(...)`

也就是说，旧路径本质是：

- 带 guard 的 eval recording 路径

而不是：

- 纯粹 direct inference

### 10.2 `FileExistsError` 的真正根因

之前用户已经删掉了 `outputs/eval_groot_so101_block/`

但运行时还是报：

- `FileExistsError: [Errno 17] File exists`

根因不是用户没删干净，而是：

- `JsonlStepLogger` 会先创建 `events_jsonl_path.parent`
- 而旧默认配置里 `events_jsonl_path` 恰好放在 `dataset_root` 里面
- 结果 dataset 目录被 logger 先创建了
- 随后 `LeRobotDataset.create(... exist_ok=False)` 再创建同一目录时就报错

这说明问题的本质是：

- 推理路径和录制路径耦合得太紧
- 日志路径和 dataset 根目录设计得不合理

---

## 11. 本轮如何修正了旧推理路径

### 11.1 先做了过渡性修复

在完全改成 no-save 之前，先做过一轮保守修补：

1. headless 环境自动关闭 `DISPLAY_DATA`
2. `events_jsonl_path` 默认移到 dataset 根目录之外
3. 如果默认 `dataset_root` 已存在，则自动生成带时间戳的新目录

这些修复解决了：

- 没有 `DISPLAY` 时的 rerun / winit 报错
- `events_jsonl_path` 提前创建 dataset 根目录的问题

### 11.2 后来做了真正的结构性修复

更重要的修复不是继续“绕着 dataset_root 打补丁”，而是直接把 GROOT SO101 路径改成：

- direct no-save inference

现在的推理主入口是：

- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_run_so101_pickplace.py`

它的核心做法是：

1. 直接从 checkpoint 加载 policy
2. 直接从 checkpoint 加载 preprocessor / postprocessor
3. 复用 SO101 runtime 的 bridge / guard / logger
4. 复用共享控制循环 `record_loop(...)`
5. 但是明确传 `dataset=None`

因此新的真实行为已经是：

- policy-only inference
- no dataset recording
- no default disk output

---

## 12. 直推 no-save 路径的关键代码设计

### 12.1 共享循环 `record_loop(...)` 本来已经接近可复用

共享控制循环本来就有：

- `dataset: LeRobotDataset | None = None`

并且只有在：

- `dataset is not None`

时才会写入 frame。

### 12.2 真正卡住它的只剩一个点

`record_loop(...)` 的 policy 分支原来仍然写死了：

- `make_robot_action(action_values, dataset.features)`

这意味着：

- observation 这边其实已经能脱离 dataset
- 但 action 映射这边还不行

### 12.3 本轮对共享循环做了什么小而关键的补丁

本轮给共享循环增加了：

- `policy_action_features`

于是 policy 分支现在会：

- 优先用 `policy_action_features`
- 只有没有显式传时，才回退到 `dataset.features`

这就把最后一个硬依赖点消掉了。

这类改法的优点是：

- 不需要复制第二套控制循环
- bridge / guard / logger 逻辑不分叉
- ACT、GROOT 等 direct runtime 以后都能复用同一条共享主链

---

## 13. GROOT direct runtime 现在的最终行为

当前 GROOT SO101 直推入口：

- `/data/tfj/lerobot_tfj/scripts/run_groot_so101_infer.sh`

它已经具备以下行为特征。

### 13.1 默认部署权重

默认使用：

- `/data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/last/pretrained_model`

也就是：

- stage-1 safe checkpoint

### 13.2 stage-2 RL 只做显式 opt-in

只有显式指定：

- `PREFER_STAGE2_RL=true`

时，才会切到：

- 最新 stage-2 RL checkpoint

### 13.3 默认 no-save

默认情况下：

- 不创建 dataset 根目录
- 不保存录制数据
- 不默认写 `events.jsonl`

### 13.4 可选事件日志

如果需要日志，可以显式指定：

- `EVENTS_JSONL_PATH=...`

### 13.5 可做 dry-run / preflight

可以通过：

- `PREFLIGHT_ONLY=1`

做：

- checkpoint 加载验证
- 运行链路验证

而不进入真实机器人动作循环

---

## 14. 真实机器人上 stage-1 和 stage-2 的行为对比

这是整份文档里最值得上 PPT 的一页，因为它直接决定部署结论。

### 14.1 stage-2 RL 的真实机器人行为

真实机器人上 stage-2 RL checkpoint 的事件日志显示：

- 一上来就产生了巨大动作值

例如：

- `shoulder_lift.pos = -8000.606...`
- `elbow_flex.pos = 9146.913...`
- `wrist_flex.pos = 3684.145...`

这些动作远远超出真实机器人允许范围，因此 guard 触发：

- `step_delta_exceeded`
- `action_limit_exceeded`

最终 run_end 表现为：

- `halted=true`
- `last_reason=action_limit_exceeded`

### 14.2 stage-1 的真实机器人行为

对比 stage-1 checkpoint 的事件日志可以看到：

- 动作尺度正常
- 早期 action 量级大约在百位以内

例如：

- `shoulder_lift.pos ≈ -96.9`
- `elbow_flex.pos ≈ 98.2`

并且：

- 没有 guard reject
- `run_end` 为 `halted=false`

### 14.3 这页 PPT 应该怎么讲

可以非常清楚地总结成一句话：

- 当前 stage-2 RL 在真实机器人上还不满足安全部署条件，而 stage-1 基座已经满足基本可用性，所以默认部署策略必须保持在 stage-1。

---

## 15. 为什么 stage-2 RL 会在真实机器人上失控

这部分是分析层，不是完全定论，但已经有足够强的工程推断。

最合理的解释是：

1. stage-2 使用的是合成 reward，而不是物理真实 reward。
2. offline replay 的目标是数据集上的 value / advantage 优化，不是机器人动态约束校准。
3. actor 的输出尺度在离线优化后发生了偏移。
4. 这些偏移在仿真或离线指标上可能不显著，但在真实机械臂绝对控制值上会被放大。

这意味着后续如果真想把 stage-2 RL 也用于真实机器人，至少还需要做：

- action scale 约束
- 更严格的输出归一化检查
- 真实机器人或更接近真实机器人的闭环验证
- 可能的 reward / target 重新设计

---

## 16. 本轮 direct no-save 路径的验证结果

### 16.1 共享循环回归测试

本轮新增并验证了：

- `policy + dataset=None` 的回归测试

最终结果：

- `tests/runtime/test_so101_record_loop_hooks.py` 通过
- 总结果为 `5 passed`

它验证了：

- policy-only 情况下共享 `record_loop` 可以在 `dataset=None` 时正常工作
- 动作映射使用显式的 `policy_action_features`

### 16.2 真实 checkpoint dry-run 验证

新直推入口已经实际加载了真实 stage-1 GROOT checkpoint：

- 不是 mock
- 不是伪加载
- 而是真实走到了 GROOT 初始化和权重恢复

### 16.3 no-save 行为验证

还做了一个专门的 dry-run 检查：

- 给脚本传入一个临时 `dataset_root`
- 跑完后确认该目录没有被创建
- 对应 `.events.jsonl` 也没有被创建

这说明：

- 新直推路径已经真正摆脱 dataset recording 副作用

---

## 17. OpenClaw / 统一路由相关整理

这轮工作没有只修 GROOT 单条脚本，也同步整理了周边入口的策略。

### 17.1 OpenClaw GROOT server

相关文件：

- `/data/tfj/lerobot_tfj/scripts/openclaw_groot_server.py`

保留并强化了：

- safer default policy path 选择逻辑
- 事件日志路径不要默认落在 dataset 根目录内部

### 17.2 SO101 统一推理路由脚本

相关文件：

- `/data/tfj/lerobot_tfj/scripts/run_so101_pickplace_infer.sh`

它的作用是：

- 统一分发 backend
- 支持 `groot / smolvla / pi05 / act / policy_record`

从架构角度看，它是一个 router，不是模型逻辑本身。

---

## 18. 这轮工作的关键文件索引

如果 PPT 里要加“代码落点”页，可以用下面这组文件。

### 18.1 训练相关

- `/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py`
- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_train_groot_hybrid.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/trainer.py`
- `/data/tfj/lerobot_tfj/scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`
- `/data/tfj/lerobot_tfj/scripts/train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`

### 18.2 机器人推理相关

- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_run_so101_pickplace.py`
- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record_so101_policy.py`
- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record.py`
- `/data/tfj/lerobot_tfj/scripts/run_groot_so101_infer.sh`
- `/data/tfj/lerobot_tfj/scripts/run_so101_pickplace_infer.sh`
- `/data/tfj/lerobot_tfj/scripts/openclaw_groot_server.py`

### 18.3 事件日志 / 结果对比相关

- `/data/tfj/lerobot_tfj/outputs/eval_groot_so101_block_20260318_132235.events.jsonl`
- `/data/tfj/lerobot_tfj/outputs/eval_groot_stage1_compare_132436.events.jsonl`
- `/data/tfj/lerobot_tfj/outputs/train/groot_offline_rl_stage2_smoke_20260318_1000`

---

## 19. 现在可以直接怎么用

### 19.1 直接跑 GROOT SO101 no-save inference

```bash
cd /data/tfj/lerobot_tfj
bash scripts/run_groot_so101_infer.sh
```

### 19.2 只做直推预检

```bash
cd /data/tfj/lerobot_tfj
PREFLIGHT_ONLY=1 bash scripts/run_groot_so101_infer.sh
```

### 19.3 直推但保留事件日志

```bash
cd /data/tfj/lerobot_tfj
EVENTS_JSONL_PATH=./outputs/groot_direct.events.jsonl bash scripts/run_groot_so101_infer.sh
```

### 19.4 开 stage-2 dataset-only offline RL

```bash
cd /data/tfj/lerobot_tfj
bash scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh
```

### 19.5 显式尝试 stage-2 RL 权重做推理

```bash
cd /data/tfj/lerobot_tfj
PREFER_STAGE2_RL=true bash scripts/run_groot_so101_infer.sh
```

但是这一条目前只适合实验验证，不适合默认真实部署。

---

## 20. 这轮工作里值得放到 PPT 的“方法论”总结

### 20.1 第一条：名字和真实行为必须一致

一个脚本叫 `infer`，并不代表它真的只推理。

本轮就发现：

- 名字叫 infer
- 实际还在建 dataset

所以工程上必须始终验证：

- 真实入口调用链
- 副作用
- 写盘行为

### 20.2 第二条：真实机器人默认部署策略必须保守

训练后的新权重不应因为“更新了”就自动替换默认线上权重。

必须先看：

- 真实机器人动作尺度
- safety guard 行为
- 连续运行稳定性

### 20.3 第三条：共享循环可以复用，但前提是接口真正解耦

`record_loop(...)` 能被复用，不是因为它名字通用，而是因为：

- 它已经包含完整控制链
- 并且最后那一点 dataset 硬依赖被拆掉了

所以抽象复用真正重要的是：

- 找到最后一个不必要的硬依赖
- 用最小改动把它解耦

### 20.4 第四条：测试不只看单测，也要看真实链路

本轮真正有说服力的不是“代码看起来对”，而是：

- 单测通过
- smoke 训练通过
- checkpoint 可写
- resume 可行
- 真实 checkpoint 可加载
- no-save 行为可验证
- stage1 / stage2 真实机器人行为有对比证据

---

## 21. 后续还需要做什么

这部分非常适合做 PPT 的“下一步规划”页。

### 21.1 如果目标是继续优化 stage-2 RL

建议后续重点做：

1. 输出尺度校验
2. 动作归一化 / 反归一化复查
3. 更严格的 action range 约束
4. offline target 设计复查
5. 更长程的离线验证和仿真验证

### 21.2 如果目标是提升部署可用性

建议后续重点做：

1. 把 SmolVLA / PI0.5 的 SO101 推理路径也统一成 no-save direct runtime 风格
2. 统一事件日志结构
3. 统一 preflight / dry-run / run-time-s / safety-profile 接口

### 21.3 如果目标是汇报和传播

建议把汇报分成三层：

1. 业务层：
   - 我们把训练链和部署链都打通了
2. 技术层：
   - dataset-only offline RL + direct no-save inference
3. 风险层：
   - stage-2 RL 暂未达到真实机器人默认部署标准

---

## 22. 一个直接可用的 PPT 目录建议

如果要立刻做 PPT，我建议按下面的结构拆页。

### 第 1 页：项目背景

- GROOT 在本项目里的角色
- 为什么需要 stage-2
- 为什么要区分训练和部署

### 第 2 页：整体系统结构

- stage-1 supervised training
- stage-2 dataset-only offline RL
- SO101 direct inference

### 第 3 页：本轮核心成果

- 打通 stage-2 训练
- 补齐 checkpoint 保存 / resume
- 推理路径改成 no-save direct runtime

### 第 4 页：stage-2 RL 方法定义

- dataset-only
- synthetic reward
- Monte Carlo value target
- 不是 online RL

### 第 5 页：训练链路代码结构

- config
- train script
- offline replay
- trainer

### 第 6 页：工程验证结果

- 单测通过
- smoke 训练通过
- checkpoint 落盘成功
- resume 路径可用

### 第 7 页：资源画像

- 4090
- 约 26GB 显存
- checkpoint 体积
- 保存较慢

### 第 8 页：旧 infer 路径问题

- 实际仍是 record
- `FileExistsError` 根因
- 日志路径与 dataset 路径耦合

### 第 9 页：新的 direct inference 架构

- 直接从 checkpoint 加载
- 复用 runtime / guard / logger
- `dataset=None`
- no-save

### 第 10 页：真实机器人 stage1 / stage2 对比

- stage-2 产生巨大动作值，被 guard 拒绝
- stage-1 动作尺度正常，可持续执行

### 第 11 页：最终部署策略

- 默认 stage-1
- stage-2 opt-in
- 真实机器人优先安全

### 第 12 页：后续规划

- 继续优化 stage-2 输出尺度
- 统一更多 backend 的 direct runtime
- 完善评估与日志系统

---

## 23. 最终一句话结论

如果整份 PPT 只保留一句压轴总结，那么最准确的话应该是：

- 本轮工作已经把 GROOT 的 stage-2 dataset-only offline RL 训练链路和 SO101 的 direct no-save inference 部署链路都打通了，但真实机器人默认部署仍应保持在更稳定的 stage-1 权重上，stage-2 RL 目前仍属于显式 opt-in 的实验路径。
