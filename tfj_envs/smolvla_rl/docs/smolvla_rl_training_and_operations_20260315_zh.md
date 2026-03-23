# SmolVLA RL 训练与运维说明

本文档站在工程实施角度，回答下面这些问题：

1. 现在这套东西应该怎么用。
2. 训练时到底是分段还是同时。
3. 已经成功跑通的离线 SmolVLA 训练是什么样。
4. hybrid RL 训练该怎么理解，应该如何启动。
5. SO101 上机执行时要注意什么。
6. 现在这套实现的主要风险和限制是什么。

## 1. 先给最实用的 workflow

如果目标是“先把 SmolVLA 用在当前数据上，然后再尝试 RL 微调”，建议工作流按下面的宏观阶段来理解。

### 阶段 A：准备离线基础模型

输入：

- 本地 SmolVLA base checkpoint；
- 本地 SmolVLM2-500M backbone；
- 本地 demonstration dataset。

产出：

- 一个已经能在当前任务上做出合理动作的 SmolVLA checkpoint。

为什么必须先做这一步：

- 当前 hybrid trainer 不是从零探索型 RL；
- 它依赖一个已经具备行为先验的 actor；
- online rollout 的质量取决于当前策略初始水平。

更精确地说：

- 这是强工程建议，不是 parser 或 config 的硬性语法要求；
- 当前代码并没有强制“必须从离线 checkpoint 恢复”这一条；
- 但从现有验证结果、训练稳定性和 rollout 质量看，先有一个离线基座是明显更稳的做法。

### 阶段 B：进入 hybrid RL 微调

输入：

- 阶段 A 产出的 SmolVLA checkpoint；
- 一个标准环境接口的仿真 env；
- hybrid trainer。

产出：

- 一个带有 value head 并经过 online reweighting 微调的 SmolVLA policy。

注意：

- 这里说的是“整体工作流程分阶段”；
- 不是说 hybrid trainer 内部会先纯 offline 再纯 RL。
- 这里推荐先在仿真或标准 env 接口环境里做原型验证；
- 代码层面要求的是 `env` 可配置且满足当前 collector 约束，而不是把“仿真”写成了唯一允许类型。

### 阶段 C：SO101 上机验证

输入：

- 训练好的 policy；
- SO101 follower；
- top / wrist 相机；
- 任务文本。

产出：

- 真实机器人执行结果；
- 可选 eval dataset。

这里必须再强调一次：

- 这一阶段对应的是策略执行、录制和评估数据采集；
- 不是在线训练，也不包含参数更新；
- 对应入口是 `src/lerobot/scripts/lerobot_record_so101_policy.py`。

## 2. “分段训练”与“同时训练”到底怎么理解

这是最容易混淆的地方，必须分两层讲。

### 2.1 从 trainer 单步实现看：同时训练

在 `src/lerobot/rl/smolvla_hybrid/trainer.py` 里，每个训练 step 都做：

1. 收在线 chunk；
2. 取 offline batch；
3. 取 online replay batch；
4. 算 offline loss；
5. 算 online flow loss；
6. 算 value loss；
7. 三者加权；
8. 一次反传，一次更新。

所以从“每个 step 里到底发生了什么”这个角度看，答案是：

- offline 和 RL 是同时训练的。
- 这里的“同时”指同步混合优化，不是异步 actor-learner 架构。

### 2.2 从工程使用看：分阶段使用

但从“项目应该怎么推进”这个角度看，更合理的答案是：

1. 先做 offline SmolVLA 训练；
2. 再用这个结果做 hybrid RL 微调；
3. 最后再做 SO101 上机验证。

所以不要把两个层次混成一句话。

最准确的表述是：

- hybrid trainer 的内部更新是同步混合优化；
- 但整个项目流程最好是先 offline，再 hybrid RL。
- 前者描述的是单 step 优化行为；
- 后者描述的是更合理的工程推进顺序。

## 3. 已经验证过的离线 SmolVLA 训练

### 3.1 数据集信息

当前已经实际使用并验证过的数据集根目录是：

- `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`

从 `meta/info.json` 和 `meta/tasks.parquet` 可确认：

- 机器人类型：`so101_follower`
- 总 episode 数：`122`
- 总帧数：`32385`
- fps：`30`
- 任务数：`1`
- 任务文本：`Put the block in the bin`
- 观测图像：
  - `observation.images.top`
  - `observation.images.wrist`
- 状态维度：`6`
- 动作维度：`6`

仓库中已经提供了一个辅助脚本用于直接打印这些元数据：

- `tfj_envs/smolvla_rl/scripts/show_trimmed_dataset_meta.sh`

### 3.1.1 这次离线训练所依赖的本地模型资产

当前仓库里已经实际用到、并且脚本默认会自动寻找的本地权重资产包括：

- SmolVLA base：
  - `/home/cqy/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/4d2f2b37fa245361ef1efe6d91ce96b8bd4af511`
- SmolVLM2-500M-Video-Instruct：
  - `/home/cqy/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467`

脚本之所以默认走本地 snapshot，是因为当前机器在很多时候处于离线或半离线环境，直接依赖在线拉取会让训练和上机启动变得不稳定。

### 3.2 为什么训练时必须显式使用 rename map

数据集里的图像键是：

- `observation.images.top`
- `observation.images.wrist`

而 SmolVLA 当前训练配置预期的是：

- `observation.images.camera1`
- `observation.images.camera2`

因此训练和上机都必须处理映射关系：

```json
{
  "observation.images.top": "observation.images.camera1",
  "observation.images.wrist": "observation.images.camera2"
}
```

如果这一层没配对，最常见的问题是：

- 预处理拿不到模型需要的视觉输入键；
- 模型 forward 或推理前的数据管道直接报错。

### 3.3 为什么视频后端要用 pyav

这批 trimmed dataset 的视频编码是 AV1。

当前验证过的稳定做法是：

- `--dataset.video_backend=pyav`

这样做的原因是：

- 之前尝试默认 `torchcodec` 路径时，这批视频并不稳定；
- `pyav` 是已经实际跑通的一条路径；
- 因此本目录下的离线训练脚本默认都显式指定 `pyav`。

### 3.4 已经成功完成的离线训练结果

之前已经成功跑完过一轮离线 SmolVLA 训练，输出目录是：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341`

已存在的 checkpoint 包括：

- `002000`
- `004000`
- `006000`
- `008000`
- `010000`
- `last -> 010000`

此前监控记录里，训练结束的关键信号是：

- `Checkpoint policy after step 10000`
- `End of training`

这个结果的重要含义：

- 离线训练链路是通的；
- 本地 SmolVLA base + 本地 SmolVLM2 + trimmed dataset + `pyav` + rename map 这组组合已经验证过；
- 之后 hybrid RL 应该建立在这个基础之上，而不是跳过这一步。

这次训练结束时，已确认出现过的结束信号包括：

- `Checkpoint policy after step 10000`
- `End of training`

### 3.5 为什么说“10 epoch”实际并不严格等于 10.0

这次离线训练最终使用的是 step 驱动：

- `steps=10000`

在这个数据集规模和 batch 配置下，这大约对应 `9.88` 个 epoch，而不是数学上严格等于 `10.00`。

所以后续文档和脚本里，建议优先说：

- `10000 steps`

而不要把它写死成：

- “严格 10 epoch”

这样表达更准确，也更符合训练系统的真实计数方式。

## 4. 离线训练脚本如何使用

### 4.1 前台单次启动

脚本：

- `tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh`

它默认会做这些事情：

1. 自动寻找本地 SmolVLA base snapshot；
2. 自动寻找本地 SmolVLM2-500M snapshot；
3. 设定 Hugging Face / Transformers 离线环境变量；
4. 使用 trimmed dataset；
5. 显式指定 `dataset.video_backend=pyav`；
6. 自动附带图像 rename map；
7. 把 stdout / stderr 写到训练日志里。

在当前机器上，已经实际探测到的本地默认 snapshot 路径是：

- SmolVLA base:
  - `/home/cqy/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/4d2f2b37fa245361ef1efe6d91ce96b8bd4af511`
- SmolVLM2-500M:
  - `/home/cqy/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467`

典型调用：

```bash
bash tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh
```

常用覆盖参数：

```bash
RUN_TAG=manual_1 \
STEPS=10000 \
BATCH_SIZE=32 \
NUM_WORKERS=4 \
OUTPUT_DIR=/data/tfj/lerobot_tfj/outputs/train/my_smolvla_run \
LOG_FILE=/data/tfj/lerobot_tfj/outputs/logs/my_smolvla_run.log \
bash tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh
```

### 4.2 后台启动加监控

脚本：

- `tfj_envs/smolvla_rl/scripts/start_smolvla_offline_trimmed_train_nohup.sh`
- `tfj_envs/smolvla_rl/scripts/monitor_training_process.sh`

后台脚本会：

1. 启动离线训练；
2. 保存 train pid；
3. 再启动一个 monitor 进程；
4. 周期性记录：
   - 进程存活状态
   - CPU / MEM
   - GPU 利用率和显存
   - 训练日志中的最新关键行

这样做的价值是：

- 不需要人工一直 `tail -f`；
- 也不会因为会话断开丢失监控轨迹；
- 更适合长期训练记录。

## 5. hybrid RL 训练脚本应该怎么理解

### 5.1 这个脚本不是“SO101 真机 RL 一键启动”

脚本：

- `tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh`

它的定位是：

- 给当前仓库里已经存在的 `SmolVLA hybrid trainer` 提供一个可重复调用的模板封装。

它默认假设：

- 你已经有一个离线训练好的 SmolVLA checkpoint；
- 你要在某个标准环境里做 hybrid 微调；
- 环境通过 `--env.type` 和 `--env.task` 提供；
- 训练仍然同时混合 offline 数据和 online rollout。

它不意味着：

- 当前仓库已经具备了完整 SO101 真机在线 RL 所需的 collector / reward / safety 全链路。

### 5.2 为什么必须显式传 env type 和 env task

当前 hybrid trainer 在配置校验里明确要求：

- `env` 不能为空；
- `collector.n_envs == 1`

因此这个模板脚本做成了：

```bash
bash tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh <env_type> <env_task> [extra args...]
```

例如：

```bash
bash tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh \
  libero \
  libero_object \
  --env.obs_type=pixels_agent_pos
```

如果是其他环境，则替换成对应的 `env.type` 和 `env.task` 即可。

### 5.2.1 当前仓库里能看到哪些环境类型

从 `src/lerobot/envs/configs.py` 和 `src/lerobot/envs/factory.py` 看，当前仓库里和这个话题直接相关的环境类型主要包括：

- `aloha`
- `pusht`
- `libero`
- `metaworld`
- `gym_manipulator`

但要注意三层区别：

1. 配置类存在，不等于当前 hybrid trainer 对这条链路已经成熟支持。
2. 当前 hybrid trainer 最稳的定位仍然是标准 vector env 的单环境原型。
3. `gym_manipulator` 虽然在配置层存在，但它对应的是更接近真实机器人 RL 的专用链路，当前这版 hybrid trainer 并没有把那条链路完整接起来。

因此，如果只是想验证当前 SmolVLA hybrid RL 是否工作，优先建议用标准仿真环境；如果目标是 SO101 真机在线 RL，则仍然属于后续架构扩展工作。

### 5.3 这个 hybrid 脚本实际传了哪些关键参数

脚本内部默认会传：

- 预训练 policy 路径；
- 本地 SmolVLM2 路径；
- 离线 demonstration dataset；
- `dataset.video_backend=pyav`；
- `collector.n_envs=1`；
- replay buffer 容量；
- online batch 大小；
- offline / online / value 三类 loss 权重；
- rename map。

因此这份脚本既表达了“当前代码需要什么”，也表达了“当前实现的假设边界”。

这三类 loss 的混合不是口头概念，而是 trainer 里显式控制的：

- `offline_loss_weight`
- `online_flow_loss_weight`
- `value_loss_weight`

当前仓库里这些超参定义在：

- `src/lerobot/configs/train_smolvla_hybrid.py`

而实际相加更新发生在：

- `src/lerobot/rl/smolvla_hybrid/trainer.py`

## 5.4 当前 reward 到底怎么设置

这部分之前没有单独展开，现在补全。

当前 `SmolVLA hybrid trainer` 里，reward 不是在脚本里手工写死一套“抓取成功奖励 + 距离奖励 + 惩罚项”的公式。它的实际设定是：

- reward 原样来自 `env.step(...)`；
- collector 只负责把一个 action chunk 内的多步 reward 折扣累加；
- 然后用这个 chunk 回报去训练 value head，并进一步构造 advantage。

也就是说，当前 reward 的定义边界是：

- 任务语义由 environment 提供；
- hybrid trainer 只做 chunk 级 return 聚合。

### 5.4.1 当前 hybrid trainer 使用的 reward 公式

假设一个 chunk 内执行了 `H` 个环境步，对应环境即时奖励是：

- `r_t, r_{t+1}, ..., r_{t+H-1}`

那么 collector 存进 replay buffer 的 reward 是：

`R_chunk = sum_{k=0}^{H-1} gamma^k * r_{t+k}`

其中：

- `gamma = cfg.losses.discount`

如果中途 episode 结束，那么 chunk 会提前停止，并把：

- `bootstrap_discount = 0`

如果没有结束，则：

- `bootstrap_discount = gamma^H`

### 5.4.2 为什么这样设

主要有四个理由。

第一，和 SmolVLA 的动作粒度一致。

- SmolVLA 一次预测的是 action chunk；
- 所以 reward 也按 chunk 汇总，决策单位与回报单位一致。

第二，能直接复用当前生成式 actor。

- 如果强行改成 primitive-step reward / transition，collector、buffer、loss 都要重新拆；
- 第一版原型这样做复杂度最低。

第三，value target 可以自然写成 chunk-level bootstrap。

- `target = R_chunk + beta * V(next_state)`
- 这里 `beta` 就是 chunk 执行后的剩余折扣因子。

第四，reward 语义留给环境本身，更通用。

- hybrid trainer 不需要为每个任务都重写一套 reward shaping；
- 同一套 trainer 可以服务不同 env。

### 5.4.3 当前没有做的 reward shaping

当前 hybrid 路径没有做：

- 手工距离奖励；
- success bonus shaping；
- 视觉 reward classifier 注入；
- teleop 成功事件奖励注入；
- 真实机器人安全事件惩罚注入。

所以如果你问“现在这套代码的 reward function 是什么”，最准确的回答是：

- 它不是 trainer 内部自定义的复杂奖励函数；
- 它就是环境 reward 的 chunk 级折扣聚合。

### 5.4.4 真实机器人 `gym_manipulator` 路径里另有 reward 机制

这也是之前最容易被忽略的一点。

在 `gym_manipulator` 那条代码路径里：

- `RobotEnv.step(...)` 默认 reward 是 `0.0`；
- 但 action/env processor 可以额外注入 reward；
- 例如 teleop `success` 事件可以把 reward 设为 `1.0`；
- reward classifier 也可以在成功时把 reward 设成 `success_reward`。

但当前 SmolVLA hybrid trainer 没有接这条链。

因此：

- 当前 hybrid trainer 的 reward 来自标准 env.step；
- 当前真实机器人/HIL reward processor 是另一条尚未并入 hybrid trainer 的代码路径。

## 6. 当前 hybrid 训练真实发生了什么

当你启动 hybrid trainer 时，真实流程可以分成四段。

### 6.1 初始化阶段

会完成：

- 校验配置；
- 构建 offline dataset；
- 构建 SmolVLA policy；
- 构建 preprocessor / postprocessor；
- 构建 env 及 env processors；
- 构建 optimizer 和 scheduler；
- 构建 collector；
- 构建 replay buffer。

### 6.2 可选 warmup 阶段

如果配置了：

- `collector.warmup_chunks > 0`

则 trainer 会先让 collector 收一部分 chunk 填满 buffer。

这一步不是：

- “纯 offline 训练”

也不是：

- “纯 RL 训练”

而只是：

- “先让 replay buffer 不是空的”。

### 6.3 主循环阶段

每个 step 执行：

1. `collector.collect(...)`
2. `next(dl_iter)` 拿 offline batch
3. `replay_buffer.sample(...)` 拿 online batch
4. `policy.forward(offline_batch)`
5. `compute_online_losses(...)`
6. `total_loss = offline + online + value`
7. `backward()`
8. `optimizer.step()`
9. `scheduler.step()`

### 6.4 日志 / 保存 / 评估阶段

主循环里还会按频率触发：

- `log_freq`
- `save_freq`
- `eval_freq`

当前模板里默认把：

- `eval_freq=0`

这是因为当前重点是先把训练链路稳定起来，而不是在这里额外堆复杂评估。

## 7. SO101 上机执行与录制

### 7.1 为什么单独做了 `lerobot_record_so101_policy.py`

之前直接用通用 `lerobot_record.py` 时，SO101 场景里存在几个现实问题：

1. CLI 面太大，不利于快速复用；
2. SmolVLA 需要 rename map；
3. 网络隔离环境下，Transformers tokenizer 可能会反复尝试联网；
4. 需要更直接地封装 SO101 follower、top/wrist camera、可选 leader 这些默认参数。

因此新加了：

- `src/lerobot/scripts/lerobot_record_so101_policy.py`

它的职责边界要说得很硬：

- 这是策略执行 / 录制 / 评估数据采集 wrapper；
- 不是 hybrid trainer；
- 不做在线参数更新；
- 不承担 RL reward 计算或 replay buffer 采样职责。

### 7.2 这个上机脚本解决了什么问题

这个脚本已经内置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `HF_HUB_DISABLE_TELEMETRY=1`
- `TOKENIZERS_PARALLELISM=false`

这能直接规避此前出现过的这类现象：

- 模型明明本地已有快照；
- 但 tokenizer 或配置仍在尝试访问 `huggingface.co`；
- 结果在无网环境下一路重试，拖慢甚至阻塞启动。

### 7.3 当前 SO101 默认参数

当前脚本默认的 SO101 相关参数是：

- `robot_id = my_so101`
- `robot_port = /dev/ttyACM0`
- `robot_calibration_dir = /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`
- `top_camera_index = 4`
- `wrist_camera_index = 6`
- `camera_width = 640`
- `camera_height = 480`
- `camera_fps = 30`

任务文本默认：

- `Put the block in the bin`

### 7.4 上机执行脚本

本目录已经把调用它的 shell wrapper 也整理好了：

- `tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh`

典型调用：

```bash
bash tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh
```

如果要显式指定模型和输出目录：

```bash
POLICY_PATH=/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model \
DATASET_ROOT=/data/tfj/lerobot_tfj/outputs/eval_smolvla_so101_manual \
NUM_EPISODES=1 \
EPISODE_TIME_S=300 \
bash tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh
```

## 8. 现阶段最重要的限制与风险

### 8.1 不要把 hybrid trainer 误认为真实机器人 RL 已完成

这是当前最需要反复强调的结论。

原因包括：

- 当前 collector 只支持单 env；
- 当前实现是标准 env 路径，不是专门的 SO101 真机 RL collector；
- reward 链路没有完成真实机器人闭环；
- 没有安全控制壳；
- 没有 actor-learner 解耦；
- 没有更稳的 critic / Q 网络体系。

把这件事放到 reward 上就是：

- 当前 hybrid trainer 已经有“如何消费 reward”的机制；
- 但还没有一套面向 SO101 真机任务、经过完整工程定义和验证的 reward 生成机制。

### 8.2 value critic 仍然比较弱

目前只有：

- 单个 `V(s)` 头；
- 一步 bootstrap；
- 无 target network；
- 无 double critic；
- 无 Q(s, a)。

这意味着：

- 这套 RL 信号更像“对 flow loss 做方向性加权”；
- 而不是完整、强力、稳定的大规模 off-policy RL。

### 8.3 当前 collector 的 credit assignment 是 chunk 粒度

现在 replay buffer 里一条样本对应一个动作块。

好处：

- 与 SmolVLA 的 chunk actor 对齐；
- 实现简单；
- 第一版更容易跑通。

代价：

- reward 分配更粗；
- chunk 内每一步的差异没有被精细建模；
- 若未来要强化 credit assignment，collector 与 loss 都需要进一步细化。

## 9. 推荐的项目推进方式

如果后面要继续把这项工作往前推，建议节奏如下。

### 第一步：固定离线训练基线

先确保下面这组组合始终稳定：

- 本地基础权重路径；
- trimmed dataset；
- `pyav`；
- rename map；
- 输出目录管理；
- 日志与监控。

### 第二步：在仿真环境里做 hybrid RL 原型验证

目标不是一开始就追求“机器人上线”，而是先验证：

- online flow loss 是否能带来可见提升；
- value head 是否稳定；
- warmup / replay / chunks_per_step 等超参是否合理；
- 训练是否会出现 value collapse 或权重爆炸。

### 第三步：再考虑真实机器人 collector 与 reward 闭环

只有当第二步结论稳定之后，才值得继续做：

- SO101 真机 collector
- reward processor
- 安全限幅
- 人工接管
- 失败恢复

否则会很容易把“算法不稳定”和“系统接口没接好”混在一起。

## 10. 本目录文件应该怎么读

如果你是从零接手这个目录，建议阅读顺序如下。

1. 先读 `smolvla_rl_architecture_and_integration_20260315_zh.md`
   - 理解当前 hybrid 实现到底是什么。

2. 再读本文档
   - 理解项目应该怎么推进、脚本怎么用、哪些地方已经验证过。

3. 最后看 `scripts/`
   - 直接拿现成命令跑。

## 11. 最终结论

当前仓库里的 SmolVLA RL 工作，最适合被理解为：

- 一个建立在已训练 SmolVLA 基座之上的 hybrid 微调原型；
- trainer 内部是 offline + online 同步优化；
- 工程流程上仍然应采用“先离线、再 hybrid RL、最后上机验证”的三段式推进；
- 现阶段最稳妥的实践路径，是先把离线训练和仿真 hybrid 微调做扎实，再考虑真实 SO101 在线 RL 闭环。
