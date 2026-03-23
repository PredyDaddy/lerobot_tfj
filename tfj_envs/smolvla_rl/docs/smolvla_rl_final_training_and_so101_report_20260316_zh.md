# SmolVLA RL 最终训练完成与 SO101 上机报告

本文档记录 2026-03-16 这次 SmolVLA hybrid RL 训练完成、checkpoint 固化、SO101 上机命令整理，以及相关工程修复项。

这份报告只讲已经实际完成并验证过的内容，不讲未落地的设想。

## 1. 最终结论

截至 2026-03-16，本仓库已经实际完成一轮 SmolVLA hybrid RL 训练，结果如下：

- 训练输出目录：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221`
- 最终 checkpoint：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/005000`
- 当前 `last` 软链接：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last -> 005000`
- 最终可直接用于上机的模型目录：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model`
- 训练结束信号：
  - 训练日志在 `step=5000` 后打印 `End of SmolVLA hybrid training`

这次训练不是单纯离线 SmolVLA。

训练日志中持续出现：

- `online_policy_loss`
- `value_loss`
- `buffer_size`
- `done_ratio`

这说明 RL 分支在训练过程中确实参与了参数更新。

## 2. 本次训练建立在什么基础上

这次 hybrid RL 训练不是从零开始，而是建立在一轮已经跑通的离线 SmolVLA 结果之上。

离线基座 checkpoint：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model`

离线 demonstration 数据集：

- `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`

本地视觉 backbone：

- `/home/cqy/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467`

这意味着当前项目的实际训练路径是：

1. 先离线训练 SmolVLA 行为策略。
2. 再把这个离线 checkpoint 作为 hybrid RL 的初始化策略。
3. 再得到一份经过 RL 微调后的 actor 权重。
4. 最后再把这份 RL 微调后的 actor 拿去 SO101 上机。

## 3. 这次为了让 RL 真正跑起来，做了哪些工程修复

这部分很重要，因为如果只看最终命令，会误以为 RL 一开始就能直接训练。

实际上，为了让这条链路真的可跑，中间做了几处必要修复。

### 3.1 补了本地 `gym_aloha` 代理环境

当前机器缺少外部环境依赖：

- `gym_aloha`
- `gym_pusht`
- `libero`
- `metaworld`
- `gym_hil`

因此新增了一个最小本地环境包，用来把 hybrid trainer 的 RL 链路跑通：

- `/data/tfj/lerobot_tfj/gym_aloha/__init__.py`
- `/data/tfj/lerobot_tfj/gym_aloha/simple_aloha_env.py`

它注册的环境 id 是：

- `gym_aloha/AlohaInsertion-v0`

这个环境提供：

- 6 维状态 `agent_pos`
- top / wrist 两路图像
- 6 维连续动作
- 一个可计算的 reward

这一步的作用不是“伪造真实任务成功”，而是：

- 先让 SmolVLA hybrid RL 的训练闭环真正跑起来
- 验证 replay buffer、collector、online flow loss、value loss、checkpoint 保存、恢复路径是否都正确

### 3.2 修复 hybrid 启动脚本的 `PYTHONPATH`

之前 `launch_smolvla_hybrid_train.sh` 只把 `src` 加进了 `PYTHONPATH`，导致新建的本地 `gym_aloha` 包无法被导入。

修复后，脚本同时加入：

- repo root
- repo root 下的 `src`

这样环境包和源码包都能被正确导入。

### 3.3 修复 SmolVLA prefix-only 编码崩溃

修复文件：

- `/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py`

修复点：

- `encode_prefix_context(...)` 内部把 `fill_kv_cache=use_cache` 改成了 `fill_kv_cache=True`

原因：

- prefix-only 路径下没有 expert suffix embedding
- 但 cross-attention expert 分支仍会尝试走依赖 cache 的逻辑
- 最终会触发 `NoneType` 相关崩溃

这个问题不修，RL 训练过程中的 prefix 编码不会稳定工作。

### 3.4 修复 online action 维度不匹配

修复文件：

- `/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py`

问题现象：

- online env 给出的 action 维度是 6
- 但 SmolVLA 内部 flow matching 路径按 `max_action_dim=32` 组织张量
- 直接计算会出现矩阵乘法维度错误

修复方式：

- 小于 `max_action_dim` 时对 online action 右侧补零
- 大于 `max_action_dim` 时做截断

这一步修完以后，RL online loss 才能稳定回传。

### 3.5 明确了后台训练监控方式

在当前代理执行环境里，普通 `nohup ... &` 启动的后台子进程会被宿主会话清理。

这不是仓库本身的问题，而是代理执行环境的生命周期限制。

因此这次训练监控采用了持久会话方式，而不是依赖普通 shell 后台。

这条结论只影响“代理帮你监控”的方式，不影响你在自己机器 shell 里正常用 `nohup` 或 `tmux`。

## 4. 本次 RL 训练到底用的是什么 reward

这次训练的 reward 来自本地代理环境：

- `/data/tfj/lerobot_tfj/gym_aloha/simple_aloha_env.py`

核心逻辑是：

1. 环境维护一个 6 维状态向量 `state`
2. 环境有一个 6 维目标向量 `target`
3. 每步执行动作后，计算当前状态到目标状态的欧氏距离 `dist`
4. 奖励定义为：

```text
reward = -dist
if success:
    reward += 2.0
```

其中 success 条件是：

```text
dist < 0.12
```

也就是说，这次用于 RL 微调的即时奖励函数是：

```text
r(s, a, s') = - || state - target ||_2 + 2.0 * 1[|| state - target ||_2 < 0.12]
```

这不是 SO101 真机任务的语义奖励，而是一个工程代理奖励。

它的定位是：

- 用来驱动 hybrid RL 训练链路跑通
- 让 actor 在一个可计算 reward 的连续动作环境里接受 online signal

它不能直接等价解释为真实的 “Put the block in the bin” 奖励函数。

## 5. 训练时实际优化的损失函数是什么

这次训练的总损失不是单个 loss，而是三部分加权和。

对应代码：

- `/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/trainer.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py`

### 5.1 offline actor loss

这部分来自 SmolVLA 原本的离线监督训练目标。

在 trainer 里表现为：

```text
offline_loss = policy.forward(offline_batch)
```

它的作用是：

- 继续保持离线 demonstration 行为先验
- 防止 online RL 信号把 actor 快速拉偏

### 5.2 online actor loss

这部分是 advantage-weighted flow matching loss。

当前实现流程是：

1. 从 replay buffer 采样在线 chunk
2. 对目标动作加噪
3. 调 `policy.compute_fm_score(...)`
4. 计算 flow matching MSE
5. 用 advantage 形成样本权重，再加权平均

核心形式可以写成：

```text
L_online_actor = mean( w_i * || f_theta(x_i, noisy_a_i, t_i) - target_flow_i ||^2 )
```

其中：

- `target_flow = noise - action`
- `w_i = exp(clipped_advantage / temperature)`，再做上限截断

### 5.3 value loss

value head 的目标是拟合 bootstrapped target：

```text
value_target = chunk_reward + bootstrap_discount * next_value
```

实际 value loss 是：

```text
L_value = MSE(value, value_target)
```

### 5.4 总损失

这次训练使用的权重是：

- `offline_loss_weight = 1.0`
- `online_flow_loss_weight = 0.3`
- `value_loss_weight = 1.0`

因此总损失为：

```text
L_total = 1.0 * L_offline + 0.3 * L_online_actor + 1.0 * L_value
```

这说明：

- value branch 在训练里是强约束，不是装饰项
- RL actor loss 存在，但权重小于 offline actor loss
- 当前实现更接近“保守的在线微调”，而不是强探索型 RL

## 6. chunk reward 是怎么从环境奖励聚合出来的

这里还有一层经常会被忽略：

- hybrid trainer 不是一步动作一步训练
- 它是按 action chunk 进行 rollout 和存储

对应代码：

- `/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/collector.py`

collector 会在一个 action chunk 内：

1. 逐步执行动作
2. 每步从 `env.step(...)` 拿到即时 reward
3. 用 discount 进行折扣累加
4. 最后把这个 chunk 的折扣累计 reward 写进 replay buffer

也就是说，value 学到的不是单步 reward，而是 chunk 级别的 bootstrapped target。

## 7. 本次实际训练配置

本次最终完成的 hybrid RL 训练主要参数如下：

- `env.type=aloha`
- `env.task=AlohaInsertion-v0`
- `env.obs_type=pixels_agent_pos`
- `steps=5000`
- `batch_size=8`
- `num_workers=4`
- `save_freq=500`
- `log_freq=20`
- `collector.n_envs=1`
- `collector.use_async_envs=false`
- `collector.chunks_per_step=1`
- `collector.warmup_chunks=0`
- `replay_buffer.capacity=4096`
- `replay_buffer.online_batch_size=16`
- `losses.offline_loss_weight=1.0`
- `losses.online_flow_loss_weight=0.3`
- `losses.value_loss_weight=1.0`
- `losses.discount=0.99`
- `losses.advantage_temperature=1.0`

初始化 actor 来自：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model`

## 8. 可直接复制执行的训练指令

这一节不讲原理，只给可以直接用的命令。

### 8.1 离线 SmolVLA 训练

最推荐的前台启动方式：

```bash
cd /data/tfj/lerobot_tfj
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh
```

这条命令会默认使用：

- 数据集：
  - `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`
- 视频后端：
  - `pyav`
- batch size：
  - `32`
- steps：
  - `10000`
- save freq：
  - `2000`
- device：
  - `cuda`

如果要显式指定参数，可以这样写：

```bash
cd /data/tfj/lerobot_tfj
RUN_TAG=offline_$(date +%Y%m%d_%H%M%S) \
BATCH_SIZE=32 \
STEPS=10000 \
SAVE_FREQ=2000 \
NUM_WORKERS=4 \
DEVICE=cuda \
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh
```

如果你想在你自己的 shell 里后台跑：

```bash
cd /data/tfj/lerobot_tfj
RUN_TAG=offline_$(date +%Y%m%d_%H%M%S) \
MONITOR_INTERVAL=60 \
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/start_smolvla_offline_trimmed_train_nohup.sh
```

### 8.2 Hybrid RL 训练

最推荐的前台启动方式：

```bash
cd /data/tfj/lerobot_tfj
STEPS=5000 \
SAVE_FREQ=500 \
LOG_FREQ=20 \
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh \
  aloha \
  AlohaInsertion-v0 \
  --env.obs_type=pixels_agent_pos
```

这条命令默认会：

- 读取离线基座：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341/checkpoints/last/pretrained_model`
- 读取数据集：
  - `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`
- 使用本地 SmolVLM2 backbone
- 训练 5000 step
- 每 500 step 保存一次 checkpoint

如果你想在自己的 shell 里后台跑：

```bash
cd /data/tfj/lerobot_tfj
RUN_TAG=hybrid_$(date +%Y%m%d_%H%M%S) \
STEPS=5000 \
SAVE_FREQ=500 \
LOG_FREQ=20 \
MONITOR_INTERVAL=90 \
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/start_smolvla_hybrid_train_nohup.sh \
  aloha \
  AlohaInsertion-v0 \
  --env.obs_type=pixels_agent_pos
```

### 8.3 复现这次已经完成的 RL 训练结果

如果你的目标不是“重新设计参数”，而是尽量复现本次已经完成的那轮训练，那么建议按下面这条思路来理解：

- 训练配置：见本报告第 7 节
- 最终产物：
  - `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model`

也就是说，当前最重要的不是重新跑一遍一模一样的 `RUN_TAG`，而是直接使用已经训练好的这份 RL checkpoint 做上机验证。

## 9. 最终训练结果

### 8.1 输出目录与日志

训练输出目录：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221`

训练日志：

- `/data/tfj/lerobot_tfj/outputs/logs/smolvla_hybrid_aloha_live_20260316_111221.train.log`

### 8.2 最终 checkpoint 列表

当前该目录下已保存的 checkpoint 包括：

- `000500`
- `001000`
- `001500`
- `002000`
- `002500`
- `003000`
- `003500`
- `004000`
- `004500`
- `005000`
- `last -> 005000`

### 8.3 最终一步指标

日志最后一条 step 指标是：

- `step=5000`
- `loss=2767.977296447754`
- `offline_loss=0.024926905147731306`
- `online_policy_loss=0.06116455681622028`
- `value_loss_total=2767.9340270996095`
- `buffer_size=4096.0`
- `done_ratio=0.128125`

训练结束信号是：

- `End of SmolVLA hybrid training`

### 8.4 这说明了什么

这组结果至少证明了下面几点：

- hybrid RL 训练主循环可以稳定跑满 5000 step
- replay buffer 能正常装满到 4096
- online actor loss 和 value loss 全程参与更新
- checkpoint 保存链路、`last` 软链接更新、训练结束标志都正常

但这组结果不等价于下面这些结论：

- 不等价于 “真实 SO101 任务已经被 RL 成功解决”
- 不等价于 “真实机器人上一定优于离线基座”
- 不等价于 “代理 reward 已经准确表达真实抓放任务目标”

## 10. SO101 上机指令

### 10.1 当前默认上机脚本

已经更新的上机 wrapper：

- `/data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh`

仓库通用入口也同步更新：

- `/data/tfj/lerobot_tfj/scripts/run_so101_policy_record.sh`

这两个脚本现在默认都会指向：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model`

### 10.2 推荐直接执行命令

最省事的跑法：

```bash
cd /data/tfj/lerobot_tfj
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh
```

如果要先清理旧评测目录再跑：

```bash
cd /data/tfj/lerobot_tfj
CLEAR_DATASET_ROOT=1 bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh
```

### 10.3 对应的显式 Python 命令

```bash
PYTHONPATH=/data/tfj/lerobot_tfj/src python /data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record_so101_policy.py \
  --policy.path=/data/tfj/lerobot_tfj/outputs/train/smolvla_hybrid_aloha_live_20260316_111221/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --robot_port=/dev/ttyACM0 \
  --robot_id=so101_follower \
  --robot_calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top_camera_index=4 \
  --wrist_camera_index=6 \
  --camera_width=640 \
  --camera_height=480 \
  --camera_fps=30 \
  --camera_warmup_s=1 \
  --task="Put the block in the bin" \
  --dataset_repo_id=local/eval_smolvla_rl_so101 \
  --dataset_root=./outputs/eval_smolvla_rl_so101 \
  --dataset_fps=30 \
  --num_episodes=1 \
  --episode_time_s=300 \
  --reset_time_s=15 \
  --dataset_video=true \
  --display_data=false \
  --play_sounds=false \
  --clear_dataset_root=false
```

如果要清空旧目录再上机，把最后一项改成：

```bash
--clear_dataset_root=true
```

## 11. 这次上机命令踩到的实际坑

这次已经真实踩到一个参数解析坑：

- `--clear_dataset_root=0`

会报错：

- `Couldn't parse '0' into a bool`

原因是：

- `draccus` 对布尔值要求是 `true/false`
- 不接受 `0/1`

因此我已经把两个 shell wrapper 都改成了布尔兼容模式：

- `0` 会自动转成 `false`
- `1` 会自动转成 `true`
- `true/false/yes/no/on/off` 也都能识别

这意味着：

- 直接调 Python 时请写 `true/false`
- 调 wrapper 时写 `0/1` 或 `true/false` 都可以

## 12. RL 在上机推理时到底有没有被用到

这个问题必须说清楚。

答案是：

- 有，用到了 RL 微调后的 actor 权重
- 但没有在上机过程中再做 RL 更新
- value head 也不参与机器人动作决策

更精确地说：

1. 训练阶段：
   - actor 通过 offline loss 和 online flow loss 一起被更新
   - value head 通过 value loss 被更新
2. 上机推理阶段：
   - 用的是训练后保存下来的 policy actor 权重
   - 机器人执行时不会再在线反传或更新参数
   - value head 不负责出最终动作

所以：

- “RL 有用到”指的是 actor 已经被 RL 训练过
- 不是指“上机时仍在做 RL”

## 13. 当前结果的正确边界

这次成果可以准确描述成：

- 已经把 RL 真正接进了 SmolVLA 的训练链路
- 已经完成一轮从离线基座出发的 hybrid RL 微调
- 已经产出可供 SO101 上机的 RL 微调 checkpoint
- 已经把 SO101 上机默认入口改到这份 RL checkpoint

但当前还不能过度描述成：

- 已经完成真实 SO101 任务的在线 RL 闭环
- 已经得到真实任务语义级奖励函数
- 已经证明 RL checkpoint 在真机上稳定优于离线 checkpoint

这些结论还需要下一阶段的真实上机对比验证。

## 14. 本次涉及的关键文件

本次实际新增或修改过、和最终结果直接相关的关键文件包括：

- `/data/tfj/lerobot_tfj/gym_aloha/__init__.py`
- `/data/tfj/lerobot_tfj/gym_aloha/simple_aloha_env.py`
- `/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py`
- `/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py`
- `/data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh`
- `/data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/start_smolvla_hybrid_train_nohup.sh`
- `/data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh`
- `/data/tfj/lerobot_tfj/scripts/run_so101_policy_record.sh`
- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record_so101_policy.py`

## 15. 现在最推荐的下一步

如果目标是继续推进真实机器人验证，建议按这个顺序来：

1. 先用当前 RL checkpoint 完成一轮 SO101 上机测试。
2. 保留 eval dataset，记录成功率、轨迹长度、动作稳定性。
3. 用同样的上机设置，对比离线基座 checkpoint 和 RL 微调 checkpoint。
4. 如果 RL 版本没有明显收益，再决定是否要把代理环境 reward 换成更贴近任务语义的设计。

这样推进才是可验证、可归因的工程路径。
