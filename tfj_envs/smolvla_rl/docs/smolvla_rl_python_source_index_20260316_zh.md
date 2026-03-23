# SmolVLA RL Python 源码索引

这份索引只回答一个问题：

- 当前 SmolVLA hybrid RL 相关的真正 Python 代码在哪些文件里；
- 各文件分别负责什么；
- reward、loss、collector、value head、训练入口分别在哪里。

## 1. 先说结论

`./tfj_envs/smolvla_rl/` 下面我放的是：

- 文档
- shell 启动脚本

真正的训练逻辑和模型逻辑，不在 `tfj_envs/` 下面，而是在仓库源码目录：

- `src/lerobot/...`

这是因为：

- `tfj_envs/smolvla_rl/` 的定位是“归档与操作入口”
- 真正的 Python 实现必须留在 LeRobot 的源码路径里，才能被训练入口和包导入机制直接使用

## 2. 我新增或改动过的 Python 文件

### 2.1 新增的训练与运行入口

- [src/lerobot/scripts/lerobot_train_smolvla_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_train_smolvla_hybrid.py)
  - hybrid RL 的训练入口

- [src/lerobot/scripts/lerobot_record_so101_policy.py](/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record_so101_policy.py)
  - SO101 上机执行 / 录制入口
  - 这个是我明确新增的独立 Python 脚本

### 2.2 新增的 hybrid RL 核心实现

- [src/lerobot/configs/train_smolvla_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_smolvla_hybrid.py)
  - hybrid trainer 的配置定义

- [src/lerobot/rl/smolvla_hybrid/trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/trainer.py)
  - hybrid 训练主循环

- [src/lerobot/rl/smolvla_hybrid/losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py)
  - online RL 损失函数定义

- [src/lerobot/rl/smolvla_hybrid/collector.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/collector.py)
  - 在线收集 action chunk transition

- [src/lerobot/rl/smolvla_hybrid/buffer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/buffer.py)
  - chunk replay buffer

- [src/lerobot/policies/smolvla/rl_types.py](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/rl_types.py)
  - prefix context / chunk prediction 的结构体定义

### 2.3 改动过的 SmolVLA 模型文件

- [src/lerobot/policies/smolvla/configuration_smolvla.py](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/configuration_smolvla.py)
  - 增加 value head 相关配置

- [src/lerobot/policies/smolvla/modeling_smolvla.py](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py)
  - 增加 shared prefix context、value head、RL 调用接口

## 3. reward 代码在哪

### 3.1 当前 hybrid trainer 实际使用的 reward

如果你要看“当前 SmolVLA hybrid trainer 实际怎么吃 reward”，先看：

- [collector.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/collector.py#L126)

这里的关键逻辑是：

- `discounted_reward = 0.0`
- `bootstrap_discount = 1.0`
- 每执行一步：
  - `next_observation, reward, terminated, truncated, _ = self.env.step(action_np)`
  - `discounted_reward += bootstrap_discount * float(reward[0])`
  - `bootstrap_discount *= self.discount`

也就是说：

- 当前 hybrid trainer 的 reward 不是在 trainer 里手写的任务奖励函数
- 它直接用 `env.step(...)` 返回的 reward
- 然后把一个 action chunk 内的 reward 按折扣累加成 chunk reward

### 3.2 当前 hybrid trainer 的 value target 在哪

看：

- [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L41)

关键代码：

- `values = policy.get_value(online_batch.observation)`
- `next_values = policy.get_value(online_batch.next_observation)`
- `value_targets = online_batch.reward + online_batch.bootstrap_discount * next_values`

也就是说：

- value target = chunk reward + 残余折扣 * next value

## 4. RL 损失函数代码在哪

如果你要看“损失函数到底怎么写的”，直接看：

- [src/lerobot/rl/smolvla_hybrid/losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py)

这里有三块核心逻辑。

### 4.1 online actor loss

关键代码在：

- [losses.py:31](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L31)
- [losses.py:38](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L38)
- [losses.py:54](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L54)

逻辑是：

- 对 online batch action 加噪
- 调 `policy.compute_fm_score(...)`
- 计算 flow matching MSE
- 再用 advantage-derived weight 重加权

### 4.2 value loss

关键代码在：

- [losses.py:55](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L55)

就是：

- `value_loss = F.mse_loss(values, value_targets)`

### 4.3 advantage 权重

关键代码在：

- [losses.py:46](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L46)
- [losses.py:50](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L50)
- [losses.py:52](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py#L52)

这部分就是：

- `advantages = value_targets - values.detach()`
- clip
- temperature scaling
- `weights = exp(...)`

## 5. 总损失代码在哪

如果你要看“offline + RL + value 最后怎么合起来”，看：

- [trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/trainer.py#L203)

关键代码是：

- `offline_loss, _ = policy.forward(offline_batch)`
- `online_policy_loss, value_loss, online_metrics = compute_online_losses(...)`
- `total_loss = offline_weight * offline_loss + online_flow_weight * online_policy_loss + value_weight * value_loss`

也就是当前训练总损失：

- 离线 SmolVLA loss
- 在线 advantage-weighted flow loss
- value regression loss

三者一起优化。

## 6. collector 和 replay buffer 代码在哪

### 6.1 collector

- [src/lerobot/rl/smolvla_hybrid/collector.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/collector.py)

它负责：

- 从当前 observation 构造 policy 输入
- 调 `policy.predict_action_chunk_with_info(...)`
- 把 action chunk 执行进环境
- 聚合 chunk reward
- 构造 transition

### 6.2 replay buffer

- [src/lerobot/rl/smolvla_hybrid/buffer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/buffer.py)

它负责：

- 定义 `SmolVLAChunkTransition`
- 定义 `SmolVLAChunkBatch`
- 存储 `reward` / `done` / `bootstrap_discount`
- 提供 `sample(...)`

## 7. shared prefix context 和 value head 代码在哪

如果你要看“共享前缀表征”和“value head”是怎么写进 SmolVLA 的，看：

- [src/lerobot/policies/smolvla/modeling_smolvla.py](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py)

### 7.1 value head 的定义

关键位置：

- [modeling_smolvla.py:593](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L593)
- [modeling_smolvla.py:610](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L610)

### 7.2 prefix context 的编码

关键位置：

- [modeling_smolvla.py:657](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L657)

### 7.3 value 从 prefix context 里读

关键位置：

- [modeling_smolvla.py:693](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L693)

### 7.4 flow score 从 prefix context 里读

关键位置：

- [modeling_smolvla.py:708](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L708)

### 7.5 推理时一边采样动作块，一边拿 value

关键位置：

- [modeling_smolvla.py:960](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py#L960)

## 8. hybrid trainer 的配置代码在哪

如果你要看：

- reward discount 是多少
- online loss weight 是多少
- value loss weight 是多少
- replay buffer 大小是多少

看：

- [src/lerobot/configs/train_smolvla_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_smolvla_hybrid.py)

尤其是：

- [train_smolvla_hybrid.py:41](/data/tfj/lerobot_tfj/src/lerobot/configs/train_smolvla_hybrid.py#L41)

这里定义了：

- `offline_loss_weight = 1.0`
- `online_flow_loss_weight = 0.3`
- `value_loss_weight = 1.0`
- `discount = 0.99`
- `advantage_temperature = 1.0`
- `normalize_advantage = True`

## 9. SO101 上机 Python 文件在哪

如果你要看“上机执行的 Python 入口”，看：

- [src/lerobot/scripts/lerobot_record_so101_policy.py](/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record_so101_policy.py)

这个文件负责：

- 强制离线模式环境变量
- SO101 follower 默认参数
- top / wrist camera 默认参数
- SmolVLA 的 rename map 自动补齐

它不是 RL loss 文件，也不是 reward 文件，而是上机执行 / 录制入口。

## 10. 最后一句最直接的话

如果你现在只想看“奖励函数代码”和“损失函数代码”，你先打开这三个文件就够了：

1. [src/lerobot/rl/smolvla_hybrid/collector.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/collector.py)
2. [src/lerobot/rl/smolvla_hybrid/losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/losses.py)
3. [src/lerobot/rl/smolvla_hybrid/trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/smolvla_hybrid/trainer.py)

如果你想看“SmolVLA 模型内部是怎么被改成带 value head 的”，再看：

4. [src/lerobot/policies/smolvla/modeling_smolvla.py](/data/tfj/lerobot_tfj/src/lerobot/policies/smolvla/modeling_smolvla.py)
