# SmolVLA RL 架构与接入说明

本文档只描述当前仓库里已经存在的 SmolVLA hybrid RL 实现，不把“未来可能怎么改”混进“现在代码实际上怎么跑”里。目标是把下面几件事讲清楚：

1. SmolVLA 原本是怎么生成动作的。
2. RL 现在是怎么接进 SmolVLA 的。
3. `value_head`、`prefix context`、`flow score` 分别承担什么职责。
4. 当前 trainer 到底是“分段训练”还是“同一步混合训练”。
5. 这套实现更像哪一种 RL 近似，以及它和真正完整在线 RL 还有哪些差距。

## 1. 先给结论

当前仓库里的实现不是把 SmolVLA 改造成了一个完整的 PPO / SAC / DDPG 风格在线 RL 框架，而是做了一套更接近如下形式的混合训练：

- actor 主体仍然是 SmolVLA 原本的 flow-matching 动作块生成器；
- 在同一个观测前缀表征上新增了一个 `state value` 头；
- 在线数据上的 actor 更新，不是传统 policy gradient，而是 advantage-weighted 的 flow-matching loss；
- 训练循环里同时吃 offline dataset batch 和 online replay batch；
- 因此它更像一个 `advantage-weighted generative actor + value critic` 的原型实现。

一句话压缩：

- 微观上：每个 optimizer step 同时优化 offline loss 和 online RL loss。
- 宏观上：更合理的使用方式仍然是先拿离线数据把 SmolVLA 训到能用，再进入 hybrid RL 微调。

## 2. 相关代码入口

这套实现的主入口分成两层。

第一层是训练入口：

- `src/lerobot/scripts/lerobot_train_smolvla_hybrid.py`
- `src/lerobot/configs/train_smolvla_hybrid.py`
- `src/lerobot/rl/smolvla_hybrid/trainer.py`

第二层是模型侧 RL 接口：

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/rl_types.py`

配套的在线数据链路：

- `src/lerobot/rl/smolvla_hybrid/collector.py`
- `src/lerobot/rl/smolvla_hybrid/buffer.py`
- `src/lerobot/rl/smolvla_hybrid/losses.py`

## 3. SmolVLA 原本的动作生成机制

### 3.1 它不是直接回归 action

SmolVLA 在这里不是“看观测，直接吐出一个动作向量”的结构，也不是“离散 token 自回归”的结构。它是条件 flow-matching 动作生成器。

核心特征：

- 输入前缀是图像、语言、状态；
- 输出不是一步 action，而是一整个 `action chunk`；
- 推理从噪声动作块开始；
- 模型在给定时间步 `t` 和观测前缀的条件下，预测一个 flow / velocity；
- 再通过离散积分把噪声逐步推回到可执行动作块。

关键代码都在 `src/lerobot/policies/smolvla/modeling_smolvla.py`。

### 3.2 训练目标本质

SmolVLA 原始训练的目标可以概括为：

1. 从真实动作块 `actions` 出发；
2. 采样噪声 `noise`；
3. 采样时间 `t`；
4. 构造 noisy action：
   - `x_t = t * noise + (1 - t) * actions`
5. 构造目标 flow：
   - `u_t = noise - actions`
6. 模型预测 `v_t`；
7. 用 `MSE(u_t, v_t)` 训练。

这件事的直觉是：

- 模型学习的是“如何在给定观测条件下，把一个 noisy action chunk 朝着真实 action chunk 推过去”。

### 3.3 推理时怎么生成动作块

推理路径关键在：

- `SmolVLAPolicy.predict_action_chunk_with_info(...)`
- `VLAFlowMatching.sample_actions_with_info(...)`
- `VLAFlowMatching.compute_fm_score_from_prefix_context(...)`

执行逻辑可以概括为：

1. 编码观测前缀；
2. 初始化高斯噪声动作块；
3. 反复调用 flow score 网络；
4. 用 Euler 风格更新 `x_t`；
5. 最后得到整个 action chunk；
6. 执行时再从 chunk queue 里逐个取出 action。

当前 hybrid RL 没有把这套 actor 换掉，而是保留了它。

## 4. RL 是怎么接进 SmolVLA 的

### 4.1 总体思路

当前实现的核心思路不是替换 actor，而是在原 actor 周围补出 RL 所需的最小闭环：

1. 保留原始 SmolVLA flow-matching actor。
2. 在共享的观测前缀表征上增加 `value_head`。
3. 让模型能够单独暴露：
   - 观测值函数 `get_value(...)`
   - 给 noisy action 打分的 `compute_fm_score(...)`
   - 带额外诊断信息的动作块采样 `sample_actions_with_info(...)`
4. 在在线 batch 上，根据 `reward + discount * next_value` 计算 advantage。
5. 用 advantage 去重加权 actor 的 flow loss。

也就是说，RL 不是把动作头替换成全新的策略头，而是把“flow-matching 生成器”继续当 actor，然后通过 value-guided reweighting 去做策略改进。

### 4.2 配置层面加了什么

在 `src/lerobot/policies/smolvla/configuration_smolvla.py` 里新增了 value head 相关超参：

- `value_head_hidden_dim`
- `value_head_num_layers`
- `value_head_dropout`
- `value_head_pooling`

这说明 RL 接入的第一步，是承认需要一个额外的 critic 分支。

### 4.3 模型层面加了什么

在 `src/lerobot/policies/smolvla/modeling_smolvla.py` 里，关键新增包括：

- `self.value_head = self._make_value_head()`
- `encode_prefix_context(...)`
- `get_value_from_prefix_context(...)`
- `get_value(...)`
- `compute_fm_score_from_prefix_context(...)`
- `compute_fm_score(...)`
- `sample_actions_with_info(...)`
- `predict_action_chunk_with_info(...)`

这些函数不是零散补丁，它们共同构成了一个很明确的结构：

- 观测前缀先被统一编码成 `prefix context`；
- critic 使用 pooled prefix feature 估计 `V(s)`；
- actor 使用同一个 prefix context 对 noisy action 计算 flow score；
- 采样动作块时可以一边生成 chunk，一边拿到对应 value 诊断。

### 4.4 为什么要引入 prefix context

`prefix context` 是当前接法里最关键的工程抽象之一。

它的作用有两个。

第一，避免重复算前缀：

- 在 flow matching 推理时，动作块是一步步去噪的；
- suffix 会随着时间步变化；
- 但图像、语言、状态组成的 prefix 在一次动作块采样中通常不变；
- 所以可以先把 prefix 编码好，再重复复用。

第二，让 actor / critic 共用观测表征：

- critic 不需要重新做一遍完整观测编码；
- actor 和 critic 在同一前缀表征上分叉；
- 这能让实现更紧凑，也让值函数与策略看到的是同一份条件信息。

### 4.5 `value_head` 在这套架构里的职责

当前 `value_head` 估计的是 `V(s)`，不是 `Q(s, a)`。

这点很关键，因为它直接决定了当前实现更像哪一类 RL。

`value_head` 的使用方式是：

1. 对当前 observation 估计 `values = V(s)`；
2. 对 next observation 估计 `next_values = V(s')`；
3. 构造 bootstrap target：
   - `target = reward + bootstrap_discount * next_values`
4. 用 MSE 回归 value；
5. 同时通过
   - `advantage = target - values.detach()`
   构造 actor reweighting 的权重。

这意味着：

- value head 同时服务 critic 学习和 actor 加权；
- 但它不是 Q critic，也没有 target network。

### 4.6 `compute_fm_score` 在 RL 里的地位

`compute_fm_score(...)` 仍然是 actor 的核心输出。

在线 RL 接入之后，它并没有变成“只在离线监督里用”的遗留函数，而是在线损失里继续被直接调用：

1. 从 replay buffer 取出在线 chunk action；
2. 对 action chunk 加噪；
3. 让模型预测 noisy action 在时间 `t` 的 flow；
4. 用目标 flow 和预测 flow 算每个样本的 flow loss；
5. 再由 advantage 派生的权重去重加权这些样本损失。

换句话说：

- RL 更新的 actor loss 仍然是 flow-matching loss；
- 只是这个损失不再一视同仁，而是被 value 信息加权。

## 5. 在线数据链路是什么样的

### 5.1 当前 replay buffer 存的不是 primitive action，而是 action chunk

`src/lerobot/rl/smolvla_hybrid/buffer.py` 里定义了：

- `SmolVLAChunkTransition`
- `SmolVLAChunkBatch`
- `SmolVLAChunkReplayBuffer`

它们存储的是 chunk-level transition，而不是传统单步 action-level transition。

transition 字段包括：

- `observation`
- `action`
- `reward`
- `next_observation`
- `done`
- `bootstrap_discount`

其中 `action` 是整个 chunk，`reward` 是 chunk rollout 后聚合出的折扣回报。

### 5.2 collector 如何收集 chunk transition

`src/lerobot/rl/smolvla_hybrid/collector.py` 中的 `SmolVLAChunkCollector.collect(...)` 做了以下事情：

1. 读取当前环境 observation；
2. 预处理 observation；
3. 调用 `policy.predict_action_chunk_with_info(...)` 得到动作块；
4. 后处理动作块，映射到 env action 空间；
5. 在环境中逐步执行 chunk 内的若干动作；
6. 把 rollout 期间的 reward 累加成 chunk-level discounted reward；
7. 构造 chunk transition；
8. 写入 replay buffer。

这里的设计选择是：

- 用动作块作为 RL 的 decision unit；
- 不是每个 primitive action 都单独做 transition。

这带来的好处是和 SmolVLA 的 chunk-based actor 对齐。

这带来的代价是：

- credit assignment 变粗；
- 一个 value 估计和一个聚合回报对应一整块动作。

## 6. 当前 online loss 到底怎么定义

`src/lerobot/rl/smolvla_hybrid/losses.py` 里的 `compute_online_losses(...)` 可以概括成下面的顺序。

### 6.1 先构造在线 actor 的 flow matching 目标

对于 replay batch 中的动作块：

1. 采样噪声；
2. 采样时间步；
3. 构造 noisy actions；
4. 构造 target flow；
5. 用 `policy.compute_fm_score(...)` 预测 flow；
6. 计算每个样本的 flow MSE。

### 6.2 再构造 value target

然后计算：

- `values = V(s)`
- `next_values = V(s')`
- `value_targets = reward + bootstrap_discount * next_values`

### 6.3 再用 advantage 给 actor loss 加权

当前 advantage 定义是：

- `advantage = value_targets - values.detach()`

之后会做：

- 可选标准化；
- clip；
- temperature scaling；
- 指数化得到权重；
- 再把权重乘到每个样本的 flow loss 上。

最终得到：

- `policy_loss = mean(weight * per_sample_flow_loss)`
- `value_loss = mse(values, value_targets)`

这就能看出它的 RL 风格：

- 不是 PPO 的 ratio-clipping；
- 不是 SAC 的 Q-learning；
- 不是直接对 log-prob 做 policy gradient；
- 而更像 AWR / IQL / AWAC 这一类“advantage-weighted regression”思想的生成式版本。

## 6.5 当前实现里的奖励函数到底是什么

这是一个必须单独讲清楚的问题。

当前仓库里的 `SmolVLA hybrid trainer` 自己并没有额外手写一套任务特定 reward function。它对 reward 的处理逻辑是：

1. reward 的原始来源不是 policy，也不是 hybrid trainer 本身；
2. reward 直接来自环境执行时的 `env.step(action)` 返回值；
3. collector 不做任务级启发式 shaping；
4. collector 只负责把一个 action chunk 内多步环境 reward 聚合成一个 chunk-level return。

也就是说，当前 hybrid trainer 的 reward 设计原则是：

- reward 定义属于 environment；
- hybrid trainer 只做 chunk 级聚合和 bootstrap。

### 6.5.1 当前 hybrid collector 实际使用的 reward 公式

在 `src/lerobot/rl/smolvla_hybrid/collector.py` 中，collector 对一个 action chunk 的处理是：

1. 让 policy 预测一个 action chunk；
2. 将 chunk 中的动作逐步送入环境；
3. 每一步读取环境返回的 `reward`；
4. 用折扣系数累加成一个 chunk 回报。

如果一个 chunk 内实际执行了 `H` 步，那么 collector 存进 replay buffer 的 reward 是：

`R_chunk = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(H-1) * r_{t+H-1}`

这里的 `gamma` 就是：

- `cfg.losses.discount`

collector 代码里对应的是：

- 初始 `discounted_reward = 0.0`
- 初始 `bootstrap_discount = 1.0`
- 每执行一步：
  - `discounted_reward += bootstrap_discount * reward`
  - `bootstrap_discount *= discount`

如果中途 `terminated` 或 `truncated`，就提前停止 chunk rollout。

### 6.5.2 value target 是怎么接上 reward 的

在 `src/lerobot/rl/smolvla_hybrid/losses.py` 里，当前 value target 不是单步 TD，而是 chunk 粒度的 n-step bootstrap target：

`value_target = R_chunk + beta * V(next_state_after_chunk)`

其中：

- `R_chunk` 是上面 collector 聚合出来的 chunk 折扣回报；
- `beta` 就是 collector 存下来的 `bootstrap_discount`；
- 如果 chunk 正常执行完 `H` 步且没有 done，那么 `beta = gamma^H`；
- 如果 chunk 中途 done / truncated，那么 `beta = 0`。

因此当前 value target 的含义其实很明确：

- 用一个 chunk 内的实际环境回报；
- 再加上 chunk 末状态的 bootstrap value；
- 构成 chunk-level return target。

### 6.5.3 为什么 reward 要按 chunk 聚合

当前这么设，不是随手写的，而是因为 actor 的决策单位本来就是 chunk。

更具体地说，有三个原因。

第一，决策粒度对齐。

- SmolVLA 一次输出的是整个 action chunk；
- collector 执行的也是整个 chunk；
- 因此 replay buffer 如果仍然按 primitive action 单步切，会让 actor 结构和 RL decision unit 不一致。

第二，训练实现更直接。

- 当前 hybrid trainer 是第一版原型；
- chunk-level transition 能最大程度复用现有 SmolVLA actor 结构；
- 不需要先把 generative chunk actor 打散成每步独立策略。

第三，bootstrap 更自然。

- collector 执行完一个 chunk 后，正好来到一个新的 chunk 边界状态；
- 在这个边界上接 `V(next_state)`，和当前 actor 的调用方式是对齐的。

所以这里的“奖励函数”更准确地说不是“人为设计了一条复杂 reward 公式”，而是：

- 直接使用环境 reward；
- 再在 chunk 级别做折扣聚合。

### 6.5.4 当前 hybrid trainer 没有做哪些 reward shaping

当前这条 hybrid 路径没有做下面这些事情：

- 没有额外密集 shaping reward；
- 没有手工距离奖励；
- 没有额外 success bonus；
- 没有直接在 hybrid collector 里接 reward classifier；
- 没有在 hybrid trainer 里接 teleop success / intervention reward。

这也是为什么我前面一直强调：

- 当前 hybrid trainer 是一个通用原型；
- reward 的任务语义主要由环境本身决定；
- 它还不是完整的真实机器人 reward 工程系统。

## 6.6 `gym_manipulator` 路径里另外存在的 reward 链路

为了避免混淆，还必须把另一条代码路径说出来。

在 `src/lerobot/rl/gym_manipulator.py` 和 `src/lerobot/processor/hil_processor.py` 里，真实机器人 / HIL 这条链路里确实存在额外 reward 处理逻辑，但它和当前 hybrid trainer 不是同一条执行路径。

### 6.6.1 RobotEnv 自身默认 reward

`gym_manipulator` 里的 `RobotEnv.step(...)` 默认返回：

- `reward = 0.0`
- `terminated = False`
- `truncated = False`

也就是说，如果只看这个最底层 env，本身没有内建任务成功奖励。

### 6.6.2 action / env processor 可以额外改 reward

在这条路径里，reward 可以被 processor 改写。

主要来源包括：

1. `InterventionActionProcessorStep`
   - 如果 teleop 事件里有 `success=True`；
   - 它会把 `TransitionKey.REWARD` 设成 `float(success)`，也就是典型的成功即 `1.0`。

2. `RewardClassifierProcessorStep`
   - 如果配置了 `reward_classifier.pretrained_path`；
   - 它会根据图像观测判断是否成功；
   - 成功时把 reward 设为 `success_reward`；
   - 并可选地 `terminate_on_success`。

3. 环境本身 reward 与 action processor reward 会在 `step_env_and_process_transition(...)` 中相加。

因此那条路径的 reward 语义更像：

- 原始环境 reward
- 加上 action / teleop / success 信号修正
- 再加上可选 reward classifier 成功奖励

### 6.6.3 为什么这条 reward 链路当前没有并入 hybrid trainer

原因很直接：

- 当前 hybrid trainer 没有调用 `step_env_and_process_transition(...)`；
- 当前 hybrid trainer 走的是 `make_env(...) + SmolVLAChunkCollector` 这条标准环境路径；
- 它不是 `gym_manipulator` 的专用机器人 RL 训练入口。

所以现在这两个判断要严格分开：

- 仓库里存在面向真实机器人/HIL 的 reward processing 代码；
- 但当前 SmolVLA hybrid trainer 并没有把这条 reward processing 链路接进去。

## 7. trainer 的真实训练结构

### 7.1 训练不是“纯 offline 训完再纯 RL”

`src/lerobot/rl/smolvla_hybrid/trainer.py` 里的主循环非常清楚：

1. collect 在线 chunk；
2. 从 offline dataset 拿一批数据；
3. 从 replay buffer 采样一批 online 数据；
4. 计算 offline SmolVLA loss；
5. 计算 online policy loss；
6. 计算 value loss；
7. 三者加权求和；
8. 一次 `backward()`；
9. 一次 `optimizer.step()`。

也就是说，代码层面的事实是：

- 在同一个 optimizer step 里，同时训练 offline 和 online。

### 7.2 但工程上仍然是“先有一个能用的 SmolVLA”

虽然 step 级别是混合训练，但更合理的整体使用方式仍然是：

1. 先训练或准备好一个能工作的 SmolVLA checkpoint；
2. 再用这个 checkpoint 作为 hybrid trainer 的初始化；
3. 然后做 hybrid RL fine-tuning。

理由非常现实：

- hybrid trainer 本身没有替代 offline 行为先验；
- collector 只会把当前 policy 的 chunk 执行进环境；
- 如果初始策略完全不会做事，在线回报和 replay buffer 质量都会很差；
- 当前实现的 RL 也不是那种能从零强力探索起来的完整大规模 RL。

所以应当区分两层语义：

- 代码 step 语义：offline loss 和 online loss 同步优化。
- 工程阶段语义：先离线训练，再 hybrid 微调。

## 8. 当前实现更像哪一种 RL 近似

最接近的类比是：

- AWR / AWAC / IQL 一类 advantage-weighted regression；
- 但 actor 不是 Gaussian policy，而是 flow-matching generative policy。

它不是 PPO，原因包括：

- 没有 log-prob ratio；
- 没有 clipped surrogate objective；
- 没有 GAE；
- 没有 entropy regularization 这套标准 PPO 结构。

它也不是 SAC，原因包括：

- 没有 `Q(s, a)` critic；
- 没有双 Q；
- 没有 target network；
- 没有基于 Q 的 actor gradient；
- 没有温度自动调节。

因此更准确的描述应该是：

- 一个 state-value-augmented advantage-weighted flow-matching actor；
- 或者说，一个 value-guided generative actor-critic 原型。

## 9. 当前实现的能力边界

### 9.1 它适合什么

当前实现更适合：

- 标准 `gymnasium` / `gym.vector.VectorEnv` 风格环境；
- 单环境版本；
- reward 定义清楚的模拟任务；
- 先做原型验证的场景。

从当前仓库能看到的环境配置类型看，`src/lerobot/envs/configs.py` 中已经注册或定义了这些主类：

- `aloha`
- `pusht`
- `libero`
- `metaworld`
- `gym_manipulator`

从代码限制可以直接看出：

- `TrainSmolVLAHybridConfig.validate()` 强制 `policy.type == "smolvla"`；
- `env` 不能为空；
- `collector.n_envs == 1`；
- `resolve_single_vector_env(...)` 只支持单 suite / 单 task；
- `SmolVLAChunkCollector` 只支持 `env.num_envs == 1`。

### 9.2 它不适合什么

当前实现不适合直接拿去做完整的 SO101 真机在线 RL。

原因不是“模型一定不行”，而是整条数据采集和奖励闭环还没接好：

- 当前 hybrid trainer 走的是 `make_env(...)` 标准环境路径；
- 它没有真正接入 `gym_manipulator` 的真实机器人 RL 链路；
- 也没有完整的真实机器人 reward 处理闭环；
- 也没有安全约束、人工接管、失败恢复、异步采样这些真实系统常见组件。

因此要把“仓库里存在 `gym_manipulator` 配置”与“当前 hybrid trainer 已经稳定支持真实 SO101 在线 RL”这两件事严格分开。

因此如果把它直接说成“SmolVLA 实机 RL 已经打通”，这是不准确的。

## 10. 如果要走向真实 SO101 在线 RL，还缺什么

这一部分不是当前代码已经完成的内容，而是从架构上看清楚下一步缺口。

至少还需要补以下几类东西：

1. 真实机器人 collector
   - 不是标准 vector env；
   - 要能接 SO101 机械臂、相机、控制周期和安全限幅。

2. 奖励闭环
   - 成功判定；
   - 失败惩罚；
   - 必要时的 reward classifier 或人工反馈。

3. 更稳的 critic 设计
   - target network；
   - 更可靠的 return / advantage 估计；
   - 甚至可能需要引入 Q critic。

4. 更真实的训练编排
   - actor-learner 解耦；
   - 异步 rollout；
   - 多阶段恢复；
   - buffer 管理和优先级策略。

5. 机器人安全壳
   - 动作裁剪；
   - torque / speed / workspace 限制；
   - 超时与急停；
   - 人工接管回落。

6. 训练系统层的异步化
   - 当前 trainer 是单循环同步收集、同步计算、同步更新；
   - 它不是 actor-learner 异步并行架构；
   - 如果未来要做更大规模在线 RL，这一层也要重构。

## 11. 与当前仓库里 SO101 推理 / 录制脚本的关系

当前仓库里已经有一条面向 SO101 的 policy 录制入口：

- `src/lerobot/scripts/lerobot_record_so101_policy.py`

它的定位不是 RL trainer，而是：

- 把一个已经训练好的 policy 接到 SO101 follower 机器人上执行；
- 同时可选地录 eval dataset。

这个脚本里已经处理了几个关键工程问题：

1. 默认离线加载 Hugging Face / Transformers 资源；
2. SmolVLA 自动加图像键重命名：
   - `observation.images.top -> observation.images.camera1`
   - `observation.images.wrist -> observation.images.camera2`
3. 默认 SO101 follower 参数：
   - `robot_id=my_so101`
   - `robot_port=/dev/ttyACM0`
   - top camera index `4`
   - wrist camera index `6`

这条脚本对“训练后上机验证”很重要，但它不等于“在线 RL 已经打到机器人里”。

## 12. 最终判断

如果只允许一句话总结当前仓库里的 SmolVLA RL 方案，那么最准确的表述是：

- 这是一个保留 SmolVLA flow-matching actor、引入共享 prefix context 和 state value head、并用 advantage-weighted online flow loss 进行策略改进的单环境 hybrid RL 原型。

如果再补一句工程判断：

- 它适合作为“离线 SmolVLA 之后的模拟环境 hybrid 微调”起点，但还不能直接等同于一套可投入 SO101 真机在线 RL 的完整系统。
