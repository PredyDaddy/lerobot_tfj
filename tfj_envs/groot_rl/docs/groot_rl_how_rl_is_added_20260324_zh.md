# GROOT 里的 RL 到底是怎么加进去的

这份文档专门回答一个非常具体的问题：

- 这套 GROOT stage-2 代码里，RL 到底是怎么接进去的？

它不是一份泛泛的强化学习教程，而是尽量贴着当前仓库的真实实现来讲。目标是让你读完之后，能把这套代码在脑子里压缩成一个清晰的结构，而不是继续停留在“好像哪里都像 RL，又哪里都不像”的状态。

---

## 1. 先给结论

如果只用一句话概括当前实现，那么最准确的说法是：

- 这版 GROOT 不是把原模型整体改造成了一个完整的 PPO / SAC actor-critic，而是在原来的动作监督学习之上，额外加了 transition replay、value target、value loss 和 advantage-weighted action loss。

再换一种更直白的说法：

1. 原来的 GROOT 继续学“动作怎么做”。
2. 新加的 RL 部分负责学“这个状态值不值”和“哪些动作样本更值得学”。
3. 最后把这两部分 loss 混在一起训练。

所以它更像：

- `监督学习 + 轻量 RL 化`

而不是：

- `从零重新造了一个完整的 RL 算法框架`

---

## 2. 为什么这套代码会让人看不懂

你会觉得它难懂，不是因为你理解有问题，而是因为这套实现本身就有三个“容易迷惑人”的特点。

### 2.1 第一，它保留了原来的 GROOT 训练主干

GROOT 本来就是学动作 chunk 的模型。现在加 RL 以后，这条主线没有消失，还是在那儿。

也就是说：

- `offline_loss` 还在
- `policy.forward(...)` 还在
- 原来的动作拟合训练还在

所以你一眼看过去，会觉得：

- 这不还是监督学习吗？

是的，它确实还是监督学习，只不过又额外加了 RL 的信号。

### 2.2 第二，它没有长成教科书里的 PPO / SAC 样子

很多人对“RL 加进来”的心理预期是：

- 一个很清晰的 actor
- 一个很清晰的 critic
- 一个很清晰的 policy gradient 公式

但当前这版更像：

- 共享 GROOT 表征
- 在这个共享表征上读 value
- 再用 advantage 去重加权动作损失

所以它不像标准 PPO，也不像标准 SAC。

### 2.3 第三，它把同一份 demo 数据用了两次

同一份数据集在这套代码里同时扮演两个角色：

1. 作为普通离线 batch，用来做原来的动作监督学习。
2. 又被重排成 transition，用来做 value / advantage 相关训练。

所以如果你脑子里一直在找“哪份是 BC 数据，哪份是 RL 数据”，就会觉得绕。

当前实现里其实是：

- 一份 demo 数据
- 两种消费方式

---

## 3. 先把整体训练图画出来

你可以先把每个训练 step 想成下面这件事：

```text
                同一份 demo dataset
                       |
        -----------------------------------
        |                                 |
   普通 dataloader                    offline replay buffer
        |                                 |
   offline batch                      transition batch
        |                                 |
   offline_loss                    online_flow_loss + value_loss
        |                                 |
        ---------------合并-----------------
                       |
                   total_loss
                       |
                    反向传播
```

也就是说，每个 step 同时做两件事：

1. 用普通 batch 保持原来的行为克隆能力。
2. 用 transition batch 注入“RL 味道”的 value / advantage 信号。

---

## 4. 第一层：RL 是从配置上被打开的

相关代码在：

- [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L49)

最关键的配置有三组。

### 4.1 loss 配置

在 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L49) 到 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L60)：

- `offline_loss_weight`
- `online_flow_loss_weight`
- `value_loss_weight`
- `discount`
- `use_advantage_weighting`
- `advantage_temperature`
- `normalize_advantage`

这组配置定义的是：

- 训练总 loss 由几部分组成
- RL 的那两部分权重多大
- advantage weighting 怎么做

### 4.2 offline replay 配置

在 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L63) 到 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L71)：

- `enabled`
- `transition_stride`
- `value_target_mode`
- `terminal_reward`
- `step_reward`
- `success_value`

这组配置定义的是：

- 是否走 dataset-only offline replay
- transition 怎么切
- reward 怎么合成
- value target 怎么构造

### 4.3 为什么你这次这条路不是在线 RL

在 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L101) 到 [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L159) 里，已经明确限制：

- `offline_replay.enabled=true` 时，必须 `env=None`
- `num_workers=0`
- `collector.chunks_per_step=0`
- `collector.warmup_chunks=0`

这说明当前这条路就是：

- dataset-only offline RL

不是：

- 边跑环境边收集 rollout 的 online RL

---

## 5. 第二层：RL 的第一步，是把 demo 数据改造成 transition

这一步是 RL 真正开始接进来的地方。

相关代码在：

- [offline_replay.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py#L49)
- [offline_replay.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py#L115)

### 5.1 原来普通监督学习只需要什么

原来的动作监督学习只需要：

- `observation`
- `action`

这就够了。

### 5.2 RL 额外需要什么

RL 还需要：

- `reward`
- `next_observation`
- `done`
- `bootstrap_discount`

所以必须把原来的 episode 数据重新切成 transition。

### 5.3 这段代码具体做了什么

在 [_build_transitions(...)](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py#L49) 里，它会沿着每条 episode 往前扫：

1. 取一个起点 `start_index`
2. 决定执行多少步 `executed_steps`
3. 定义一个 `next_index`
4. 构造：
   - reward
   - done
   - bootstrap_discount

最后形成一个 `GrootOfflineTransition`。

### 5.4 这一步最重要的理解

这里最重要的一点是：

- 它没有创造新数据，只是把原来的 demo episode 重新解释成了 RL 的 transition 格式。

也就是说，RL 不是凭空来的，它是：

- 从 demo 里“重新切”出来的

---

## 6. 第三层：reward 不是环境实时给的，而是离线合成的

这是理解这版实现的关键。

相关代码在：

- [offline_replay.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py#L80)

### 6.1 `monte_carlo` 模式

如果 `value_target_mode == "monte_carlo"`：

- 当前 transition 的 reward 会直接包含从当前位置到 episode 终点的折扣累计回报
- 此时 `bootstrap_discount = 0.0`

直观理解是：

- “把未来这段回报一口气都算进来”

### 6.2 `n_step` 模式

如果不是 `monte_carlo`，而是 `n_step`：

- 只先算一小段回报
- 如果还没结束，就把剩下的未来价值交给 `V(next_obs)` 去补

于是目标会变成：

- `r + gamma^n * V(next_obs)`

### 6.3 这说明什么

当前 stage-2 的 reward 不是：

- 真实环境在线采集到的 reward

而是：

- 从 demo episode 位置合成出来的 reward / value target

所以你可以把它理解成：

- 用离线示教数据给模型增加一个 value 视角

---

## 7. 第四层：policy 这边并没有被推翻，还是原来的动作学习

相关代码在：

- [modeling_groot.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/modeling_groot.py#L111)
- [modeling_groot.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/modeling_groot.py#L204)

GROOT 本来就支持：

- `forward_action_chunk(...)`
- `predict_action_chunk(...)`

而普通监督训练的那条线仍然在 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L147)：

```python
compute_offline_loss(policy, offline_batch)
```

本质上就是：

- `policy.forward(offline_batch)`

所以请先记住一件事：

- RL 不是替代原来的动作训练
- RL 是叠加在原来的动作训练之上的

---

## 8. 第五层：RL 新加了一个 value 预测能力

这部分代码在：

- [modeling_groot.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/modeling_groot.py#L161)
- [modeling_groot.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/modeling_groot.py#L179)
- [groot_n1.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/groot_n1.py#L354)

### 8.1 `predict_value(...)` 是做什么的

它的作用是：

- 给当前 observation 估一个 value

也就是：

- `V(s)`

### 8.2 它不是一个特别独立显眼的 critic

这里恰恰是很多人容易卡住的地方。

当前实现并不是那种很标准的：

- 单独定义一个 critic MLP
- 单独定义一个 critic optimizer
- 然后特别明确地把 actor 和 critic 分开

当前实现更像：

- 先得到 GROOT 的共享 hybrid context
- 再从这个共享 context 里直接读一个 value

看 [groot_n1.py](/data/tfj/lerobot_tfj/src/lerobot/policies/groot/groot_n1.py#L354) 就能看出来：

- `get_value_from_hybrid_context(...)`

它本质是在共享表征上做 pooled value 读取。

### 8.3 这也是为什么它看起来“不像 RL”

因为它没有长成你脑子里预期的那种 actor-critic 架子。

所以你会觉得：

- “value 到底加哪儿了？”

答案是：

- 加在共享 GROOT context 上了，不是另起炉灶造了一个特别大的 critic 子系统。

---

## 9. 第六层：value loss 是怎么计算出来的

相关代码在：

- [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L181)

这部分其实是当前实现里最标准、最像 RL 的一段。

### 9.1 当前值

先算：

- `values = V(s)`

### 9.2 目标值

再算：

- `next_values = V(s')`
- `target = reward + bootstrap_discount * next_values`

对应 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L191) 到 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L194)。

### 9.3 value loss

然后用：

- `MSE(V(s), target)`

对应 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L196)。

### 9.4 advantage

最后顺手得到：

- `advantage = target - V(s)`

对应 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L197)。

这一项非常关键，因为后面动作 loss 的加权就靠它。

---

## 10. 第七层：actor 这边不是 PPO policy gradient，而是 advantage-weighted action loss

这是整套实现最容易被讲错的地方。

相关代码在：

- [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L151)
- [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L206)

### 10.1 `online_flow_loss` 本质上是什么

`compute_online_flow_loss(...)` 干的事情，本质还是：

- 让 policy 去拟合 action chunk

只不过拟合的不是普通 offline dataloader batch，而是 replay batch。

### 10.2 advantage weighting 在哪里发生

看 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L219) 到 [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L223)：

1. 先根据 advantage 算每个样本的 weight
2. 再用这个 weight 去乘每个样本自己的 flow loss

也就是：

- 好样本多学一点
- 差样本少学一点

### 10.3 这和标准 PPO 的区别

它不是：

- `log pi(a|s) * A`

而更像：

- `A-weighted imitation / reconstruction`

所以更准确的理解是：

- RL 信号通过样本加权的方式，间接影响动作学习

而不是：

- 直接用经典 policy gradient 公式改写整个 actor 训练

---

## 11. 第八层：训练时总 loss 是怎么拼起来的

这一步在 [trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/trainer.py#L666) 到 [trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/trainer.py#L699)。

你可以直接把它记成下面这个式子：

```text
total_loss =
    offline_loss_weight * offline_loss
  + online_flow_loss_weight * online_policy_loss
  + value_loss_weight * value_loss
```

这就是 RL 真正“加进去”的地方。

### 11.1 `offline_loss`

这部分保持原来的行为克隆能力。

### 11.2 `online_policy_loss`

这部分用 replay transition 继续学动作，但会受 advantage weighting 影响。

### 11.3 `value_loss`

这部分让模型学会估值。

### 11.4 关键理解

这套实现不是：

- “先训练 actor，再单独训练 critic”

而是：

- 同一个 step 里，三种 loss 一起算，一起反传

---

## 12. 第九层：为什么我说它是“轻量 RL 化”，而不是完整 RL 框架

这个判断很重要。

### 12.1 有 RL 味道的部分

它确实已经有这些 RL 元素：

- transition replay
- reward
- next observation
- done
- bootstrap target
- value loss
- advantage weighting

### 12.2 但它又不是很“完整”

当前这版还有几个明显特征：

1. `value.*` 配置虽然存在，但没有长成一个很显眼的独立 critic 子网络构造流程。
2. actor 训练仍然是动作拟合主导，而不是标准 policy gradient 主导。
3. 这条主线目前是 dataset-only offline replay，不是成熟的在线 RL 训练系统。

所以如果非要给它贴一个更准确的标签，我会建议你用：

- hybrid RL fine-tuning
- advantage-weighted offline RL stage
- value-augmented action training

而不要过度简化成：

- “这就是标准 PPO/SAC”

---

## 13. 用伪代码把整条链压缩一下

这段最适合帮助你彻底理顺。

```python
# 1. 普通监督学习 batch
offline_batch = next(dataloader)
offline_loss = policy.forward(offline_batch)

# 2. 从 demo 数据集构出来的 replay transition batch
online_batch = replay_buffer.sample()

# 3. 继续拟合动作 chunk
online_flow_loss = action_chunk_loss(policy, online_batch)

# 4. 学 value
values = V(s)
next_values = V(s_next)
targets = reward + bootstrap_discount * next_values
value_loss = mse(values, targets)

# 5. 用 advantage 给动作损失加权
advantages = targets - values.detach()
online_flow_loss = weighted_by_advantage(online_flow_loss, advantages)

# 6. 三部分一起训练
total_loss = (
    a * offline_loss
    + b * online_flow_loss
    + c * value_loss
)
total_loss.backward()
optimizer.step()
```

只要你把这段伪代码记住，后面再回头看真实代码，就不会那么乱了。

---

## 14. 我建议你接下来按什么顺序读代码

如果你现在还没完全吃透，不要一上来就啃整个 trainer 文件。我们可以按下面顺序读，会轻松很多。

### 第一步：先看配置

看：

- [train_groot_hybrid.py](/data/tfj/lerobot_tfj/src/lerobot/configs/train_groot_hybrid.py#L49)

重点只看三组字段：

- `losses.*`
- `offline_replay.*`
- `offline_replay.enabled` 的约束

你要先知道系统想干什么。

### 第二步：再看 replay

看：

- [offline_replay.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/offline_replay.py#L49)

重点只看两件事：

- transition 怎么切
- reward / target 怎么构造

你要知道 RL 的 batch 是怎么来的。

### 第三步：再看 losses

看：

- [losses.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/losses.py#L147)

重点只看三段：

- `compute_offline_loss`
- `compute_online_value_loss`
- `compute_online_losses`

你要知道 RL 具体是怎么变成 loss 的。

### 第四步：最后再看 trainer.step()

看：

- [trainer.py](/data/tfj/lerobot_tfj/src/lerobot/rl/groot_hybrid/trainer.py#L666)

这一步你会发现，整个 trainer 并不神秘，它只是把前面几件事按顺序拼起来了。

---

## 15. 最后再压成三句话

如果你现在只想先记住最关键的三句话，就记下面这三句。

### 第一句

- RL 的第一步，不是先改模型，而是先把 demo episode 变成 `(s, a, r, s')` transition。

### 第二句

- RL 的第二步，是在 GROOT 共享表征上学一个 value，并得到 advantage。

### 第三句

- RL 的第三步，不是完全替代原动作训练，而是用 `offline_loss + online_flow_loss + value_loss` 的混合方式继续训练。

---

## 16. 一个最诚实的最终总结

如果你问“这套代码里的 RL 到底是怎么加进去的”，最诚实也最准确的回答就是：

- 它不是另起炉灶造了一套全新的 RL 系统，而是保留原来的 GROOT 动作监督学习主线，再用离线 replay 构造 value target，用 value 学出 advantage，最后把 advantage 作为样本权重重新作用到动作 chunk 学习上。

你要是愿意，我们下一步可以直接继续做两种之一：

1. 我再给你写一份“逐行读 `trainer.step()` 的超白话版”。
2. 我给你画一张真正的流程图文档，把 `dataset -> replay -> value -> advantage -> total_loss` 画出来。
