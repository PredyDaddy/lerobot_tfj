# VLA / SmolVLA / Flow Matching / Hybrid RL 综合知识文档

本文档的目标不是记录某一次实验日志，而是把当前对话中涉及的核心知识点系统化整理出来，供下游 AI 直接据此制作 PPT、讲稿或培训材料。

写作原则：

- 尽量去项目化，不依赖具体机器路径、具体日志文件、具体 shell 历史。
- 尽量把“概念”“实现”“训练流程”“部署经验”“易错点”放在同一份文档里。
- 尽量把局部结论上升为可复用的方法论，而不是只描述某个个例。

---

## 1. 先讲清楚：这个主题到底是什么

这里讨论的是一类将视觉、语言和机器人状态结合起来生成动作的策略系统，以及如何把 RL 信号接进这类策略中。

更具体地说，核心主题包括：

1. 什么是 VLA。
2. SmolVLA 这类模型的结构长什么样。
3. 为什么它用的是 flow matching，而不是传统的动作回归。
4. “共享前缀表征”到底是什么意思。
5. 如果要把 RL 接进 SmolVLA，应该接在什么位置。
6. 这种接法与 PPO、SAC、DDPG 的关系和区别是什么。
7. 实际训练时应该先做什么、后做什么。
8. 部署到机器人上时，哪些东西真的在用，哪些东西其实不在用。

这份文档既包含模型知识，也包含工程思路。

---

## 2. 什么是 VLA

VLA 是 Vision-Language-Action 的缩写。

它的目标不是只做“看图说话”，也不是只做“状态到动作”的传统控制，而是：

- 输入视觉观测；
- 输入语言任务描述；
- 输入机器人状态；
- 输出一段动作或者一个动作块。

VLA 的核心价值在于：

- 任务条件不再只是数值状态，而是可以直接用自然语言指定；
- 视觉和动作之间不再是弱耦合，而是直接在一个统一的条件生成框架里建模；
- 更适合做跨任务、跨场景、带语言条件的机器人策略。

从工程视角看，VLA 通常包含三类输入：

- 视觉输入：相机图像，可能是单视角或多视角。
- 语言输入：任务文本、指令文本、语义提示。
- 状态输入：关节角、夹爪状态、末端姿态、历史观测等。

输出通常是：

- 单步动作；
- 或者一个 action chunk，也就是未来若干步的动作序列。

---

## 3. SmolVLA 的定位与大致规模

SmolVLA 可以理解为“小型化的 VLA 实现”，它不是一个纯 MLP policy，也不是一个离散 token 自回归动作语言模型，而是一个条件 flow-matching 动作生成器。

如果它基于一个 500M 级别的视觉语言骨干，再叠加状态投影、动作 expert、value head 等模块，那么总参数规模大约处于“几亿参数”的量级。

一个典型配置下可以记住这样一个量级印象：

- 总参数量大约 4.5 亿左右。
- 在冻结视觉主干、只训练部分模块的设置下，可训练参数可能降到 1 亿左右。

这里有两个重要提醒：

1. “总参数量”不等于“训练时真正更新的参数量”。
2. “VLA 一定比 ACT 更大”不是普遍规律，具体取决于骨干网络、hidden size、chunk size、视觉编码器规模、是否冻结等。

因此，参数量讨论要始终带着配置上下文去看。

---

## 4. SmolVLA 的整体结构

从功能上看，SmolVLA 可以拆成下面几个部分。

### 4.1 视觉编码模块

作用：

- 把一个或多个相机图像编码成高维视觉特征。
- 为后续动作生成提供场景理解信息。

输入通常是：

- 顶视角图像；
- 腕部图像；
- 或更多相机。

视觉模块的本质不是直接输出动作，而是把“场景里发生了什么”变成可供后续条件生成使用的中间表征。

### 4.2 语言编码模块

作用：

- 把任务文本映射成语义条件。
- 告诉策略“当前到底要做什么”。

语言的意义不只是辅助说明，而是条件生成的一部分。  
同样的视觉画面，在不同语言任务下，动作目标可以完全不同。

### 4.3 状态投影模块

作用：

- 把机器人低维状态映射到和视觉、语言表征兼容的隐藏空间。

状态信息通常包括：

- 关节位置；
- 夹爪状态；
- 其他传感状态。

状态信息之所以必要，是因为图像只能告诉模型“外界长什么样”，却未必能精确告诉模型“机械臂当前在哪个精细配置上”。

### 4.4 前缀编码模块

SmolVLA 不会直接把观测扔给动作头，而是先把视觉、语言、状态这些条件统一编码成一个“前缀上下文”。

这个前缀上下文可以理解为：

- 当前这一步的“任务条件总结”；
- actor 和 critic 都能消费的共享观测表征；
- 后续动作生成和价值估计的共同输入基础。

### 4.5 动作生成模块

SmolVLA 的动作生成不是传统“直接回归动作向量”，而是：

- 先给动作加噪；
- 在给定时间步 `t` 和条件前缀的情况下预测一个 flow / velocity；
- 通过多步迭代把噪声逐步还原成动作块。

因此它更接近一个条件生成器，而不是一个直接回归器。

### 4.6 Value Head

如果把 RL 接入 SmolVLA，通常不会重写整套 actor，而是在共享前缀表征上额外挂一个 value head。

它的作用是：

- 对当前状态估计一个 state value；
- 为 advantage 计算提供基础；
- 给 actor 的在线更新提供权重或方向性信号。

它不是动作头，也不是直接负责出最终动作的模块。

---

## 5. SmolVLA 的动作生成为什么是 Flow Matching

这是理解这类模型最关键的点之一。

### 5.1 直观理解

可以把 flow matching 理解成：

- 从一个随机噪声动作序列开始；
- 模型在每个时间步告诉你“应该朝哪个方向把它往真实动作推”；
- 经过多次迭代，噪声被逐渐变成合理动作。

这和“直接回归一个动作向量”相比，有几个特点：

- 更像生成模型；
- 更适合建模多峰动作分布；
- 更容易与时间步、噪声过程、迭代求解结合。

### 5.2 训练时在学什么

训练时通常会：

1. 采样一个真实动作 `a`。
2. 采样噪声 `n`。
3. 采样时间步 `t`。
4. 构造 noisy action：

```text
x_t = t * n + (1 - t) * a
```

5. 构造目标 flow：

```text
u_t = n - a
```

6. 让模型预测：

```text
v_theta(s, x_t, t)
```

7. 用均方误差拟合：

```text
|| v_theta - u_t ||^2
```

这意味着 actor 的核心学习目标不是“直接猜动作”，而是“在条件上下文下，给 noisy action 指出正确的恢复方向”。

### 5.3 推理时在做什么

推理时通常从纯噪声开始：

```text
x_1 ~ N(0, I)
```

然后反复做：

```text
x_{t+dt} = x_t + dt * v_theta(s, x_t, t)
```

其中 `dt` 为负，表示从高噪声逐渐走向低噪声。

最终得到：

- 一段动作块；
- 而不是单一标量或单个 token。

### 5.4 为什么动作块比单步动作更常见

因为真实机器人控制常常存在：

- 执行延迟；
- 观测与动作不同步；
- 高频控制带来的瞬时噪声；
- 单步预测过于脆弱的问题。

所以 action chunk 的好处是：

- 一次出若干步动作；
- 减少每一步都重新完整推理的代价；
- 给控制器更平滑的时间窗口。

---

## 6. “共享前缀表征”到底是什么

这也是对话里反复出现的核心概念。

### 6.1 定义

共享前缀表征，或者 shared prefix representation，本质上是：

- 由视觉、语言、状态共同编码得到的一份统一观测条件表征；
- 后续的 actor 和 critic 都从这份表征出发。

它不是原始图像，也不是最终动作，而是位于中间的一份“已经被模型理解过”的条件摘要。

### 6.2 为什么重要

它的重要性体现在三个层面。

#### 第一，避免重复编码

在 flow matching 推理里，一次动作块采样要做多轮 denoise。

但在这一整个 denoise 过程中：

- 图像没变；
- 语言没变；
- 状态通常视为同一步观测下固定。

所以没有必要每一步 denoise 都重新做一遍完整的视觉语言编码。  
先把 prefix 编码好，再反复复用，是更合理的设计。

#### 第二，为 actor 和 critic 提供统一语义空间

如果 actor 和 critic 都基于同一份 prefix 表征：

- critic 学到的价值判断和 actor 看到的条件更一致；
- RL 信号更容易反馈到 actor 真正依赖的那部分表示空间；
- 模块边界更清晰。

#### 第三，降低工程复杂度

把“观测编码”和“动作生成/价值估计”拆开后：

- 系统更容易调试；
- 更容易插入缓存、监控和辅助输出；
- 更容易做推理阶段复用。

### 6.3 一个微妙但重要的实现细节

从概念上看，actor 和 critic 共享 prefix representation。  
但从具体实现上看，要分两种情况：

1. 在推理/采样路径中，前缀表征通常真的会先编码一次，然后在 denoising 循环中复用。
2. 在训练路径中，如果 `get_value(...)` 和 `compute_fm_score(...)` 是两个独立调用，它们可能会各自重新编码 prefix，而不是物理上复用同一份张量。

所以要区分：

- 概念上的共享；
- 实现上的一次编码复用。

这对做 PPT 时的表述很重要。  
如果想讲严谨，可以写成：

- “共享前缀表征是当前架构的核心抽象；在采样路径中已明显复用，在训练路径中可进一步做更彻底的计算共享。”

---

## 7. SmolVLA 中各模态分别在做什么

### 7.1 视觉

视觉负责回答：

- 物体在哪里；
- 容器/目标区域在哪里；
- 场景布局如何；
- 末端与目标的相对关系如何。

### 7.2 语言

语言负责回答：

- 当前任务是什么；
- 当前动作目标是什么；
- 同一场景下应优先完成哪类行为。

### 7.3 状态

状态负责回答：

- 机器人当前配置是什么；
- 图像中看不准的细节状态是什么；
- 低维控制量和当前动力学上下文是什么。

### 7.4 Prefix Context

prefix context 负责把视觉、语言、状态统一成：

- 可被 actor 用于动作生成的条件；
- 可被 critic 用于价值估计的状态表征。

### 7.5 Flow Matching Actor

它负责：

- 接收 noisy action 和时间步；
- 在 prefix context 条件下预测 flow；
- 通过多次 denoise 还原动作块。

### 7.6 Value Head

它负责：

- 从 pooled prefix feature 上估计 `V(s)`；
- 提供 critic 学习目标；
- 为 online actor loss 提供 advantage 权重。

---

## 8. Value Head 是怎么接到 SmolVLA 上的

如果要把 RL 接进 SmolVLA，最稳妥的一种方式不是把 actor 全部换掉，而是：

1. 保留原来的 flow-matching actor。
2. 在共享 prefix 特征上增加一个 value head。
3. 用这个 value head 估计 `V(s)`。
4. 用 advantage 对 actor 的 online flow loss 做重加权。

### 8.1 为什么是 `V(s)` 而不是 `Q(s, a)`

当前这种接法更偏向：

- state-value augmented actor
- 而不是完整的 action-value actor-critic

这有几个原因：

- prefix context 本身最自然地描述的是“状态条件”；
- actor 已经是一个复杂生成器，不适合再直接套一个标准 Q 学习结构；
- 用 `V(s)` 构造 advantage 足以提供一个相对稳定的策略改进信号。

### 8.2 Value Head 的输入是什么

输入通常不是原始图像，而是：

- pooled prefix hidden states。

也就是：

- 视觉、语言、状态已经融合后的 prefix hidden states；
- 再经过 pooling 得到一个固定长度向量；
- 然后喂给 value MLP。

### 8.3 Value Head 的输出是什么

输出是一个标量：

```text
V(s)
```

表示：

- 当前观测条件下未来回报的估计。

### 8.4 Pooling 为什么重要

因为 prefix hidden states 本质上是一个序列。  
value head 通常需要一个固定长度向量作为输入，所以要做 pooling。

常见做法：

- mean pooling；
- last-token pooling。

这本质上是在问：

- 应该把整个前缀平均汇总；
- 还是取序列末端作为摘要。

---

## 9. 把 RL 接入 SmolVLA 的总体思路

这部分是整个项目方案的核心。

### 9.1 总体原则

最合理的接法通常不是：

- 推翻原来的 SmolVLA actor；
- 另起一套完全独立的 RL policy。

而是：

- 保留 SmolVLA 原有的动作生成机制；
- 在其上叠加 value estimation；
- 用 online 数据和 advantage 去微调 actor。

### 9.2 为什么不直接换成 PPO/SAC/DDPG 的标准 actor

因为 SmolVLA 的 actor 不是普通的：

- Gaussian policy；
- deterministic MLP policy；
- 离散 logits policy。

它是一个：

- generative actor；
- action-chunk flow-matching actor；
- 依赖视觉、语言、状态共同条件的生成器。

如果硬换成标准 PPO/SAC/DDPG actor，代价会非常大：

- 原有 flow-matching 优势会丢掉；
- 多模态条件生成结构要大改；
- 现有离线 imitation 先验难以平滑继承。

### 9.3 更自然的做法

更自然的做法是把它理解成：

- actor 仍是生成式 flow-matching policy；
- critic 是基于 prefix 的 state value estimator；
- RL 信号不直接替代 actor loss，而是调制 actor loss。

这样可以做到：

- 保留离线行为先验；
- 逐步引入 online 反馈；
- 避免 RL 一上来把行为策略拉崩。

---

## 10. 当前这种 Hybrid RL 的损失函数长什么样

当前这种接法里，总损失通常由三部分组成。

### 10.1 离线 actor loss

这是 SmolVLA 原本的 imitation / flow matching loss。

作用：

- 保持 demonstration 行为先验；
- 防止 online RL 微调把策略快速拖偏。

### 10.2 在线 actor loss

这部分不是标准 policy gradient，而是：

- advantage-weighted online flow matching loss。

形式上可以写成：

```text
L_online_actor = mean( w_i * L_fm_i )
```

其中：

- `L_fm_i` 是每个在线样本的 flow matching loss；
- `w_i` 是由 advantage 推出来的权重。

### 10.3 Value loss

这部分用来训练 critic：

```text
L_value = MSE( V(s), target )
```

其中 target 一般是：

```text
target = reward_chunk + bootstrap_discount * V(s')
```

### 10.4 总损失

最终一般是：

```text
L_total = w_offline * L_offline
        + w_online  * L_online_actor
        + w_value   * L_value
```

这说明：

- actor 并没有脱离离线监督；
- RL 是以“混合微调”的形式进入；
- critic 是显式训练的，不是隐含存在。

---

## 11. Advantage 是怎么在这里起作用的

这是当前接法区别于单纯 imitation 的关键。

### 11.1 先估计 value

先计算：

```text
V(s)
V(s')
```

### 11.2 再构造 target

```text
target = r + gamma * V(s')
```

### 11.3 再构造 advantage

```text
A = target - V(s)
```

### 11.4 再把 advantage 变成权重

常见做法是：

1. 对 advantage 做标准化；
2. 做 clip；
3. 除以温度；
4. 再取指数：

```text
w = exp(clipped_advantage / temperature)
```

然后再设置一个最大权重上限。

### 11.5 直观含义

如果某个在线样本比 critic 原先估计得更好：

- 它的 advantage 更高；
- 它的 flow loss 权重更大；
- actor 更倾向于往这类样本靠。

如果某个在线样本更差：

- 它的权重更低；
- actor 不会被它强行拉过去。

这是一种“用 critic 给生成式 actor 做方向性重加权”的方法。

---

## 12. Reward 在这类系统里应该怎么理解

### 12.1 奖励有两种层次

第一种是任务语义奖励：

- 真正衡量任务有没有完成；
- 通常最难定义。

第二种是工程代理奖励：

- 用来先验证训练链路；
- 不一定等价于真实任务目标。

### 12.2 一个常见的代理奖励设计

如果只是为了先把训练链路跑通，一个常见设计是：

```text
r = - distance_to_target + success_bonus
```

也就是：

- 距离越近奖励越高；
- 达到阈值后给额外成功奖励。

这种奖励的优点是：

- 容易计算；
- 稠密；
- 便于快速观察 RL 是否真的在学习。

缺点是：

- 可能并不真正等价于任务完成；
- 容易把策略引向“优化代理指标”而不是“完成真实操作”。

### 12.3 所以奖励设计的正确态度是什么

正确态度是：

- 先用简单代理奖励验证训练闭环；
- 再逐步把奖励做得更接近真实任务；
- 不要把“代理奖励跑通”误解成“真机任务已经学会”。

---

## 13. Chunk-Level RL 是什么意思

当前这类 SmolVLA hybrid 设计往往不是逐步一步一步存 transition，而是按 action chunk 存。

### 13.1 为什么这样做

因为 actor 本来就是输出 action chunk。

如果 replay buffer 只按单步存：

- 会割裂 actor 原本的预测结构；
- 会让 online 数据和 actor 输出单位不一致；
- 价值目标也不容易对齐。

### 13.2 这意味着什么

collector 通常会：

1. 生成一个动作块；
2. 在环境中逐步执行若干步；
3. 把这一段内的奖励按折扣累加；
4. 得到 chunk reward；
5. 存入 replay buffer。

所以 critic 学到的更像是：

- “在这个观测条件下，执行这段动作块会带来怎样的累计效果”。

这不是最标准的单步 MDP actor-critic 写法，但和 action-chunk actor 的结构是一致的。

---

## 14. 训练流程到底应该是分段还是同时

这个问题必须分两层讲。

### 14.1 从工程流程上看：分阶段

更合理的项目推进顺序是：

1. 先做离线 imitation 训练。
2. 再做 hybrid RL 微调。
3. 最后做机器人上机验证。

原因：

- RL 阶段需要一个已有行为先验的 actor；
- 否则 online rollout 质量太差；
- critic 也更难学稳定。

### 14.2 从 hybrid trainer 的单个 step 看：同时优化

在 hybrid trainer 的一个 step 内，往往会同时计算：

- offline loss；
- online actor loss；
- value loss。

再把三者加权求和，一次反传更新。

所以最准确的表述是：

- 项目推进上，建议先 offline，再 hybrid。
- hybrid trainer 内部，则是同步混合优化。

这两句话并不矛盾。

---

## 15. 纯离线训练阶段到底有没有训练 RL

没有。

如果一个训练阶段只跑了：

- 离线 SmolVLA trainer；
- demonstration 数据；
- imitation / flow matching loss；

那么它训练的只是 actor 的离线行为先验，不算 RL。

在这个阶段通常：

- value head 即使已经挂在模型里，也没有得到真正的 RL 监督；
- replay buffer、collector、online reward、advantage weighting 都没有起作用。

所以要严格区分：

- “模型代码里有 RL 相关模块”
- 和
- “这次训练实际上在用 RL”

这是两回事。

---

## 16. 推理时到底有没有在用 RL

这个问题也要分清楚。

### 16.1 如果使用的是 RL 微调后的 checkpoint

那么回答是：

- 有，在用 RL 训练过的 actor 参数。

因为 actor 的权重已经在 hybrid RL 阶段被更新过了。

### 16.2 但推理时不等于还在做 RL

机器人上机执行时通常只会：

- 加载一个训练好的 actor；
- 做前向推理；
- 输出动作。

它不会：

- 在现场继续采样 replay buffer；
- 继续反传；
- 继续更新参数。

所以更准确的说法是：

- 上机推理使用的是 RL 微调后的策略；
- 但上机过程本身通常不是在线 RL。

### 16.3 Value Head 在推理时通常也不负责出动作

它的主要作用是训练期：

- 估计 `V(s)`；
- 构造 advantage；
- 帮助 actor 更新。

最终上机动作主要还是由 actor 给出。

---

## 17. 这套方法为什么不是 PPO、SAC、DDPG

这是做 PPT 时特别容易被讲错的地方。

### 17.1 为什么不是 PPO

PPO 的典型特征包括：

- on-policy；
- 显式概率比值 `ratio`；
- clipped surrogate objective；
- 通常是随机策略；
- 常见地配一个 value baseline。

而当前这种 SmolVLA hybrid 接法：

- 没有 ratio-clipping；
- 没有显式 log-prob policy gradient 目标；
- actor 不是普通概率策略头，而是 flow-matching 生成器；
- online actor 更新是 weighted flow loss，不是 PPO surrogate。

所以它不是 PPO。

### 17.2 为什么不是 SAC

SAC 的典型特征包括：

- off-policy；
- stochastic actor；
- `Q(s, a)` critic；
- 最大熵目标；
- 常用于连续动作。

而当前这种接法：

- 虽然用了 replay buffer，带有 off-policy 味道；
- 但没有标准 `Q(s, a)`；
- 没有 entropy temperature；
- actor 不是 Gaussian actor，而是 flow-matching actor。

所以它也不是 SAC。

### 17.3 为什么不是 DDPG

DDPG 的典型特征包括：

- off-policy；
- deterministic actor；
- `Q(s, a)` critic；
- actor 通过 critic 的动作梯度更新。

当前这种接法：

- 没有标准动作价值 critic；
- actor 不是一个直接输出确定性动作的 MLP；
- actor 更新不是靠 `∇_a Q(s,a)`，而是靠 weighted flow-matching loss。

所以它也不是 DDPG。

### 17.4 最准确的说法

更准确的描述是：

- 这是一个保留生成式 flow actor、增加 state value head、并使用 advantage-weighted online flow loss 的 hybrid RL 原型。

也可以更口语化地说：

- 它更像“带 value 的生成式 actor 微调”，而不是教科书式 PPO/SAC/DDPG。

---

## 18. PPO、SAC、DDPG 分别是什么

下面这一节是给下游 AI 做 PPT 时直接使用的理论材料。

### 18.1 PPO

PPO 全称 Proximal Policy Optimization。

核心思想：

- 用 on-policy 数据训练；
- 每次策略更新不要走太大；
- 用 clipped objective 限制策略变化幅度。

适合：

- 稳定性要求高；
- 理论和工程都相对成熟；
- 常见于连续动作和离散动作任务。

优点：

- 稳定；
- 实现成熟；
- 广泛使用。

缺点：

- 样本效率一般；
- 需要不断采集新数据；
- 对昂贵环境不够友好。

### 18.2 SAC

SAC 全称 Soft Actor-Critic。

核心思想：

- off-policy；
- 使用 replay buffer；
- 学 `Q(s, a)`；
- 同时鼓励高回报和高熵。

适合：

- 连续动作控制；
- 样本效率要求高；
- 希望策略更稳定、更鲁棒。

优点：

- 样本效率高于 PPO；
- 在连续控制里很强；
- 训练常较稳。

缺点：

- 结构更复杂；
- 对 Q 学习稳定性、target network、温度系数等更敏感。

### 18.3 DDPG

DDPG 全称 Deep Deterministic Policy Gradient。

核心思想：

- off-policy；
- actor 输出确定性动作；
- critic 学 `Q(s, a)`；
- 用 critic 指导 actor 更新。

适合：

- 连续动作控制；
- 想要低方差 deterministic policy。

优点：

- 概念直接；
- 连续动作适配自然。

缺点：

- 容易不稳定；
- 对超参数较敏感；
- 现代实践中经常被 TD3、SAC 等替代。

---

## 19. On-Policy 与 Off-Policy 的区别

### 19.1 On-Policy

on-policy 的意思是：

- 训练当前策略时，主要依赖当前策略自己新采样出来的数据。

特点：

- 数据“新鲜”；
- 分布和当前策略一致；
- 稳定但样本效率较低。

典型算法：

- PPO。

### 19.2 Off-Policy

off-policy 的意思是：

- 当前策略训练时，可以使用过去策略采集的数据。

特点：

- 可以用 replay buffer；
- 样本效率更高；
- 但训练更容易出现分布偏移与估计偏差问题。

典型算法：

- DDPG；
- SAC。

### 19.3 当前这类 Hybrid RL 更靠近哪一边

它通常更靠近 off-policy 风格，因为：

- 使用了 replay buffer；
- online batch 并不要求必须来自当前一步刚采到的数据。

但它又不是标准 off-policy Q-learning，因为：

- actor 更新形式不是标准 Q-backprop；
- 仍保留大量离线 imitation 成分。

---

## 20. 随机策略与确定性策略的区别

### 20.1 随机策略

随机策略输出的是一个动作分布：

```text
pi(a | s)
```

然后从这个分布里采样动作。

优点：

- 天然带探索；
- 更适合与熵正则化结合；
- 容易表达多峰行为。

缺点：

- 方差更高；
- 控制上可能不够稳定。

### 20.2 确定性策略

确定性策略直接输出一个动作：

```text
a = mu(s)
```

优点：

- 推理简单；
- 低方差；
- 连续控制中执行稳定。

缺点：

- 需要额外探索机制；
- 容易陷入局部模式。

### 20.3 Flow-Matching Actor 属于哪类

它不完全等价于经典“随机策略”或“确定性策略”。

更准确地说：

- 它是一个生成式策略；
- 推理时可以从噪声初始化，因此具有随机性来源；
- 最终生成动作块后，执行时又像一个确定结果。

所以它更适合被理解为：

- “带采样过程的生成式 actor”。

---

## 21. `V(s)` 和 `Q(s, a)` 的区别

### 21.1 `V(s)`

`V(s)` 表示：

- 在状态 `s` 下，未来总体回报的期望。

它不直接区分具体动作。

### 21.2 `Q(s, a)`

`Q(s, a)` 表示：

- 在状态 `s` 下执行动作 `a`，然后继续下去，未来总体回报的期望。

它显式依赖动作。

### 21.3 两者的关系

可以粗略理解为：

- `V(s)` 更像“这个局面好不好”；
- `Q(s, a)` 更像“在这个局面下做这个动作好不好”。

### 21.4 当前这种 SmolVLA-RL 为什么更常用 `V(s)`

因为：

- prefix context 天然描述状态条件；
- actor 已经是复杂生成器，不适合再直接把动作和状态拼给一个标准 critic；
- 用 `V(s)` 构造 advantage 更自然。

---

## 22. 连续动作与离散动作的区别

### 22.1 连续动作

连续动作表示：

- 动作值是实数；
- 例如关节角速度、末端位移、夹爪开度。

优点：

- 更适合机器人控制；
- 动作平滑。

难点：

- 搜索空间大；
- 学习更难。

### 22.2 离散动作

离散动作表示：

- 动作是有限个类别；
- 例如“抓/放/不动”。

优点：

- 学习和搜索更简单；
- 适合高层决策。

难点：

- 不适合精细低层控制；
- 需要额外机制把高层动作变成低层控制。

### 22.3 机器人操作里通常怎么用

常见做法是：

- 低层臂控制用连续动作；
- 某些开关型执行器可以离散化；
- 或做混合动作空间。

SmolVLA、ACT 这类操作策略通常更自然地工作在连续动作空间或连续动作块上。

---

## 23. SmolVLA 与 ACT 的一般性差异

这部分非常适合做一页对比 slide。

### 23.1 ACT 的核心思想

ACT 可以理解为：

- Action Chunking Transformer；
- 用 transformer 直接预测未来一段动作；
- 训练时可选 VAE latent；
- 解码头最终是动作回归头。

它的典型结构是：

- 图像 backbone；
- 状态 token；
- 可选 latent token；
- transformer encoder/decoder；
- 动作回归 head。

### 23.2 SmolVLA 的核心思想

SmolVLA 更像：

- 用视觉语言骨干先理解多模态条件；
- 再通过 flow matching 生成动作块；
- actor 本质是一个条件生成器。

### 23.3 二者最大的建模差异

ACT：

- 更接近“直接把条件映射成动作块”。

SmolVLA：

- 更接近“在条件下从噪声逐步生成动作块”。

### 23.4 一般性对比

ACT 的典型特点：

- 结构更直接；
- 动作头更像回归器；
- 训练目标更接近监督学习；
- 更容易先快速跑通。

SmolVLA 的典型特点：

- 生成式建模能力更强；
- 多模态条件耦合更自然；
- 更适合往 flow / generative policy 路径扩展；
- 接 RL 时也更容易走“weighted generative actor”路线。

### 23.5 关于参数量，不要讲错

不能简单讲：

- “VLA 一定比 ACT 大”

或者：

- “ACT 一定更轻”

更准确的说法是：

- 参数量取决于具体 backbone 和实现配置；
- VLM backbone 可能很大，但冻结后可训练部分未必更大；
- 模型能力不只看参数量，还看动作建模方式、条件利用方式和训练数据。

---

## 24. 当前这套 Hybrid RL 的训练范式应该怎么总结

最好的总结方式是：

- 先离线 imitation 学会“怎么像示范那样做”；
- 再用 RL 告诉它“哪些在线行为更值得保留”；
- actor 继续保持生成式 flow 模式，不被改写成标准 RL actor；
- critic 只负责给 actor 一个更有方向性的评价信号。

用一句更适合 PPT 的话说就是：

> 这不是把 SmolVLA 改造成标准 PPO/SAC，而是把 RL 作为生成式 actor 的在线偏好校正器。

---

## 25. 实际部署到机器人上时，哪些模块真的在工作

### 25.1 上机时实际会用到的

- 训练好的 actor 参数；
- 视觉预处理；
- 状态预处理；
- 语言任务文本；
- 多相机输入；
- 动作后处理；
- 机器人通信和控制接口。

### 25.2 上机时通常不会发生的

- 不会现场继续做 RL 更新；
- 不会继续训练 value head；
- 不会现场维护 replay buffer 并反传。

### 25.3 所以部署语言要讲准确

如果用的是 RL 微调后的 checkpoint，应说：

- “部署的是经过 RL 微调的 actor”

而不要说：

- “机器人上机时正在跑 RL”。

---

## 26. 机器人上机封装应该遵循什么原则

从工程视角看，一个好用的机器人上机脚本应该：

1. 比通用 recorder 更聚焦。
2. 固定机器人类型、相机数量、常用标定目录。
3. 保留策略路径、设备、任务文本、录制目录这些核心参数。
4. 支持可选 teleop 或 leader 设备。
5. 支持评测数据录制。
6. 尽量默认离线运行，避免现场临时下载依赖。

### 26.1 为什么离线模式重要

机器人现场最怕的是：

- 模型或 tokenizer 临时联网拉取；
- 网络不可用导致程序卡住；
- 推理前还在做外部依赖解析。

所以部署时常见建议是：

- 本地缓存模型资产；
- 开启 HF / Transformers 离线模式；
- 让推理路径尽可能确定、可复现。

### 26.2 为什么 observation key rename 很重要

真实机器人、数据集、模型期望输入之间，视觉键名很容易不一致。

例如：

- 相机驱动叫 `top`、`wrist`
- 模型却期望 `camera1`、`camera2`

这类不一致如果不显式做 rename，推理往往会直接失败。

因此“观测键名映射层”是一个非常重要但经常被忽略的工程部件。

---

## 27. 训练和部署中的典型工程坑

### 27.1 环境依赖不齐

很多 RL 训练脚本假设：

- 目标环境包已经存在；
- gym id 已注册；
- 所有 suite/task 都能直接创建。

现实中经常不是这样。

因此一个实用策略是：

- 先用一个最小代理环境验证训练闭环；
- 再替换成真实环境。

### 27.2 动作维度与模型内部维度不一致

在很多实现里：

- 机器人真实动作维度可能是 6；
- 但模型内部为了统一接口，把动作 padding 到更大的 `max_action_dim`，比如 32。

如果 online RL 样本直接把 6 维动作丢进 32 维动作投影层，就会出维度错误。

因此需要：

- 小维度时补零；
- 大维度时截断；
- 训练和推理都保持一致。

### 27.3 Prefix-only 编码路径容易出错

如果前缀编码阶段没有 suffix，但底层网络仍尝试走依赖 suffix 或 expert cross-attention 的路径，就可能出现：

- `NoneType` 消费；
- cache 不一致；
- 只在特定模式下才触发的隐蔽 bug。

这类问题说明：

- prefix-only 和 full prefix+suffix 不是同一条路径；
- 需要单独验证。

### 27.4 纯离线阶段其实不训练 critic

这是最容易被忽视的认知误区。

很多人看到模型里已经有 value head，就误以为离线阶段 critic 也在学。  
其实如果训练目标没有 value loss，那么 critic 基本没有真正被训练。

所以一定要明确：

- 离线阶段主要训练 actor；
- hybrid 阶段才开始真正训练 value head。

### 27.5 Actor 和 Critic 的“共享表示”不一定意味着“共享一次前向”

概念共享不等于计算共享。

如果训练代码中：

- `get_value()` 调一次 prefix encoder；
- `compute_fm_score()` 再调一次 prefix encoder；

那就说明：

- 语义上在共享前缀抽象；
- 但计算上还没有完全复用。

### 27.6 CLI 的布尔值解析很容易出错

很多配置系统要求：

- `true/false`

而不是：

- `0/1`

因此 shell wrapper 最好做一层兼容：

- `0 -> false`
- `1 -> true`

否则部署时会因为参数解析这种低级问题中断。

---

## 28. 如果要向别人解释这套方案，最推荐的表述方式

可以按下面这条主线讲：

1. 先介绍 VLA 是什么。
2. 再介绍 SmolVLA 用多模态条件加 flow matching 生成动作块。
3. 再介绍共享前缀表征，把视觉、语言、状态统一编码。
4. 再介绍 value head 是怎么接在共享前缀上的。
5. 再介绍 hybrid RL 不是替换 actor，而是对 actor 做 advantage-weighted 微调。
6. 再介绍它和 PPO/SAC/DDPG 的差异。
7. 再介绍训练流程：离线 imitation -> hybrid RL -> 真机部署。
8. 最后讲工程难点：环境依赖、奖励设计、动作维度对齐、部署离线化。

这条线最顺，也最不容易讲错。

---

## 29. 适合做 PPT 的推荐结构

如果下游 AI 要做 PPT，可以按下面结构切页。

### 第一部分：背景

- 机器人策略为什么需要视觉、语言、动作统一建模
- VLA 的价值

### 第二部分：SmolVLA 架构

- 多模态输入
- prefix 编码
- flow-matching actor
- action chunk

### 第三部分：动作生成原理

- noisy action
- timestep
- target flow
- iterative denoising

### 第四部分：共享前缀表征

- 定义
- 为什么要共享
- 为什么能复用

### 第五部分：RL 接入点

- 为什么保留 actor
- 为什么加 value head
- 为什么用 advantage-weighted flow loss

### 第六部分：损失函数

- offline loss
- online actor loss
- value loss
- total loss

### 第七部分：训练流程

- offline imitation
- hybrid RL
- robot deployment

### 第八部分：与 PPO / SAC / DDPG 的比较

- 不是什么
- 更像什么

### 第九部分：SmolVLA 与 ACT 对比

- 动作建模方式
- 条件建模方式
- 参数量与复杂度的正确看法

### 第十部分：工程经验与坑

- 奖励设计
- 环境依赖
- 维度对齐
- 离线部署
- CLI 配置坑

### 第十一部分：结论

- 这类系统的核心不是“把 RL 硬加进去”
- 而是“让生成式 actor、共享表征和价值信号协同起来”

---

## 30. 一页版总结

如果最终只能保留一页总结，可以写成：

> SmolVLA 是一种多模态条件下的生成式机器人策略，它通过共享前缀表征融合视觉、语言和状态，再用 flow matching 生成动作块。把 RL 接入这类模型时，最自然的做法不是把 actor 替换成标准 PPO/SAC/DDPG 策略，而是在共享前缀上增加一个 value head，用 advantage-weighted online flow loss 去微调原有生成式 actor。工程上最稳妥的流程是先做离线 imitation，再做 hybrid RL，再把 RL 微调后的 actor 部署到机器人上；部署时真正使用的是经过 RL 微调的 actor，而不是现场继续跑 RL。核心难点在于奖励设计、环境依赖、动作维度对齐、共享前缀复用以及部署链路的稳定性。 

