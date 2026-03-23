# SmolVLA RL 训练策略审查报告

## 审查范围

本报告审查以下文档与实现之间的一致性，重点关注两件事：

- 是否把“step 级同时训练”和“工程上分阶段推进”区分清楚。
- 是否把“离线训练 -> hybrid 微调 -> SO101 上机”这三段说清楚。

对照文件：

- `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md`
- `src/lerobot/rl/smolvla_hybrid/trainer.py`
- `src/lerobot/configs/train_smolvla_hybrid.py`
- `src/lerobot/scripts/lerobot_record_so101_policy.py`

## 正确表述

- 文档第 2 节已经把两个层次拆开讲，方向是对的。文档在 `2.1` 里把 hybrid trainer 解释为“每个 step 同时取 offline batch 和 online batch，再合并 loss 做一次更新”，这和 `src/lerobot/rl/smolvla_hybrid/trainer.py:191-231` 的真实实现一致。代码确实是在同一个训练循环里先 `collector.collect(...)`，再取 `offline_batch`、`online_batch`，然后计算 `offline_loss`、`online_policy_loss`、`value_loss`，最后做一次 `backward()` 和 `optimizer.step()`。

- 文档第 1 节和第 11 节给出的三段式工程路径也是成立的。`阶段 A：离线基础模型`、`阶段 B：hybrid RL 微调`、`阶段 C：SO101 上机验证` 这一拆分，与代码里的职责边界匹配。`src/lerobot/configs/train_smolvla_hybrid.py:61-86` 约束的是 hybrid trainer 的训练配置，`src/lerobot/scripts/lerobot_record_so101_policy.py:17-27,223-237` 则明确是一个 SO101 上机执行/录制 wrapper，而不是训练入口。

- 文档第 5 节把 hybrid 脚本解释为“不是 SO101 真机 RL 一键启动”，这点表述准确。`src/lerobot/configs/train_smolvla_hybrid.py:68-73` 只要求 `env` 已配置且 `collector.n_envs == 1`，`src/lerobot/rl/smolvla_hybrid/trainer.py:139-160` 也是通过 `make_env(...)`、collector、replay buffer 走标准环境接口，并没有出现真实机器人在线 RL 所需的专用 collector、安全壳或 actor-learner 解耦链路。

- 文档第 6 节对 hybrid trainer 的阶段拆分总体准确。初始化、warmup、主循环、日志/保存/评估四段，基本都能在 `src/lerobot/rl/smolvla_hybrid/trainer.py:114-310` 中找到一一对应的实现。尤其是 `warmup_chunks > 0` 时只是先往 replay buffer 填数据，这和文档“不是纯 offline，也不是纯 RL，只是先让 buffer 不是空的”的说法一致。

- 文档第 7 节对 SO101 wrapper 的价值描述也基本准确。`src/lerobot/scripts/lerobot_record_so101_policy.py:55-59` 确实预设了离线环境变量，`:149-157` 会在 SmolVLA policy 下自动补 rename map，`:164-237` 负责把 SO101 follower、相机、可选 leader、数据录制参数拼成 `RecordConfig` 后交给通用录制循环执行。

## 可能误解点

- “先做离线基础模型”在文档里有时写成了近似“必须先做”的语气，这容易被读成代码硬约束，但从实现上看并不是。`src/lerobot/configs/train_smolvla_hybrid.py:54-59` 给了 `policy: SmolVLAConfig` 的默认值，`src/lerobot/rl/smolvla_hybrid/trainer.py:126-131` 也可以直接构建 policy。也就是说，当前代码没有强制要求必须从离线 checkpoint 恢复；更准确的说法应是“从工程验证路径和稳定性角度，强烈建议先有离线基座”。

- “阶段 B 输入是仿真 env”这句话作为工程建议没问题，但容易让人误以为 hybrid trainer 在代码层面只接受仿真环境。实际上 `src/lerobot/configs/train_smolvla_hybrid.py:68-73` 只校验 `env` 非空和 `collector.n_envs == 1`，没有把“仿真”写成类型约束。这里应明确区分“当前建议在仿真里先做原型验证”和“代码只要求标准 env 接口”。

- “阶段 C：SO101 上机验证”虽然整体方向对，但如果只看标题，仍可能被理解成“训练流程的最后一个在线训练阶段”。对照 `src/lerobot/scripts/lerobot_record_so101_policy.py:17-27,223-237`，这个入口本质上是执行/录制/eval 数据采集 wrapper，不做参数更新，不包含 reward 计算，也不是 RL collector。这个边界最好再写得更硬一点。

- “同时训练”一词仍可能被部分读者误解成“多进程并发 actor-learner”或者“异步联训”。当前 `src/lerobot/rl/smolvla_hybrid/trainer.py:191-214` 体现的是单循环、同步收集、同步算 loss、同步更新，不是异步架构。文档第 8 节虽然提到“没有 actor-learner 解耦”，但这一点如果能前置到第 2 节，会更不容易误解。

## 建议补充

- 建议在文档第 1 节或第 2 节加一句总括性澄清：`“分阶段”描述的是工程推进顺序，`“同时训练”描述的是 hybrid trainer 单个 step 的优化行为；前者不是 parser/config 的硬约束，后者也不等于异步 actor-learner 架构。`

- 建议把“为什么必须先做离线基础模型”改成更精确的工程表述，例如：`“当前代码并未强制要求从离线 checkpoint 启动，但从现有验证结果、online rollout 质量和训练稳定性看，推荐先做离线基线，再进入 hybrid 微调。”`

- 建议在“阶段 C：SO101 上机验证”或第 7 节开头显式补一句：`对应入口是 src/lerobot/scripts/lerobot_record_so101_policy.py，它负责策略执行、录制和可选评估数据采集，不负责在线训练或参数更新。`

- 建议在第 5 节补一条“训练路径和上机路径分离”的说明：`hybrid trainer 走的是 train config + env + collector + replay buffer 路径；SO101 wrapper 走的是 robot/camera/record 路径。二者共享的是 policy checkpoint 和 rename map，不共享训练循环。`

- 建议在第 6 节主循环说明后加一句关于 loss 权重的注释，点明同步混合优化不是抽象概念，而是由 `offline_loss_weight`、`online_flow_loss_weight`、`value_loss_weight` 三项显式控制，对应 `src/lerobot/configs/train_smolvla_hybrid.py:41-50` 与 `src/lerobot/rl/smolvla_hybrid/trainer.py:204-214`。

## 总体结论

这份文档整体上已经把你关心的主线讲对了，尤其是“step 级同时训练”和“工程上分阶段推进”这两个层次，已经明显分开，没有把它们混成一句话。离线训练、hybrid 微调、SO101 上机这三段也基本独立成章，结构上是清楚的。

剩下的主要问题不是方向错误，而是少数措辞还可以更严一点，避免读者把“工程建议”误听成“代码硬约束”，或者把“上机验证”误听成“在线 RL 训练”。如果补上上面几句澄清，这份文档在训练策略层面的表达会更稳，也更不容易被后续接手的人误解。
