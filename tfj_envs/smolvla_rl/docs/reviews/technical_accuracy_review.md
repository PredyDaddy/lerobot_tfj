# SmolVLA-RL 技术准确性审查报告

## 审查范围

本报告以当前仓库代码为准，对照审查以下文档与实现：

- `tfj_envs/smolvla_rl/docs/smolvla_rl_architecture_and_integration_20260315_zh.md`
- `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md`
- `src/lerobot/rl/smolvla_hybrid/trainer.py`
- `src/lerobot/rl/smolvla_hybrid/losses.py`
- `src/lerobot/rl/smolvla_hybrid/collector.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/configuration_smolvla.py`

为核对文档中直接引用的训练/上机脚本默认值，我额外辅助查看了 `src/lerobot/configs/train_smolvla_hybrid.py`、`src/lerobot/scripts/lerobot_record_so101_policy.py` 以及 `tfj_envs/smolvla_rl/scripts/` 下对应 wrapper，但审查判断仍以核心实现文件为主。

## 准确之处

1. 文档对当前方法的定性基本准确。两份文档都把当前实现描述为“保留 SmolVLA flow-matching actor，在其上增加 value head，并用 advantage-weighted online flow loss 做 hybrid 微调”，而不是把它误写成标准 PPO 或 SAC。这个判断与代码一致：`trainer.py` 在同一个 step 中同时计算 `offline_loss`、`online_policy_loss` 和 `value_loss`，再做一次反向传播与参数更新，见 `src/lerobot/rl/smolvla_hybrid/trainer.py:191-214`。

2. 文档对模型和配置层新增接口的描述基本准确。`SmolVLAConfig` 确实新增了 `value_head_hidden_dim`、`value_head_num_layers`、`value_head_dropout`、`value_head_pooling` 等 critic 相关配置，见 `src/lerobot/policies/smolvla/configuration_smolvla.py:106-139`。模型侧也确实新增了 `value_head`、`encode_prefix_context(...)`、`get_value(...)`、`compute_fm_score(...)`、`sample_actions_with_info(...)`、`predict_action_chunk_with_info(...)` 等接口，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:328-403`、`src/lerobot/policies/smolvla/modeling_smolvla.py:657-762`、`src/lerobot/policies/smolvla/modeling_smolvla.py:960-1038`。

3. 文档对 online loss 的拆解与代码实现一致。`compute_online_losses(...)` 的实际流程确实是：对 replay 中的 chunk action 采样噪声和时间步，构造 noisy action 与 target flow，用 `policy.compute_fm_score(...)` 做 flow 预测，再用 `V(s)` 与 `reward + discount * V(s')` 构造 advantage，对 per-sample flow loss 进行指数加权，同时单独回归 value loss。见 `src/lerobot/rl/smolvla_hybrid/losses.py:31-55`。

4. 文档对 collector 和 replay buffer 的基本描述是对的。当前 buffer 存储的是 chunk-level transition，不是 primitive-action transition；collector 也只支持单环境，且 reward 是按 chunk rollout 期间累计的折扣回报。对应实现见 `src/lerobot/rl/smolvla_hybrid/buffer.py:29-47`、`src/lerobot/rl/smolvla_hybrid/collector.py:43-58`、`src/lerobot/rl/smolvla_hybrid/collector.py:121-156`。

5. 文档对“训练循环微观上是同时优化、工程流程宏观上仍建议先 offline 再 hybrid”的表述是成立的。代码层面，hybrid trainer 的确没有“先纯 offline 跑完再纯 RL”的双阶段内循环；但从工程上先准备一个可用的离线策略，再进入 hybrid 微调，确实更符合当前实现能力边界。这个判断与 `trainer.py` 的主循环和 `TrainSmolVLAHybridConfig.validate()` 的限制相符，见 `src/lerobot/rl/smolvla_hybrid/trainer.py:191-214` 和 `src/lerobot/configs/train_smolvla_hybrid.py:61-85`。

6. 文档对当前能力边界的总体判断比较审慎，没有把现状误写成“SO101 真机在线 RL 已打通”。从实现看，`TrainSmolVLAHybridConfig` 强制 `collector.n_envs == 1`，`resolve_single_vector_env(...)` 也只接受单 suite/task 环境，collector 本身再限制 `env.num_envs == 1`。因此文档把当前实现定位为“单环境 hybrid RL 原型”是正确的，见 `src/lerobot/configs/train_smolvla_hybrid.py:68-85`、`src/lerobot/rl/smolvla_hybrid/collector.py:33-37`、`src/lerobot/rl/smolvla_hybrid/collector.py:52-58`。

## 需要补充或修正之处

1. 高优先级：当前 hybrid online 路径存在明显的 action 维度对齐风险，文档没有明确提醒。文档把 replay buffer 中的 `action` 描述为“整个 chunk”，并默认它可以直接进入 online FM loss，见 `tfj_envs/smolvla_rl/docs/smolvla_rl_architecture_and_integration_20260315_zh.md:209-240`。但实际代码链路是：`predict_action_chunk_with_info()` 会先把动作截断到原始 action 维度 `self.config.action_feature.shape[0]`，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:329-351`；collector 又把这个截断后的动作直接写入 buffer，见 `src/lerobot/rl/smolvla_hybrid/collector.py:145-149`；随后 `compute_online_losses()` 直接把它传回 `compute_fm_score()`，见 `src/lerobot/rl/smolvla_hybrid/losses.py:31-39`；而 `embed_suffix()` 的 `self.action_in_proj` 期待的输入最后一维是 `max_action_dim`，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:858-866`。在本文档自己的语境下，SO101 这批数据的动作维度写的是 6，见 `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md:134-138`，而 `max_action_dim` 默认是 32，见 `src/lerobot/policies/smolvla/configuration_smolvla.py:106-110`。如果中间没有额外 padding，这条 online loss 路径静态上并不闭合。建议文档显式补一句：当前 hybrid 训练代码仍需确认或修补 action padding/对齐问题，不应让读者理解为该链路已经完成端到端验证。

2. 中优先级：文档对 `prefix context` 的“复用程度”写得略强，容易让人误以为 actor 和 critic 在训练时已经共享同一份已编码前缀。文档在 `tfj_envs/smolvla_rl/docs/smolvla_rl_architecture_and_integration_20260315_zh.md:142-166` 中写到“actor 使用同一个 prefix context”“critic 不需要重新做一遍完整观测编码”。这在 `sample_actions_with_info()` 的推理路径里是对的，因为 `prefix_context` 会先编码一次，再被 value 预测和 denoising 循环复用，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:978-1037`。但在当前训练路径里，`policy.get_value(...)` 与 `policy.compute_fm_score(...)` 是两次独立调用，两者都会各自重新走一次 `encode_prefix_context(..., use_cache=False)`，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:387-403`、`src/lerobot/policies/smolvla/modeling_smolvla.py:697-706`、`src/lerobot/policies/smolvla/modeling_smolvla.py:743-762`。更准确的写法应当是：`prefix context` 提供了“可共享”的抽象，并已在动作采样阶段复用；但在当前 online loss 训练路径里，actor/critic 并未真正共用一次前缀编码结果。

3. 中优先级：文档没有把“离线阶段并不会训练 value head”说清楚，这对理解 staged workflow 很关键。当前模型类里虽然始终存在 `value_head`，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:593-628`，但离线 `policy.forward()` 只会进入 flow-matching 的 actor loss 计算，并不会调用 `value_head`，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:405-434` 和 `src/lerobot/policies/smolvla/modeling_smolvla.py:901-937`。critic 的监督只出现在 hybrid 的 `compute_online_losses(...)` 中，见 `src/lerobot/rl/smolvla_hybrid/losses.py:41-55`。因此更严谨的表述应该是：阶段 A 主要训练的是 flow-matching actor；阶段 B 才开始真正给 value head 提供学习信号。当前文档虽然强调了“先 offline 再 hybrid”，但还缺少这句会直接影响用户预期的说明。

4. 中优先级：文档把“action 是整个 chunk，reward 是该 chunk 的聚合回报”写成了无条件成立，但代码里其实有一个隐藏前提。collector 的 rollout horizon 实际上是 `min(n_action_steps, env_action_chunk.shape[1], max_steps_per_chunk)`，见 `src/lerobot/rl/smolvla_hybrid/collector.py:121-124`；然而写入 buffer 的 `action` 却始终是完整的 `prediction.actions[0]`，见 `src/lerobot/rl/smolvla_hybrid/collector.py:145-149`。这意味着只有在“执行 horizon 恰好等于存储 chunk 长度”时，文档里的说法才严格成立。如果未来把 `n_action_steps` 调小，或者设置了 `max_steps_per_chunk`，那么 reward 实际对应的只是 chunk 前缀，而不是完整 chunk。建议文档把这一假设条件补上，否则后续调参时容易误判 credit assignment 的语义。

5. 一般优先级：关于图像键名的表述需要收窄适用范围。文档在 `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md:155-179` 中写道 “SmolVLA 当前训练配置预期的是 `observation.images.camera1/camera2`”。这对于当前这套 checkpoint、脚本和 rename map 约定是对的，但对 `SmolVLA` 这个策略实现本身并不普遍成立。`prepare_images()` 实际上读取的是 `self.config.image_features` 中当前配置出的键，见 `src/lerobot/policies/smolvla/modeling_smolvla.py:436-473`；`SmolVLAConfig.validate_features()` 也只负责补充 empty camera，不会硬编码 `camera1/camera2`，见 `src/lerobot/policies/smolvla/configuration_smolvla.py:141-145`。建议把文档措辞改成“本文所用脚本/预训练配置预期的是 `camera1/camera2`”，避免读者把脚本约定误读成模型类的通用约束。

6. 一般优先级：文档中关于本机数据集、snapshot 路径、历史训练结果和 `9.88 epoch` 的段落，本质上属于“运行记录”，不是“代码事实”。例如 `tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md:146-153`、`tfj_envs/smolvla_rl/docs/smolvla_rl_training_and_operations_20260315_zh.md:195-219` 这些内容更适合标注为“本机已验证资产/已跑通记录”，并附日志或脚本来源。它们本身不一定错误，但如果不加限定，读者会误以为这些路径和结果是仓库实现天然保证的，而不是当前机器上的经验性结论。

7. 一般优先级：文档可以补充一个训练细节，说明 online batch 在训练初期可能小于配置值。`SmolVLAChunkReplayBuffer.sample()` 会取 `min(configured_batch_size, len(buffer))`，见 `src/lerobot/rl/smolvla_hybrid/buffer.py:90-110`。这意味着即使把 `online_batch_size` 设成 16，训练早期如果 buffer 里只有 1 到 8 条样本，online loss 实际仍然在用更小 batch。这个细节不影响架构判断，但会影响对早期 loss 抖动和监控日志的理解，值得在运维文档里提醒。

## 总体判断

总体上，这两份文档对 SmolVLA-RL 的架构思路、loss 构成、collector 粒度、trainer 的混合优化方式以及“先 offline 再 hybrid 再上机验证”的工程节奏，描述是比较准确的。就“文档有没有把未实现的算法硬说成 PPO/SAC，或者把 SO101 真机在线 RL 写成已经完全打通”这一层面看，未发现严重表述失真。

但从代码静态链路看，当前实现本身存在至少一个高风险未闭环点：online FM loss 路径里的 action 维度对齐问题。除此之外，文档还有几处把“工程意图/脚本约定/本机运行经验”写得过于像“代码已经保证的事实”。因此我的结论是：

- 文档可以作为当前 SmolVLA-RL 方案的架构说明和操作指南使用。
- 文档目前不应被理解为“hybrid 训练链路已经完成端到端实证验证”。
- 最值得优先修订的，不是大改主线判断，而是把上述几个实现 caveat 明确写出来，尤其是 action padding、value head 训练阶段边界，以及 prefix context 在训练路径中的实际复用程度。
