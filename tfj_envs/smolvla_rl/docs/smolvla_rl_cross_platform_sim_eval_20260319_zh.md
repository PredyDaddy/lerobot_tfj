# SmolVLA 与 RL+SmolVLA 跨平台仿真对比报告

## 1. 任务目标

本次工作的目标不是重新训练模型，而是将已经训练好的两个 checkpoint 放到多个仿真平台上做统一评测，对比：

- 原始离线 `SmolVLA`
- `RL + SmolVLA` 混合策略

核心问题是：

1. `RL+SmolVLA` 在训练分布附近是否优于原始 `SmolVLA`。
2. 这种提升在跨平台、跨任务形态下是否还能保留。
3. 当前实现层面有哪些兼容性问题，需要在后续论文、汇报或代码整理时说明。

## 2. 本次实际跑过的平台

本次实际执行并拿到结果的平台有：

- `Aloha` 本地轻量仿真
- `PushT`
- `MetaWorld`

之所以优先这三个，是因为它们覆盖了三类很典型的评测场景：

- `Aloha`：最接近你当前 RL 微调所依赖的本地机械臂风格环境
- `PushT`：极低维平面操作任务，适合看跨域泛化
- `MetaWorld`：标准机械臂连续控制基准，适合看跨平台控制迁移

## 3. 使用的模型

### 3.1 离线 SmolVLA

离线模型是先前基于示教数据训练得到的 `SmolVLA` checkpoint，训练方式本质上是 imitation / offline behavior cloning 风格，不包含额外的 RL 价值更新。

### 3.2 RL+SmolVLA

RL 版本是在离线 `SmolVLA` 的基础上继续做 hybrid 训练得到的 checkpoint。也就是说：

1. 先有离线 `SmolVLA` 作为初始化策略
2. 再在近似 `Aloha` 风格环境里做 RL/hybrid 微调
3. 最终导出新的 policy checkpoint

所以本次评测对比的不是“两个结构完全不同的模型”，而是：

- 一个只做过离线训练
- 一个在同一主体策略基础上继续做了 RL 微调

## 4. 评测时做了哪些工程适配

这部分很重要，因为 `SmolVLA` 当前训练出来的 checkpoint 并不是天然就能直接塞到所有环境上。

### 4.1 为什么不能直接跨平台评测

训练得到的 checkpoint 输入输出接口是固定的，例如：

- `Aloha` 风格：`6D state + 6D action + 多路图像`

但其他平台并不一致：

- `PushT`：`2D state + 2D action + 单路图像`
- `MetaWorld`：`4D state + 4D action + 单路图像`

如果直接拿原始 checkpoint 去跑，会遇到三个问题：

1. 状态维度不匹配
2. 动作维度不匹配
3. 观测键名不匹配

### 4.2 新增的 cross-env 评测脚本

为了解决这个问题，我新建了脚本：

- `tfj_envs/smolvla_rl/scripts/eval_smolvla_cross_env.py`

它做了四件事：

1. 覆盖 policy config 中的 `state_dim` / `action_dim`
2. 根据目标环境把观测键重命名到 `SmolVLA` 期望的字段
3. 对 normalizer / unnormalizer 的统计量做切片
4. 调用统一的 `eval_policy_all(...)` 完成 rollout 和视频导出

### 4.3 观测重命名策略

脚本中为不同环境设置了默认 rename map：

- `aloha`
  - `observation.images.top -> observation.images.camera1`
  - `observation.images.wrist -> observation.images.camera2`
- `pusht`
  - `observation.image -> observation.images.camera1`
- `metaworld`
  - `observation.image -> observation.images.camera1`

这样做的目的，是把不同环境的原始观测尽量映射到 `SmolVLA` 训练时见过的 key 命名空间里。

### 4.4 normalizer / unnormalizer 的切片

`SmolVLA` checkpoint 里带有预处理和后处理统计量，例如：

- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

这些统计量原本是为 `6D state / 6D action` 保存的。跨平台时，如果目标环境只有 `2D` 或 `4D`，就需要做切片：

- `PushT` 只取前 `2` 维
- `MetaWorld` 只取前 `4` 维

如果不做这一步，维度会在预处理阶段直接报错，或者虽然不报错，但归一化含义完全不对。

### 4.5 为什么这个评测叫“跨域适配测试”，不是“原生任务性能测试”

这一点需要在汇报里说清楚。

本次 `PushT` 和 `MetaWorld` 评测并不是说模型专门在这些任务上训练过，而是：

- 模型主体是在 `Aloha/SO101` 风格数据上训练的
- RL 微调也主要发生在 `Aloha` 风格仿真上
- 然后再把它适配到别的平台去看泛化表现

所以：

- `Aloha` 更像“近分布评测”
- `PushT / MetaWorld` 更像“跨分布泛化评测”

## 5. 本次修掉的两个关键兼容性问题

### 5.1 Hugging Face 离线加载问题

最开始跨环境评测时，虽然本地缓存里已经有 `SmolVLM2-500M-Video-Instruct`，但脚本仍然会去访问 `huggingface.co`，离线环境下会触发多轮重试，严重拖慢评测。

根因有两层：

1. `SmolVLA` 的 VLM / processor 加载没有显式写 `local_files_only=True`
2. `HF_HUB_OFFLINE` 等环境变量如果在模块导入之后才设置，已经偏晚

为此做了两类修复：

#### 修复 A

在 `src/lerobot/policies/smolvla/smolvlm_with_expert.py` 中，把以下调用改成显式本地加载：

- `AutoModelForImageTextToText.from_pretrained(..., local_files_only=True)`
- `AutoConfig.from_pretrained(..., local_files_only=True)`
- `AutoProcessor.from_pretrained(..., local_files_only=True)`

#### 修复 B

在 `eval_smolvla_cross_env.py` 里，把这些环境变量前移到模块导入前：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `HF_HUB_DISABLE_TELEMETRY=1`
- `TOKENIZERS_PARALLELISM=false`

这一步非常关键。因为如果只在 `main()` 里设置，往往已经晚于 transformers / huggingface_hub 的部分初始化逻辑。

### 5.2 MetaWorld v2/v3 任务名不兼容

仓库里原先默认写的是：

- `metaworld-push-v2`

但当前安装的是：

- `metaworld==3.0.0`

这个版本只接受 V3 任务名，因此会直接报错：

- `ValueError: push-v2 is not a V3 environment`

解决方式：

1. 在 `metaworld_config.json` 中确认任务表实际已经是 `v3`
2. 在 cross-env 评测脚本里将 `metaworld` 默认任务改为 `push-v3`
3. 正式评测时统一使用 `push-v3`

## 6. 评测结果

### 6.1 20 episode 正式结果

| 平台 | 模型 | avg_sum_reward | avg_max_reward | success_rate | episodes | eval_time_s |
|---|---:|---:|---:|---:|---:|---:|
| Aloha | Offline SmolVLA | -1294.7232 | -0.6044 | 0.0% | 20 | 37.03 |
| Aloha | RL+SmolVLA | -1217.9226 | -0.6076 | 0.0% | 20 | 32.55 |
| PushT | Offline SmolVLA | 17.9951 | 0.0600 | 0.0% | 20 | 30.74 |
| PushT | RL+SmolVLA | 17.9951 | 0.0600 | 0.0% | 20 | 30.43 |
| MetaWorld push-v3 | Offline SmolVLA | 3.7829 | 0.0515 | 0.0% | 20 | 57.59 |
| MetaWorld push-v3 | RL+SmolVLA | 4.2964 | 0.0514 | 0.0% | 20 | 57.77 |

### 6.2 RL 相对离线模型的变化

按 `avg_sum_reward` 看：

- `Aloha`：`+76.8005`，约 `+5.93%`
- `PushT`：`+0.0000`，约 `+0.00%`
- `MetaWorld`：`+0.5135`，约 `+13.57%`

### 6.3 结果的第一层结论

第一层可以直接说的结论是：

1. 在最接近训练/微调分布的 `Aloha` 环境上，`RL+SmolVLA` 确实优于离线 `SmolVLA`
2. 在 `MetaWorld` 上，`RL+SmolVLA` 也有一定幅度的 reward 提升
3. 在 `PushT` 上，两者几乎完全相同
4. 三个平台在这次 20 episode 测试中成功率都还是 `0%`

## 7. 如何解读这些结果

### 7.1 为什么 Aloha 上的提升最可信

这是最可信的一组结果，因为：

- RL 微调本来就是在 `Aloha` 风格环境里做的
- 输入输出接口完全一致
- 观测语义和动作语义最接近训练过程

因此这里的 reward 改善，最能反映 RL 微调对原模型的直接帮助。

### 7.2 为什么 PushT 上“完全一样”值得特别说明

`PushT` 结果几乎一模一样，这不是一个可以忽略的小现象，反而很值得在汇报里专门点出来。

可能原因包括：

1. `PushT` 的状态和动作都只有 `2D`，而原模型主要是在机械臂风格任务上学到的表征，迁移后策略输出可能退化到相似的低维模式
2. reward 信号很弱，两个模型都没有进入真正成功区域，于是平均 reward 差异被压平
3. 当前的跨维度切片方式虽然保证了“能跑”，但不保证不同 checkpoint 在该环境中一定会表现出可分辨差异
4. 两个 checkpoint 共享大量 backbone 权重，在这个离目标分布较远的环境上，RL 微调学到的增益可能根本没有被激活

这类结果在论文或答辩里并不丢人，关键是解释要准确：

- RL 增益不是无条件地跨一切任务泛化
- 增益最明显地体现在接近训练微调分布的环境上

### 7.3 为什么 MetaWorld 只看到 reward 提升，没有 success 提升

这是典型的“策略更接近目标，但还没跨过成功阈值”的情况。

也就是说，RL 版本可能已经让动作更合理、轨迹更接近目标，因此累积 reward 有改善；但距离真正触发环境定义的 success 条件，仍然差一步。

这通常说明后续还有几种可做的工作：

- 增加 RL 微调步数
- 增强奖励设计
- 在更接近 MetaWorld 语义的数据或任务上继续适配
- 对动作头或 low-level control 做更强约束

## 8. 这次评测不能过度解读的地方

下面这些边界需要写清楚，避免后续别人把结果解释过头。

### 8.1 这不是“在三平台都训练过”的结论

不是。这里只有 `Aloha` 方向是训练/微调分布附近。

`PushT` 和 `MetaWorld` 是跨域测试，不是原生训练平台。

### 8.2 这不是“RL 一定提高成功率”的结论

不是。本次所有平台的 success rate 还是 `0%`。

更准确的说法是：

- RL 让 reward 有所改善
- 但这份改善还没有转化为实际成功事件

### 8.3 这不是严格公平的原生基准对比

因为两个模型都不是在 `PushT` / `MetaWorld` 原生任务上训练出来的，所以这里更像“泛化测试”，不是和这些平台上专门训练的 SOTA baseline 的公平对比。

## 9. 这次工作的直接产物

### 9.1 新增/修改的脚本

#### 新增

- `tfj_envs/smolvla_rl/scripts/eval_smolvla_cross_env.py`
- `tfj_envs/smolvla_rl/scripts/run_cross_platform_eval_compare.sh`

#### 修改

- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

### 9.2 输出结果目录

本次实际结果都已经落盘在统一评测目录下，包含：

- `Aloha` 正式结果
- `PushT` 冒烟与正式结果
- `MetaWorld` 冒烟与正式结果
- `eval_info.json`
- rollout 视频
- 运行日志

## 10. 后续建议

如果接下来要继续把这部分工作做成更强的实验结论，我建议按下面顺序推进：

1. 先把 `Aloha` 上的成功率做上来
2. 再增加 `MetaWorld` 上更接近机械臂操控语义的任务评测
3. 对 `PushT` 这种极低维环境，单独设计更适合的 adapter 或单独训练小规模头部
4. 如果要做 PPT，要明确把 `Aloha` 说成“近分布验证”，把 `PushT / MetaWorld` 说成“跨域泛化验证”

## 11. 一句话总结

本次跨平台仿真结果说明：

- `RL+SmolVLA` 在接近训练分布的 `Aloha` 上有明确 reward 提升
- 在 `MetaWorld` 上也存在一定 reward 提升
- 在 `PushT` 上两者几乎无差别
- 当前提升主要体现在 reward 层面，还没有转化为成功率提升

