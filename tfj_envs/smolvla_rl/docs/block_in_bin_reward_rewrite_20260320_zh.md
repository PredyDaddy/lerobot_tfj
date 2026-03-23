# SmolVLA RL Block-In-Bin 奖励函数重写说明

这份说明对应 2026-03-20 新增的本地代理环境：

- `gym_aloha/AlohaBlockInBin-v0`

对应代码：

- `/data/tfj/lerobot_tfj/gym_aloha/simple_aloha_env.py`

## 1. 为什么要重写

之前 hybrid RL 训练跑通时使用的是一个非常简化的代理奖励：

```text
reward = -||state - target||_2 + success_bonus
```

那套奖励只适合验证：

- collector 能不能收集 online chunk
- replay buffer 能不能工作
- value loss 能不能反传
- hybrid 训练链路能不能跑通

它不适合表达真实的 “Put the block in the bin” 任务语义。

这次重写的目标不是把 toy env 变成真实机器人仿真，而是把 reward 改成更接近任务流程的 staged reward：

1. 靠近物块
2. 抓住物块
3. 抬起物块
4. 搬运到箱子上方
5. 放入箱子

## 2. 新环境的状态和动作

为了兼容现有 SmolVLA 配置，仍然保持：

- 状态维度：6
- 动作维度：6
- 两路图像：`top` / `wrist`

观测状态定义为：

```text
state = [
  ee_x,
  ee_y,
  ee_z,
  gripper_open,
  block_x,
  block_y,
]
```

其中：

- `ee_x, ee_y, ee_z`：末端执行器位置
- `gripper_open`：夹爪开合程度，`0` 表示闭合，`1` 表示张开
- `block_x, block_y`：物块平面位置

环境内部还维护一些隐藏量：

- `block_z`
- `bin_xy`
- `grasped`
- `placed`

动作的主要语义是：

```text
action[:2]   -> 末端平面移动
action[2]    -> 末端高度
action[3]    -> 夹爪开合
action[4:6]  -> 平面细调
```

## 3. 新奖励函数

### 3.1 总公式

新环境的单步奖励定义为：

```text
reward =
    reach_reward
  + grasp_hold_reward
  + grasp_bonus
  + lift_reward
  + transport_reward
  + place_bonus
  - action_penalty
  - time_penalty
```

## 3.2 各项含义

### 3.2.1 接近物块奖励

```text
ee_to_block_xy = || ee_xy - block_xy ||_2
reach_reward = 0.8 * (1 - tanh(4.0 * ee_to_block_xy))
```

作用：

- 鼓励机械臂先移动到物块附近
- 距离越近，reward 越高
- 用 `tanh` 是为了避免距离过大时 reward 数值爆炸

### 3.2.2 抓持持续奖励

```text
grasp_hold_reward = 1.0 if grasped else 0.0
```

作用：

- 只要已经抓住，就持续给一个正奖励
- 避免策略抓住以后立刻松手

### 3.2.3 抓取瞬时奖励

```text
grasp_bonus = 2.0 if just_grasped else 0.0
```

抓取判定条件：

```text
||ee_xy - block_xy||_2 < 0.06
ee_z < 0.09
gripper_open < 0.35
```

作用：

- 鼓励策略完成“靠近 + 下降 + 闭爪”这一关键动作
- 这是一个阶段切换奖励

### 3.2.4 抬升奖励

```text
lift_progress = clip((block_z - 0.02) / 0.18, 0, 1)
lift_reward = 1.5 * lift_progress
```

作用：

- 物块刚离桌面时开始获得奖励
- 抬得越高，reward 越大
- 上限截断，避免无限推高

### 3.2.5 搬运到箱子上方奖励

```text
block_to_bin = ||block_xy - bin_xy||_2
transport_reward = 2.5 * (1 - tanh(3.5 * block_to_bin))
```

注意：

- 只有在 `grasped` 或已经 `placed` 时才计算这一项

作用：

- 鼓励已经抓起的物块朝箱子移动
- 这是任务中最重要的中间 shaping 之一

### 3.2.6 放置成功奖励

```text
place_bonus = 8.0 if just_placed else 0.0
```

放置成功的触发逻辑是：

```text
releasing      = gripper_open > 0.78
over_bin       = ||block_xy - bin_xy||_2 < 0.07
high_enough    = block_z > 0.12
```

在已经抓住物块的情况下，如果：

- 到了箱子上方
- 物块已经抬起来
- 夹爪张开释放

那么环境会把物块判成已放入箱子，并给一大笔终结奖励。

### 3.2.7 动作惩罚

```text
action_penalty = 0.02 * ||action[:4]||_2
```

作用：

- 防止动作过于剧烈
- 抑制高频抖动

### 3.2.8 时间惩罚

```text
time_penalty = 0.01
```

作用：

- 鼓励更快完成任务
- 避免一直拖时间吃中间奖励

## 4. 成功判定

最终 `success` 定义为：

```text
placed == True and ||block_xy - bin_xy||_2 < 0.05
```

一旦成功：

- `terminated = True`
- `info["is_success"] = True`

这意味着：

- reward 用于训练
- `is_success` 用于成功率评测

这两个概念被明确分开了。

## 5. 为什么这版比旧版更适合 block-in-bin

旧版奖励的问题是：

- 只有“离某个抽象目标向量近不近”
- 没有抓取阶段
- 没有搬运阶段
- 没有放入箱子的语义

新版奖励更贴近这个任务的原因是：

- 它把任务拆成了机械操作里真正有意义的阶段
- 每个阶段都有连续 shaping
- 最终成功有明确大 bonus
- 同时保留动作惩罚和时间惩罚，避免策略钻空子

## 6. 重要边界

这版奖励仍然是“代理奖励”，不是 SO101 真机真实世界奖励。

也就是说：

- 它比旧的 `-distance to target` 更像 block-in-bin
- 但它仍然不是现实物理仿真
- 它更适合做 hybrid RL 链路调试和 reward 结构验证

如果后面要做真正面向 SO101 的 RL，建议下一步走两条路线之一：

1. 引入真实成功事件，人工按键或脚本标记 success
2. 使用 reward classifier，对真实相机画面直接判 success

## 7. 现在怎么启动

直接用新脚本：

```bash
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_block_in_bin.sh
```

或者手动指定 task：

```bash
bash /data/tfj/lerobot_tfj/tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh \
  aloha \
  AlohaBlockInBin-v0
```

## 8. 这版奖励和 hybrid loss 的关系

这点很重要。

SmolVLA hybrid trainer 里：

- 环境先输出 step reward
- collector 把一个 action chunk 内的多步 reward 折扣累加成 chunk reward
- 这个 chunk reward 再进入 value target
- advantage 再去加权 online flow matching loss

所以严格说，训练里真正被消费的是：

```text
chunk_return = r_0 + gamma * r_1 + gamma^2 * r_2 + ...
```

而不是单个 primitive step reward。

这也是为什么 reward 设计必须：

- 稳定
- 不要全靠最后一步稀疏成功
- 中间阶段要能持续提供有方向的信号
