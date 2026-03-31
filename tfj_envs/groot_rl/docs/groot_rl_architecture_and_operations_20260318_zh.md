# GROOT RL 架构与操作说明

## 1. 这份 bundle 里放了什么

这份 `tfj_envs/groot_rl` 目录整理的是本轮会话里围绕 GROOT 做的两条主线：

- `stage-2 dataset-only offline RL`
- `SO101 上的 GROOT direct no-save inference`

它不是一个新的独立 Python 包，而是一个便于查阅的归档目录。真正运行时仍然使用仓库根目录下的原始实现；这里同时保留了用户入口脚本副本和关键源码快照。

## 2. 训练路径

训练入口脚本：

- `scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`
- `scripts/train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`

这一阶段的核心特点：

- 不接在线 env
- 只使用数据集 replay
- reward 由 demo 轨迹合成
- trainer 现在支持正常写出 checkpoint，便于 resume 和后续部署核查

核心源码快照：

- `reference_src/src/lerobot/scripts/lerobot_train_groot_hybrid.py`
- `reference_src/src/lerobot/configs/train_groot_hybrid.py`
- `reference_src/src/lerobot/rl/groot_hybrid/offline_replay.py`
- `reference_src/src/lerobot/rl/groot_hybrid/trainer.py`

## 3. 机器人推理路径

当前机器人入口脚本：

- `scripts/run_groot_so101_infer.sh`

这一条路径已经改成真正的 direct inference：

- 默认不创建 dataset 目录
- 默认不保存录制数据
- 默认也不写 `events.jsonl`
- 如需事件日志，显式传 `EVENTS_JSONL_PATH=...`
- 默认部署权重仍然是较稳的 stage-1 GROOT
- 只有显式 `PREFER_STAGE2_RL=true` 时才会切到 stage-2 RL checkpoint

核心源码快照：

- `reference_src/src/lerobot/scripts/lerobot_run_so101_pickplace.py`
- `reference_src/src/lerobot/scripts/lerobot_record_so101_policy.py`

## 4. 为什么 stage-1 仍是默认

本轮会话里已经验证过：

- stage-2 RL checkpoint 在真实机器人上会出现过大的动作请求，guard 会拒绝并停机
- stage-1 checkpoint 的动作尺度正常，能稳定通过 guard

因此 bundle 里保留了明确的部署原则：

- real robot default: stage-1 safe checkpoint
- stage-2 RL: opt-in only

## 5. 关键运行命令

### 5.1 直接推理

```bash
cd /data/tfj/lerobot_tfj
bash scripts/run_groot_so101_infer.sh
```

### 5.2 只做直推加载预检

```bash
cd /data/tfj/lerobot_tfj
PREFLIGHT_ONLY=1 bash scripts/run_groot_so101_infer.sh
```

### 5.3 直推但保留事件日志

```bash
cd /data/tfj/lerobot_tfj
EVENTS_JSONL_PATH=./outputs/groot_direct.events.jsonl bash scripts/run_groot_so101_infer.sh
```

### 5.4 开 stage-2 dataset-only offline RL

```bash
cd /data/tfj/lerobot_tfj
bash scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh
```

## 6. 使用这份 bundle 时要注意什么

- `scripts/` 里的文件是本轮整理时的副本，后续如果仓库原文件继续变化，这里不会自动同步。
- `reference_src/` 也是源码快照，不应被当成新的 import 路径。
- 真正执行时，请优先在仓库根目录使用原路径命令。
