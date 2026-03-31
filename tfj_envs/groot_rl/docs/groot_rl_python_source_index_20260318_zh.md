# GROOT RL Python 源码索引

## 1. 用户入口脚本

### `scripts/run_groot_so101_infer.sh`

作用：

- GROOT SO101 direct inference 的 shell 入口
- 默认选择 stage-1 safe policy
- 支持 `PREFER_STAGE2_RL=true`
- 支持 `PREFLIGHT_ONLY=1`
- 默认 no-save / no-dataset / no-events

### `scripts/train_groot_grasp_block_in_bin1_offline_rl_stage2.sh`

作用：

- 启动 dataset-only offline RL stage-2
- 做 PyAV / 数据集 / checkpoint 预检
- 配置 offline replay 相关超参

### `scripts/train_groot_grasp_block_in_bin1_offline_stage2_rl.sh`

作用：

- stage-2 训练脚本兼容别名
- 转发到 canonical launcher

### `scripts/openclaw_groot_server.py`

作用：

- OpenClaw 侧 GROOT server 入口
- 保留了 safer default policy 选择逻辑

### `scripts/run_so101_pickplace_infer.sh`

作用：

- SO101 inference backend router
- 可转发到 groot / smolvla / pi05 / act / policy_record

### `scripts/run_so101_policy_record.sh`

作用：

- 仍然保留的录制路径入口
- 适用于需要 dataset recording 的场景

## 2. 核心 Python 实现快照

### `reference_src/src/lerobot/scripts/lerobot_run_so101_pickplace.py`

作用：

- 真实的 GROOT SO101 direct no-save runtime
- 直接从 checkpoint 加载 policy 和 processor
- 构建 robot runtime
- 调用共享 `record_loop(..., dataset=None, policy=...)`

### `reference_src/src/lerobot/scripts/lerobot_record_so101_policy.py`

作用：

- SO101 runtime config / bridge / safety guard / JSONL logger 的主要实现
- direct runtime 和 record runtime 都会复用这里的装配逻辑

### `reference_src/src/lerobot/scripts/lerobot_train_groot_hybrid.py`

作用：

- GROOT hybrid trainer 的 CLI 入口

### `reference_src/src/lerobot/configs/train_groot_hybrid.py`

作用：

- GROOT hybrid train config
- 包含 offline replay 相关配置项

### `reference_src/src/lerobot/rl/groot_hybrid/offline_replay.py`

作用：

- dataset-only offline replay buffer
- 负责 transition sampling、value target 构造等逻辑

### `reference_src/src/lerobot/rl/groot_hybrid/trainer.py`

作用：

- GROOT hybrid training 主流程
- 负责 offline replay、训练 step 和 checkpoint 落盘

## 3. 当前最重要的行为结论

- 机器人默认不要直接上 stage-2 RL checkpoint
- 机器人默认应该继续用 stage-1 safe checkpoint
- direct inference 现在已经不需要 dataset 目录
- 如果要日志，单独开 `EVENTS_JSONL_PATH`
- stage-2 训练主要用于离线继续优化，不应自动覆盖 real-robot 默认部署路径
