# ACT 蒸馏工作区说明

## 1. 目录目的

这个目录把本次 ACT 蒸馏相关的训练脚本、上机脚本和说明文档统一收敛在：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill`

目录结构：

- `scripts/`
  - 训练 smoke / full 启动脚本
  - 一键训练脚本
  - 后台训练脚本
  - SO101 上机推理脚本
- `docs/`
  - 训练说明
  - 代码修改说明
  - 工作区使用说明
- `README.md`
  - 最短使用入口

## 2. 当前默认训练方案

教师模型：

- `/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last`

学生模型：

- policy type: `act`
- `dim_model=512`
- `n_heads=8`
- `chunk_size=100`
- `n_action_steps=100`
- `n_encoder_layers=2`
- `dim_feedforward=1024`
- `latent_dim=16`
- `n_vae_encoder_layers=2`

说明：

- 当前 Stage-2 `decoder_kd` 默认路径还没有把 projection 接入训练主干，所以学生保持 `dim_model=512`，只压缩层数、FFN 和 latent。
- 这次学生大约 `23.1M` 参数，教师约 `51.6M` 参数。

## 3. 训练入口

### 3.1 smoke

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash scripts/train_act_distill_smoke.sh
```

### 3.2 full

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash scripts/train_act_distill_full.sh
```

### 3.3 一键训练

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
MODE=smoke bash scripts/launch_act_distill_train.sh
```

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
MODE=full bash scripts/launch_act_distill_train.sh
```

### 3.4 后台训练

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
MODE=full bash scripts/start_act_distill_train_nohup.sh
```

## 4. SO101 上机入口

### 4.1 argparse 风格 Python 入口

文件：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/lerobot_run_act_so101.py`

典型用法：

```bash
PYTHONPATH=/data/tfj/lerobot_tfj/src python \
  /data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/lerobot_run_act_so101.py \
  --policy-path /data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740 \
  --policy-device cuda \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --fps 30 \
  --run-time-s 300 \
  --task "Put the block in the bin" \
  --display-data
```

### 4.2 一键 shell 入口

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 \
ROBOT_PORT=/dev/ttyACM0 \
TOP_CAM_INDEX=4 \
WRIST_CAM_INDEX=6 \
TASK_TEXT="Put the block in the bin" \
bash scripts/run_act_distill_so101_infer.sh
```

## 5. 已确认的环境注意事项

### 5.1 视频后端

本次训练脚本固定使用：

- `--dataset.video_backend=pyav`

原因：

- 该数据集缓存视频是 `AV1`
- 当前环境里 `PyAV/libdav1d` 能正常解码
- `torchcodec` 在训练 DataLoader 路径上出现过 `No valid stream found in input file`

### 5.2 训练入口

必须使用 repo-local 入口：

- `PYTHONPATH=/data/tfj/lerobot_tfj/src python -m lerobot.scripts.lerobot_train`

不要使用全局安装的：

- `lerobot-train`

因为该 wrapper 可能导向另一个安装目录，不是当前仓库。

## 6. 关键输出路径

训练默认输出：

- smoke:
  - `/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_smoke_20260315_153740`
- full:
  - `/data/tfj/lerobot_tfj/outputs/act_distill_grasp_block_in_bin1_stage2_20260315_153740`

训练日志：

- 后台模式默认写到：
  - `/data/tfj/lerobot_tfj/tfj_envs/act_distill/logs/`
