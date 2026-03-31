# ACT 蒸馏一键命令

这个文档只保留可直接复制执行的命令。

工作目录：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
```

## 1. smoke 训练

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_smoke.sh
```

自定义输出目录：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
OUTPUT_DIR=/data/tfj/lerobot_tfj/outputs/act_distill_smoke_manual bash train_smoke.sh
```

## 2. full 训练

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_full.sh
```

自定义输出目录：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
OUTPUT_DIR=/data/tfj/lerobot_tfj/outputs/act_distill_full_manual bash train_full.sh
```

## 3. 后台 full 训练

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_full_nohup.sh
```

后台训练后查看日志：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ls -lt logs
```

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
tail -f logs/act_distill_train_full_*.log
```

## 4. SO101 上机推理

最短命令：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
bash infer_so101.sh
```

带任务文本：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 \
ROBOT_PORT=/dev/ttyACM0 \
TOP_CAM_INDEX=4 \
WRIST_CAM_INDEX=6 \
TASK_TEXT="Put the block in the bin" \
bash infer_so101.sh
```

关闭可视化：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
DISPLAY_DATA=false bash infer_so101.sh
```

## 5. 直接用 Python 入口

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
PYTHONPATH=/data/tfj/lerobot_tfj/src python scripts/lerobot_run_act_so101.py \
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

## 6. 最常用环境变量

训练：

- `OUTPUT_DIR`
- `BASE_CONFIG`
- `TEACHER_PATH`
- `PYTHON_BIN`

上机：

- `POLICY_PATH`
- `POLICY_DEVICE`
- `ROBOT_ID`
- `ROBOT_PORT`
- `ROBOT_CALIB_DIR`
- `TOP_CAM_INDEX`
- `WRIST_CAM_INDEX`
- `TASK_TEXT`
- `DISPLAY_DATA`
- `RUN_TIME_S`

## 7. 以后默认就记这 4 个文件

- `train_smoke.sh`
- `train_full.sh`
- `train_full_nohup.sh`
- `infer_so101.sh`
