# ACT Distill Workspace

这个目录是本次 ACT 蒸馏的统一工作区。

## 结构

- 根目录快捷入口
  - `train_smoke.sh`
  - `train_full.sh`
  - `train_full_nohup.sh`
  - `infer_so101.sh`
  - `ONE_CLICK_COMMANDS_zh.md`
- `scripts/`
  - 训练 smoke / full 脚本
  - 一键训练脚本
  - 后台训练脚本
  - SO101 上机推理脚本
- `docs/`
  - 训练说明、代码修改说明、工作区使用说明
  - 通用 ACT 蒸馏知识文档，可直接用于生成 PPT

## 最常用入口

训练 smoke：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_smoke.sh
```

训练 full：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_full.sh
```

一键训练：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
MODE=full bash scripts/launch_act_distill_train.sh
```

后台训练：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash train_full_nohup.sh
```

SO101 上机推理：

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
bash infer_so101.sh
```

## 推荐阅读顺序

1. `docs/ACT_DISTILL_PPT_KNOWLEDGE_BASE_20260317_zh.md`
2. `docs/ACT_DISTILL_WORKSPACE_GUIDE_20260316_zh.md`
3. `docs/ACT_DISTILL_CODE_CHANGES_20260316_zh.md`
4. `scripts/README.md`
5. `ONE_CLICK_COMMANDS_zh.md`
