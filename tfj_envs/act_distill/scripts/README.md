# ACT Distill Scripts

Main entrypoints:

- `train_act_distill_smoke.sh`
  - 4-step smoke run for config, KD wiring, processor compatibility, and checkpoint save.
- `train_act_distill_full.sh`
  - full Stage-2 ACT distillation run.
- `launch_act_distill_train.sh`
  - one-click training launcher with `MODE=smoke` or `MODE=full`.
- `start_act_distill_train_nohup.sh`
  - background training launcher with log redirection.
- `lerobot_run_act_so101.py`
  - argparse-style SO101 on-robot ACT inference entrypoint.
- `run_act_distill_so101_infer.sh`
  - one-click shell wrapper for real-robot inference.

Examples:

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
bash scripts/train_act_distill_smoke.sh
```

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
MODE=full bash scripts/start_act_distill_train_nohup.sh
```

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/act_distill
ROBOT_ID=my_so101 ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
bash scripts/run_act_distill_so101_infer.sh
```
