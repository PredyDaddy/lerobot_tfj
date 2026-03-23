# 训练日志 2（train_log2.md）

> 说明：该文档用于完整记录本次 ACT 训练（torchcodec 方案）的所有关键信息。尽量“可复现实验”的粒度。

## 1. 训练目标
- 策略：ACT（offline imitation）
- 任务：grasp_block_in_bin
- 目的：提高训练吞吐、降低数据解码瓶颈

## 2. 环境信息
- 机器：ubuntu-Z790-EAGLE-AX
- Conda 环境：`lerobot`
- Python：3.11
- CUDA：可用（GPU 总显存 ~47GB）
- 日期：2026-02-28

## 3. 数据集信息
- repo_id：`admin123/grasp_block_in_bin_clean`
- root：`/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin_clean`
- 帧数：~53K
- 轨迹数：118

## 4. 关键改动（相对你之前慢的训练）
1. **视频解码后端**
   - 从 `pyav` 切换到 `torchcodec`
   - 目的：降低 `data_s` 读取时间
   - 备注：`decord` 在本仓库不支持（会报 Unsupported video backend）

2. **数据加载与序列长度**
   - `--num_workers=8`
   - `--policy.chunk_size=50`
   - `--policy.n_action_steps=50`
   - `--policy.use_vae=false`
   - 目的：降低计算量、减轻 I/O 与模型开销

3. **显存控制**
   - `batch_size=64`
   - 原先 `batch=128` 曾触发 OOM

## 5. 训练命令（推荐）
```bash
source /home/cqy/miniconda3/etc/profile.d/conda.sh && conda activate lerobot

nohup lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=admin123/grasp_block_in_bin_clean \
  --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin_clean \
  --dataset.video_backend=torchcodec \
  --batch_size=64 \
  --steps=20000 \
  --num_workers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.use_vae=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=20 \
  --save_freq=4000 \
  --output_dir=outputs/train/act_grasp_fast_$(date +%Y%m%d_%H%M%S) \
  --job_name=act_grasp_fast \
  > /data/tfj/lerobot/outputs/train/act_grasp_fast_latest.log 2>&1 &
```

## 6. 日志位置
- 主日志：`/data/tfj/lerobot/outputs/train/act_grasp_fast_latest.log`
- PID 文件：`/data/tfj/lerobot/outputs/train/act_grasp_fast_latest.pid`

## 7. 训练检查命令
```bash
pgrep -af 'lerobot-train'

# 查看实时日志
 tail -n 200 /data/tfj/lerobot/outputs/train/act_grasp_fast_latest.log
```

## 8. 已知问题与处理
- `torchcodec` 在未安装 FFmpeg 时会报错：`Could not load libtorchcodec`。
- 解决方式：
  ```bash
  conda install -y -c conda-forge ffmpeg libffi
  ```

## 9. 验证 torchcodec 是否可用
```bash
python - <<'PY'
import torchcodec
print('torchcodec_ok', torchcodec.__version__)
PY
```

## 10. 训练速度参考（期望）
- 首个 batch `data_s` 可能较高，后续稳定在 ~0.02s（torchcodec 正常时）
- `updt_s` ~ 0.4–0.8s 区间

## 11. 备注
- 该文档只描述“加速 + 可用”的 ACT 训练配置。
- 若要回退到 `pyav`，只需替换：`--dataset.video_backend=pyav`。
- 若仍遇到慢速问题，建议检查磁盘读写负载和 CPU 使用率。



nohup lerobot-train \
   --policy.type=groot \
   --dataset.repo_id=admin123/grasp_block_in_bin1 \
   --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1 \
   --dataset.video_backend=torchcodec \
   --output_dir=./outputs/groot_grasp_block_in_bin1 \
   --steps=18000 \
   --save_freq=2000 \
   --eval_freq=-1 \
   --batch_size=32 \
   --num_workers=8 \
   --policy.device=cuda \
   --wandb.enable=false \
   --policy.push_to_hub=false \
   --policy.repo_id=Tao/groot_grasp_block_in_bin1 \
   --policy.base_model_path=/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B \
   > /data/tfj/lerobot/outputs/groot_grasp_block_in_bin1.log 2>&1 &


#GR00T训练代码


nohup lerobot-train \
   --policy.type=groot \
   --policy.repo_id=robotech/groot \
   --dataset.repo_id=admin123/grasp_block_in_bin1 \
   --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1 \
   --batch_size=64 \
   --steps=10000 \
   --output_dir=tmp/train/groot_grasp \
   --job_name=groot_training \
   --policy.device=cuda \
   --wandb.enable=false \
   --policy.base_model_path=/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B \
   --save_freq=2000 \
   > /data/tfj/lerobot/outputs/groot_grasp_block_in_bin1.log 2>&1 &

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1003/gdm/Xauthority

python /data/tfj/lerobot_tfj/scripts/camera_overlay_align.py \
   --reference-image /data/tfj/lerobot_tfj/outputs/reference_first_frame.png \
   --camera-id 4 \
   --display-mode side_by_side
```

conda run -n lerobot_flex
python /data/tfj/lerobot_tfj/doc/lerobot_record_act_onnx.py \
   --robot.id=my_so101 \
   --robot.type=so101_follower \
   --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
   --robot.port=/dev/ttyACM0 \
   --robot.cameras='{"top":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":
{"type":"opencv","index_or_path":6,"width":640,"height":480,"fps":30}}' \
   --display_data=false \
   --play_sounds=false \
   --dataset.repo_id=admin123/eval_grasp_block_in_bin1 \
   --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/eval_grasp_block_in_bin2 \
   --dataset.push_to_hub=false \
   --dataset.num_episodes=5 \
   --dataset.episode_time_s=300 \
   --dataset.reset_time_s=10 \
   --dataset.single_task="grasp block in bin" \
   --policy.path=/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model \
   --policy.device=cuda \
   --onnx.path=/data/tfj/lerobot_tfj/outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model/act_core.onnx \
   --onnx.provider=auto





  python /data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_onnx_infer_so101.py \
      --robot-id my_so101 \
      --robot-port /dev/ttyACM0 \
      --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
      --robot-max-relative-target 1 \
      --top-cam-index 4 \
      --wrist-cam-index 6 \
      --camera-width 640 \
      --camera-height 480 \
      --camera-fps 30 \
      --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
      --policy-device cpu \
      --policy-n-action-steps 8 \
      --policy-num-inference-steps 10 \
      --policy-fixed-noise \
      --policy-noise-seed 0 \
      --onnx-path /data/tfj/lerobot_tfj/tfj_envs/pi_trt/docs/results/pi05_onnx_fix_20260311_230500/onnx \
      --onnx-provider cuda \
      --local-tokenizer-path /home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224 \
      --joint-delta-limit 0.7 \
      --gripper-delta-limit 1.5 \
      --joint-action-alpha 0.2 \
      --gripper-action-alpha 0.35 \
      --task "grasp block in bin" \
      --run-time-s 10 \
      --log-interval 5


  python /data/tfj/lerobot_tfj/tfj_envs/pi_trt/scripts/run_pi05_torch_infer_so101.py \
      --robot-id my_so101 \
      --robot-port /dev/ttyACM0 \
      --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
      --robot-max-relative-target 1 \
      --top-cam-index 4 \
      --wrist-cam-index 6 \
      --camera-width 640 \
      --camera-height 480 \
      --camera-fps 30 \
      --policy-path /data/tfj/lerobot_tfj/pi_model/pretrained_model \
      --policy-device cuda \
      --policy-n-action-steps 8 \
      --policy-num-inference-steps 10 \
      --local-tokenizer-path /home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224 \
      --joint-delta-limit 0.7 \
      --gripper-delta-limit 1.5 \
      --joint-action-alpha 0.2 \
      --gripper-action-alpha 0.35 \
      --task "grasp block in bin" \
      --run-time-s 10 \
      --log-interval 5



  cd /data/tfj/lerobot_tfj

  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  python /data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record.py \
    --robot.type=so101_follower \
    --robot.id=my_so101 \
    --robot.port=/dev/ttyACM0 \
    --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
    --robot.max_relative_target=0.5 \
    --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 5}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480,
  fps: 5}}' \
    --dataset.repo_id=local/eval_pi05_so101_debug \
    --dataset.single_task='grasp block in bin' \
    --dataset.root=/tmp/lerobot_record_pi05 \
    --dataset.fps=5 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=0 \
    --dataset.num_episodes=1 \
    --dataset.video=false \
    --dataset.push_to_hub=false \
    --dataset.num_image_writer_threads_per_camera=1 \
    --policy.path=/data/tfj/lerobot_tfj/pi_model/pretrained_model \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.dtype=float32 \
    --policy.n_action_steps=1 \
    --policy.num_inference_steps=10 \
    --display_data=false \
    --play_sounds=false




     source /home/cqy/miniconda3/etc/profile.d/conda.sh
  conda activate lerobot_flex
  cd /data/tfj/lerobot_tfj

  PYTHONPATH=/data/tfj/lerobot_tfj/src \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PI05_LOCAL_TOKENIZER_PATH=/home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224 \
  python /data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_record.py \
    --robot.type=so101_follower \
    --robot.id=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
    --robot.max_relative_target=1.0 \
    --robot.cameras='{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 5}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 5}}' \
    --dataset.repo_id=local/eval_pi05_so101_debug \
    --dataset.single_task='grasp block in bin' \
    --dataset.root=/tmp/lerobot_record_pi05 \
    --dataset.fps=5 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=0 \
    --dataset.num_episodes=1 \
    --dataset.video=false \
    --dataset.push_to_hub=false \
    --dataset.num_image_writer_threads_per_camera=1 \
    --policy.path=/data/tfj/lerobot_tfj/pi_model/pretrained_model \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.dtype=float32 \
    --policy.n_action_steps=1 \
    --policy.num_inference_steps=10 \
    --display_data=false \
    --play_sounds=false



lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30,fourcc: "MJPG"}}" \
  --robot.id=so101_follower \
  --display_data=false \
  --dataset.repo_id=local/eval_pi05_so101_debug \
  --dataset.single_task="Clean the desk" \
  --policy.path=/data/tfj/lerobot_tfj/pi_model/pretrained_model