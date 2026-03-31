# GROOT 训练记录（grasp_block_in_bin1）

> 日志来源：`/data/tfj/lerobot/outputs/groot_grasp_block_in_bin1.log`

## 1. 数据集信息
- Dataset repo_id: `admin123/grasp_block_in_bin1`
- Dataset root: `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- 视频编码：AV1（见 `meta/info.json`）
- 解码后端（训练时）：`torchcodec`

## 2. 基模与资源
- GROOT 基模路径：`/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B`
- Eagle 资产缓存：`/home/cqy/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5`

## 3. 训练输出位置
- 训练输出目录：`/data/tfj/lerobot/tmp/train/groot_grasp`
- Checkpoints：`/data/tfj/lerobot/tmp/train/groot_grasp/checkpoints`
- 最新 checkpoint：`/data/tfj/lerobot/tmp/train/groot_grasp/checkpoints/last`

## 4. 训练命令（原始）
```bash
lerobot-train \
  --policy.type=groot \
  --policy.repo_id=robotech/groot \
  --dataset.repo_id=admin123/grasp_block_in_bin1 \
  --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1 \
  --batch_size=32 \
  --steps=10000 \
  --output_dir=tmp/train/groot_grasp \
  --job_name=groot_training \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.base_model_path=/home/cqy/.cache/modelscope/hub/models/nv-community/GR00T-N1.5-3B \
  --save_freq=2000
```

## 5. 断点续训命令
```bash
lerobot-train \
  --resume=true \
  --config_path=/data/tfj/lerobot/tmp/train/groot_grasp/checkpoints/last/pretrained_model/train_config.json
```

## 6. 训练超参数（来自日志）
- batch_size: 64（effective batch size: 64 x 1）
- steps: 10000
- save_freq: 2000
- eval_freq: 20000
- optimizer: adamw, lr=1e-4, betas=(0.95,0.999), wd=1e-5
- scheduler: cosine_decay_with_warmup
- n_action_steps: 50
- n_obs_steps: 1
- normalization_mapping: ACTION/STATE=MEAN_STD, VISUAL=IDENTITY

## 7. 训练进度片段（日志摘要）
- Step 200: loss ~0.191
- Step 1000: loss ~0.024
- Step 2000: loss ~0.019
- Step 4000: loss ~0.013
- Step 6000: loss ~0.009
- Step 8000: loss ~0.006

## 8. 备注
- FlashAttention 版本：2.6.3（日志显示加载成功）
- tokenizer regex 警告可忽略（不会阻断训练）
