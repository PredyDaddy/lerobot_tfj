# Pi0 / Pi05 训练文档（ModelScope 基模版）

本文档用于在当前 `lerobot` 仓库中快速启动 `pi0` 或 `pi05` 微调训练，基模通过 **ModelScope** 下载到本地后再训练。

## 1. 模型名称

推荐优先使用以下模型名：

- `lerobot/pi0_base`
- `lerobot/pi05_base`

说明：

- `pi0` 对应策略类型：`--policy.type=pi0`
- `pi05` 对应策略类型：`--policy.type=pi05`

## 2. 环境准备

在仓库根目录执行：

```bash
pip install -U modelscope
pip install -e ".[pi]"
```

## 3. 使用 ModelScope 下载基模

新建并执行下载脚本（或直接复制到终端）：

```bash
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

pi0_dir = snapshot_download(
    "lerobot/pi0_base",
    cache_dir="./pretrained_models",
)
print("PI0_BASE_DIR=", pi0_dir)

pi05_dir = snapshot_download(
    "lerobot/pi05_base",
    cache_dir="./pretrained_models",
)
print("PI05_BASE_DIR=", pi05_dir)
PY
```

下载完成后，一般会在以下目录看到模型文件：

- `./pretrained_models/lerobot/pi0_base`
- `./pretrained_models/lerobot/pi05_base`

如果 `pi05_base` 暂时拉不到，可先只训练 `pi0`。

## 4. 训练命令

先替换以下占位符：

- `your_dataset`：数据集 repo id，例如 `your_name/your_dataset`
- `your_repo_id`：训练输出模型 repo id，例如 `your_name/pi0_finetune_xxx`

### 4.1 训练 pi0

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_dataset \
  --policy.type=pi0 \
  --output_dir=./outputs/pi0_training \
  --job_name=pi0_training \
  --policy.repo_id=your_repo_id \
  --policy.pretrained_path=./pretrained_models/lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=3000
```

### 4.2 训练 pi05

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_dataset \
  --policy.type=pi05 \
  --output_dir=./outputs/pi05_training \
  --job_name=pi05_training \
  --policy.repo_id=your_repo_id \
  --policy.pretrained_path=./pretrained_models/lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=3000
```

## 5. Pi05 数据集 quantiles 问题处理

若训练 `pi05` 报错提示数据集没有 quantiles，可先执行：

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id=your_dataset
```

或在训练命令中补充如下归一化映射：

```bash
--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

## 6. 显存不够时的建议

如果出现 OOM（显存不足）：

1. 将 `--batch_size=32` 改为 `16` 或 `8`
2. 保持 `--policy.gradient_checkpointing=true`
3. 保持 `--policy.dtype=bfloat16`
4. 先用 `--steps=500` 做冒烟训练，确认流程没问题再拉长步数

## 7. 最小可跑版本（先验证流程）

### pi0 最小验证

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_dataset \
  --policy.type=pi0 \
  --output_dir=./outputs/pi0_smoke \
  --job_name=pi0_smoke \
  --policy.repo_id=your_repo_id \
  --policy.pretrained_path=./pretrained_models/lerobot/pi0_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=8 \
  --steps=200
```

### pi05 最小验证

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=your_dataset \
  --policy.type=pi05 \
  --output_dir=./outputs/pi05_smoke \
  --job_name=pi05_smoke \
  --policy.repo_id=your_repo_id \
  --policy.pretrained_path=./pretrained_models/lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=8 \
  --steps=200
```

---

如果你已经确定了 `your_dataset` 和 `your_repo_id`，可以把值贴出来，我可以直接给你生成“最终可执行版命令”（无需手动替换占位符）。
