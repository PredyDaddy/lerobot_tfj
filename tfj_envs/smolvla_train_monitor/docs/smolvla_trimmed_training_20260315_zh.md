# SmolVLA 裁剪数据集训练复现简版

## 1. 这份文档是干什么的

这是一份中文精简版说明，专门记录这次已经跑通的 SmolVLA 离线训练流程，方便后面直接复现。

本次成功训练的数据集是：

- 数据集 `repo_id`：
  `admin123/grasp_block_in_bin1_trimmed_static_tail`
- 数据集本地目录：
  `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`

对应的成功训练输出目录是：

- `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341`

完整英文版过程记录见：

- `docs/smolvla_trimmed_training_20260315.md`

## 2. 这次最终跑通的关键配置

这次能稳定跑通，靠的是下面这几个点同时成立：

1. 必须显式传 `dataset.repo_id` 和 `dataset.root`。
2. 视频后端不能继续用 `torchcodec`，要改成 `pyav`。
3. 强制走本地离线缓存，避免训练时再去访问 Hugging Face。
4. 数据集里的图像键要通过 `rename_map` 映射成 SmolVLA 训练期望的名字。

最终稳定配置的核心参数是：

- `--dataset.repo_id=admin123/grasp_block_in_bin1_trimmed_static_tail`
- `--dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`
- `--dataset.video_backend=pyav`
- `--batch_size=32`
- `--steps=10000`
- `--num_workers=4`

## 3. 为什么一开始会失败

这次排查里，主要踩了下面几个坑。

### 3.1 缺依赖

一开始缺：

- `num2words`

需要补：

```bash
python -m pip install num2words
```

### 3.2 `torchcodec` 解不了这批视频

这套裁剪数据集的视频是 `AV1` 编码。用 `torchcodec` 试跑时，训练在取 batch 的阶段报错：

```text
ValueError: No valid stream found in input file. Is -1 of the desired media type?
```

所以最后改成：

```bash
--dataset.video_backend=pyav
```

### 3.3 不能只给本地路径当 `dataset.repo_id`

直接把本地绝对路径塞进 `--dataset.repo_id`，不是这次最稳的写法。稳定组合是：

```bash
--dataset.repo_id=admin123/grasp_block_in_bin1_trimmed_static_tail
--dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail
```

### 3.4 训练不是按 epoch 停，是按 step 停

这次不是“训练满 10 个 epoch 自动停”，而是：

```bash
--steps=10000
```

所以它会在第 `10000` step 正常结束。

## 4. 一键复现方式

### 4.1 直接启动训练

```bash
bash tfj_envs/smolvla_train_monitor/scripts/launch_smolvla_trimmed_train.sh \
  /data/tfj/lerobot_tfj/outputs/train/my_smolvla_trimmed_run \
  /data/tfj/lerobot_tfj/outputs/logs/my_smolvla_trimmed_run.train.log \
  10000 \
  32 \
  4
```

### 4.2 后台启动训练并同时启动监控

```bash
bash tfj_envs/smolvla_train_monitor/scripts/start_smolvla_trimmed_train_nohup.sh
```

这个脚本会自动生成：

- 输出目录
- 训练日志
- 监控日志
- 训练 PID 文件
- 监控 PID 文件

### 4.3 如果你要至少跑满 10 个 epoch

建议直接用：

```bash
bash tfj_envs/smolvla_train_monitor/scripts/start_smolvla_trimmed_train_nohup.sh \
  10121 \
  32 \
  4 \
  60
```

这几个参数分别是：

1. `steps`
2. `batch_size`
3. `num_workers`
4. `监控间隔秒数`

## 5. 这次训练的最终结果

这次训练是正常结束，不是卡死，不是中断。

训练日志最后几行是：

```text
INFO 2026-03-15 14:12:31 ot_train.py:562 step:10K smpl:320K ep:1K epch:9.88 loss:0.019 grdn:0.190 lr:2.5e-06 updt_s:0.399 data_s:0.012
INFO 2026-03-15 14:12:31 ot_train.py:569 Checkpoint policy after step 10000
INFO 2026-03-15 14:12:32 ot_train.py:640 End of training
```

最终 checkpoint 目录里有：

- `002000`
- `004000`
- `006000`
- `008000`
- `010000`
- `last -> 010000`

也就是说，模型确实已经完整保存到了第 `10000` step。

## 6. 为什么不是正好 10 个 epoch

因为这次数据集大小和 batch size 决定了：

- `dataset.num_frames = 32385`
- `batch_size = 32`

所以：

- `10000 step` 一共训练了 `320000` 个 sample
- `320000 / 32385 = 9.8811`

也就是说，这次实际只训练到了大约：

- `9.88 epoch`

如果你想至少完整跑满 `10.00 epoch`，需要：

- `steps = ceil(10 * 32385 / 32) = 10121`

## 7. 本次建议保留的结论

以后你再跑这个裁剪数据集，建议默认沿用下面这套结论：

1. `dataset.video_backend` 默认先用 `pyav`。
2. `dataset.repo_id` 和 `dataset.root` 一起显式传。
3. 默认加离线环境变量，减少外部依赖。
4. 如果目标是“至少 10 epoch”，不要再用 `10000 step`，改成 `10121 step`。

## 8. 相关脚本

本目录下可直接复用的脚本有：

- `scripts/launch_smolvla_trimmed_train.sh`
- `scripts/monitor_train_process.sh`
- `scripts/start_smolvla_trimmed_train_nohup.sh`

如果后面你还要继续扩展这个目录，建议把新实验也按这个结构放：

- `docs/`
- `scripts/`
- 单独记录每次成功 run 的结论
