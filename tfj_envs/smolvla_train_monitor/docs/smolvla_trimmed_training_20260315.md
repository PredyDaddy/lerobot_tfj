# SmolVLA Trimmed Training And Monitoring Notes

## 1. Scope

This document records the full workflow used on `2026-03-15` to launch, debug, monitor, and complete a
SmolVLA offline training run on the trimmed SO101 dataset:

- Dataset root:
  `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`
- Dataset repo id:
  `admin123/grasp_block_in_bin1_trimmed_static_tail`
- Policy:
  `smolvla`
- Machine:
  RTX 4090

This is the exact path that was actually run to completion. The final completed run was:

- Run name:
  `smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341`
- Output dir:
  `/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341`
- Train log:
  `/data/tfj/lerobot_tfj/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341.train.log`
- Monitor log:
  `/data/tfj/lerobot_tfj/outputs/logs/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341.monitor.log`

## 2. What Was Done

The original intent in the broader session was related to SmolVLA and RL integration, but the actual runnable
path for this machine was narrowed to a proven offline SmolVLA training run on the trimmed dataset.

The final execution path was:

1. Verify the dataset metadata and task.
2. Verify the model checkpoints exist locally.
3. Check GPU availability.
4. Probe the training command with 1 step until the runtime stack was stable.
5. Launch the full training run.
6. Monitor the training log and process state until completion.

## 3. Important Findings Before Launch

### 3.1 Dataset facts

From the dataset metadata:

- `robot_type = so101_follower`
- `fps = 30`
- `observation.state` shape is `[6]`
- `action` shape is `[6]`
- image keys:
  `observation.images.top`
  `observation.images.wrist`

The task text from `meta/tasks.parquet` was:

- `Put the block in the bin`

### 3.2 Why this was run as offline SmolVLA training

For immediate execution, online RL on real SO101 hardware was not used here. The reason was practical:

- the current hybrid RL path was not yet wired cleanly for `gym_manipulator`
- reward handling for the real robot path was not yet ready to trust
- the offline SmolVLA training path on this dataset was already known to work

So the run described here is an offline supervised SmolVLA training run, not online RL.

### 3.3 Runtime issues that had to be fixed

#### Missing dependency

The training environment was missing:

- `num2words`

This was fixed with:

```bash
python -m pip install num2words
```

#### Dataset invocation detail

Passing only a local filesystem path through `--dataset.repo_id` was not the robust path for this run.
The stable configuration was:

- `--dataset.repo_id=admin123/grasp_block_in_bin1_trimmed_static_tail`
- `--dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`

#### Video backend issue

This trimmed dataset uses `AV1` video files. A probe run with `torchcodec` failed with:

- `ValueError: No valid stream found in input file. Is -1 of the desired media type?`

The stable workaround was:

- `--dataset.video_backend=pyav`

This is the main reason the final run used `pyav` instead of `torchcodec`.

#### Hugging Face offline loading

The stable run forced local-only loading:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `HF_HUB_DISABLE_TELEMETRY=1`

Without this, the runtime could spend time trying to reach the Hub even though the local snapshots already existed.

#### Tokenizer fork warning

During multiprocess data loading, the standard warning appeared:

- `huggingface/tokenizers: The current process just got forked...`

Training still worked. In the reusable launch script below, this is muted with:

- `TOKENIZERS_PARALLELISM=false`

### 3.4 Why plain `nohup` was not the reliable path inside Codex

In the Codex tool environment, plain detached child processes could be reaped after the tool call returned.
Because of that, the final successful run was kept alive using persistent PTY-backed sessions during the interactive
assistant workflow.

For a normal shell session on the machine, `nohup` is still a practical option. A wrapper script is provided below
for that purpose.

## 4. Scripts In This Directory

This directory contains three reusable scripts:

- `scripts/launch_smolvla_trimmed_train.sh`
  Runs the actual SmolVLA training command.
- `scripts/monitor_train_process.sh`
  Monitors a training PID, GPU usage, and the latest step/loss line in the log.
- `scripts/start_smolvla_trimmed_train_nohup.sh`
  Convenience wrapper to launch training and monitoring together from a normal shell.

## 5. Exact Stable Training Configuration

The stable full training configuration used for the successful run was:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
PYTHONPATH=/data/tfj/lerobot_tfj/src \
python /data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_train.py \
  --policy.path=/home/cqy/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/4d2f2b37fa245361ef1efe6d91ce96b8bd4af511 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.vlm_model_name=/home/cqy/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --dataset.repo_id=admin123/grasp_block_in_bin1_trimmed_static_tail \
  --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=10000 \
  --num_workers=4 \
  --save_freq=2000 \
  --save_checkpoint=true \
  --eval_freq=0 \
  --log_freq=50 \
  --wandb.enable=false \
  --output_dir=/data/tfj/lerobot_tfj/outputs/train/smolvla_grasp_block_in_bin1_trimmed_static_tail_20260315_130341 \
  --job_name=smolvla_grasp_block_in_bin1_trimmed_static_tail \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

## 6. Final Outcome Of The Completed Run

The run completed normally.

The ending lines in the training log were:

```text
INFO 2026-03-15 14:12:31 ot_train.py:562 step:10K smpl:320K ep:1K epch:9.88 loss:0.019 grdn:0.190 lr:2.5e-06 updt_s:0.399 data_s:0.012
INFO 2026-03-15 14:12:31 ot_train.py:569 Checkpoint policy after step 10000
INFO 2026-03-15 14:12:32 ot_train.py:640 End of training
```

Recent loss progression near the end was:

- `step:9K loss:0.019`
- `step:9K loss:0.018`
- `step:10K loss:0.019`
- `step:10K loss:0.018`
- `step:10K loss:0.019`

The checkpoint directory contents after completion were:

- `002000`
- `004000`
- `006000`
- `008000`
- `010000`
- `last -> 010000`

## 7. Why It Stopped And Why It Was Not Exactly 10 Epochs

The run stopped because it was configured with:

- `--steps=10000`

This training script stops on step count, not on exact epoch count.

For this dataset:

- `dataset.num_frames = 32385`
- `batch_size = 32`

So:

- `10000 * 32 = 320000` samples
- `320000 / 32385 = 9.8811` epochs

That is why the log ended around:

- `epch:9.88`

If you want at least a full `10.00` epochs on this dataset with `batch_size=32`, use:

- `steps = ceil(10 * 32385 / 32) = 10121`

In practice, either of these are reasonable:

- `10121`
- `10200`

## 8. How To Reuse This Setup

### 8.1 Launch the training command directly

```bash
bash tfj_envs/smolvla_train_monitor/scripts/launch_smolvla_trimmed_train.sh \
  /data/tfj/lerobot_tfj/outputs/train/my_smolvla_trimmed_run \
  /data/tfj/lerobot_tfj/outputs/logs/my_smolvla_trimmed_run.train.log \
  10000 \
  32 \
  4
```

### 8.2 Launch training plus monitoring in a normal shell

```bash
bash tfj_envs/smolvla_train_monitor/scripts/start_smolvla_trimmed_train_nohup.sh
```

This will:

- create a timestamped output dir
- create a timestamped train log
- create a timestamped monitor log
- launch training in the background
- launch the monitor loop in the background

### 8.3 Run a 10-epoch version

```bash
bash tfj_envs/smolvla_train_monitor/scripts/start_smolvla_trimmed_train_nohup.sh \
  10121 \
  32 \
  4 \
  60
```

Arguments are:

1. `steps`
2. `batch_size`
3. `num_workers`
4. `monitor_interval_seconds`

## 9. Recommended Next Step

If the goal is to rerun the same trimmed dataset training but make it complete at least 10 full epochs,
the cleanest next run is:

- `steps=10121`
- `batch_size=32`
- `num_workers=4`
- `dataset.video_backend=pyav`

If the goal is to move back toward RL, do not use this document as proof that the real SO101 RL stack is ready.
This document only records the successful offline SmolVLA training and monitoring workflow.
