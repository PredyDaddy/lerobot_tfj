# OpenClaw 调用 GROOT 说明

这个目录提供一个可安装到 OpenClaw 的本地插件，用来调用当前仓库里的 `GROOT` 实机 HTTP 服务。

## 目录结构

- `openclaw-groot-tool/`: OpenClaw 本地插件
- `openclaw.json.example`: OpenClaw 配置示例

## 先决条件

1. 你的 GROOT 本地服务已经能启动：

```bash
bash scripts/start_openclaw_groot_server.sh
```

2. 本地健康检查正常：

```bash
curl http://127.0.0.1:8765/health
```

## 安装插件

建议用本地 link 安装，便于后续继续改：

```bash
openclaw plugins install -l /data/tfj/lerobot_tfj/openclaw调用/openclaw-groot-tool
```

安装后重启 OpenClaw Gateway。

## 配置 OpenClaw

把 `openclaw.json.example` 里的内容合并到你的 `~/.openclaw/openclaw.json`。

关键配置有两部分：

1. 启用插件并配置 HTTP 地址
2. 给目标 agent 追加放开这三个工具：
   - `groot_run`
   - `groot_job_status`
   - `groot_job_stop`

## OpenClaw 可调用的工具

### 1. `groot_run`

启动一次 GROOT 实机任务，最少只需要：

```json
{
  "task": "Pick up the block with the GROOT policy"
}
```

也可以覆盖默认串口、相机和模型路径。

### 2. `groot_job_status`

查询任务状态：

```json
{
  "job_id": "your_job_id"
}
```

### 3. `groot_job_stop`

停止任务：

```json
{
  "job_id": "your_job_id"
}
```

## 推荐给 OpenClaw 的提示词

你可以直接这样告诉 OpenClaw：

```text
使用 groot_run 工具，让机器人用 GROOT 策略抓取桌上的方块；如果任务已启动，继续用 groot_job_status 跟踪状态。
```

或者更简洁一点：

```text
Use the GROOT policy to pick up the block.
```

## 当前默认参数

插件实际调用的是你前面已经写好的：

- `scripts/run_groot_so101_infer.sh`

它默认绑定以下参数：

- 模型：`/data/tfj/lerobot_tfj/tmp/train/groot_grasp1/checkpoints/last/pretrained_model`
- follower 串口：`/dev/ttyACM0`
- top 相机：`4`
- wrist 相机：`6`
- follower 标定目录：`/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`
- leader 标定目录：`/home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader`

## 联调建议

先按下面顺序测：

1. `bash scripts/start_openclaw_groot_server.sh`
2. `curl http://127.0.0.1:8765/health`
3. `curl -X POST http://127.0.0.1:8765/run -H 'Content-Type: application/json' -d '{"task":"Pick up the block with the GROOT policy"}'`
4. 再让 OpenClaw 调 `groot_run`，并在需要时轮询 `groot_job_status`
