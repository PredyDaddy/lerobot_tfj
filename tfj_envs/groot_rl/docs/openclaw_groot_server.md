# OpenClaw GROOT 本地服务说明

这个服务用于把 OpenClaw 的一句话任务，转成对 `scripts/run_groot_so101_infer.sh` 的 HTTP 调用。

## 1. 启动服务

```bash
bash scripts/start_openclaw_groot_server.sh
```

默认监听：

- `http://127.0.0.1:8765`

可通过环境变量改地址和端口：

```bash
OPENCLAW_GROOT_HOST=0.0.0.0 OPENCLAW_GROOT_PORT=8765 bash scripts/start_openclaw_groot_server.sh
```

## 2. 健康检查

```bash
curl http://127.0.0.1:8765/health
```

## 3. 发起一次抓取任务

```bash
curl -X POST http://127.0.0.1:8765/run \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "Pick up the block with the GROOT policy",
    "display_data": true,
    "num_episodes": 1,
    "episode_time_s": 30,
    "reset_time_s": 15
  }'
```

返回结果里会带：

- `job_id`
- `pid`
- `log_path`

## 4. 查询任务状态

```bash
curl http://127.0.0.1:8765/jobs
```

查询单个任务：

```bash
curl http://127.0.0.1:8765/jobs/<job_id>
```

## 5. 停止任务

```bash
curl -X POST http://127.0.0.1:8765/jobs/<job_id>/stop
```

## 6. 可覆盖参数

`POST /run` 支持这些字段：

- `task`
- `leader_port`
- `policy_path`
- `policy_device`
- `robot_port`
- `top_camera_index`
- `wrist_camera_index`
- `dataset_repo_id`
- `dataset_root`
- `num_episodes`
- `episode_time_s`
- `reset_time_s`
- `display_data`

## 7. 你当前的默认配置

服务会默认调用下面这套配置：

- 模型：`/data/tfj/lerobot_tfj/tmp/train/groot_grasp1/checkpoints/last/pretrained_model`
- follower 串口：`/dev/ttyACM0`
- top 相机：`4`
- wrist 相机：`6`
- follower 标定目录：`/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`
- leader 标定目录：`/home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader`

## 8. OpenClaw 对接建议

如果 OpenClaw 支持 HTTP 工具，可以让它调用：

- `POST /run`：启动抓取
- `GET /jobs/<job_id>`：读取执行状态和日志
- `POST /jobs/<job_id>/stop`：中断任务

推荐让 OpenClaw 发送英文任务，例如：

```json
{
  "task": "Pick up the block with the GROOT policy"
}
```
