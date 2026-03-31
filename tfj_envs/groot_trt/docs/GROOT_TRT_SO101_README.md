# GROOT TRT SO101 上机 README

这份文档专门回答 4 个问题：

1. GROOT 的 TensorRT engine 导出到了哪里
2. 真机上机脚本在哪里
3. 端口号、相机号、标定目录要怎么传
4. 你应该抄哪条命令直接运行

如果你现在只关心 GROOT 真机 SO101 上机，不想翻长文档，先看这篇。

## 1. 先记住 GROOT TRT 不是单个 engine

GROOT 和 ACT 不一样。

ACT 通常是一个主 engine 文件。

GROOT 当前这套 TensorRT 部署不是单个文件，而是一个目录，里面放 7 个子 engine。

固定 7 个 engine 名字是：

- `vit_fp16.engine`
- `llm_fp16.engine`
- `vlln_vl_self_attention.engine`
- `state_encoder.engine`
- `action_encoder.engine`
- `DiT_fp16.engine`
- `action_decoder.engine`

所以你以后找 GROOT TRT，不要找一个总的 `model.engine`，而是找：

- `gr00t_engine_api_trt1013/`

这个目录。

## 2. 你的 GROOT TRT 导出到了哪里

当前仓库里已经存在这些 GROOT TRT 导出目录：

- `/data/tfj/lerobot_tfj/outputs/trt/groot_grasp_trt_repo_20260311_121701/gr00t_engine_api_trt1013`
- `/data/tfj/lerobot_tfj/outputs/trt/groot_export_only_20260311_140937/gr00t_engine_api_trt1013`
- `/data/tfj/lerobot_tfj/outputs/trt/groot_engine_export_20260311_144026/gr00t_engine_api_trt1013`
- `/data/tfj/lerobot_tfj/outputs/trt/groot_export_verify_20260311_152547/gr00t_engine_api_trt1013`
- `/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013`

当前真机脚本默认指向的是这一套：

- `RUN_DIR`
  `/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210`
- `ENGINE_DIR`
  `/data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013`

这两个默认值写在：

- [run_groot_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py)

你可以把它理解成：

- `RUN_DIR` 是这次 GROOT TRT 导出的总目录
- `ENGINE_DIR` 是真正放 7 个 `.engine` 文件的目录

## 3. 上机脚本在哪里

### 3.1 真正的 Python 上机入口

路径：

- [run_groot_trt_infer_so101.py](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py)

这个脚本是真正干活的真机脚本。

### 3.2 推荐你用的 shell 包装入口

路径：

- [one_click_run_groot_trt_so101.sh](/data/tfj/lerobot_tfj/tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh)

推荐原因：

- 少敲很多路径
- 自动带 conda 环境
- 自动处理 `TMPDIR`
- 你只需要传 `POLICY_PATH`、`RUN_DIR`、`ENGINE_DIR`

这个 shell 脚本的前 3 个位置参数是：

1. `POLICY_PATH`
2. `RUN_DIR`
3. `ENGINE_DIR`

后面再跟 `--robot-port`、`--robot-calibration-dir`、`--top-cam-index` 这些参数。

## 4. 端口号、相机号、标定目录默认值

当前 Python 真机脚本里的默认值是：

- `--robot-id my_so101`
- `--robot-port /dev/ttyACM0`
- `--robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`
- `--top-cam-index 4`
- `--wrist-cam-index 6`
- `--camera-width 640`
- `--camera-height 480`
- `--camera-fps 30`

也就是说，你问的端口号不是没有，而是脚本里已经写了默认值。

## 5. 你的标定文件在哪里

你机器上当前可用的 SO101 follower 标定目录是：

- `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`

这个目录里我确认存在这些文件：

- `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_so101.json`
- `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower.json`
- `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_so101_follower.json`

这里最容易搞错的一点是：

- `--robot-calibration-dir` 传的是目录，不是某个 json 文件

脚本会根据：

- `--robot-id`

去这个目录里找对应的标定文件。

举例：

- 如果你传 `--robot-id my_so101`
- 同时传 `--robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`

那它会去匹配：

- `my_so101.json`

所以你现在这套最自然的组合就是：

- `--robot-id my_so101`
- `--robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`

## 6. 最推荐的上机命令

### 6.1 先做 preflight，不真正跑控制

这是第一次最推荐的命令。

```bash
cd /data/tfj/lerobot_tfj

bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin" \
  --preflight-only
```

这条命令的作用是：

- 检查 checkpoint 能不能加载
- 检查 7 个 TensorRT engine 能不能加载
- 检查标定目录是否存在
- 检查机器人配置是否能构建
- 检查相机是否能打开

如果这一步都过不了，不要直接进入正式上机。

### 6.2 真机短时运行

```bash
cd /data/tfj/lerobot_tfj

bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin" \
  --run-time-s 10
```

这条命令会实际跑 10 秒。

第一次建议永远先加：

- `--run-time-s 10`

不要第一次就长时间跑。

## 7. 如果你想直接运行 Python 脚本

也可以不用 shell 包装，直接跑 Python：

```bash
cd /data/tfj/lerobot_tfj

conda run --no-capture-output -n lerobot_flex python tfj_envs/groot_trt/scripts/run_groot_trt_infer_so101.py \
  --policy-path /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  --run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  --engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin" \
  --preflight-only
```

## 8. 这些参数分别是什么意思

- `--policy-path`
  GROOT checkpoint 路径。可以传到 `010000`，脚本会自己解析到真实的 `pretrained_model`。

- `--run-dir`
  GROOT TRT 这次导出的总目录。

- `--engine-dir`
  真正放 7 个 `.engine` 文件的目录。

- `--robot-id`
  机器人 ID。它会和标定目录里的 json 文件名对应。

- `--robot-port`
  机械臂串口。你现在默认是 `/dev/ttyACM0`。

- `--robot-calibration-dir`
  标定目录，不是某个具体 json。

- `--top-cam-index`
  顶部相机 index。

- `--wrist-cam-index`
  腕部相机 index。

- `--preflight-only`
  只做预检查，不真正进入控制循环。

- `--run-time-s 10`
  实际运行 10 秒后退出。

## 9. 如果你换了一套新的导出结果

你只需要改这两项：

- `RUN_DIR`
- `ENGINE_DIR`

比如你想改成另外一套导出：

- `/data/tfj/lerobot_tfj/outputs/trt/groot_export_verify_20260311_152547`

那就把命令里的：

- `--run-dir /data/tfj/lerobot_tfj/outputs/trt/groot_export_verify_20260311_152547`
- `--engine-dir /data/tfj/lerobot_tfj/outputs/trt/groot_export_verify_20260311_152547/gr00t_engine_api_trt1013`

一起改掉。

## 10. 如何自己检查这些路径存不存在

### 10.1 看 engine 目录

```bash
find /data/tfj/lerobot_tfj/outputs/trt -maxdepth 2 -type d | grep groot
```

### 10.2 看某个 engine 目录下的 7 个文件

```bash
find /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 -maxdepth 1 -type f
```

### 10.3 看 SO101 标定目录

```bash
find /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower -maxdepth 1 -type f
```

### 10.4 看串口

```bash
ls /dev/ttyACM*
```

## 11. 最常见的坑

### 11.1 把 `ENGINE_DIR` 当成单个文件

错。

GROOT 这里传的是目录，不是一个总的 `.engine` 文件。

### 11.2 把 `--robot-calibration-dir` 传成 json 文件

错。

这里应该传目录，例如：

- `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`

### 11.3 `robot-id` 和标定文件名对不上

比如你传：

- `--robot-id my_so101`

那标定目录里最好真的有：

- `my_so101.json`

你这台机器目前是有的。

### 11.4 没先做 preflight 就直接上机

不推荐。

第一次一定先跑：

- `--preflight-only`

## 12. 你只要记住这一条

如果你今天只想先确认 GROOT TRT 上机入口没问题，先跑这条：

```bash
cd /data/tfj/lerobot_tfj

bash tfj_envs/groot_trt/scripts/one_click_run_groot_trt_so101.sh \
  /data/tfj/lerobot_tfj/tmp/train/groot_grasp/checkpoints/010000 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210 \
  /data/tfj/lerobot_tfj/outputs/trt/groot_self_run_20260311_161210/gr00t_engine_api_trt1013 \
  --robot-id my_so101 \
  --robot-port /dev/ttyACM0 \
  --robot-calibration-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --task "grasp block in bin" \
  --preflight-only
```

只要这条能过，说明：

- 路径基本对
- 7 个 engine 基本对
- 机器人配置基本对
- 标定目录基本对
- 真机脚本入口基本对
