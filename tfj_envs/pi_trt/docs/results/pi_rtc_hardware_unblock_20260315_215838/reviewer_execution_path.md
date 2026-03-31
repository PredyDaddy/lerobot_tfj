# Reviewer Execution Path

## 结论

- 当前 `scripts/run_pi05_torch_infer_so101.py` 的真实执行路线已经可以清楚分成三段：
  1. 环境与资产校验
  2. preflight
  3. 真机连接、warmup、动作发送
- 这条路线在当前机器上并非“完全阻塞”：
  - 默认 checkpoint 资产齐全
  - 离线 tokenizer 可解析
  - `cuda` 可用，当前 GPU 为 `NVIDIA GeForce RTX 4090`
  - 默认相机索引 `4` 和 `6` 的 preflight 实测通过
  - 但默认串口 `--robot-port /dev/ttyACM0` 仍然存在设备歧义，因为当前机器上同时有 `/dev/ttyACM0` 和 `/dev/ttyACM1`
- 真机前的最稳妥路线不是“直接 RTC 20 秒实跑”，而是：
  1. `--dry-run`
  2. `--preflight-only`
  3. 非 RTC 的短时实机 smoke
  4. 显式 `--rtc-enable` 的短时 RTC smoke

## 代码里的真实执行顺序

依据 `scripts/run_pi05_torch_infer_so101.py` 当前实现，顺序如下。

### 1. Validate environment

入口在 `main()` 的 `scripts/run_pi05_torch_infer_so101.py:633-668`。

这一段实际做了这些事：

- `validate_paths(...)`
  - 解析 checkpoint 目录
  - 校验 calibration 目录存在
  - 尝试发现 tokenizer 目录
- `load_policy_config(...)`
  - 读取 `config.json`
  - 写入 `policy_cfg.device`
  - 限制 policy type 必须是 `pi05`
- `apply_pi_runtime_overrides(...)`
  - 处理 `policy_*` 参数
  - 处理 `rtc_*` 参数
  - 写回 `policy_cfg.rtc_config`
- 运行参数合法性检查
  - `joint_action_alpha` / `gripper_action_alpha`
  - `joint_delta_limit` / `gripper_delta_limit`
  - `robot_max_relative_target`
- `load_policy_preprocessor_from_checkpoint(...)`
  - 这里会强制要求离线 tokenizer 可用
  - 也会做 runtime compatibility shim
- `load_postprocessor(...)`
- `print_summary(...)`

注意：

- `--dry-run` 不是“只做 parser 检查”。它会完整走完上面这段，包括 preprocessor / postprocessor 加载，然后才在 `scripts/run_pi05_torch_infer_so101.py:670-672` 退出。
- 也就是说，`--dry-run` 足以暴露 checkpoint、tokenizer、processor 配置层的问题，但不会触碰相机、串口、机器人连接。

### 2. Preflight checks

入口在 `scripts/run_pi05_torch_infer_so101.py:674-684`。

这一段顺序是：

- 如果没有 `--skip-camera-preflight`
  - 运行 `preflight_cameras(...)`
  - 逐个 `cv2.VideoCapture(index)` 打开并读一帧
- 如果没有 `--skip-policy-preflight`
  - 运行 `preflight_policy(...)`
  - 加载 `PI05Policy.from_pretrained(...)`
  - 仅验证模型可加载
  - 不做 `predict_action_chunk(...)`
- 如果传了 `--preflight-only`
  - 到此直接退出
  - 还没有 robot build
  - 还没有 `robot.connect()`
  - 还没有任何动作发送

注意：

- 当前 `--preflight-only` 只证明“相机能开、模型能载入”，不证明“串口对、robot.connect 能成功、warmup chunk 能成功、动作循环能成功”。
- 当前 `preflight_cameras(...)` 只验证“能打开并读一帧”，没有验证后续 runtime 真正使用的 `width/height/fps` 设置是否被设备接受。

### 3. Build robot, connect, warmup, send actions

入口在 `scripts/run_pi05_torch_infer_so101.py:686-998`。

这一段顺序是：

- `build_robot_config(...)`
- `make_robot_from_config(...)`
- `make_default_processors()`
- 如果之前 `--skip-policy-preflight`
  - 在这里才真正加载 policy
- `get_safe_torch_device(...)`
- 构建 `TorchChunkPolicyRuntime`
- `robot.connect()`
- `policy.reset()` / preprocessor reset / postprocessor reset
- `build_dataset_features(robot)`
- 构建 `AsyncChunkPrefetcher`
- 构建 `ActionQueue`
- `Warm up initial chunk`
  - `robot.get_observation()`
  - 处理观测
  - `prefetcher.predict_sync(...)`
  - merge initial chunk
- 进入主循环
  - 读观测
  - collect async chunk
  - 估计 prefetch threshold / predicted delay
  - submit async chunk
  - 需要时 async wait
  - 队列见底时 hold 或 sync_refill
  - `action_queue.get()`
  - postprocess 到 robot action
  - smoothing / delta clamp / finite check
  - `robot.send_action(...)`

关键点：

- 真正的“第一条动作”不是在 `robot.connect()` 后立刻发，而是在 warmup initial chunk 完成、进入主循环并从 queue 取出第一条 action 后才发送。
- 因此，`--preflight-only` 和真实 smoke 的差距不小；真正的 smoke 至少要覆盖：
  - `robot.connect()`
  - `Warm up initial chunk`
  - 至少 1 次 `robot.send_action(...)`

## 当前机器上的实际核对结果

我实际做了以下核对。

### 资产与环境

- `checkpoint_dir=/data/tfj/lerobot_tfj/pi_model/pretrained_model`
  - `config.json` 存在
  - `model.safetensors` 存在
  - `policy_preprocessor.json` 存在
  - `policy_postprocessor.json` 存在
- calibration 目录存在：
  - `/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower`
- tokenizer 自动发现成功：
  - `/home/cqy/.cache/modelscope/hub/models/google/paligemma-3b-pt-224`
- `ensure_pi_runtime_compatibility(require_local_tokenizer=True)` 返回 `ready=True`
- 当前 conda 环境检测为 `lerobot`
- 当前 GPU 可用：
  - `torch.cuda.is_available() == True`
  - `NVIDIA GeForce RTX 4090`

### 设备可见性

- 当前存在 `/dev/ttyACM0`
- 当前存在 `/dev/ttyACM1`
- 当前存在 `/dev/video0` 到 `/dev/video7`

### 实际命令验证

已执行：

```bash
python scripts/run_pi05_torch_infer_so101.py --dry-run
python scripts/run_pi05_torch_infer_so101.py --preflight-only
python scripts/run_pi05_torch_infer_so101.py --preflight-only --skip-camera-preflight --skip-policy-preflight
```

结果：

- `--dry-run` 成功返回，说明环境、checkpoint、tokenizer、processor 加载链路当前可过。
- `--preflight-only` 成功返回，且默认 camera index `4` / `6` 当前能打开并读帧，policy 也能加载。
- `--preflight-only --skip-camera-preflight --skip-policy-preflight` 也会成功返回，并打印 `Preflight completed. Exiting before robot connect.`。这说明当前存在明显的“参数组合假阳性”。

## 建议的实际执行路线

### 阶段 0：只确认资产和参数绑定

建议命令：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --dry-run \
  --policy-device cuda
```

目的：

- 确认 checkpoint 路径、tokenizer、preprocessor、postprocessor、policy config 解析无误
- 确认当前环境确实能走到 summary 输出

判断标准：

- 看到 `Policy path`
- 看到 `Resolved RTC config`
- 看到 `Tokenizer path`
- 最后看到 `Dry run only. Exiting before any preflight or hardware access.`

### 阶段 1：真实 preflight，但不触碰机器人连接

建议命令：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --preflight-only \
  --policy-device cuda
```

目的：

- 确认默认 camera index 真能打开
- 确认 policy 真能从 safetensors 加载

判断标准：

- 看到 `Camera 4 OK`
- 看到 `Camera 6 OK`
- 看到 `PI05 PyTorch policy OK`
- 最后看到 `Preflight completed. Exiting before robot connect.`

注意：

- 这一步仍然没有验证 `--robot-port`
- 这一步也没有验证 warmup chunk、主循环、动作发送

### 阶段 2：先做非 RTC 的短时真机 smoke

建议先不要在第一条真机 smoke 上叠加 RTC 和 AMP，先把变量数压到最少。

建议命令：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --policy-device cuda \
  --run-time-s 8 \
  --log-interval 1 \
  --joint-delta-limit 0.10 \
  --robot-max-relative-target 0.10
```

目的：

- 先验证最基础的 robot connect、warmup、queue、send_action 路径
- 尽快暴露串口、robot config、action postprocess、runtime loop 层的问题

为什么建议先不加 `--policy-use-amp`：

- `policy_use_amp` 只是 launcher 本地 wrapper 上的一层 autocast，不是当前执行路线里必须先验证的东西
- 第一次短时 smoke 应优先减少变量

### 阶段 3：显式开启 RTC 的短时真机 smoke

在非 RTC 短时 smoke 干净之后，再切 RTC。

建议命令：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --policy-device cuda \
  --rtc-enable \
  --rtc-execution-horizon 8 \
  --run-time-s 8 \
  --log-interval 1 \
  --joint-delta-limit 0.10 \
  --robot-max-relative-target 0.10
```

建议重点盯住：

- `Resolved RTC config`
- `Initial chunk ready`
- 周期日志中的：
  - `queue_size`
  - `prefetch_pending`
  - `sync_refill_count`
  - `refill_mode`
  - `rtc_enabled`
  - `real_delay`

如果再往后验证 `AMP + RTC`，再单独增加：

```bash
--policy-use-amp
```

但不建议把它并入第一条 RTC smoke。

## 参数层面的坑和潜在阻塞

### 1. `--preflight-only` 不能证明 robot 侧没问题

- `--preflight-only` 在 `robot.connect()` 之前就退出
- 所以它不能证明：
  - `--robot-port` 是否正确
  - 机器人是否真能连上
  - warmup chunk 是否能跑
  - `robot.send_action(...)` 是否能执行

### 2. `--skip-camera-preflight` 和 `--skip-policy-preflight` 会制造假阳性

这个是我已经实际验证过的。

- 命令：

```bash
python scripts/run_pi05_torch_infer_so101.py \
  --preflight-only \
  --skip-camera-preflight \
  --skip-policy-preflight
```

- 结果：
  - 返回成功
  - 打印 `Preflight completed. Exiting before robot connect.`

结论：

- 这组参数不应作为任何“已完成 preflight”的证据
- 如果现场要做准入检查，不能允许把它当作成功背书

### 3. 多个参数把字符串 `"0"` 吃成 `None`

这是当前 parser 的真实行为，因为它使用了 `parse_optional_int()` / `parse_optional_float()`，会把 `"0"`、`"none"`、`"null"` 都解释成 `None`。

受影响的参数包括：

- `--robot-max-relative-target`
- `--policy-n-action-steps`
- `--policy-num-inference-steps`
- `--policy-temporal-ensemble-coeff`
- `--joint-delta-limit`
- `--gripper-delta-limit`
- `--joint-action-alpha`
- `--gripper-action-alpha`

实际影响：

- `--joint-delta-limit 0` 不会报错，而是等价于“不设限”
- `--joint-action-alpha 0` 不会报错，而是等价于“不做 smoothing”
- `--policy-num-inference-steps 0` 不会报错，而是 silently 回退为默认值
- `--policy-temporal-ensemble-coeff 0` 甚至会绕过当前的“不支持 temporal ensembling”报错，因为它在 parser 阶段就被吃成了 `None`

结论：

- 现场命令不要用 `0` 表示“禁用”或“最小值”
- 要么不传，要么传明确的正值

### 4. 任意 `--rtc-*` override 都会隐式打开 RTC

当前 `resolve_rtc_runtime_config(...)` 的逻辑是：

- 只要传了任一 `--rtc-*` override
- 即使没写 `--rtc-enable`
- 也会把 `rtc_config.enabled` 设为 `True`

实际影响：

- `--rtc-debug-maxlen 32`
- `--rtc-execution-horizon 8`
- `--rtc-max-guidance-weight 5`

这些都属于“隐式 RTC on”。

建议：

- 现场 smoke 不要依赖这种隐式行为
- 要么明确 `RTC off`
- 要么明确写出 `--rtc-enable`

### 5. `--run-time-s <= 0` 不是“短跑”，而是无限跑

当前 summary 已经写明：

- `run_time_s: 0.0 (<=0 means until Ctrl+C)`

所以：

- `--run-time-s 0`
- 不会做 0 秒 smoke
- 会变成一直跑，直到人工中断

结论：

- 短时 smoke 必须显式给正数，比如 `5`、`8`、`10`

### 6. 默认 `--log-interval 30` 不适合短时 smoke

如果只跑 5 到 10 秒，默认 `30` 很可能看不到足够多的周期日志。

建议：

- 短时 smoke 固定带上 `--log-interval 1`

### 7. 当前机器上的默认 robot port 有歧义

当前机器上同时存在：

- `/dev/ttyACM0`
- `/dev/ttyACM1`

但 launcher 默认只写：

```bash
--robot-port /dev/ttyACM0
```

结论：

- 真机 smoke 前，必须先人工确认 follower 真实挂载在哪个 ACM 口
- 这一点 `--preflight-only` 不会帮你发现

### 8. `--skip-policy-preflight` 会把 policy 失败推迟到更晚阶段

如果传了 `--skip-policy-preflight` 但没有 `--preflight-only`：

- 脚本会继续走到 `Build robot and processors`
- 然后才在 `scripts/run_pi05_torch_infer_so101.py:691-693` 加载 policy

这意味着：

- 这不是“更快更安全”
- 只是把失败从 preflight 延后到硬件阶段

### 9. 相机 preflight 通过，不等于 runtime camera config 全通过

`preflight_cameras(...)` 只做：

- open device
- read one frame

它没有验证：

- `640x480`
- `30 fps`
- runtime camera config 是否被实际设备接受

结论：

- 当前 camera preflight 通过是好信号
- 但不能把它当作 runtime camera config 的完全背书

## 当前我对“短时实机 smoke”是否建议推进的判断

- 可以推进，但不建议一步到位上 `RTC + AMP + 长时运行`
- 推荐顺序必须是：
  1. `--dry-run`
  2. `--preflight-only`
  3. 非 RTC 的短时 smoke
  4. RTC 的短时 smoke
- 当前最现实的阻塞不是 checkpoint、GPU、tokenizer，而是：
  - robot port 选错
  - 现场误用 `--skip-*`
  - 用 `0` 值参数却以为真的生效
  - 直接把 `--run-time-s 0` 当短时 smoke

## 最终建议

- 先按上面的四阶段顺序推进，不要跳步骤。
- 第一条真机 smoke 不建议带 `--policy-use-amp`。
- RTC smoke 必须显式写 `--rtc-enable`，不要只靠某个 `--rtc-*` override 让它隐式打开。
- 任何带 `--skip-camera-preflight` 或 `--skip-policy-preflight` 的 `--preflight-only` 成功结果，都不应被当作准入结论。
