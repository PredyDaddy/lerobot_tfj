# SmolVLA 脚本束审查报告

## 1. 审查范围

本次只检查以下脚本与 `src/lerobot/scripts/lerobot_record_so101_policy.py` 的关系，以及它们和现有文档的一致性、显著易踩坑、路径和环境变量假设是否清楚：

- `tfj_envs/smolvla_rl/scripts/README.md`
- `tfj_envs/smolvla_rl/scripts/launch_smolvla_offline_trimmed_train.sh`
- `tfj_envs/smolvla_rl/scripts/start_smolvla_offline_trimmed_train_nohup.sh`
- `tfj_envs/smolvla_rl/scripts/launch_smolvla_hybrid_train.sh`
- `tfj_envs/smolvla_rl/scripts/start_smolvla_hybrid_train_nohup.sh`
- `tfj_envs/smolvla_rl/scripts/monitor_training_process.sh`
- `tfj_envs/smolvla_rl/scripts/run_so101_policy_record.sh`
- `tfj_envs/smolvla_rl/scripts/show_trimmed_dataset_meta.sh`
- `src/lerobot/scripts/lerobot_record_so101_policy.py`

## 2. 总体结论

这组脚本整体是可用的，但可用性高度依赖“当前这台机器”的缓存、输出目录、标定目录、GPU 和设备编号现实。它们更像是“当前工作站上的稳定操作封装”，不是可直接搬到另一台机器的一般化脚本包。

和文档的关系上，整体叙述基本一致：

- `run_so101_policy_record.sh` 的确是 `lerobot_record_so101_policy.py` 的直接 shell wrapper。
- 训练脚本和 `lerobot_record_so101_policy.py` 没有直接调用关系，但在 bundle 叙事上是同一条链路的不同阶段：
  - 离线训练
  - hybrid 微调
  - SO101 上机执行/录制
- 文档对离线模式、rename map、SO101 默认端口/相机索引的描述基本和脚本一致。

但需要强调的是：文档现在更像“操作说明”，而不是“环境前提清单”。很多脚本的真实前提写在默认值里，没有被明确抬到文档前置条件。

## 3. 脚本覆盖范围

| 脚本 | 角色 | 与 `lerobot_record_so101_policy.py` 的关系 | 与文档一致性 |
| --- | --- | --- | --- |
| `scripts/README.md` | 脚本目录索引 | 间接，描述 bundle 中各脚本定位 | 基本一致 |
| `launch_smolvla_offline_trimmed_train.sh` | 前台离线训练入口 | 无直接关系，属于上游训练阶段 | 基本一致 |
| `start_smolvla_offline_trimmed_train_nohup.sh` | 后台离线训练包装器 | 无直接关系 | 基本一致 |
| `launch_smolvla_hybrid_train.sh` | 前台 hybrid 训练入口 | 无直接关系，属于上游微调阶段 | 基本一致 |
| `start_smolvla_hybrid_train_nohup.sh` | 后台 hybrid 训练包装器 | 无直接关系 | README 覆盖到，训练操作文档覆盖不足 |
| `monitor_training_process.sh` | 通用训练监控器 | 无直接关系 | 基本一致 |
| `run_so101_policy_record.sh` | SO101 policy 录制包装器 | 直接调用 | 基本一致，但默认值会遮蔽 Python 入口的一部分自动行为 |
| `show_trimmed_dataset_meta.sh` | 数据集元信息查看工具 | 无直接关系 | 基本一致 |
| `src/lerobot/scripts/lerobot_record_so101_policy.py` | SO101 专用 Python 录制入口 | 被 `run_so101_policy_record.sh` 直接调用 | 文档描述与实现基本一致 |

## 4. 关系梳理

### 4.1 直接关系

`run_so101_policy_record.sh` 是唯一一个直接调用 `src/lerobot/scripts/lerobot_record_so101_policy.py` 的脚本：

- shell wrapper 组装命令并传入 policy 路径、机器人串口、标定目录、相机参数、任务文本和数据集路径。
- Python 入口负责：
  - 读取 `--policy.path`
  - 自动在 SmolVLA 情况下补 `rename_map`
  - 构造 `SO101FollowerConfig`
  - 构造 `DatasetRecordConfig`
  - 最终转调通用 `lerobot_record.py`

所以这两层关系是清晰的：

- shell 层负责“本机默认值和运行姿势”
- Python 层负责“SO101 专用配置裁剪”

### 4.2 间接关系

其他训练脚本和 `lerobot_record_so101_policy.py` 没有直接调用关系，但共享几个隐含假设：

- 默认都围绕 SmolVLA 工作流；
- 默认都使用同一套图像 rename map：
  - `observation.images.top -> observation.images.camera1`
  - `observation.images.wrist -> observation.images.camera2`
- 默认都倾向本地离线 Hugging Face/Transformers 资源；
- 默认都把当前工作目录定位到仓库根目录，并注入 `PYTHONPATH=${REPO_ROOT}/src`

也就是说，这些脚本不是孤立的，而是在 bundle 层面共享“当前机器、本地缓存、SmolVLA 路径布局”的操作约定。

## 5. 可用性判断

### 5.1 当前机器上的可用性

按当前工作区实际检查，这套脚本在本机上是“可运行概率高”的：

- 默认 Hugging Face cache 根目录存在；
- 默认 trimmed dataset 根目录存在；
- 默认 SO101 follower / leader 标定目录存在；
- `run_so101_policy_record.sh` 默认的 `POLICY_PATH` 存在；
- `rg`、`nvidia-smi`、`transformers`、`accelerate`、`safetensors`、`num2words`、`pyarrow` 都存在；
- `scripts/` 下 shell 文件都有可执行位。

结论：

- 在当前这台机器上，这组脚本可以视为“有较强现实可用性”。
- 但这个结论不应外推到别的机器。

### 5.2 可移植性判断

可移植性偏弱，原因不是脚本写错，而是默认值太具体：

- 大量默认值写死到 `/home/cqy/...`
- 默认 `DEVICE=cuda`、`POLICY_DEVICE=cuda`
- 默认 HF 离线
- 默认 checkpoint 名字带具体时间戳
- 默认相机索引固定为 `4` 和 `6`
- 默认串口固定为 `/dev/ttyACM0`

因此这组脚本更适合作为“当前环境的操作封装”，不适合作为“拿来即用的通用发版脚本”。

## 6. 风险点

### 6.1 强依赖本机绝对路径和本地缓存，文档没有把这件事说透

以下脚本都把关键前提埋进了默认值：

- `launch_smolvla_offline_trimmed_train.sh`
- `launch_smolvla_hybrid_train.sh`
- `run_so101_policy_record.sh`
- `show_trimmed_dataset_meta.sh`
- `src/lerobot/scripts/lerobot_record_so101_policy.py`

典型表现：

- HF cache 默认是 `/home/cqy/.cache/huggingface/hub`
- dataset 默认是 `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1_trimmed_static_tail`
- 标定目录默认是 `/home/cqy/.cache/huggingface/lerobot/calibration/...`
- record wrapper 默认 checkpoint 直接指向具体时间戳目录

文档虽然说明了“离线模式”和“本地快照”，但没有把“第一次迁移到别的机器时必须先具备这些本地资产”明确写成前置条件清单。这会导致用户误以为脚本本身具备自动准备能力。

### 6.2 `run_so101_policy_record.sh` 缺少 train 脚本级别的前置检查

训练脚本至少会提前检查：

- 本地 base checkpoint 是否存在
- 本地 SmolVLM2 snapshot 是否存在
- 一部分 Python 依赖是否存在

但 `run_so101_policy_record.sh` 没有同等级别的 preflight 检查。它默认直接调用 Python 入口，以下问题会在更深层才暴露：

- `POLICY_PATH` 不存在
- 串口不存在
- 标定目录不存在
- 相机 index 不存在或被占用
- 当前机器没有 CUDA，但默认 `POLICY_DEVICE=cuda`

这不是功能性错误，但会把问题发现时机推迟到运行期，增加定位成本。

### 6.3 `run_so101_policy_record.sh` 的默认命名会遮蔽 Python 入口的自动命名逻辑

`src/lerobot/scripts/lerobot_record_so101_policy.py` 里有一层自动命名逻辑：

- 当 `dataset_repo_id == local/eval_so101_policy` 时，会自动改成 `local/eval_{policy.type}_so101`
- 当 `dataset_root == ./outputs/eval_so101_policy` 时，会自动改成 `./outputs/eval_{policy.type}_so101`

但 `run_so101_policy_record.sh` 直接显式传入：

- `DATASET_REPO_ID=local/eval_smolvla_so101`
- `DATASET_ROOT=./outputs/eval_smolvla_so101`

这意味着：

- 对当前默认 SmolVLA 模型来说，这没有问题；
- 但如果用户只改 `POLICY_PATH` 去录别的 policy，而没同步改 `DATASET_REPO_ID` / `DATASET_ROOT`，最终仍会落到 `smolvla` 命名下。

这会造成数据目录和 repo id 命名误导，不利于后期整理。

### 6.4 rename map 在多处重复，后续漂移风险高

同一套 SmolVLA camera rename map 至少出现在三类位置：

- `launch_smolvla_offline_trimmed_train.sh`
- `launch_smolvla_hybrid_train.sh`
- `src/lerobot/scripts/lerobot_record_so101_policy.py`

当前三处是一致的，这是好的；但后续如果 camera key 命名、数据集结构或 policy 输入约定变化，容易出现“训练脚本改了、录制脚本没改”的漂移。

### 6.5 hybrid 后台包装器的文档覆盖不完整

`start_smolvla_hybrid_train_nohup.sh` 已经存在，也在 bundle README 和 `scripts/README.md` 里列出，但训练操作文档的 hybrid 章节主要讲的是：

- `launch_smolvla_hybrid_train.sh`

没有像离线训练那样，单独把 hybrid 的后台启动和监控方式作为一节完整说明。这会导致用户知道脚本存在，但不知道推荐怎么用。

### 6.6 训练脚本“前台启动”实际上默认不输出训练流到终端

`launch_smolvla_offline_trimmed_train.sh` 和 `launch_smolvla_hybrid_train.sh` 最后都是：

- `exec "${CMD[@]}" >"${LOG_FILE}" 2>&1`

这意味着它们虽是“前台运行”，但训练过程日志默认被重定向到文件，不会持续打印在当前终端。对熟悉这套脚本的人不是问题，但第一次使用时容易误判成“脚本卡住了”。

离线训练文档已经提到日志写文件，hybrid 部分则没有把这一点说得同样明确。

### 6.7 hybrid 训练脚本的依赖前置检查弱于离线训练脚本

`launch_smolvla_offline_trimmed_train.sh` 有一段显式 Python 依赖探测。

`launch_smolvla_hybrid_train.sh` 没有对应检查，只检查了：

- `POLICY_PATH`
- `SMOLVLM2_PATH`

如果 hybrid 依赖链不完整，失败会落在 Python 入口后才显现，错误信息通常也更绕。

### 6.8 监控脚本默认假设有 NVIDIA GPU 和 `rg`

`monitor_training_process.sh` 使用了：

- `nvidia-smi`
- `rg`

在当前机器上它们都存在，因此可用；但脚本本身并未把这两个前提写清楚。尤其如果未来有人拿去在 CPU-only 或更精简环境里复用，监控脚本的行为就会变得不可预期。

## 7. 建议

### 7.1 保留当前脚本，但明确标注“机器绑定型默认值”

建议在 `scripts/README.md` 和训练操作文档最前面补一个“前置条件”小节，明确写出：

- 默认值按当前工作站配置编写；
- 迁移到新机器时，至少需要检查：
  - HF cache
  - dataset root
  - calibration dirs
  - checkpoint path
  - CUDA 可用性
  - 相机 index
  - 串口路径

### 7.2 给 `run_so101_policy_record.sh` 增加最小 preflight

建议至少在 shell 层提前检查：

- `POLICY_PATH` 是否存在
- `ROBOT_CALIB_DIR` 是否存在
- 如果传了 `LEADER_PORT`，则 `LEADER_CALIB_DIR` 是否存在
- 当 `POLICY_DEVICE=cuda` 时，是否存在可用 GPU

这样可以把错误提前到脚本入口，减少“进了 Python 才炸”的情况。

### 7.3 让 shell wrapper 不去覆盖 Python 入口的自动命名

建议 `run_so101_policy_record.sh` 改成只在用户显式设置时才传：

- `--dataset_repo_id`
- `--dataset_root`

或者至少把默认值改回：

- `local/eval_so101_policy`
- `./outputs/eval_so101_policy`

让 `lerobot_record_so101_policy.py` 自己根据 `policy.type` 决定最终命名。这样对非 SmolVLA policy 也更稳。

### 7.4 把共用 rename map 抽到单一来源

建议避免在多个 shell 脚本和一个 Python 入口里重复写同一段 JSON / dict。可以选一种方式：

- 放到单独的 shell 变量文件
- 放到单独的 Python helper
- 或至少在文档里声明“这三处必须保持一致”

### 7.5 给 hybrid nohup 包装器补齐文档

建议在 `docs/smolvla_rl_training_and_operations_20260315_zh.md` 的 hybrid 章节，像离线训练一样补一节：

- `start_smolvla_hybrid_train_nohup.sh`
- `monitor_training_process.sh`
- 如何传 `env_type` / `env_task`
- 输出哪些 pid/log 文件

### 7.6 把“前台但不直播日志”写清楚

建议在脚本 README 和操作文档中明确写一句：

- `launch_*` 是前台运行，但训练 stdout/stderr 默认重定向到日志文件，不会持续刷屏。

这能减少第一次使用时的误判。

### 7.7 把环境变量分为“必须改”和“通常不用改”

当前脚本的环境变量很多，但优先级不清晰。建议在 README 中分两组列出：

- 必须按机器调整：
  - `HF_CACHE_DIR`
  - `DATASET_ROOT`
  - `POLICY_PATH`
  - `ROBOT_PORT`
  - `ROBOT_CALIB_DIR`
  - `TOP_CAMERA_INDEX`
  - `WRIST_CAMERA_INDEX`
- 通常按实验调整：
  - `RUN_TAG`
  - `OUTPUT_DIR`
  - `LOG_FILE`
  - `STEPS`
  - `BATCH_SIZE`
  - `NUM_EPISODES`
  - `EPISODE_TIME_S`

## 8. 结论性判断

如果目标是“在当前这台机器上服务当前 SmolVLA 项目推进”，这组脚本是够用的，且 bundle 叙事与实现基本对齐。

如果目标是“让另一个人或另一台机器无痛接手”，现在还差两件事：

1. 把机器绑定型前提显式文档化。
2. 给 record wrapper 和 hybrid wrapper 补更扎实的前置检查与命名约束。

一句话总结：

- 这是一组“当前环境可用、但默认值过于本机化”的脚本束；
- 与 `lerobot_record_so101_policy.py` 的关系清楚；
- 文档基本一致，但对路径、缓存、设备和默认命名的隐含前提说明还不够。
