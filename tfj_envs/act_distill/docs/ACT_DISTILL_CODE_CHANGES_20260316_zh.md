# ACT 蒸馏代码修改说明

## 1. 这次新增或整理的运行资产

训练资产：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/train_act_distill_smoke.sh`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/train_act_distill_full.sh`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/launch_act_distill_train.sh`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/start_act_distill_train_nohup.sh`

上机资产：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/lerobot_run_act_so101.py`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/scripts/run_act_distill_so101_infer.sh`

文档资产：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/docs/ACT_DISTILL_LAUNCH_NOTES_20260315.md`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/docs/ACT_DISTILL_WORKSPACE_GUIDE_20260316_zh.md`
- `/data/tfj/lerobot_tfj/tfj_envs/act_distill/docs/ACT_DISTILL_CODE_CHANGES_20260316_zh.md`

## 2. 为了打通训练做的代码修复

### 2.1 ACT processor compatibility 修复

文件：

- `/data/tfj/lerobot_tfj/src/lerobot/policies/act/processor_act.py`

原因：

- 原先的 KD 启动期检查要求 student / teacher 的 entire stats dict 逐 tensor 全等。
- 实际同一条数据线也可能因为序列化、dtype 或额外 dataset metadata 字段导致不完全相等。
- 这会把本来等价的归一化路径错误拦掉。

修复内容：

- 只比较真正参与归一化的 feature。
- 只比较当前 normalization mode 必需的统计量。
- 允许极小浮点误差。
- 不放宽 image key 顺序、rename map、norm mode 这些真正危险的边界。

### 2.2 processor 回归测试补充

文件：

- `/data/tfj/lerobot_tfj/tests/processor/test_act_processor.py`

新增了一条回归测试，覆盖：

- 等价 normalization stats
- dtype 不同
- 带额外 metadata key

这条路径应该判定为 compatible。

## 3. 为了打通数据读取做的训练侧冻结

训练脚本固定：

- `--dataset.video_backend=pyav`

原因：

- 当前数据集视频为 `AV1`
- `PyAV/libdav1d` 可读
- `torchcodec` 在本机训练时出现了解码失败

## 4. 为了上机做的新入口

文件：

- `/data/tfj/lerobot_tfj/src/lerobot/scripts/lerobot_run_act_so101.py`

这个脚本的目标不是录数据，而是：

- 直接在 `so101_follower` 上运行 ACT 蒸馏模型
- 使用 `argparse` 风格参数
- 兼容你给的这组参数名：
  - `--robot-id`
  - `--robot-port`
  - `--robot-calibration-dir`
  - `--top-cam-index`
  - `--wrist-cam-index`

同时额外支持：

- `--policy-path`
- `--policy-device`
- `--fps`
- `--run-time-s`
- `--task`
- `--display-data`

## 5. 为什么又复制到 tfj_envs/act_distill

原因很直接：

- 训练时临时目录和源码目录分散
- 上机入口在 `src/lerobot/scripts`
- shell wrapper 在 `scripts/`

你现在要的是一个统一工作区，所以把可运行资产复制并整理到：

- `/data/tfj/lerobot_tfj/tfj_envs/act_distill`

这里保留：

- 可直接执行的 shell 入口
- 可直接阅读的文档
- 可直接调用的 Python 上机脚本
