# RTC 实施计划执行审查

## 结论

实施计划整体可执行，代码改动面控制得比较合理，没有出现“为了 RTC 去碰 engine/export/build”这种过度扩张。

## 主要认可点

- 把第一改动点放在 `trt_pi_adapter.py` 是对的，因为 RTC 数学逻辑本来就应该先回到 denoise loop。
- 新增共享 `pi05_chunk_runtime.py` 是对的，否则 `run_pi05_trt_infer_so101.py` 会很快变成一个巨型脚本。
- 保留 `RTC off` 快路径很重要，这能保证当前可部署链路不被破坏。

## 执行层风险

- `control_utils.predict_action()` 不能半改半留。如果 RTC 模式还从这条路径进去，最终一定会和 `select_action()` 的断言冲突。
- `ChunkPredictionResult` 字段需要一次定义好，后续 launcher、benchmark、日志都要吃这份 schema。
- 如果不先做 mock runtime benchmark，直接上真机会很难分辨问题来自 queue 逻辑还是来自机器人端。

## 建议补强

- 在实施计划中加一个 “Step 0: schema 定义” 小节，先定义 runtime result / stats / log fields。
- 在计划中加一条：`run_pi05_trt_infer_so101.py --dry-run` 和 `--preflight-only` 需要能覆盖 RTC 配置路径，不只是普通路径。
- 如果第一轮时间有限，可以暂时不迁移 ONNX launcher，只把共享 helper 设计成 TRT 先用、ONNX 后迁移。

## 最终意见

计划可以执行，但强烈建议先把 schema 和 mock benchmark 固定，再进入真机运行。
