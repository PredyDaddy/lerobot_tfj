# RTC 技术方案架构审查

## 结论

当前技术方案方向是对的，核心判断也准确：

- RTC 应该放在 runtime，而不是 engine
- `select_action()` 不是正确入口
- 共享 helper 能减少 TRT / ONNX orchestration 漂移

我没有看到架构级阻塞项。

## 主要认可点

- 方案把 RTC 定义成“运行时时延隐藏与 queue 策略”，而不是“kernel 加速”，这个口径是正确的。
- 方案尊重了 `PI05` 当前的真实边界：RTC 在 `sample_actions()` 的 denoise loop 中，而不是在导出图里。
- 方案没有碰现有 Stage 2-5 工件链路，降低了工程风险。

## 架构级提醒

- `ActionQueue` 和 `AsyncChunkPrefetcher` 的职责必须明确分开。不要把 queue merge 语义塞进 prefetcher，否则后面很难维护。
- 共享 helper 建议只抽 orchestration，不要把 TRT / ONNX adapter 的 backend 细节混进去。
- 如果第一轮同时改 TRT 和 ONNX，两条链可能一起出问题。实施上可以先让 TRT 路径落地，再迁移 ONNX。

## 建议补强

- 在技术方案中补一句：第一阶段默认仍保持 `RTC off`，RTC 作为显式开启的新运行模式。
- 在方案中明确 `predict_action_chunk()` 的 metadata 输出约定，否则 queue merge 阶段的状态传递会散落在不同函数签名里。

## 最终意见

可以进入实施阶段，但建议按“TRT 先落地，共享 helper 后抽”的顺序推进。
