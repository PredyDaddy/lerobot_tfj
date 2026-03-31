# Worker A Round 4: ONNX Provenance / Coherence

## 改动摘要

- 在 `scripts/run_pi05_onnx_infer_so101.py` 内新增 `OnnxArtifactSafetyReport`，把 ONNX 启动前的 provenance/coherence 校验集中到 `assess_onnx_artifact_safety(...)`，并在 `resolve_onnx_artifacts(...)` 内直接 hard fail，不再允许 fallback 混拼三件套。
- `resolve_onnx_artifacts(...)` 现在要求同时解析到 `stage2_export_onnx.json` 与 `stage3_verify_onnx.json`。若任一报告缺失、格式不是对象、关键 gate 状态缺失或不是 `pass`，会在进入 `PI05 ONNX policy OK` 前阻断。
- `validate_paths(...)` 现在会对请求的 `--policy-path` 与 `stage2/stage3` 报告记录的 `policy_dir` 做阻断式一致性校验，不再只是 warning。
- `print_summary(...)` 不再只打印单个误导性的 `ONNX dir`，而是打印 `stage2/stage3` 报告路径、gate 状态、provenance 的 `policy_dir/run_dir/onnx_dir`，以及 vision/prefix/denoise 三件套的实际路径。
- `main()` 里把 `--joint-delta-limit 0`、`--gripper-delta-limit 0`、`--robot-max-relative-target 0` 统一收口为 fail-fast，和 TRT launcher 当前的 `> 0` 策略保持一致。
- 在 `robot.send_action(...)` 前新增 `assert_finite_robot_action(...)`，把 ONNX 发送前 finite guard 补齐到和 TRT 同级的安全边界。

关键实现位置：

- provenance 数据结构与 gate 抽取：`scripts/run_pi05_onnx_infer_so101.py:69-168`
- ONNX provenance 阻断规则：`scripts/run_pi05_onnx_infer_so101.py:522-799`
- summary/provenance logging：`scripts/run_pi05_onnx_infer_so101.py:910-981`
- `0` 值参数阻断：`scripts/run_pi05_onnx_infer_so101.py:1126-1142`
- 发送前 finite guard：`scripts/run_pi05_onnx_infer_so101.py:1086-1093`、`scripts/run_pi05_onnx_infer_so101.py:1419-1428`

## 阻断规则

当前 ONNX launcher 会在进入 ONNX preflight 前阻断以下情况：

1. 找不到 `stage2_export_onnx.json`。
2. 找不到 `stage3_verify_onnx.json`。
3. `stage2` 报告的 `stage` 不是 `stage2_export_onnx`。
4. `stage3` 报告的 `stage` 不是 `stage3_verify_onnx`。
5. `stage2` gate 状态缺失或不是 `pass`。
   读取顺序是 `stage2_acceptance.status`，若不存在则回退到 `overall_status`。
6. `stage3` gate 状态缺失或不是 `pass`。
   读取顺序是 `stage3_acceptance.status`，若不存在则回退到 `overall_status`。
7. `stage2/stage3` 任一报告缺失 `policy_dir`、`run_dir`、`onnx_dir`。
8. `stage2/stage3` 的 `policy_dir`、`run_dir` 或 `onnx_dir` 彼此不一致。
9. `stage2` 的 `onnx_paths` 缺 vision/prefix/denoise 任一项，或路径落不到实际文件。
10. 三个 ONNX 不在同一个实际父目录下。
11. `stage3` 的 `artifact_paths` 与 `stage2` 解析出的三件套路径不一致。
12. `stage3.stage2_context.stage2_report_path` 或 `stage2_onnx_paths` 与已解析的 `stage2` provenance 不一致。
13. 显式传入单个 `--onnx-path /path/to/*.onnx`，但该文件不属于 coherent 的三件套集合。
14. 请求的 `--policy-path` 与 `stage2/stage3` 报告记录的 `policy_dir` 不一致。

补充说明：

- `stage3_verify_onnx` 的 `overall_status` 目前只作为 note 打印，不作为 gate；真正的阻断状态取 `stage3_acceptance.status`。这是为了延续现有 stage3 报告的 gate 语义，避免把 `overall_status=warn` 但 gate 已 `pass` 的历史有效 run 一刀切误伤。

## 与 TRT 对齐情况

- provenance/coherence：ONNX 现在也在 launcher 入口做阻断式校验，不再允许“文件能打开就继续”。粒度上仍然是 `stage2/stage3`，不像 TRT 那样覆盖 `stage4/stage5`，但对 ONNX 自身来源闭合已经补到同一级别的 fail-fast 口径。
- `0` 值 CLI 语义：`--joint-delta-limit`、`--gripper-delta-limit`、`--robot-max-relative-target` 现在和 TRT 一样，提供时必须 `> 0`；`0` 不再在 ONNX 路径里被解释成“关闭限幅”。
- 发送前安全边界：ONNX 已补 `assert_finite_robot_action(...)`，与 TRT 的发送前 finite 检查对齐。
- summary/logging：ONNX 现在显式打印三件套路径与 stage2/stage3 provenance，不再用单个 `ONNX dir` 伪装成已闭合来源。

## 自检命令

执行过的最小必要自检如下：

```bash
python -m py_compile scripts/run_pi05_onnx_infer_so101.py
```

```bash
python scripts/run_pi05_onnx_infer_so101.py --help
```

```bash
python - <<'PY'
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path('scripts').resolve()))
import run_pi05_onnx_infer_so101 as module

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')

results = []
with tempfile.TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)

    mixed_root = root / 'mixed_case'
    onnx_a = mixed_root / 'bundle_a' / 'onnx'
    onnx_b = mixed_root / 'bundle_b' / 'onnx'
    touch(onnx_a / 'pi_shared_vision_encoder.onnx')
    touch(onnx_b / 'pi_shared_prefix_cache.onnx')
    touch(onnx_b / 'pi05_denoise_step.onnx')
    stage2_mixed = mixed_root / 'stage2_export_onnx.json'
    stage3_mixed = mixed_root / 'stage3_verify_onnx.json'
    write_json(stage2_mixed, {
        'stage': 'stage2_export_onnx',
        'overall_status': 'pass',
        'policy_dir': str(root / 'policy_ok'),
        'run_dir': str(mixed_root / 'run_dir'),
        'onnx_dir': str(onnx_a),
        'onnx_paths': {
            'vision_encoder': str(onnx_a / 'pi_shared_vision_encoder.onnx'),
            'prefix_cache': str(onnx_b / 'pi_shared_prefix_cache.onnx'),
            'denoise_step': str(onnx_b / 'pi05_denoise_step.onnx'),
        },
    })
    write_json(stage3_mixed, {
        'stage': 'stage3_verify_onnx',
        'overall_status': 'pass',
        'stage3_acceptance': {'status': 'pass', 'hard_fail': False, 'failed_checks': []},
        'policy_dir': str(root / 'policy_ok'),
        'run_dir': str(mixed_root / 'run_dir'),
        'onnx_dir': str(onnx_a),
        'artifact_paths': {
            'vision_encoder': str(onnx_a / 'pi_shared_vision_encoder.onnx'),
            'prefix_cache': str(onnx_b / 'pi_shared_prefix_cache.onnx'),
            'denoise_step': str(onnx_b / 'pi05_denoise_step.onnx'),
        },
        'stage2_context': {
            'stage2_report_path': str(stage2_mixed),
            'stage2_onnx_paths': {
                'vision_encoder': str(onnx_a / 'pi_shared_vision_encoder.onnx'),
                'prefix_cache': str(onnx_b / 'pi_shared_prefix_cache.onnx'),
                'denoise_step': str(onnx_b / 'pi05_denoise_step.onnx'),
            },
        },
    })
    try:
        module.resolve_onnx_artifacts(str(onnx_a), None)
    except Exception as exc:
        results.append(('mixed_artifacts', type(exc).__name__, str(exc).splitlines()[0]))

    policy_a = root / 'policy_a'
    policy_b = root / 'policy_b'
    calib_dir = root / 'calib'
    policy_a.mkdir(parents=True)
    policy_b.mkdir(parents=True)
    calib_dir.mkdir(parents=True)
    (policy_a / 'config.json').write_text('{}\n', encoding='utf-8')
    (policy_b / 'config.json').write_text('{}\n', encoding='utf-8')
    coherent_root = root / 'policy_mismatch_case'
    coherent_onnx = coherent_root / 'onnx'
    for filename in module.ONNX_FILENAMES.values():
        touch(coherent_onnx / filename)
    stage2_policy = coherent_root / 'stage2_export_onnx.json'
    stage3_policy = coherent_root / 'stage3_verify_onnx.json'
    paths = {name: str(coherent_onnx / filename) for name, filename in module.ONNX_FILENAMES.items()}
    write_json(stage2_policy, {
        'stage': 'stage2_export_onnx',
        'overall_status': 'pass',
        'policy_dir': str(policy_b),
        'run_dir': str(coherent_root / 'run_dir'),
        'onnx_dir': str(coherent_onnx),
        'onnx_paths': paths,
    })
    write_json(stage3_policy, {
        'stage': 'stage3_verify_onnx',
        'overall_status': 'pass',
        'stage3_acceptance': {'status': 'pass', 'hard_fail': False, 'failed_checks': []},
        'policy_dir': str(policy_b),
        'run_dir': str(coherent_root / 'run_dir'),
        'onnx_dir': str(coherent_onnx),
        'artifact_paths': paths,
        'stage2_context': {'stage2_report_path': str(stage2_policy), 'stage2_onnx_paths': paths},
    })
    args = argparse.Namespace(
        policy_path=str(policy_a),
        robot_calibration_dir=str(calib_dir),
        onnx_path=str(coherent_onnx),
        onnx_stage2_report_path=None,
        local_tokenizer_path=None,
    )
    try:
        module.validate_paths(args)
    except Exception as exc:
        results.append(('policy_mismatch', type(exc).__name__, str(exc).splitlines()[0]))

    missing_root = root / 'missing_status_case'
    missing_onnx = missing_root / 'onnx'
    for filename in module.ONNX_FILENAMES.values():
        touch(missing_onnx / filename)
    stage2_missing = missing_root / 'stage2_export_onnx.json'
    stage3_missing = missing_root / 'stage3_verify_onnx.json'
    missing_paths = {name: str(missing_onnx / filename) for name, filename in module.ONNX_FILENAMES.items()}
    write_json(stage2_missing, {
        'stage': 'stage2_export_onnx',
        'policy_dir': str(policy_a),
        'run_dir': str(missing_root / 'run_dir'),
        'onnx_dir': str(missing_onnx),
        'onnx_paths': missing_paths,
    })
    write_json(stage3_missing, {
        'stage': 'stage3_verify_onnx',
        'overall_status': 'pass',
        'stage3_acceptance': {'status': 'pass', 'hard_fail': False, 'failed_checks': []},
        'policy_dir': str(policy_a),
        'run_dir': str(missing_root / 'run_dir'),
        'onnx_dir': str(missing_onnx),
        'artifact_paths': missing_paths,
        'stage2_context': {'stage2_report_path': str(stage2_missing), 'stage2_onnx_paths': missing_paths},
    })
    try:
        module.resolve_onnx_artifacts(str(missing_onnx), None)
    except Exception as exc:
        results.append(('missing_stage2_status', type(exc).__name__, str(exc).splitlines()[0]))

for name, exc_type, message in results:
    print(f'{name}: {exc_type}: {message}')
PY
```

负向 smoke 结果：

- `mixed_artifacts: ValueError: Refusing to launch PI05 ONNX runtime without coherent stage2/stage3 provenance:`
- `policy_mismatch: ValueError: Policy path does not match stage2_export_onnx policy_dir: ...`
- `missing_stage2_status: ValueError: Refusing to launch PI05 ONNX runtime without coherent stage2/stage3 provenance:`

## 剩余风险

- 这轮没有做正向的真实 artifact smoke，也没有连硬件。当前自检只证明 fail-fast 规则和 CLI/help/语法无回归，不证明任意现有 ONNX bundle 都满足新 gate。
- 历史 ONNX 产物如果 `stage2_export_onnx.json` 缺 `overall_status` / `stage2_acceptance.status`，现在会被明确拒绝。这是有意收紧，但意味着旧 run 可能需要补报告或重导出。
- `stage3_verify_onnx` 目前按 canonical 文件名 `stage3_verify_onnx.json` 自动解析；若未来同一 run 有多份并行 stage3 变体报告但没有统一 canonical 文件名，launcher 还需要更细的 report 选择策略。
