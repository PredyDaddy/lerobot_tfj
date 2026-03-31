# PI TensorRT Workspace

This workspace hosts the phase-1 PI TensorRT flow under `tfj_envs/pi_trt/`.

Phase-1 scope is intentionally narrow:

- Variant focus: `pi05` is the only required deployment target.
- Engine contract: one run directory maps to one checkpoint and one fixed variant.
- Frozen boundaries: `vision_encoder`, `prefix_cache`, `denoise_step`.
- Validation style: script-driven stage reports instead of mainline CI integration.

The first deliverable in this workspace is the stage-0 environment and preflight gate:

- Shared helpers live in `scripts/common.py`.
- Environment + checkpoint preflight CLI lives in `scripts/check_env.py`.
- Output JSON defaults to `tfj_envs/pi_trt/runs/<run_name>/stage0_env_check.json`.

Recommended phase-1 order:

1. Run `check_env.py` to verify Python packages, CUDA visibility, TensorRT import, and checkpoint assets.
2. Use the generated run directory as the anchor for later checkpoint inspection, ONNX export, and TRT build stages.
3. Keep every stage artifact inside the same run directory so later compare scripts can consume a stable layout.

Quick start:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --checkpoint-path /path/to/pretrained_model \
  --strict \
  --print-json
```

Environment-only probe before a checkpoint is ready:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --allow-missing-checkpoint \
  --print-json
```

Custom run directory:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --checkpoint-path /path/to/pretrained_model \
  --run-dir tfj_envs/pi_trt/runs/manual_stage0
```

Outputs created by stage-0:

- `stage0_env_check.json`: environment probe + checkpoint asset summary.
- `pi_trt_metadata.json`: initial run metadata skeleton for later stages.

For dependency notes and JSON semantics, see `docs/PI_TRT_ENVIRONMENT.md`.
