# PI TRT Environment

This note documents the phase-1 environment gate for the PI TensorRT workspace.

## Phase-1 goal

Stage-0 establishes whether a machine is ready for the rest of the PI TensorRT flow:

- Python modules import cleanly: `torch`, `onnx`, `onnxruntime`, `tensorrt`
- CUDA is visible through `torch.cuda.is_available()`
- Optional GPU inventory can be queried with `nvidia-smi`
- Required checkpoint assets are present when a checkpoint path is supplied

The script does not build ONNX or TensorRT engines yet. It only produces a reliable preflight JSON report and a metadata skeleton for the run directory.

## Required checkpoint assets

When `--checkpoint-path` is provided, the script resolves one of these layouts:

- `<path>/`
- `<path>/pretrained_model/`
- `<path>/checkpoints/last/pretrained_model/`

The resolved checkpoint directory must contain:

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

## CLI usage

Strict preflight against a concrete checkpoint:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --checkpoint-path /path/to/pretrained_model \
  --strict
```

Environment-only probe:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --allow-missing-checkpoint \
  --print-json
```

Explicit output path:

```bash
python tfj_envs/pi_trt/scripts/check_env.py \
  --variant pi05 \
  --checkpoint-path /path/to/pretrained_model \
  --run-dir tfj_envs/pi_trt/runs/demo \
  --output tfj_envs/pi_trt/runs/demo/stage0_env_check.json
```

## Output files

The default run directory is `tfj_envs/pi_trt/runs/pi_trt_<variant>_<timestamp>/`.

Stage-0 writes:

- `stage0_env_check.json`
- `pi_trt_metadata.json`

The JSON report contains:

- `preflight.ready`: overall pass/fail gate
- `preflight.errors`: blocking issues
- `preflight.warnings`: non-blocking notes
- `preflight.environment.modules`: import/version information
- `preflight.environment.cuda`: CUDA and `nvidia-smi` probe details
- `preflight.checkpoint`: required asset status

The metadata file contains the initial run contract and reserved stage artifact paths so later scripts can attach their results without re-deriving directory structure.
