# ACT TRT SO101 Real-Robot Debugging Notes

## Scope

This note records the ACT TensorRT real-robot deployment issues encountered while bringing
`outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model` onto an SO101 follower robot
in `/data/tfj/lerobot_tfj`.

It focuses on the practical boundary between:

- exported ACT deployment artifacts
- the LeRobot real-robot runtime
- the local machine's camera / keyboard / dataset state

This is not an export tutorial. For export/build/verify steps, see:

- [ACT_TRT_REAL_ROBOT_USAGE.md](/data/tfj/lerobot_tfj/doc/ACT_TRT_REAL_ROBOT_USAGE.md)
- [ACT_TRT_REFERENCE_EXPORT_REPORT.md](/data/tfj/lerobot_tfj/doc/ACT_TRT_REFERENCE_EXPORT_REPORT.md)

## Final Working Direction

The stable real-robot approach is:

1. Export and validate the ACT TensorRT engine first.
2. Run real-robot deployment from the `lerobot` conda environment, not `lerobot_flex`.
3. Use a direct Python launcher that prints stage-by-stage progress, instead of hiding the runtime behind multiple wrappers.
4. Preflight the TensorRT engine, cameras, keyboard listener, and dataset root before sending any robot action.

The direct runner added during this debugging pass is:

- [run_act_trt_so101_eval.py](/data/tfj/lerobot_tfj/tfj_envs/run_act_trt_so101_eval.py)

## Main Findings

### 1. Export / verify and runtime do not need the same conda environment

The repo reference flow is still best handled in `lerobot_flex`:

- ONNX export
- TensorRT engine build
- Torch / ONNX / TRT consistency checks

But the real-robot runner can use `lerobot` once that environment has:

- `onnxruntime-gpu`
- `tensorrt`
- working OpenCV camera access
- working X session variables for `pynput`

Practical consequence:

- Do not force all ACT TRT work into one environment.
- Split the problem into:
  - artifact generation and numerical validation
  - runtime deployment and hardware integration

### 2. `conda run` can look like a hang even when the process is alive

Using:

```bash
conda run -n lerobot python ...
```

for a long-running real-robot process often looks like a freeze because output is buffered.

Use one of these instead:

```bash
conda run --live-stream -n lerobot python ...
```

or:

```bash
conda activate lerobot
python ...
```

Practical consequence:

- Treat "no terminal output" as an observability problem first, not necessarily a runtime crash.

### 3. TensorRT engine I/O names are not guaranteed to match the reference ACT names

The reference exporter/build flow often uses:

- `obs_state_norm`
- `img0_norm`
- `img1_norm`
- `actions_norm`

But the engine actually used on the checkpoint runtime path here exposed:

- `observation_state`
- `observation_images_top`
- `observation_images_wrist`
- `action_chunk`

The matching metadata file was:

- `act_core_b1.metadata.json`

Practical consequence:

- Never hardcode reference ACT tensor names into the real-robot path without inspecting the real engine.
- Always resolve runtime I/O from:
  - engine inspection
  - export metadata
  - policy config camera keys

### 4. `last` is an alias, not the real checkpoint step

The path:

- `outputs/act_grasp_block_in_bin1/checkpoints/last/pretrained_model`

resolved to:

- `outputs/act_grasp_block_in_bin1/checkpoints/018000/pretrained_model`

Practical consequence:

- In logs and debugging output, print the resolved checkpoint path.
- Otherwise users think they are testing one checkpoint while the runtime is actually loading another.

### 5. A half-created dataset root is worse than a missing dataset root

The runtime was pointed at:

- `/home/cqy/.cache/huggingface/lerobot/admin123/eval_grasp_block_in_bin2`

That directory existed, but contained only:

- `meta/info.json`

This is an empty dataset skeleton, not a usable existing dataset.

In that state, auto-resume logic is unsafe because LeRobot may try to open it as a real dataset and then
fail or behave inconsistently due to missing tasks / stats / episodes metadata.

Practical consequence:

- Treat a root containing only `meta/info.json` as broken state.
- Delete and recreate it, or pick a clean root.
- Do not blindly map "directory exists" to "resume safely".

### 6. Hardware preflight should be explicit

Before connecting the robot and entering the control loop, these checks passed independently:

- TensorRT engine loads
- keyboard listener initializes
- camera `4` opens and reads `(480, 640, 3)`
- camera `6` opens and reads `(480, 640, 3)`

Practical consequence:

- Add a `--preflight-only` mode to the real-robot script.
- Keep hardware checks separate from motion control.
- When debugging, prove each subsystem first, then compose them.

### 7. `pynput` is not just a Python dependency issue

`pynput` was installed, but it still fails without a valid X session.

The working environment variables were:

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1003/gdm/Xauthority
```

Practical consequence:

- If keyboard listener init fails, check X session wiring before reinstalling Python packages.

### 8. Direct runners are easier to debug than wrapper chains

The earlier wrapper approach delegated into another script with limited stage visibility.
That made it harder to distinguish between:

- env validation
- engine loading
- camera init
- dataset opening
- robot connection
- actual control loop

The rewritten direct runner prints explicit stage markers:

- `Validate environment`
- `Build configs`
- `Preflight checks`
- `Create robot and dataset pipeline`
- `Open or create dataset`
- `Load processors and TRT policy`
- `Connect robot`
- `Start record loop`

Practical consequence:

- For real-robot control code, favor a direct control path with visible stage logging.
- Avoid stacked wrappers unless they add clear value and preserve observability.

## Practical Checklist

Run this before real deployment:

1. Confirm the engine and metadata exist and match the target checkpoint.
2. Print the resolved checkpoint path behind `last`.
3. Verify the runtime environment can `import tensorrt`.
4. Use `conda run --live-stream` or activate the environment manually.
5. Verify `DISPLAY` and `XAUTHORITY` before using `pynput`.
6. Preflight both cameras with one frame read.
7. Preflight the TRT adapter and print actual engine tensor names.
8. Check whether `dataset.root` is:
   - absent and safe to create
   - a valid existing dataset
   - an empty skeleton that should be deleted

## Lessons Learned

- A numerically correct engine is only half the job. Real-robot deployment fails just as often at the runtime boundary as at the model boundary.
- "It builds" is not enough. "It loads in the target environment with the target camera keys and dataset state" is the real deployment gate.
- Observability matters more than elegance for hardware scripts. A slightly longer direct runner is better than a compact wrapper that hides where it failed.
- Deployment tooling should aggressively detect invalid local state instead of assuming any existing path is reusable.

## Recommended Commands

Preflight only:

```bash
conda run --live-stream -n lerobot \
python /data/tfj/lerobot_tfj/tfj_envs/run_act_trt_so101_eval.py --preflight-only
```

Full run:

```bash
conda run --live-stream -n lerobot \
python /data/tfj/lerobot_tfj/tfj_envs/run_act_trt_so101_eval.py
```
