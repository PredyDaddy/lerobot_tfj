# so101_control_pause_20260319

Current clean entry point for the control stack:
- `/data/tfj/lerobot_tfj/tfj_envs/so101_control_suite`

This folder remains the broader pause-state archive.

This folder is a pause-state archive of the SO-101 control/runtime project files.

Moved on: 2026-03-19
Workspace root: /data/tfj/lerobot_tfj

Layout keeps original relative paths under:
- scripts/
- src/lerobot/scripts/
- src/lerobot/runtime/so101_pickplace/
- tests/runtime/
- docs/train_logs/

Purpose:
- Pause current project iteration
- Keep all related files centralized under ./tfj_envs/

## Manual arm control console

Script:
- `scripts/so101_manual_console.py`

Quick examples:

```bash
# 1) Read current joint state only (no movement)
scripts/so101_manual_console.py --robot-port /dev/ttyACM0 --cmd "obs"

# 2) Safe tiny movement: up 1cm, left 1cm, open gripper
scripts/so101_manual_console.py \
  --robot-port /dev/ttyACM0 \
  --max-command-delta-deg 2 \
  --max-relative-target-deg 2 \
  --gripper-max-relative-target 30 \
  --cmd "up 1; left 1; open; obs"

# 3) Interactive mode
scripts/so101_manual_console.py --robot-port /dev/ttyACM0

# 4) Closed-loop IK mode (recommended for Cartesian commands)
scripts/so101_manual_console.py \
  --robot-port /dev/ttyACM0 \
  --control-mode ik \
  --tcp-offset-xyz "0 0 0" \
  --max-command-delta-deg 8 \
  --max-relative-target-deg 8
```

Supported commands in interactive mode:
- `obs`
- `frame [show|set <direction> <x+|x-|y+|y-|z+|z->|reset|save|load]`
- `calibrate [cm]` (IK mode)
- `up [cm]`, `down [cm]`, `left [cm]`, `right [cm]`, `forward [cm]`, `back [cm]`
- `up_exact [cm]`, `down_exact [cm]`, `left_exact [cm]`, `right_exact [cm]`
- `forward_exact [cm]`, `back_exact [cm]` (IK mode)
- `open`, `close`
- `j <joint> <delta_deg>`
- `set <joint> <value>`
- `home`
- `quit`

Safety notes:
- Keep an operator on-site while running commands.
- Start with small values and strict limits before increasing motion.
- In IK mode, keep `--no-ik-auto-flip-axis` (default) for deterministic user directions.
- Use `frame set` + `frame save` once if left/right/forward mapping differs from your setup.
- If the commanded point is not the real tool tip, tune `--tcp-offset-xyz "x y z"` in meters.

## Python SDK wrapper

Script:
- `scripts/so101_sdk.py`

Example (recommended IK runtime):

```python
from so101_sdk import SO101SDK, SO101SDKConfig

cfg = SO101SDKConfig(
    robot_port="/dev/ttyACM0",
    control_mode="ik",
    tcp_offset_xyz=(0.0, 0.0, 0.0),
    max_command_delta_deg=8,
    max_relative_target_deg=8,
)

with SO101SDK(cfg) as arm:
    arm.up(1.0)             # move end-effector up 1cm
    arm.left(1.0)           # move left 1cm
    arm.open_gripper()      # open gripper
    print(arm.state())      # read current joint state
```

Run with IK dependencies:

```bash
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python your_script.py
```

## Web UI

Script:
- `scripts/so101_web_ui.py`

Start server:

```bash
# Recommended (IK dependencies available)
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python scripts/so101_web_ui.py --host 127.0.0.1 --port 8765
```

Then open in browser:

```text
http://127.0.0.1:8765
```

If you only want a dry-run UI check:

```bash
python3 scripts/so101_web_ui.py --host 127.0.0.1 --port 8765 --control-mode joint --dry-run
```

Notes:
- The UI has `TCP X/Y/Z(mm)` inputs. Those values are converted to meters and applied as `tcp_offset_xyz`.
- For real IK control, start the UI with `/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python`, not plain `python3`.
