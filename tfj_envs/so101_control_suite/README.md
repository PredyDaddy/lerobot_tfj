# so101_control_suite

This is the cleaned SO101 control workspace under `tfj_envs/`.

Use this folder if you want to:
- write your own SO101 control script
- use the Python SDK directly
- start a local Web UI for manual control

Main files:
- `scripts/so101_manual_console.py`: low-level manual console and IK control
- `scripts/so101_sdk.py`: Python SDK wrapper for your own scripts
- `scripts/so101_web_ui.py`: local browser-based control panel
- `examples/demo_sdk.py`: minimal SDK example
- `runtime/`: saved direction map and future runtime files

## Recommended Python

For real IK control, use:

```bash
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python
```

Reason:
- `ik` mode requires `placo`
- the system `python3` may not have that dependency

## Quick Start

### 1) Write your own Python script

Start from:

```python
import sys
sys.path.insert(0, "/data/tfj/lerobot_tfj/tfj_envs/so101_control_suite/scripts")

from so101_sdk import SO101SDK, SO101SDKConfig

cfg = SO101SDKConfig(
    robot_port="/dev/ttyACM0",
    control_mode="ik",
    tcp_offset_xyz=(0.0, 0.0, 0.0),
    max_command_delta_deg=8,
    max_relative_target_deg=8,
)

with SO101SDK(cfg) as arm:
    print("state:", arm.state())
    print("tcp:", arm.tcp_xyz())
    arm.up(1.0)
    arm.left(1.0)
    arm.open_gripper()
```

Run it with:

```bash
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python your_script.py
```

### 2) Start the Web UI

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/so101_control_suite
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python scripts/so101_web_ui.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

### 3) Use the manual console

```bash
cd /data/tfj/lerobot_tfj/tfj_envs/so101_control_suite
/data/tfj/lerobot_tfj/.venvs/geom_grasp/bin/python scripts/so101_manual_console.py \
  --robot-port /dev/ttyACM0 \
  --control-mode ik \
  --tcp-offset-xyz "0 0 0" \
  --cmd "obs"
```

## TCP Note

If the controlled point is not the real tool tip, tune:

```bash
--tcp-offset-xyz "x y z"
```

Units are meters in the tool frame.

In the Web UI, the `TCP X/Y/Z(mm)` fields are the same parameter, but entered in millimeters.

## History

The older archive remains here:
- `/data/tfj/lerobot_tfj/tfj_envs/so101_control_pause_20260319`

That folder keeps the broader pause-state project.
This folder is the cleaned control-focused entry point.
