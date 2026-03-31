#!/usr/bin/env python3
from __future__ import annotations

import sys

sys.path.insert(0, "/data/tfj/lerobot_tfj/tfj_envs/so101_control_suite/scripts")

from so101_sdk import SO101SDK, SO101SDKConfig


def main() -> int:
    cfg = SO101SDKConfig(
        robot_port="/dev/ttyACM0",
        control_mode="ik",
        tcp_offset_xyz=(0.0, 0.0, 0.0),
        max_command_delta_deg=8,
        max_relative_target_deg=8,
        dry_run=True,
    )

    with SO101SDK(cfg) as arm:
        print("state_before:", arm.state())
        print("tcp_before:", arm.tcp_xyz())
        arm.up(1.0)
        arm.left(1.0)
        arm.open_gripper()
        print("state_after:", arm.state())
        print("tcp_after:", arm.tcp_xyz())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
