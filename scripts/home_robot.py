#!/usr/bin/env python3
"""Home the Dobot and apply the persisted post-home j4 offset.

Every home must go through this script (or ``DobotRobot.home``) so the suction
cup returns to the same yaw.  The offset is read from
``configs/robot_limits.yaml`` (``post_home_j4_offset_deg``); 88 degrees is the
default and leaves a 2 degree margin inside the j4 [-90, 90] limit.

Usage:
    python scripts/home_robot.py
    python scripts/home_robot.py --port /dev/dobot_follower --timeout-s 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dobot_rua.hardware.dobot import DobotRobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/dobot_follower")
    parser.add_argument("--config", default="configs/robot_limits.yaml")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    home_deg = tuple(float(value) for value in data["home_deg"])
    offset = float(data.get("post_home_j4_offset_deg", 88.0))

    robot = DobotRobot(
        port=args.port,
        base_j1_scale=float(data.get("base_j1_scale", 1.0)),
    )
    robot.connect()
    try:
        state = robot.home(
            home_deg=home_deg,
            post_home_j4_offset_deg=offset,
            timeout_s=args.timeout_s,
        )
    finally:
        # Keep the vacuum state as-is; the caller may still be holding a board.
        robot.close(release=False)

    print(
        "home+lock: "
        f"x={state.x_mm:.2f} y={state.y_mm:.2f} z={state.z_mm:.2f} "
        f"r={state.r_deg:.2f} "
        f"j1={state.j1_deg:.2f} j2={state.j2_deg:.2f} "
        f"j3={state.j3_deg:.2f} j4={state.j4_deg:.2f}"
    )


if __name__ == "__main__":
    main()
