#!/usr/bin/env python3
"""Keyboard jog for the Dobot Magician.

Run it in a real terminal (not through a pipe). One keypress = one small move.

    w / s   x + / -          a / d   y + / -
    r / f   z + / -          q / e   j4 - / +
    1 2 3   1 / 2 / 5 mm     5 / 0   step 10 mm / 0.5 mm
    p       print pose       v       toggle the vacuum
    k       record a touch into the current touch session
    ?       help             x       quit (vacuum left as-is)

Every target is clamped to a safe box and IK-checked before it is sent.
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import time
import tty
from pathlib import Path

from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.hardware.jog import JogBox, next_target
from dobot_rua.kinematics import inverse_kinematics, load_fk, load_limits

HELP = __doc__


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _status(robot: DobotRobot, target) -> None:
    state = robot.read_state()
    print(
        f"  target ({target[0]:6.1f},{target[1]:6.1f},{target[2]:6.1f},r{target[3]:5.1f})  "
        f"actual ({state.x_mm:6.1f},{state.y_mm:6.1f},{state.z_mm:6.1f})  "
        f"suction={'ON' if state.suction_on else 'off'}"
    )


def _record_touch(session: Path, robot: DobotRobot) -> None:
    print("\n-- record touch --")
    answer = input("marker id: ").strip()
    if not answer:
        print("cancelled")
        return
    try:
        marker = int(answer)
    except ValueError:
        print("marker id must be an integer")
        return
    payload = {"touches": []}
    if session.exists():
        payload = json.loads(session.read_text(encoding="utf-8"))
        payload.setdefault("touches", [])
    state = robot.read_state()
    payload["touches"].append(
        {
            "marker": marker,
            "base_mm": [state.x_mm, state.y_mm, state.z_mm],
            "time": time.strftime("%H:%M:%S"),
        }
    )
    session.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"recorded marker {marker} at ({state.x_mm:.1f}, {state.y_mm:.1f}, "
        f"{state.z_mm:.1f}) mm -> {session}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-port", default="/dev/dobot_follower")
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--session", type=Path, default=Path("audit/touch_session.json"))
    parser.add_argument("--no-ik-check", action="store_true")
    args = parser.parse_args()

    params = load_fk("configs/fk_magician.yaml")
    limits = load_limits("configs/robot_limits.yaml")
    box = JogBox(x=(90.0, 230.0), y=(-90.0, 90.0), z=(0.0, 120.0), r=(-45.0, 45.0))

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    robot.set_speed(args.speed, args.speed)
    state = robot.read_state()
    target = (state.x_mm, state.y_mm, state.z_mm, state.r_deg)
    step_mm = float(args.step_mm)
    step_deg = 5.0

    print(__doc__)
    _status(robot, target)
    print("press ? for help, x to quit")

    try:
        while True:
            key = _getch()
            if key in ("x", "\x1b"):
                break
            if key == "?":
                print(HELP)
                continue
            if key == "p":
                _status(robot, target)
                continue
            if key == "v":
                robot.set_suction(not robot.read_state().suction_on)
                print(f"  suction {'ON' if robot.read_state().suction_on else 'off'}")
                continue
            if key == "k":
                _record_touch(args.session, robot)
                target = (
                    robot.read_state().x_mm,
                    robot.read_state().y_mm,
                    robot.read_state().z_mm,
                    robot.read_state().r_deg,
                )
                continue
            if key in "123450":
                step_mm = {"1": 1.0, "2": 2.0, "3": 5.0, "5": 10.0, "0": 0.5}[key]
                print(f"  step {step_mm} mm")
                continue
            step = next_target(target, key, step_mm, step_deg, box)
            if step is None:
                continue
            if step.clamped:
                print("  clamped to the safe box")
            x, y, z, r = step.target
            if not args.no_ik_check and not inverse_kinematics(
                params, x, y, z, limits, margin_deg=2.0
            ):
                print("  unreachable for the three axes; move not sent")
                continue
            robot.move_pose(x, y, z, r, timeout_s=12.0)
            target = (x, y, z, r)
            _status(robot, target)
    finally:
        robot.close(release=False)
        print("closed; vacuum left as-is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
