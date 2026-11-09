#!/usr/bin/env python3
"""Jog-based independent validation of the front planar calibration.

For each physical mark, the front camera + calibration estimates its world XY,
the robot moves the suction tip near that XY, the operator jogs the tip onto
the mark, and the robot pose provides the actual XY.  The estimated-minus-actual
error is the independent measurement; it does not reuse the camera mapping for
the actual point.

Run this after ``calibrate_front_planar.py`` and before formal trials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.front_planar import (
    load_planar_calibration,
    nearest_solution,
)
from dobot_rua.calibration.validation import summarize_validation
from dobot_rua.hardware.cameras import load_tip_length_mm
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.kinematics import inverse_kinematics, load_fk, load_limits
from dobot_rua.perception.blocks import detect_blocks


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plane", choices=("table", "top", "both"), default="both")
    p.add_argument("--points", type=int, default=3)
    p.add_argument("--port", default="/dev/dobot_follower")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--robot-config", type=Path, default=Path("configs/robot_limits.yaml"))
    p.add_argument("--fk-config", type=Path, default=Path("configs/fk_magician.yaml"))
    return p


def _capture_front(device: str, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open front camera {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    frame = None
    for _ in range(10):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    if frame is None:
        raise RuntimeError("front camera returned no frame")
    return frame


def _move_tip(
    robot: DobotRobot,
    fk,
    limits,
    tip_length_mm: float,
    x_mm: float,
    y_mm: float,
    tip_z_mm: float,
    j4_deg: float,
) -> bool:
    state = robot.read_state()
    solutions = inverse_kinematics(
        fk,
        float(x_mm),
        float(y_mm),
        float(tip_z_mm) + tip_length_mm,
        limits,
        margin_deg=5.0,
    )
    solution = nearest_solution(
        solutions, (state.j1_deg, state.j2_deg, state.j3_deg)
    )
    if solution is None:
        return False
    robot.move_joints(solution[0], solution[1], solution[2], j4_deg)
    return True


def main() -> int:
    args = parser().parse_args()
    calibration = load_planar_calibration(args.camera_config, "front")
    if calibration is None:
        print("front planar calibration is missing or invalid")
        return 2
    task = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
    camera_config = yaml.safe_load(
        args.camera_config.read_text(encoding="utf-8")
    ) or {}
    front = (camera_config.get("cameras") or {}).get("front") or {}
    device = str(front["device"])
    width, height = (camera_config.get("capture") or {}).get(
        "default_size", [1280, 720]
    )
    tip_length_mm = load_tip_length_mm(args.camera_config)
    robot_config = yaml.safe_load(
        args.robot_config.read_text(encoding="utf-8")
    ) or {}
    j4_deg = float(robot_config.get("post_home_j4_offset_deg", 88.0))
    table_z_m = float(task.get("table_z_m", 0.0))
    block_height_m = float(
        task.get("block_height_m", task.get("block_size_m", 0.025))
    )
    planes = ["table", "top"] if args.plane == "both" else [args.plane]
    plane_z = {
        "table": table_z_m,
        "top": table_z_m + block_height_m,
    }
    fk = load_fk(args.fk_config)
    limits = load_limits(args.robot_config)
    robot = DobotRobot(port=args.port)
    robot.connect()
    all_passed = True
    try:
        for plane in planes:
            z_m = plane_z[plane]
            image = _capture_front(device, int(width), int(height))
            blocks = detect_blocks(
                image,
                calibration.intrinsic_K,
                calibration.T_world_from_optical,
                z_m,
                distortion=calibration.distortion,
            )
            if not blocks:
                print(f"{plane}: no mark detected; place a colored mark on the plane")
                all_passed = False
                continue
            errors = []
            for index, block in enumerate(blocks[: max(1, args.points)]):
                estimated = np.asarray(block.center_xyz_m[:2], dtype=float)
                if not _move_tip(
                    robot,
                    fk,
                    limits,
                    tip_length_mm,
                    estimated[0] * 1000.0,
                    estimated[1] * 1000.0,
                    z_m * 1000.0 + 50.0,
                    j4_deg,
                ):
                    print(f"{plane}[{index}]: unreachable estimated XY")
                    continue
                input(
                    f"{plane}[{index}]: jog the tip onto the physical mark, "
                    "then press Enter..."
                )
                state = robot.read_state()
                actual = np.array(
                    [state.x_mm / 1000.0, state.y_mm / 1000.0], dtype=float
                )
                error_mm = (estimated - actual) * 1000.0
                errors.append(error_mm)
                print(
                    f"{plane}[{index}]: estimated=({estimated[0]*1000:.1f},"
                    f"{estimated[1]*1000:.1f}) actual=({actual[0]*1000:.1f},"
                    f"{actual[1]*1000:.1f}) error=({error_mm[0]:.2f},"
                    f"{error_mm[1]:.2f}) mm"
                )
            if not errors:
                print(f"{plane}: no validated points")
                all_passed = False
                continue
            report = summarize_validation(np.asarray(errors))
            print(
                f"{plane}: bias=({report.bias_xy_mm[0]:.2f},"
                f"{report.bias_xy_mm[1]:.2f}) norm={report.bias_norm_mm:.2f} mm "
                f"random={report.random_mm:.2f} mm "
                f"{'PASS' if report.passed else 'FAIL'}"
            )
            all_passed = all_passed and report.passed
    finally:
        robot.close(release=False)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
