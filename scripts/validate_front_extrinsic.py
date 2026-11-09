#!/usr/bin/env python3
"""Task-side validation of the front-camera extrinsic.

Detects the task features (coloured blocks, black/red line intersections),
maps them to base coordinates with the calibrated extrinsic, and optionally
moves the robot tip over each one so the picture shows whether it lands there.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.geometry import project_point_to_pixel
from dobot_rua.hardware.cameras import load_tip_length_mm
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.perception.blocks import detect_blocks
from dobot_rua.perception.lines import detect_line_intersection


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--plane-z-mm", type=float, default=1.0)
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--move", action="store_true", help="move the tip over each feature")
    p.add_argument("--hover-mm", type=float, default=30.0)
    p.add_argument(
        "--tip-length-mm",
        type=float,
        default=None,
        help="override the flange-to-suction-cup-lip distance in the camera config",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p


def _capture(device: str, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(12):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    if frame is None:
        raise SystemExit(f"no frame from {device}")
    return frame


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    front = config["cameras"]["front"]
    if not front.get("T_world_from_optical"):
        raise SystemExit("front extrinsic missing; run finish_front_extrinsic first")
    intrinsic = np.asarray(front["intrinsic_K"], dtype=float)
    transform = np.asarray(front["T_world_from_optical"], dtype=float)
    plane_z = args.plane_z_mm / 1000.0
    tip_length_mm = (
        float(args.tip_length_mm)
        if args.tip_length_mm is not None
        else load_tip_length_mm(args.camera_config)
    )
    print(f"front tip_length (mm): {tip_length_mm:.1f}")

    frame = _capture(front["device"], args.width, args.height)
    debug = frame.copy()
    targets: list[tuple[str, float, float]] = []

    for block in detect_blocks(frame, intrinsic, transform, plane_z):
        x_mm, y_mm = block.center_xyz_m[0] * 1000, block.center_xyz_m[1] * 1000
        u, v = block.center_uv
        targets.append((f"block {block.color} #{block.id}", x_mm, y_mm))
        print(f"block {block.color:6s} id={block.id} conf={block.confidence:.2f}  "
              f"pixel=({u:.0f},{v:.0f})  base=({x_mm:.1f},{y_mm:.1f}) mm")
        cv2.circle(debug, (int(u), int(v)), 12, (255, 0, 255), 2)
        cv2.putText(debug, f"{block.color} ({x_mm:.0f},{y_mm:.0f})", (int(u) + 14, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    for colour in ("black", "red"):
        hit = detect_line_intersection(frame, intrinsic, transform, plane_z, colour)
        if hit is None:
            print(f"{colour:5s} line intersection: not found")
            continue
        u, v = hit.intersection_uv
        x_mm, y_mm = hit.intersection_xyz_m[0] * 1000, hit.intersection_xyz_m[1] * 1000
        targets.append((f"{colour} intersection", x_mm, y_mm))
        print(f"{colour:5s} intersection: pixel=({u:.0f},{v:.0f})  base=({x_mm:.1f},{y_mm:.1f}) mm  "
              f"conf={hit.confidence:.2f}")
        cv2.drawMarker(debug, (int(u), int(v)), (0, 165, 255), cv2.MARKER_CROSS, 36, 2)
        cv2.putText(debug, f"{colour} ({x_mm:.0f},{y_mm:.0f})", (int(u) + 10, int(v) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    Path("audit").mkdir(exist_ok=True)
    cv2.imwrite("audit/validate_front_features.png", debug)
    print("wrote audit/validate_front_features.png")

    if not args.move or not targets:
        return 0

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    robot.set_speed(args.speed, args.speed)
    start = robot.read_state()
    try:
        for name, x_mm, y_mm in targets:
            u, v = project_point_to_pixel(
                intrinsic, transform, np.array([x_mm, y_mm, args.hover_mm]) / 1000.0
            )
            flange = np.array(
                [x_mm, y_mm, args.hover_mm + tip_length_mm], dtype=float
            )
            print(
                f"move tip over {name}: tip ({x_mm:.1f},{y_mm:.1f},"
                f"{args.hover_mm:.1f}) -> flange ({flange[0]:.1f},"
                f"{flange[1]:.1f},{flange[2]:.1f}) -> predicted pixel ({u:.0f},{v:.0f})"
            )
            robot.move_pose(
                float(flange[0]),
                float(flange[1]),
                float(flange[2]),
                0.0,
                timeout_s=25.0,
            )
            time.sleep(0.6)
            shot = _capture(front["device"], args.width, args.height)
            cv2.drawMarker(shot, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            slug = name.replace(" ", "_").replace("#", "")
            out = Path(f"audit/validate_{slug}.png")
            cv2.imwrite(str(out), shot)
            print(f"  wrote {out}")
    finally:
        robot.move_pose(start.x_mm, start.y_mm, start.z_mm, start.r_deg, timeout_s=25.0)
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
