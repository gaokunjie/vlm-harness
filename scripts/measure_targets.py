#!/usr/bin/env python3
"""Measure black_A / red_B line intersections on a front-camera frame.

If the front camera has K and T_world_from_optical, the intersections are
written in both pixels and world metres; otherwise only pixels are written.
The agent never reads this file; GT and offline checks do.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.hardware.cameras import load_camera_specs
from dobot_rua.perception.lines import detect_line_intersection


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, help="existing front frame; otherwise capture")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--targets", type=Path, default=Path("configs/targets.yaml"))
    p.add_argument("--device", default="/dev/video2")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p


def _capture(device: str, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(20):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    if frame is None:
        raise SystemExit("no frame")
    return frame


def main() -> int:
    args = parser().parse_args()
    image = cv2.imread(str(args.image)) if args.image else _capture(
        args.device, args.width, args.height
    )
    front = load_camera_specs(args.camera_config)["front"]
    if front.planar is None or front.intrinsic_K is None or front.T_world_from_optical is None:
        raise SystemExit(
            "front planar calibration is missing or invalid; run "
            "scripts/calibrate_front_planar.py first"
        )
    intrinsic = np.asarray(front.intrinsic_K, dtype=float)
    world_from_optical = np.asarray(front.T_world_from_optical, dtype=float)
    distortion = np.asarray(front.planar["distortion"], dtype=float).reshape(-1)
    calibration_hash = str(front.planar.get("calibration_hash", ""))
    task = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
    table_z_m = float(task.get("table_z_m", 0.0))

    data = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    for target_id, color in (("black_A", "black"), ("red_B", "red")):
        result = detect_line_intersection(
            image,
            intrinsic,
            world_from_optical,
            table_z_m,
            color,
            distortion=distortion,
        )
        if result is None:
            print(f"{target_id}: {color} intersection not found")
            continue
        entry = data["targets"][target_id]
        entry["intersection_px"] = [
            round(result.intersection_uv[0], 1),
            round(result.intersection_uv[1], 1),
        ]
        entry["measured_at"] = time.strftime("%Y-%m-%d")
        entry["intersection_gt_xy_m"] = [
            round(result.intersection_xyz_m[0], 6),
            round(result.intersection_xyz_m[1], 6),
        ]
        entry["calibration_hash"] = calibration_hash
        print(target_id, entry["intersection_px"], entry["intersection_gt_xy_m"])
    args.targets.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
