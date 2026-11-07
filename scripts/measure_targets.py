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

from dobot_rua.perception.lines import detect_line_intersection


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, help="existing front frame; otherwise capture")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
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
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    front = config["cameras"]["front"]
    intrinsic = front.get("intrinsic_K")
    world_from_optical = front.get("T_world_from_optical")
    if intrinsic is None or world_from_optical is None:
        intrinsic = np.array(
            [[800.0, 0.0, image.shape[1] / 2], [0.0, 800.0, image.shape[0] / 2], [0.0, 0.0, 1.0]]
        )
        world_from_optical = np.eye(4)
        world_from_optical[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        calibrated = False
    else:
        intrinsic = np.asarray(intrinsic, dtype=float)
        world_from_optical = np.asarray(world_from_optical, dtype=float)
        calibrated = True

    data = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    for target_id, color in (("black_A", "black"), ("red_B", "red")):
        result = detect_line_intersection(
            image, intrinsic, world_from_optical, 0.0, color
        )
        if result is None:
            print(f"{target_id}: {color} intersection not found")
            continue
        entry = data["targets"][target_id]
        entry["intersection_px"] = [round(result.intersection_uv[0], 1), round(result.intersection_uv[1], 1)]
        entry["measured_at"] = time.strftime("%Y-%m-%d")
        if calibrated:
            entry["intersection_gt_xy_m"] = [
                round(result.intersection_xyz_m[0], 6),
                round(result.intersection_xyz_m[1], 6),
            ]
            entry["calibration_hash"] = "front-calibrated"
        print(target_id, entry["intersection_px"], entry.get("intersection_gt_xy_m"))
    args.targets.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
