#!/usr/bin/env python3
"""Calibrate the front camera from an ArUco marker on the end effector.

The marker is flat on the end-effector face and centred on the suction axis, so
its centre is the flange position plus a small constant offset. Sweeping the arm
through varied x/y/z and j1 poses and detecting the marker gives a dense,
human-error-free correspondence set.

    python scripts/calibrate_front_ee_marker.py            # sweep + solve
    python scripts/calibrate_front_ee_marker.py --save      # also write cameras.yaml
    python scripts/calibrate_front_ee_marker.py --replay    # solve saved samples
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.ee_marker import (
    solve_camera_from_ee_marker,
    solve_camera_from_marker_corners,
)
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.kinematics import inverse_kinematics, load_fk, load_limits, within_limits

# x/y/z in millimetres, j1 in degrees. z is high enough that the marker clears
# the table and stays visible to the tilted front camera.
POSES: list[tuple[float, float, float, float]] = [
    (140.0, 0.0, 10.0, 0.0),
    (170.0, 0.0, 0.0, 0.0),
    (170.0, 0.0, 25.0, 0.0),
    (200.0, 0.0, 0.0, 0.0),
    (200.0, 0.0, 30.0, 0.0),
    (230.0, 0.0, 10.0, 0.0),
    (160.0, -35.0, 20.0, 0.0),
    (200.0, -35.0, 20.0, 0.0),
    (230.0, -35.0, 30.0, 0.0),
    (150.0, -20.0, 35.0, 15.0),
    (210.0, -30.0, 15.0, -15.0),
    (180.0, 0.0, 40.0, 25.0),
    (180.0, 0.0, 40.0, -25.0),
    (220.0, 0.0, 35.0, 20.0),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--samples", type=Path, default=Path("audit/ee_marker_samples.json"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--speed", type=int, default=40)
    p.add_argument("--save", action="store_true")
    p.add_argument("--replay", action="store_true", help="solve the saved samples only")
    p.add_argument("--marker-height-mm", type=float, default=0.0)
    p.add_argument("--marker-side-mm", type=float, default=18.0)
    return p


def detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 7
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
    )


def capture(device: str) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(12):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    if frame is None:
        raise SystemExit(f"no frame from {device}")
    return frame


def largest_marker(
    frame: np.ndarray, det: cv2.aruco.ArucoDetector
) -> tuple[int, np.ndarray, float] | None:
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    best = None
    for corner_set, marker_id in zip(corners, ids.reshape(-1), strict=True):
        points = np.asarray(corner_set).reshape(4, 2)
        side = float(np.mean([np.linalg.norm(points[k] - points[(k + 1) % 4]) for k in range(4)]))
        if best is None or side > best[2]:
            best = (int(marker_id), points, side)
    return best


def joint_safe(params, limits, x: float, y: float, z: float, margin: float = 5.0) -> bool:
    """True if the pose has an IK solution at least ``margin`` inside the limits."""
    solutions = inverse_kinematics(params, x, y, z, limits)
    return any(within_limits(s, limits, margin_deg=margin) for s in solutions)


def sweep(args, config: dict) -> list[dict]:
    front = config["cameras"]["front"]
    det = detector()
    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    robot.set_speed(args.speed, args.speed)
    start = robot.read_state()
    samples: list[dict] = []
    params = load_fk("configs/fk_magician.yaml")
    limits = load_limits("configs/robot_limits.yaml")
    try:
        for x, y, z, j1 in POSES:
            if not joint_safe(params, limits, x, y, z):
                print(f"({x:6.1f},{y:6.1f},{z:6.1f}) rejected: outside the joint limits")
                continue
            robot.move_pose(x, y, z, j1, timeout_s=25.0)
            time.sleep(0.4)
            state = robot.read_state()
            if abs(state.x_mm - x) > 4.0 or abs(state.y_mm - y) > 4.0 or abs(state.z_mm - z) > 4.0:
                print(
                    f"({x:6.1f},{y:6.1f},{z:6.1f}) rejected: reported "
                    f"({state.x_mm:.1f},{state.y_mm:.1f},{state.z_mm:.1f})"
                )
                continue
            frame = capture(front["device"])
            found = largest_marker(frame, det)
            if found is None:
                print(f"({x:6.1f},{y:6.1f},{z:6.1f},j1={j1:5.1f}) marker not detected")
                continue
            marker_id, corners, side = found
            centre = corners.mean(axis=0)
            samples.append(
                {
                    "flange_mm": [state.x_mm, state.y_mm, state.z_mm],
                    "j4_deg": state.j4_deg,
                    "j1_deg": state.j1_deg,
                    "marker_id": marker_id,
                    "pixel": [float(centre[0]), float(centre[1])],
                    "corners": corners.tolist(),
                    "side_px": side,
                }
            )
            print(
                f"({x:6.1f},{y:6.1f},{z:6.1f},j1={j1:5.1f}) id {marker_id} "
                f"pixel ({centre[0]:7.1f},{centre[1]:6.1f}) side {side:6.1f} px"
            )
    finally:
        try:
            robot.move_pose(start.x_mm, start.y_mm, start.z_mm, start.r_deg, timeout_s=25.0)
        finally:
            robot.close(release=False)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.samples.write_text(json.dumps({"samples": samples}, indent=1), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {args.samples}")
    return samples


def solve(args, config: dict, samples: list[dict]) -> int:
    front = config["cameras"]["front"]
    intrinsic = np.asarray(front["intrinsic_K"], dtype=float)
    distortion = np.asarray(front.get("distortion") or np.zeros(5), dtype=float)
    corners = np.array([s["corners"] for s in samples], dtype=float)
    flange = np.array([s["flange_mm"] for s in samples], dtype=float)
    j4 = np.array([s["j4_deg"] for s in samples], dtype=float)
    j1 = np.array([s["j1_deg"] for s in samples], dtype=float)
    solution = solve_camera_from_marker_corners(
        corners, flange, j4, j1, args.marker_side_mm, intrinsic,
        marker_height_mm=args.marker_height_mm,
    )
    print(f"poses {len(samples)}  reprojection RMS {solution.rms_px:.2f} px  "
          f"max {solution.max_px:.2f} px")
    print(f"marker offset in the flange frame (mm): {np.round(solution.marker_offset_mm, 2)}")
    print(f"camera position (mm): {np.round(solution.camera_position_mm, 1)}")
    transform = np.eye(4)
    transform[:3, :3] = solution.rotation_world_optical
    transform[:3, 3] = solution.camera_position_mm / 1000.0
    print("T_world_from_optical =\n", np.array2string(transform, precision=6))

    if not args.save:
        return 0
    if solution.rms_px > 2.0:
        print("reprojection too large; not saving")
        return 1
    front["T_world_from_optical"] = transform.tolist()
    front["calibration_date"] = time.strftime("%Y-%m-%d")
    front["calibration_mode"] = "ee_marker_sweep"
    front["reprojection_rms_px"] = round(solution.rms_px, 3)
    front["reprojection_max_px"] = round(solution.max_px, 3)
    front["marker_offset_mm"] = [round(float(v), 3) for v in solution.marker_offset_mm]
    front["poses"] = len(samples)
    args.camera_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"wrote {args.camera_config}")
    return 0


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    if args.replay:
        samples = json.loads(args.samples.read_text(encoding="utf-8"))["samples"]
    else:
        samples = sweep(args, config)
    if not samples:
        print("no samples; nothing to solve")
        return 1
    return solve(args, config, samples)


if __name__ == "__main__":
    raise SystemExit(main())
