#!/usr/bin/env python3
"""Finish the front-camera extrinsic from a level table board plus touches.

The board lies flat on the table: its measured normal gives the camera tilt and
its distance gives the camera height. Two touched marker centres then fix the
camera yaw and horizontal position.

Sub-commands:
  prepare   capture the board, write audit/touch_session.json, annotate targets
  record    read the robot tip pose and record it against one marker id
  solve     fit the camera pose, validate it, write configs/cameras.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.level_board import camera_height_mm, solve_from_touches
from dobot_rua.calibration.touch import (
    camera_board_pose,
    marker_center,
    reprojection_rms,
)
from dobot_rua.hardware.dobot import DobotRobot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "record", "solve"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--session", type=Path, default=Path("audit/touch_session.json"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--board-preset", choices=("wrist", "large"), default="wrist")
    p.add_argument("--marker", type=int, help="marker id touched (record)")
    p.add_argument("--board-z-mm", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p


def _board_preset(name: str) -> tuple[int, int, float, float]:
    if name == "wrist":
        return 5, 5, 10.0, 7.5
    return 7, 8, 25.0, 18.75


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


def _detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 11
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
    )


def _camera(config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    front = config["cameras"]["front"]
    return (
        np.asarray(front["intrinsic_K"], dtype=float),
        np.asarray(front.get("distortion") or np.zeros(5), dtype=float),
        front["device"],
    )


def prepare(args) -> int:
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    intrinsic, distortion, device = _camera(config)
    squares_x, squares_y, square_mm, marker_mm = _board_preset(args.board_preset)
    board = make_board(squares_x, squares_y, square_mm, marker_mm, "DICT_4X4_50")
    frame = _capture(device, args.width, args.height)
    corners, ids, _ = _detector().detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 4:
        print("board not detected; adjust the board or lighting")
        return 1
    pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
    if pose is None:
        print("solvePnP failed")
        return 1
    rms = reprojection_rms(board, corners, ids, pose, intrinsic, distortion)
    height = camera_height_mm(pose, args.board_z_mm)

    markers = {}
    pixels = {}
    for marker_id in ids.reshape(-1):
        marker_id = int(marker_id)
        centre = marker_center(board, marker_id)
        centre_cam = pose[:3, :3] @ centre + pose[:3, 3]
        markers[str(marker_id)] = {
            "camera_mm": (centre_cam * 1000.0).tolist(),
        }
        for corner_set, corner_id in zip(corners, ids.reshape(-1), strict=True):
            if int(corner_id) == marker_id:
                pixels[marker_id] = np.asarray(corner_set).reshape(4, 2).mean(axis=0)
                break

    # two touch targets: the most separated pair of detected markers
    ordered = sorted(pixels)
    target = max(
        ((a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]),
        key=lambda pair: float(np.linalg.norm(pixels[pair[0]] - pixels[pair[1]])),
    )

    debug = frame.copy()
    for index, marker_id in enumerate(ordered, start=1):
        centre = pixels[marker_id].astype(int)
        colour = (0, 0, 255) if marker_id in target else (0, 200, 0)
        cv2.circle(debug, tuple(centre), 10, colour, 2)
        cv2.putText(
            debug,
            f"{index} (id {marker_id})",
            (centre[0] + 12, centre[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
        )
    output = Path("audit/touch_targets.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), debug)

    session = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rotation_cam_board": pose[:3, :3].tolist(),
        "board_origin_camera_mm": (pose[:3, 3] * 1000.0).tolist(),
        "plane_distance_mm": float(
            np.linalg.norm(pose[:3, 3]) * 1000.0
        ),
        "camera_height_mm": height,
        "reprojection_rms_px": rms,
        "markers": markers,
        "touches": [],
    }
    args.session.parent.mkdir(parents=True, exist_ok=True)
    args.session.write_text(json.dumps(session, indent=1), encoding="utf-8")
    print(f"markers detected: {len(ordered)}  reprojection {rms:.2f} px")
    print(f"camera height from the level board: {height:.1f} mm")
    print(f"annotated image: {output}")
    print("touch these two markers (red circles), touching the board surface:")
    for marker_id in target:
        print(f"  marker id {marker_id}")
    print(f"wrote {args.session}")
    return 0


def record(args) -> int:
    session = json.loads(args.session.read_text(encoding="utf-8"))
    if args.marker is None:
        print("--marker is required")
        return 1
    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    state = robot.read_state()
    robot.close()
    session["touches"].append(
        {
            "marker": int(args.marker),
            "base_mm": [state.x_mm, state.y_mm, state.z_mm],
            "time": time.strftime("%H:%M:%S"),
        }
    )
    args.session.write_text(json.dumps(session, indent=1), encoding="utf-8")
    print(
        f"recorded marker {args.marker} at "
        f"({state.x_mm:.1f}, {state.y_mm:.1f}, {state.z_mm:.1f}) mm"
    )
    return 0


def solve(args) -> int:
    session = json.loads(args.session.read_text(encoding="utf-8"))
    touches = session["touches"]
    if len(touches) < 2:
        print("need at least two recorded touches")
        return 1
    base = np.array([t["base_mm"] for t in touches], dtype=float) / 1000.0
    camera = np.array(
        [session["markers"][str(t["marker"])]["camera_mm"] for t in touches],
        dtype=float,
    ) / 1000.0
    solution = solve_from_touches(
        np.asarray(session["rotation_cam_board"], dtype=float), base, camera
    )
    print(f"touch fit: RMS {solution.rms_mm:.2f} mm  per point {np.round(solution.residuals_mm, 2)}")
    print(f"camera yaw about vertical: {np.degrees(solution.yaw_rad):.1f} deg")
    print(
        f"camera position (mm): {np.round(solution.camera_position * 1000.0, 1)}"
    )
    print(
        f"height from board plane {session['camera_height_mm']:.1f} mm vs "
        f"from touches {solution.camera_position[2] * 1000.0:.1f} mm"
    )
    if solution.rms_mm > 8.0:
        print("touch residual too large; not saving")
        return 1

    rotation_base_cam = solution.rotation_cam_base.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_base_cam
    transform[:3, 3] = solution.camera_position
    print("T_world_from_optical =\n", np.array2string(transform, precision=6))

    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    front = config["cameras"]["front"]
    front["T_world_from_optical"] = transform.tolist()
    front["calibration_date"] = time.strftime("%Y-%m-%d")
    front["calibration_mode"] = "level_board_two_touch"
    front["touch_residual_mm"] = round(solution.rms_mm, 3)
    front["touch_points"] = len(touches)
    front["board_reprojection_px"] = round(session["reprojection_rms_px"], 3)
    front["camera_height_mm"] = round(session["camera_height_mm"], 2)
    args.camera_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"wrote {args.camera_config}")
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "record":
        return record(args)
    return solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
