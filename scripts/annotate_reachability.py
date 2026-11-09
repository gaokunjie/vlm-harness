#!/usr/bin/env python3
"""Annotate the board with each marker's base position and IK reachability.

Uses the level-board + touched-point solution in audit/touch_session.json; also
writes the resulting camera extrinsic into configs/cameras.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.level_board import solve_from_touches
from dobot_rua.hardware.cameras import load_tip_length_mm
from dobot_rua.calibration.touch import camera_board_pose, marker_center
from dobot_rua.kinematics import inverse_kinematics, load_fk, load_limits
from dobot_rua.calibration.touch import reprojection_rms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, default=Path("audit/touch_session.json"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--board-preset", choices=("wrist", "large"), default="large")
    p.add_argument("--z-mm", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("audit/reachability.png"))
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    session = json.loads(args.session.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    front = config["cameras"]["front"]
    intrinsic = np.asarray(front["intrinsic_K"], dtype=float)
    distortion = np.asarray(front.get("distortion") or np.zeros(5), dtype=float)

    base = np.array([t["base_mm"] for t in session["touches"]], dtype=float) / 1000.0
    tip_length_m = load_tip_length_mm(args.camera_config) / 1000.0
    base[:, 2] -= tip_length_m
    camera = np.array(
        [session["markers"][str(t["marker"])]["camera_mm"] for t in session["touches"]],
        dtype=float,
    ) / 1000.0
    solution = solve_from_touches(
        np.asarray(session["rotation_cam_board"], dtype=float),
        base,
        camera,
    )
    transform = np.eye(4)
    transform[:3, :3] = solution.rotation_cam_base.T
    transform[:3, 3] = solution.camera_position
    print(f"camera (mm) {np.round(solution.camera_position * 1000, 1)}  "
          f"yaw {np.degrees(solution.yaw_rad):.1f} deg  touch rms {solution.rms_mm:.1f} mm")

    squares = (5, 5, 10.0, 7.5) if args.board_preset == "wrist" else (7, 8, 25.0, 18.75)
    board = make_board(*squares, "DICT_4X4_50")
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 11
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
    )
    cap = cv2.VideoCapture(front["device"], cv2.CAP_V4L2)
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
        print("no frame")
        return 1
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        print("board not detected")
        return 1
    pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
    rms = reprojection_rms(board, corners, ids, pose, intrinsic, distortion)
    print(f"board reprojection {rms:.2f} px over {len(ids)} markers")

    params = load_fk("configs/fk_magician.yaml")
    limits = load_limits("configs/robot_limits.yaml")
    debug = frame.copy()
    reachable_count = 0
    for corner_set, marker_id in zip(corners, ids.reshape(-1), strict=True):
        marker_id = int(marker_id)
        centre = marker_center(board, marker_id)
        centre_cam = pose[:3, :3] @ centre + pose[:3, 3]
        centre_base = transform[:3, :3] @ centre_cam + transform[:3, 3]
        base_mm = centre_base * 1000.0
        reachable = bool(
            inverse_kinematics(
                params, base_mm[0], base_mm[1], args.z_mm, limits, margin_deg=2.0
            )
        )
        reachable_count += int(reachable)
        pixel = np.asarray(corner_set).reshape(4, 2).mean(axis=0).astype(int)
        colour = (0, 200, 0) if reachable else (128, 128, 128)
        cv2.circle(debug, tuple(pixel), 12, colour, 2)
        label = f"id{marker_id} ({base_mm[0]:.0f},{base_mm[1]:.0f})"
        cv2.putText(debug, label, (pixel[0] + 12, pixel[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.imwrite(str(args.output), debug)
    print(f"reachable markers at z={args.z_mm} mm: {reachable_count}/{len(ids)}")
    print(f"wrote {args.output}")

    determinant = float(np.linalg.det(transform[:3, :3]))
    save_ok = solution.rms_mm <= 3.0 and determinant > 0.0
    if args.save and not save_ok:
        print(
            f"touch rms {solution.rms_mm:.1f} mm / det {determinant:+.3f}; "
            "refusing to save. Use finish_front_extrinsic.py solve with the TCP "
            "model instead."
        )
    if args.save and save_ok:
        front["T_world_from_optical"] = transform.tolist()
        front["calibration_date"] = time.strftime("%Y-%m-%d")
        front["calibration_mode"] = "level_board_vertical_tip"
        front["touch_residual_mm"] = round(solution.rms_mm, 3)
        front["touch_points"] = len(session["touches"])
        front["board_reprojection_px"] = round(rms, 3)
        front["camera_height_mm"] = round(session["camera_height_mm"], 2)
        args.camera_config.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"wrote {args.camera_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
