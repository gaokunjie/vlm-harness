#!/usr/bin/env python3
"""Front/side extrinsics from a table ChArUco board plus robot touch points.

Procedure:
  1. Tape the large 7x8 / 25 mm board flat on the table.
  2. Move the arm out of the board view and press Enter; the script captures
     front/side board images and saves annotated marker-ID images under
     audit/touch/.
  3. Jog the suction tip to the center of each printed marker ID it lists and
     press Enter after each touch. The script records the robot pose.
  4. It fits T_base_from_board (Kabsch), composes T_world_from_optical for each
     fixed camera, and writes it to configs/cameras.yaml.

The arm is only read (pose); the operator jogs it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.touch import (
    camera_board_pose,
    camera_extrinsic,
    extreme_marker_ids,
    fit_board_from_touches,
    marker_center,
)
from dobot_rua.hardware.dobot import DobotRobot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", default="front,side")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--board-preset", choices=("large", "wrist"), default="large")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--touch-count", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--image-dir", type=Path, default=Path("audit/touch"))
    return p


def _camera_config(config: dict, role: str):
    camera = config["cameras"][role]
    if camera.get("intrinsic_K") is None:
        raise SystemExit(f"{role}: intrinsic_K missing; run intrinsics first")
    return (
        np.asarray(camera["intrinsic_K"], dtype=float),
        np.asarray(camera.get("distortion") or np.zeros(5), dtype=float),
        camera["device"],
    )


def _capture(device: str, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(15):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    if frame is None:
        raise SystemExit(f"no frame from {device}")
    return frame


def _annotate(frame, corners, ids, selected, output_path: Path) -> None:
    debug = frame.copy()
    try:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
    except Exception:
        pass
    selected_set = {int(value) for value in selected}
    if ids is not None:
        for corner_set, marker_id in zip(corners, ids.reshape(-1), strict=True):
            marker_id = int(marker_id)
            points = np.asarray(corner_set).reshape(4, 2)
            center = points.mean(axis=0).astype(int)
            color = (0, 0, 255) if marker_id in selected_set else (0, 255, 0)
            cv2.circle(debug, tuple(center), 7, color, -1)
            cv2.putText(
                debug,
                str(marker_id),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    if args.board_preset == "wrist":
        squares_x, squares_y, square_mm, marker_mm = 5, 5, 10.0, 7.5
    else:
        squares_x, squares_y, square_mm, marker_mm = 7, 8, 25.0, 18.75
    board = make_board(
        squares_x, squares_y, square_mm, marker_mm, args.dictionary
    )
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]

    print("Move the arm out of the board view (survey pose), then press Enter.")
    input()
    camera_poses: dict[str, np.ndarray] = {}
    for role in cameras:
        intrinsic, distortion, device = _camera_config(config, role)
        frame = _capture(device, args.width, args.height)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < 4:
            raise SystemExit(f"{role}: board not detected; adjust lighting/view")
        pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
        if pose is None:
            raise SystemExit(f"{role}: solvePnP failed")
        camera_poses[role] = pose
        print(f"{role}: detected {len(ids)} markers")

    selected = extreme_marker_ids(board, args.touch_count)
    board_centers = [marker_center(board, marker_id) for marker_id in selected]
    # Annotate with the selected IDs highlighted.
    for role in cameras:
        # Re-capture is unnecessary for annotation; use the last frame? Re-read
        # the saved board image is enough for the operator, so annotate from a
        # fresh capture would require another survey. Save IDs from the pose
        # stage is done below by re-detecting on the stored image.
        pass
    # Save a camera board image with IDs for the operator.
    for role in cameras:
        intrinsic, distortion, device = _camera_config(config, role)
        frame = _capture(device, args.width, args.height)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            _annotate(frame, corners, ids, selected, args.image_dir / f"{role}_board_ids.png")
    print("Touch these marker centers (red in audit/touch/*_board_ids.png):")
    for marker_id, center in zip(selected, board_centers, strict=True):
        print(
            f"  marker ID {marker_id}: board frame "
            f"({center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}) mm"
        )

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    base_points = []
    try:
        for marker_id, center in zip(selected, board_centers, strict=True):
            input(
                f"Jog the suction tip to the center of marker ID {marker_id}, "
                "then press Enter..."
            )
            state = robot.read_state()
            base_points.append(
                [state.x_mm / 1000.0, state.y_mm / 1000.0, state.z_mm / 1000.0]
            )
            print(
                f"  recorded base ({state.x_mm:.1f}, {state.y_mm:.1f}, "
                f"{state.z_mm:.1f}) mm"
            )
    finally:
        robot.close()

    fit = fit_board_from_touches(np.asarray(board_centers), np.asarray(base_points))
    print(
        f"touch fit: RMS {fit.rms_mm:.2f} mm, max {fit.max_mm:.2f} mm"
    )
    if fit.rms_mm > 5.0:
        print("WARNING: touch residual is high; board may have moved or touches are off")
        answer = input("Save anyway? [y/N] ").strip().lower()
        if answer != "y":
            return 1

    for role, camera_from_board in camera_poses.items():
        base_from_camera = camera_extrinsic(fit.transform, camera_from_board)
        print(f"{role} T_world_from_optical=\n", np.array2string(base_from_camera, precision=6))
        camera = config["cameras"][role]
        camera["T_world_from_optical"] = base_from_camera.tolist()
        camera["calibration_date"] = time.strftime("%Y-%m-%d")
        camera["calibration_mode"] = "touch_board"
        camera["touch_residual_mm"] = round(fit.rms_mm, 3)
        camera["touch_points"] = len(base_points)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
