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
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.front_planar import (
    PlanarCalibration,
    camera_controls_from_config,
    compute_calibration_hash,
    save_planar_calibration,
)
from dobot_rua.calibration.touch import (
    camera_board_pose,
    camera_extrinsic,
    extreme_marker_ids,
    fit_board_from_touches,
    marker_center,
)
from dobot_rua.geometry import derive_homography
from dobot_rua.hardware.cameras import load_tip_length_mm
from dobot_rua.hardware.dobot import DobotRobot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", default="front,side")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument(
        "--robot-config", type=Path, default=Path("configs/robot_limits.yaml")
    )
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--board-preset", choices=("large", "wrist"), default="large")
    p.add_argument("--dictionary", default=None)
    p.add_argument("--touch-count", type=int, default=8)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument(
        "--reachable-columns",
        default=None,
        help="0-indexed board columns the arm can touch, e.g. 2-5 or 2,3,4,5",
    )
    p.add_argument(
        "--reachable-rows",
        default=None,
        help="0-indexed board rows the arm can touch, e.g. 1-5 or 2-6",
    )
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


def _parse_columns(spec: str) -> set[int]:
    columns: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            columns.update(range(int(low), int(high) + 1))
        else:
            columns.add(int(part))
    return columns


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
        dictionary_name = args.dictionary or "DICT_4X4_50"
    else:
        # The large board is printed with 7 columns and 8 rows (detected
        # layout for the current 25 mm / DICT_4X4_50 board).
        squares_x, squares_y, square_mm, marker_mm = 7, 8, 25.0, 18.75
        dictionary_name = args.dictionary or "DICT_4X4_50"
    board = make_board(
        squares_x, squares_y, square_mm, marker_mm, dictionary_name
    )
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]

    print(
        "Place the arm where enough markers are visible (partial occlusion is "
        "fine), then press Enter."
    )
    input()
    camera_poses: dict[str, np.ndarray] = {}
    visible_ids_by_camera: dict[str, set[int]] = {}
    for role in cameras:
        intrinsic, distortion, device = _camera_config(config, role)
        while True:
            frame = _capture(device, args.width, args.height)
            corners, ids, _ = detector.detectMarkers(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )
            count = 0 if ids is None else len(ids)
            if count >= 6:
                break
            input(
                f"{role}: only {count} markers visible; adjust the arm so at "
                "least 6 are visible, then press Enter..."
            )
        pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
        if pose is None:
            raise SystemExit(f"{role}: solvePnP failed")
        camera_poses[role] = pose
        visible_ids_by_camera[role] = {
            int(value) for value in np.asarray(ids).reshape(-1)
        }
        print(f"{role}: detected {len(ids)} markers (partial occlusion allowed)")

    touch_visible = visible_ids_by_camera.get("front")
    if touch_visible is None:
        touch_visible = next(iter(visible_ids_by_camera.values()))
    all_visible = [
        int(value)
        for value in np.asarray(board.getIds()).reshape(-1)
        if int(value) in touch_visible
    ]
    board_centers_map = {
        marker_id: marker_center(board, marker_id) for marker_id in all_visible
    }
    reachable_columns = _parse_columns(args.reachable_columns or "2-5")
    reachable_rows = _parse_columns(args.reachable_rows or "1-5")
    square_m = square_mm / 1000.0
    candidates = [
        marker_id
        for marker_id in all_visible
        if int(board_centers_map[marker_id][0] / square_m) in reachable_columns
        and int(board_centers_map[marker_id][1] / square_m) in reachable_rows
    ]
    if not candidates:
        raise SystemExit(
            "no visible markers in the reachable rows/columns; "
            f"columns={sorted(reachable_columns)} rows={sorted(reachable_rows)}"
        )
    centroid = np.mean(
        [board_centers_map[marker_id][:2] for marker_id in candidates], axis=0
    )
    seed = max(
        candidates,
        key=lambda marker_id: float(
            np.linalg.norm(board_centers_map[marker_id][:2] - centroid)
        ),
    )
    selected = [seed]
    while len(selected) < min(args.touch_count, len(candidates)):
        remaining = [
            marker_id for marker_id in candidates if marker_id not in selected
        ]
        best = max(
            remaining,
            key=lambda marker_id: min(
                float(
                    np.linalg.norm(
                        board_centers_map[marker_id][:2]
                        - board_centers_map[other][:2]
                    )
                )
                for other in selected
            ),
        )
        selected.append(best)
    selected = tuple(selected[: args.touch_count])
    board_centers = [board_centers_map[marker_id] for marker_id in selected]
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

    tip_length_mm = load_tip_length_mm(args.camera_config)
    if tip_length_mm <= 0.0:
        raise SystemExit("front tip_length_mm is missing; measure TCP first")
    robot_config = (
        yaml.safe_load(args.robot_config.read_text(encoding="utf-8")) or {}
    )
    robot = DobotRobot(
        port=args.robot_port,
        base_j1_scale=float(robot_config.get("base_j1_scale", 1.0)),
    )
    robot.connect()
    base_points = []
    try:
        for marker_id, center in zip(selected, board_centers, strict=True):
            input(
                f"Jog the suction tip to the center of marker ID {marker_id}, "
                "then press Enter..."
            )
            time.sleep(float(args.settle_s))
            state = robot.read_state()
            base_points.append(
                [
                    state.x_mm / 1000.0,
                    state.y_mm / 1000.0,
                    (state.z_mm - tip_length_mm) / 1000.0,
                ]
            )
            print(
                f"  recorded tip ({state.x_mm:.1f}, {state.y_mm:.1f}, "
                f"{state.z_mm - tip_length_mm:.1f}) mm"
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
    if "front" in camera_poses:
        task = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
        table_z_m = float(task.get("table_z_m", 0.0))
        block_height_m = float(
            task.get("block_height_m", task.get("block_size_m", 0.025))
        )
        board_thickness_m = float(task.get("board_thickness_m", 0.002))
        block_top_z_m = table_z_m + block_height_m
        intrinsic, distortion, _ = _camera_config(config, "front")
        front_t = camera_extrinsic(fit.transform, camera_poses["front"])
        h_table = derive_homography(front_t, intrinsic, table_z_m)
        h_top = derive_homography(front_t, intrinsic, block_top_z_m)
        # Region = metric board footprint expanded by 20 mm.
        width_m = squares_x * square_mm / 1000.0
        height_m = squares_y * square_mm / 1000.0
        board_corners = np.array(
            [[0, 0, 0], [width_m, 0, 0], [width_m, height_m, 0], [0, height_m, 0]],
            dtype=float,
        )
        world_corners = (
            fit.transform[:3, :3] @ board_corners.T
        ).T + fit.transform[:3, 3]
        polygon = world_corners[:, :2]
        centroid = polygon.mean(axis=0)
        directions = polygon - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        region = polygon + directions / np.maximum(norms, 1e-9) * 0.02
        status = (
            "VALID"
            if fit.rms_mm <= 2.0 and fit.max_mm <= 3.0
            else "PROVISIONAL"
        )
        calibration = PlanarCalibration(
            T_world_from_optical=front_t,
            H_table=h_table,
            H_top=h_top,
            intrinsic_K=np.asarray(intrinsic, dtype=float),
            distortion=np.asarray(distortion, dtype=float).reshape(-1),
            p_cup_board_mm=(0.0, 0.0, 0.0),
            table_z_m=table_z_m,
            block_top_z_m=block_top_z_m,
            block_height_m=block_height_m,
            board_thickness_m=board_thickness_m,
            calibrated_region=region,
            camera_controls=camera_controls_from_config(
                config["cameras"]["front"]
            ),
            fit_rmse_px=0.0,
            holdout_max_px=0.0,
            samples=len(base_points),
            calibration_hash="",
            measured_at=time.strftime("%Y-%m-%d"),
            calibration_status=status,
            calibration_mode="touch_board",
            touch_rms_mm=fit.rms_mm,
            touch_max_mm=fit.max_mm,
        )
        calibration = replace(
            calibration,
            calibration_hash=compute_calibration_hash(calibration),
        )
        save_planar_calibration(args.output, "front", calibration)
        print(
            f"front planar: status={status} touch_rms={fit.rms_mm:.2f} mm "
            f"touch_max={fit.max_mm:.2f} mm"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
