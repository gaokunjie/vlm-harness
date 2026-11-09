#!/usr/bin/env python3
"""Two-plane PnP calibration for the front camera.

The suction cup holds a ChArUco board by its blank margin.  At each pose the
board pose is solved and the suction-cup center (``p_cup_board``) is projected
into the image.  Sampling two known tip heights makes the 3D points
non-coplanar, so a full ``T_world_from_optical`` is observable without rotating
the robot.  The result is written into ``configs/cameras.yaml``.

Run ``--dry-run`` first: it prints the planned poses without opening the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import (
    board_pose,
    cup_center_pixel,
    detect_board,
    make_board,
    object_points_for,
)
from dobot_rua.calibration.front_planar import (
    CalibrationRejected,
    PlanarSample,
    build_pose_plan,
    fit_two_plane,
    nearest_solution,
    save_planar_calibration,
)
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.kinematics import inverse_kinematics, load_fk, load_limits


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/dobot_follower")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--task-config", type=Path, default=Path("configs/task.yaml"))
    p.add_argument("--robot-config", type=Path, default=Path("configs/robot_limits.yaml"))
    p.add_argument("--fk-config", type=Path, default=Path("configs/fk_magician.yaml"))
    p.add_argument("--board-preset", choices=("5x5_10mm", "7x8_25mm"), default="5x5_10mm")
    p.add_argument("--center-mm", nargs=2, type=float, default=(150.0, 0.0))
    p.add_argument("--spacing-mm", type=float, default=20.0)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--samples-per-plane", type=int, default=None)
    p.add_argument("--holdout", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=25.0)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _board_for_preset(preset: str):
    if preset == "5x5_10mm":
        return make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    return make_board(7, 8, 25.0, 18.75, "DICT_5X5_1000")


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


def _board_reprojection_px(
    transform: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    board,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> float:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    object_points = object_points_for(board, ids).astype(np.float64)
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    projected, _ = cv2.projectPoints(
        object_points, rvec, transform[:3, 3], intrinsic_k, distortion
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return float(np.mean(errors))


def _max_normal_spread_deg(poses: list[np.ndarray]) -> float:
    if len(poses) < 2:
        return 0.0
    normals = np.array([pose[:3, 2] for pose in poses])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    max_angle = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cosine = float(np.clip(normals[i] @ normals[j], -1.0, 1.0))
            max_angle = max(max_angle, math.degrees(math.acos(cosine)))
    return max_angle


def _cols_rows(args: argparse.Namespace) -> tuple[int, int]:
    if args.samples_per_plane is None:
        return args.cols, args.rows
    cols = max(1, int(round(math.sqrt(args.samples_per_plane))))
    rows = max(1, math.ceil(args.samples_per_plane / cols))
    return cols, rows


def main() -> int:
    args = parser().parse_args()
    task = yaml.safe_load(args.task_config.read_text(encoding="utf-8")) or {}
    camera_config = yaml.safe_load(
        args.camera_config.read_text(encoding="utf-8")
    ) or {}
    front = (camera_config.get("cameras") or {}).get("front") or {}
    robot_config = yaml.safe_load(
        args.robot_config.read_text(encoding="utf-8")
    ) or {}

    table_z_m = float(task.get("table_z_m", 0.0))
    block_height_m = float(
        task.get("block_height_m", task.get("block_size_m", 0.025))
    )
    board_thickness_m = float(task.get("board_thickness_m", 0.002))
    p_cup = tuple(
        float(value) for value in front.get("cup_center_in_board_mm", [25.0, 75.0, 0.0])
    )
    tip_length_mm = float(front.get("tip_length_mm", 0.0))
    if tip_length_mm <= 0.0:
        print("camera config has no tip_length_mm; measure TCP before calibrating")
        return 2

    cols, rows = _cols_rows(args)
    plan = build_pose_plan(
        center_xy_mm=(args.center_mm[0], args.center_mm[1]),
        spacing_mm=args.spacing_mm,
        cols=cols,
        rows=rows,
        table_z_mm=table_z_m * 1000.0,
        block_height_mm=block_height_m * 1000.0,
        board_thickness_mm=board_thickness_m * 1000.0,
    )
    if args.dry_run:
        for pose in plan:
            print(
                f"{pose.plane:5s} x={pose.x_mm:8.2f} y={pose.y_mm:8.2f} "
                f"tip_z={pose.z_mm:8.2f} mm"
            )
        print(f"planned poses: {len(plan)}")
        return 0

    intrinsic = np.asarray(front["intrinsic_K"], dtype=float)
    distortion = np.asarray(front["distortion"], dtype=float).reshape(-1)
    device = str(front["device"])
    capture = (camera_config.get("capture") or {})
    width, height = capture.get("default_size", [1280, 720])
    board = _board_for_preset(args.board_preset)
    fk = load_fk(args.fk_config)
    limits = load_limits(args.robot_config)
    home_deg = tuple(float(value) for value in robot_config["home_deg"])
    j4_offset = float(robot_config.get("post_home_j4_offset_deg", 88.0))

    print("operator: verify TCP (tip = flange - tip_length_mm) before continuing")
    robot = DobotRobot(port=args.port)
    robot.connect()
    samples: list[PlanarSample] = []
    board_poses: list[np.ndarray] = []
    try:
        robot.home(
            home_deg=home_deg,
            post_home_j4_offset_deg=j4_offset,
            timeout_s=float(robot_config.get("home_timeout_s", 120.0)),
        )
        for pose in plan:
            state = robot.read_state()
            current_joints = (state.j1_deg, state.j2_deg, state.j3_deg)
            solutions = inverse_kinematics(
                fk,
                pose.x_mm,
                pose.y_mm,
                pose.z_mm + tip_length_mm,  # IK/firmware use the flange frame
                limits,
                margin_deg=5.0,
            )
            solution = nearest_solution(solutions, current_joints)
            if solution is None:
                print(f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): unreachable")
                continue
            robot.move_joints(
                solution[0], solution[1], solution[2], j4_offset, timeout_s=args.timeout_s
            )
            state = robot.read_state()
            tip = np.array(
                [state.x_mm, state.y_mm, state.z_mm - tip_length_mm], dtype=float
            )
            if np.linalg.norm(tip - np.array([pose.x_mm, pose.y_mm, pose.z_mm])) > 2.0:
                print(f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): not settled")
                continue
            image = _capture_front(device, int(width), int(height))
            corners, ids = detect_board(board, image)
            cup = cup_center_pixel(
                corners,
                ids,
                board,
                intrinsic,
                distortion,
                cup_center_in_board_mm=p_cup,
            )
            if cup is None:
                print(f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): board not found")
                continue
            pixel, board_transform = cup
            reprojection = _board_reprojection_px(
                board_transform, corners, ids, board, intrinsic, distortion
            )
            samples.append(
                PlanarSample(
                    tip_xyz_m=(float(tip[0] / 1000.0), float(tip[1] / 1000.0), float(tip[2] / 1000.0)),
                    cup_pixel_uv=(float(pixel[0]), float(pixel[1])),
                    plane=pose.plane,
                    marker_count=int(len(ids)),
                    board_reprojection_px=reprojection,
                )
            )
            board_poses.append(board_transform)
            print(
                f"{pose.plane:5s} ({pose.x_mm:7.1f},{pose.y_mm:7.1f}) "
                f"markers={len(ids):2d} reproj={reprojection:.2f}px"
            )
    finally:
        robot.close(release=False)

    if len(samples) < 6:
        print(f"only {len(samples)} usable samples; need at least 6")
        return 1
    normal_spread = _max_normal_spread_deg(board_poses)
    print(f"board normal spread: {normal_spread:.2f} deg")
    camera_controls = {
        key: front[key]
        for key in (
            "focus_automatic_continuous",
            "focus_absolute",
            "auto_exposure",
            "exposure",
            "white_balance_automatic",
            "white_balance_temperature",
        )
        if key in front
    }
    try:
        calibration, diagnostics = fit_two_plane(
            samples,
            intrinsic,
            distortion,
            table_z_m,
            table_z_m + block_height_m,
            p_cup,
            block_height_m,
            board_thickness_m,
            camera_controls,
            holdout=args.holdout,
        )
    except CalibrationRejected as error:
        print(f"calibration rejected: {error}")
        return 1
    if normal_spread > 3.0:
        print(
            "calibration rejected: board normal moved >3 deg; use a rigid backing "
            "and an anti-rotation fixture"
        )
        return 1

    print(
        f"fit_rmse={diagnostics['fit_rmse_px']:.3f}px "
        f"holdout_max={diagnostics['holdout_max_px']:.3f}px "
        f"samples={diagnostics['usable_samples']}"
    )
    print(f"calibration_hash={calibration.calibration_hash}")
    if not args.no_save:
        save_planar_calibration(args.camera_config, "front", calibration)
        print(f"wrote {args.camera_config}")

    audit_dir = Path("audit")
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "front_planar_last.json").write_text(
        json.dumps(
            {
                "samples": [asdict(sample) for sample in samples],
                "diagnostics": diagnostics,
                "board_normal_spread_deg": normal_spread,
                "board_poses": [pose.tolist() for pose in board_poses],
                "calibration_hash": calibration.calibration_hash,
                "measured_at": time.strftime("%Y-%m-%d"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
