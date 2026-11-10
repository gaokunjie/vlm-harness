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
    cup_center_pixel_from_markers,
    detect_markers,
    make_board,
    marker_reprojection_px,
)
from dobot_rua.calibration.front_planar import (
    CalibrationRejected,
    PlanarSample,
    build_pose_plan,
    camera_controls_from_config,
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
    p.add_argument(
        "--min-markers",
        type=int,
        default=3,
        help="minimum decoded ArUco markers per sample (the arm tilts the board, "
        "so normal spread alone is not a rigidity test)",
    )
    p.add_argument("--max-normal-spread-deg", type=float, default=30.0)
    p.add_argument("--timeout-s", type=float, default=25.0)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-home",
        action="store_true",
        help="start from the current pose instead of homing (keeps j4 locked)",
    )
    p.add_argument(
        "--mask-rect",
        nargs=4,
        type=int,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="paint this image rectangle white before board detection",
    )
    p.add_argument("--settle-frames", type=int, default=3)
    p.add_argument("--settle-timeout-s", type=float, default=5.0)
    p.add_argument("--settle-translation-mm", type=float, default=1.5)
    p.add_argument("--settle-rotation-deg", type=float, default=1.2)
    p.add_argument("--settle-pixel", type=float, default=0.5)
    p.add_argument(
        "--fixed-wait-s",
        type=float,
        default=0.0,
        help="if >0, sleep this long after each move and take one frame (bypasses the stability gate)",
    )
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
    return marker_reprojection_px(
        board, corners, ids, transform, intrinsic_k, distortion
    )


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


def _rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _capture_stable_board(
    board,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
    device: str,
    width: int,
    height: int,
    mask_rect: tuple[int, int, int, int] | None,
    cup_center_in_board_mm: tuple[float, float, float] = (25.0, 75.0, 0.0),
    *,
    frames: int = 3,
    timeout_s: float = 5.0,
    translation_mm: float = 0.2,
    rotation_deg: float = 0.3,
    pixel: float = 0.5,
):
    """Capture until the board pose is stable for ``frames`` consecutive frames."""
    deadline = time.monotonic() + float(timeout_s)
    previous = None
    previous_cup = None
    stable = 0
    while time.monotonic() < deadline:
        image = _capture_front(device, int(width), int(height))
        if mask_rect:
            x0, y0, x1, y1 = mask_rect
            image[y0:y1, x0:x1] = 255
        corners, ids = detect_markers(board, image)
        if corners is None or ids is None:
            previous = None
            previous_cup = None
            stable = 0
            time.sleep(0.2)
            continue
        cup = cup_center_pixel_from_markers(
            board,
            corners,
            ids,
            intrinsic_k,
            distortion,
            cup_center_in_board_mm=cup_center_in_board_mm,
        )
        if cup is None:
            previous = None
            previous_cup = None
            stable = 0
            time.sleep(0.2)
            continue
        pixel_uv, transform = cup
        if previous is not None and previous_cup is not None:
            delta_t = float(
                np.linalg.norm(transform[:3, 3] - previous[:3, 3]) * 1000.0
            )
            delta_r = _rotation_delta_deg(previous, transform)
            delta_px = float(np.linalg.norm(np.asarray(pixel_uv) - previous_cup))
            if delta_t <= translation_mm and delta_r <= rotation_deg and delta_px <= pixel:
                stable += 1
                if stable >= frames:
                    return image, corners, ids, transform, pixel_uv
            else:
                stable = 0
        else:
            stable = 0
        previous = transform
        previous_cup = np.asarray(pixel_uv, dtype=float)
        time.sleep(0.2)
    return None


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
    robot = DobotRobot(
        port=args.port,
        base_j1_scale=float(robot_config.get("base_j1_scale", 1.0)),
    )
    robot.connect()
    samples: list[PlanarSample] = []
    board_poses: list[np.ndarray] = []
    try:
        if args.no_home:
            state = robot.read_state()
            if abs(state.j4_deg - j4_offset) > 1.0:
                print(
                    f"j4={state.j4_deg:.2f} is not at the locked offset "
                    f"{j4_offset:.2f}; home first or adjust j4"
                )
                return 2
        else:
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
            try:
                robot.move_joints(
                    solution[0],
                    solution[1],
                    solution[2],
                    j4_offset,
                    timeout_s=args.timeout_s,
                )
            except Exception as error:
                print(
                    f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): "
                    f"move failed: {error}"
                )
                continue
            state = robot.read_state()
            tip = np.array(
                [state.x_mm, state.y_mm, state.z_mm - tip_length_mm], dtype=float
            )
            if np.linalg.norm(tip - np.array([pose.x_mm, pose.y_mm, pose.z_mm])) > 2.0:
                print(f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): not settled")
                continue
            if args.fixed_wait_s > 0:
                time.sleep(float(args.fixed_wait_s))
                image = _capture_front(device, int(width), int(height))
                if args.mask_rect:
                    x0, y0, x1, y1 = args.mask_rect
                    image[y0:y1, x0:x1] = 255
                corners, ids = detect_markers(board, image)
                cup = (
                    cup_center_pixel_from_markers(
                        board,
                        corners,
                        ids,
                        intrinsic,
                        distortion,
                        cup_center_in_board_mm=p_cup,
                    )
                    if corners is not None and ids is not None
                    else None
                )
                if cup is None:
                    print(
                        f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): "
                        "board not found"
                    )
                    continue
                pixel, board_transform = cup
                stable = (image, corners, ids, board_transform, pixel)
            else:
                stable = _capture_stable_board(
                    board,
                    intrinsic,
                    distortion,
                    device,
                    int(width),
                    int(height),
                    tuple(args.mask_rect) if args.mask_rect else None,
                    p_cup,
                    frames=args.settle_frames,
                    timeout_s=args.settle_timeout_s,
                    translation_mm=args.settle_translation_mm,
                    rotation_deg=args.settle_rotation_deg,
                    pixel=args.settle_pixel,
                )
            if stable is None:
                print(
                    f"skip {pose.plane} ({pose.x_mm:.1f},{pose.y_mm:.1f}): "
                    "board did not settle"
                )
                continue
            image, corners, ids, board_transform, pixel = stable
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
    accepted_poses = [
        pose
        for pose, sample in zip(board_poses, samples)
        if sample.marker_count >= args.min_markers
    ]
    normal_spread = _max_normal_spread_deg(accepted_poses)
    print(f"board normal spread (markers>={args.min_markers}): {normal_spread:.2f} deg")
    try:
        camera_controls = camera_controls_from_config(front)
    except ValueError as error:
        print(f"calibration rejected: {error}")
        return 1
    audit_dir = Path("audit")
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "front_planar_last.json").write_text(
        json.dumps(
            {
                "samples": [asdict(sample) for sample in samples],
                "board_normal_spread_deg": normal_spread,
                "board_poses": [pose.tolist() for pose in board_poses],
                "min_markers": args.min_markers,
                "measured_at": time.strftime("%Y-%m-%d"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
            min_marker_count=args.min_markers,
        )
    except CalibrationRejected as error:
        print(f"calibration rejected: {error}")
        return 1
    if normal_spread > args.max_normal_spread_deg:
        print(
            f"calibration rejected: board normal moved "
            f"{normal_spread:.2f} deg > {args.max_normal_spread_deg:.2f} deg"
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
