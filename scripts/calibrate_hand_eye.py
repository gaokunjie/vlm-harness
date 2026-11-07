#!/usr/bin/env python3
"""Hand-eye calibration with the 8x7 / 25 mm ChArUco board.

Modes:
  wrist  board fixed in the world, wrist camera moves with the arm.
         Saves cameras.wrist.T_tool_from_optical (camera -> tool).
  fixed  board attached to the robot end effector, front/side camera fixed.
         Saves cameras.<role>.T_world_from_optical.

The robot is only read; the operator moves it to each pose and presses Enter.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.charuco import detect_board, make_board, object_points_for
from dobot_rua.calibration.hand_eye import solve_hand_eye
from dobot_rua.geometry import pose_to_matrix
from dobot_rua.hardware.dobot import DobotRobot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("wrist", "fixed"), required=True)
    p.add_argument("--camera", default="wrist")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.75)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p


def _tool_quaternion(r_deg: float) -> tuple[float, float, float, float]:
    rotation = Rotation.from_euler("x", 180.0, degrees=True) * Rotation.from_euler(
        "z", float(r_deg), degrees=True
    )
    x, y, z, w = rotation.as_quat()
    return (float(w), float(x), float(y), float(z))


def _camera_data(config: dict, role: str):
    camera = config["cameras"][role]
    intrinsic = camera.get("intrinsic_K")
    if intrinsic is None:
        raise SystemExit(f"camera {role} has no intrinsic_K; run calibrate_intrinsics.py first")
    return camera, np.asarray(intrinsic, dtype=float), np.asarray(
        camera.get("distortion") or np.zeros(5), dtype=float
    )


def _solve_pnp(board, corners, ids, intrinsic, distortion):
    points = object_points_for(board, ids).astype(np.float32)
    image_points = corners.astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        points, image_points, intrinsic, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            points, image_points, intrinsic, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return transform


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translation = np.mean([item[:3, 3] for item in transforms], axis=0)
    matrix = sum(item[:3, :3] for item in transforms)
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    camera, intrinsic, distortion = _camera_data(config, args.camera)
    board = make_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
    )
    cap = cv2.VideoCapture(camera["device"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {camera['device']}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    gripper_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    print("Move the arm to a pose where the board is visible, then press Enter.")
    try:
        while len(gripper_poses) < args.samples:
            input(f"[{len(gripper_poses) + 1}/{args.samples}] pose ready? Enter...")
            corners, ids = None, None
            for _ in range(10):
                ok, frame = cap.read()
                if not ok:
                    continue
                corners, ids = detect_board(board, frame)
                if corners is not None and len(ids) >= 8:
                    break
            if corners is None or len(ids) < 8:
                print("board not detected; adjust the pose and retry")
                continue
            target = _solve_pnp(board, corners, ids, intrinsic, distortion)
            if target is None:
                print("solvePnP failed; adjust the pose and retry")
                continue
            state = robot.read_state()
            base_from_tool = pose_to_matrix(
                (state.x_mm / 1000.0, state.y_mm / 1000.0, state.z_mm / 1000.0),
                _tool_quaternion(state.r_deg),
            )
            gripper_poses.append(base_from_tool)
            target_poses.append(target)
            print(f"accepted pose {len(gripper_poses)}/{args.samples}")
    finally:
        cap.release()
        robot.close()

    if len(gripper_poses) < 3:
        print("not enough valid poses")
        return 1
    result = solve_hand_eye(gripper_poses, target_poses)
    transform = result.transform
    if args.mode == "wrist":
        camera["T_tool_from_optical"] = transform.tolist()
        camera["hand_eye_date"] = time.strftime("%Y-%m-%d")
        print("saved T_tool_from_optical")
    else:
        estimates = [gripper @ transform for gripper in gripper_poses]
        fixed = _average_transforms(estimates)
        camera["T_world_from_optical"] = fixed.tolist()
        camera["hand_eye_date"] = time.strftime("%Y-%m-%d")
        print("saved T_world_from_optical")
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output} [{args.camera}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
