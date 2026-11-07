#!/usr/bin/env python3
"""Automatic wrist eye-in-hand calibration with small, supervised motions.

The ChArUco board must be fixed in the world in front of the wrist. The script
first raises the arm to a safe z, then samples a small box around that raised
pose (x/y +/-10 mm, z 0/-10 mm relative, yaw +/-10 deg). It returns to the
raised pose after every sample and always returns to the original pose at the
end. It aborts before any motion if the board is not visible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.hand_eye import solve_hand_eye
from dobot_rua.geometry import pose_to_matrix
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.hardware.safety import SafetyLimits
from dobot_rua.kinematics import inverse_kinematics, load_fk


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--speed", type=int, default=20)
    p.add_argument("--raised-z-mm", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.75)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    return p


def tool_quaternion(r_deg: float) -> tuple[float, float, float, float]:
    rotation = Rotation.from_euler("x", 180.0, degrees=True) * Rotation.from_euler(
        "z", float(r_deg), degrees=True
    )
    x, y, z, w = rotation.as_quat()
    return (float(w), float(x), float(y), float(z))


def _offsets() -> list[tuple[float, float, float, float]]:
    # Relative to the raised pose; z never goes below raised_z - 10 mm.
    return [
        (0, 0, 0, 0),
        (10, 0, 0, 0),
        (-10, 0, 0, 0),
        (0, 10, 0, 0),
        (0, -10, 0, 0),
        (10, 10, 0, 5),
        (-10, -10, 0, -5),
        (10, -10, 0, -5),
        (-10, 10, 0, 5),
        (0, 0, -10, 10),
        (0, 0, -10, -10),
        (10, 0, -10, 5),
        (-10, 0, -10, -5),
        (0, 10, -10, 5),
        (0, -10, -10, -5),
    ]


def _capture(cap: cv2.VideoCapture) -> np.ndarray | None:
    frame = None
    for _ in range(8):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    return frame


def _detect(
    board,
    detector,
    frame: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 3:
        return None
    board_ids = np.asarray(board.getIds()).reshape(-1)
    board_points = board.getObjPoints()
    index = {int(marker_id): i for i, marker_id in enumerate(board_ids)}
    object_points = []
    image_points = []
    for corner_set, marker_id in zip(corners, ids.reshape(-1), strict=True):
        index_in_board = index.get(int(marker_id))
        if index_in_board is None:
            continue
        object_points.append(
            np.asarray(board_points[index_in_board], dtype=np.float32)
        )
        image_points.append(np.asarray(corner_set).reshape(4, 2).astype(np.float32))
    if len(object_points) < 3:
        return None
    object_points = np.concatenate(object_points, axis=0)
    image_points = np.concatenate(image_points, axis=0)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, intrinsic, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return transform, len(object_points) // 4


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    wrist = config["cameras"]["wrist"]
    if wrist.get("intrinsic_K") is None:
        raise SystemExit("wrist intrinsics missing; run calibrate_intrinsics.py first")
    intrinsic = np.asarray(wrist["intrinsic_K"], dtype=float)
    distortion = np.asarray(wrist.get("distortion") or np.zeros(5), dtype=float)

    board = make_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    start = robot.read_state()
    print(
        f"start pose x={start.x_mm:.2f} y={start.y_mm:.2f} "
        f"z={start.z_mm:.2f} r={start.r_deg:.2f}"
    )
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        robot.close()
        raise SystemExit(f"could not open {args.device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    frame = _capture(cap)
    initial = None if frame is None else _detect(
        board, detector, frame, intrinsic, distortion
    )
    if initial is None:
        cap.release()
        robot.close()
        raise SystemExit("board not visible at the current pose; no motion sent")

    raised_z = max(start.z_mm + args.raised_z_mm, 10.0)
    raised = (start.x_mm, start.y_mm, raised_z, start.r_deg)
    targets = [
        (raised[0] + dx, raised[1] + dy, raised[2] + dz, raised[3] + dr)
        for dx, dy, dz, dr in _offsets()
    ]

    parameters = load_fk("configs/fk_magician.yaml")
    limits = SafetyLimits.from_yaml("configs/robot_limits.yaml").joint_limits_deg

    def safe(target):
        x, y, z, _ = target
        if not (-300.0 <= x <= 300.0 and -300.0 <= y <= 300.0 and -30.0 <= z <= 300.0):
            return False
        return bool(inverse_kinematics(parameters, x, y, z, limits, margin_deg=2.0))

    if args.dry_run:
        print(f"raised pose {raised}")
        for target in targets:
            print(target, "safe" if safe(target) else "SKIP")
        cap.release()
        robot.close()
        return 0

    robot.set_speed(args.speed, args.speed)
    gripper_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    current = start
    try:
        print("raising to safe z...")
        robot.move_pose(*raised)
        time.sleep(0.8)
        current = robot.read_state()
        for index, target in enumerate(targets):
            if not safe(target):
                print(f"skip unsafe target {target}")
                continue
            step = float(
                np.linalg.norm(
                    np.array(target[:3])
                    - np.array([current.x_mm, current.y_mm, current.z_mm])
                )
            )
            if step > 30.0:
                print(f"skip large step {step:.1f} mm")
                continue
            print(f"[{index + 1}/{len(targets)}] move {target}")
            robot.move_pose(*target)
            time.sleep(0.7)
            frame = _capture(cap)
            detected = None if frame is None else _detect(
                board, detector, frame, intrinsic, distortion
            )
            if detected is None:
                print("  board lost; returning to raised pose")
                robot.move_pose(*raised)
                current = robot.read_state()
                time.sleep(0.5)
                continue
            camera_from_board, markers = detected
            state = robot.read_state()
            base_from_tool = pose_to_matrix(
                (state.x_mm / 1000.0, state.y_mm / 1000.0, state.z_mm / 1000.0),
                tool_quaternion(state.r_deg),
            )
            gripper_poses.append(base_from_tool)
            target_poses.append(camera_from_board)
            print(
                f"  accepted markers={markers} "
                f"pose=({state.x_mm:.1f},{state.y_mm:.1f},{state.z_mm:.1f},{state.r_deg:.1f})"
            )
            robot.move_pose(*raised)
            current = robot.read_state()
            time.sleep(0.5)
            if len(gripper_poses) >= args.samples:
                break
    finally:
        try:
            robot.move_pose(start.x_mm, start.y_mm, start.z_mm, start.r_deg)
        except Exception as error:  # noqa: BLE001 - always release the session
            print(f"return-to-start failed: {error}")
        cap.release()
        robot.close()

    if len(gripper_poses) < 3:
        print("not enough valid samples")
        return 1
    result = solve_hand_eye(gripper_poses, target_poses)
    transform = result.transform
    estimates = [
        base @ transform @ camera
        for base, camera in zip(gripper_poses, target_poses, strict=True)
    ]
    translations = np.array([estimate[:3, 3] for estimate in estimates])
    rotations = np.array(
        [Rotation.from_matrix(estimate[:3, :3]).as_rotvec() for estimate in estimates]
    )
    translation_spread = float(np.linalg.norm(np.std(translations, axis=0)) * 1000.0)
    rotation_spread = float(np.degrees(np.linalg.norm(np.std(rotations, axis=0))))
    print("T_tool_from_optical=\n", np.array2string(transform, precision=6))
    print(f"board spread: translation std {translation_spread:.2f} mm, rotation std {rotation_spread:.2f} deg")
    if not np.isfinite(transform).all() or np.linalg.norm(transform[:3, 3]) > 1.0:
        print("hand-eye translation is not physically plausible; not saving")
        return 1
    if translation_spread > 15.0 or rotation_spread > 2.5:
        print("board spread is too large; not saving a weak hand-eye result")
        return 1
    wrist["T_tool_from_optical"] = transform.tolist()
    wrist["hand_eye_date"] = time.strftime("%Y-%m-%d")
    wrist["hand_eye_mode"] = "auto_wrist"
    args.camera_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.camera_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
