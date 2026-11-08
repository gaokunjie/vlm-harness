#!/usr/bin/env python3
"""Fixed-camera eye-to-hand calibration with the board held by the robot.

The operator suction-grips the printed board; the fixed camera (front and/or
side) sees it. The robot moves through small poses, the camera detects the
board, and the script solves T_base_from_camera (eye-to-hand). No wrist
calibration and no manual touch points are involved.

Safety: the arm is raised/moved only inside a small box around the current
pose, every target is IK-checked, the board must be visible at the start, and
the arm returns to its original pose at the end.
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
from dobot_rua.calibration.hand_eye import solve_eye_to_hand
from dobot_rua.calibration.touch import camera_board_pose
from dobot_rua.hardware.dobot import DobotRobot
from dobot_rua.hardware.safety import SafetyLimits
from dobot_rua.kinematics import inverse_kinematics, load_fk, three_axis_frame


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", default="front")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--robot-port", default="/dev/dobot_follower")
    p.add_argument("--board-preset", choices=("wrist", "large"), default="wrist")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--speed", type=int, default=50)
    p.add_argument("--raised-z-mm", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--grasp",
        action="store_true",
        help="turn the suction on at the current pose and keep it on afterwards",
    )
    p.add_argument(
        "--dump",
        type=Path,
        default=Path("audit/eye_to_hand_last.json"),
        help="write sampled poses here for offline re-solving",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


MODELS = ("fk_j4plus", "fk_j4minus", "world_yaw")


def tool_frame(params, state, model: str) -> np.ndarray:
    """End-effector frame for one candidate rotation model.

    ``fk_j4plus`` / ``fk_j4minus``: the FK three-axis frame composed with the
    j4 spin about the tool axis (which direction j4 turns is a mechanical
    convention, so both signs are tried). ``world_yaw``: the controller's
    reported ``r`` is the tool yaw about the world vertical, composed with the
    FK pitch.
    """
    frame = three_axis_frame(
        params,
        state.j1_deg,
        state.j2_deg,
        state.j3_deg,
        state.x_mm,
        state.y_mm,
        state.z_mm,
    )
    if model == "world_yaw":
        pitch = -(
            float(state.j2_deg) + params.c2_deg + float(state.j3_deg) + params.c3_deg
        )
        rotation = Rotation.from_euler(
            "z", float(state.r_deg), degrees=True
        ) * Rotation.from_euler("y", pitch, degrees=True)
        result = np.eye(4)
        result[:3, :3] = rotation.as_matrix()
        result[:3, 3] = frame[:3, 3]
        return result
    spin_deg = float(state.r_deg) if model == "fk_j4plus" else -float(state.r_deg)
    spin = np.eye(4)
    spin[:3, :3] = Rotation.from_euler("z", spin_deg, degrees=True).as_matrix()
    return frame @ spin


def _offsets() -> list[tuple[float, float, float, float]]:
    # Relative to the raised pose. Large yaw (dy) and pitch (dz, dx) changes
    # are what make the AX = XB translation well-conditioned; a +/-15 mm box
    # only rotates the tool by a few degrees and the solve blows up.
    return [
        (0, 0, 0, 0),
        (0, 30, 0, 0),
        (0, 60, 0, 0),
        (-30, 60, 0, 0),
        (-30, 30, 0, 0),
        (-30, 0, 0, 0),
        (-30, -30, 0, 0),
        (-30, -60, 0, 0),
        (0, -60, 0, 0),
        (0, -30, 0, 0),
        (30, -30, 0, 0),
        (30, 0, 0, 0),
        (30, 30, 0, 0),
        (30, 60, 0, 0),
        (30, 60, 20, 0),
        (0, 0, 20, 0),
        (-30, 0, 20, 0),
        (30, 0, 20, 0),
        (0, 30, 20, 0),
        (0, -30, 20, 0),
        (0, 0, -20, 0),
        (0, 30, -20, 0),
        (0, -30, -20, 0),
        (0, 0, 20, 30),
        (0, 0, 20, -30),
        (0, 30, 0, 30),
        (0, -30, 0, -30),
    ]


def _capture(device: str, width: int, height: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(10):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    cap.release()
    return frame


def main() -> int:
    args = parser().parse_args()
    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]
    camera_info = {}
    for role in cameras:
        camera = config["cameras"][role]
        if camera.get("intrinsic_K") is None:
            raise SystemExit(f"{role}: intrinsic_K missing")
        camera_info[role] = (
            np.asarray(camera["intrinsic_K"], dtype=float),
            np.asarray(camera.get("distortion") or np.zeros(5), dtype=float),
            camera["device"],
        )

    if args.board_preset == "wrist":
        squares_x, squares_y, square_mm, marker_mm = 5, 5, 10.0, 7.5
    else:
        squares_x, squares_y, square_mm, marker_mm = 7, 8, 25.0, 18.75
    board = make_board(squares_x, squares_y, square_mm, marker_mm, args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    robot = DobotRobot(port=args.robot_port)
    robot.connect()
    start = robot.read_state()
    print(
        f"start pose x={start.x_mm:.2f} y={start.y_mm:.2f} "
        f"z={start.z_mm:.2f} r={start.r_deg:.2f}"
    )

    def check_visible() -> list[str]:
        """Capture every camera and return one problem string per failure."""
        problems: list[str] = []
        for role, (intrinsic, distortion, device) in camera_info.items():
            frame = _capture(device, args.width, args.height)
            if frame is None:
                problems.append(f"{role}: no frame")
                continue
            corners, ids, _ = detector.detectMarkers(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )
            if ids is None or len(ids) < 4:
                problems.append(f"{role}: held board not detected; adjust the pose")
                continue
            pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
            if pose is None:
                problems.append(f"{role}: solvePnP failed on the held board")
                continue
            print(f"{role}: detected {len(ids)} markers on the held board")
        return problems

    raised_z = max(start.z_mm + args.raised_z_mm, 10.0)
    raised = (start.x_mm, start.y_mm, raised_z, start.r_deg)
    targets = [
        (raised[0] + dx, raised[1] + dy, raised[2] + dz, raised[3] + dr)
        for dx, dy, dz, dr in _offsets()
    ]
    params = load_fk("configs/fk_magician.yaml")
    limits = SafetyLimits.from_yaml("configs/robot_limits.yaml").joint_limits_deg

    def safe(target):
        x, y, z, _ = target
        if not (-300.0 <= x <= 300.0 and -300.0 <= y <= 300.0 and -30.0 <= z <= 300.0):
            return False
        return bool(inverse_kinematics(params, x, y, z, limits, margin_deg=2.0))

    if args.dry_run:
        print(f"raised pose {raised}")
        for target in targets:
            print(target, "safe" if safe(target) else "SKIP")
        robot.close(release=not args.grasp)
        return 0

    if args.grasp:
        robot.set_suction(True)
        print("suction on; waiting for vacuum")
        time.sleep(1.5)

    robot.set_speed(args.speed, args.speed)
    observations: list[dict[str, object]] = []
    current = start
    try:
        print("raising to safe z...")
        robot.move_pose(*raised, timeout_s=25.0)
        time.sleep(1.0)
        current = robot.read_state()
        problems = check_visible()
        if problems:
            for problem in problems:
                print(problem)
            raise RuntimeError("held board is not visible after raising; no sweep run")
        print("held board visible; starting the pose sweep")
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
            if step > 85.0:
                print(f"skip large step {step:.1f} mm")
                continue
            print(f"[{index + 1}/{len(targets)}] move {target}")
            robot.move_pose(*target, timeout_s=25.0)
            time.sleep(0.7)
            state = robot.read_state()
            off_target = float(
                np.linalg.norm(
                    np.array([state.x_mm, state.y_mm, state.z_mm])
                    - np.array(target[:3], dtype=float)
                )
            )
            if off_target > 3.0:
                print(f"  did not reach target ({off_target:.1f} mm off); resyncing")
                current = state
                continue
            frame = _capture(camera_info[cameras[0]][2], args.width, args.height)
            settled = robot.read_state()
            if (
                np.linalg.norm(
                    np.array([settled.x_mm, settled.y_mm, settled.z_mm])
                    - np.array([state.x_mm, state.y_mm, state.z_mm])
                )
                > 0.6
            ):
                print("  arm still moving during capture; skipping sample")
                current = settled
                continue
            row: dict[str, object] = {"state": state, "cameras": {}}
            visible = True
            for role, (intrinsic, distortion, device) in camera_info.items():
                if role != cameras[0]:
                    frame = _capture(device, args.width, args.height)
                if frame is None:
                    visible = False
                    break
                corners, ids, _ = detector.detectMarkers(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
                if ids is None or len(ids) < 4:
                    visible = False
                    break
                pose = camera_board_pose(board, corners, ids, intrinsic, distortion)
                if pose is None:
                    visible = False
                    break
                row["cameras"][role] = (pose, len(ids))  # type: ignore[index]
            if not visible:
                print("  board lost; returning to raised pose")
                robot.move_pose(*raised, timeout_s=25.0)
                current = robot.read_state()
                time.sleep(0.5)
                continue
            observations.append(row)
            print(
                f"  accepted pose=({state.x_mm:.1f},{state.y_mm:.1f},"
                f"{state.z_mm:.1f},{state.r_deg:.1f}) "
                + ", ".join(
                    f"{role}:{data[1]} markers"  # type: ignore[index]
                    for role, data in row["cameras"].items()  # type: ignore[union-attr]
                )
            )
            robot.move_pose(*raised, timeout_s=25.0)
            current = robot.read_state()
            time.sleep(0.5)
            if len(observations) >= args.samples:
                break
    except RuntimeError as error:
        print(f"aborted: {error}")
    finally:
        try:
            robot.move_pose(start.x_mm, start.y_mm, start.z_mm, start.r_deg, timeout_s=25.0)
        except Exception as error:  # noqa: BLE001
            print(f"return-to-start failed: {error}")
        robot.close(release=not args.grasp)

    if len(observations) < 3:
        print("not enough valid observations")
        return 1

    if args.dump is not None:
        payload = {
            "cameras": cameras,
            "board_preset": args.board_preset,
            "states": [
                {
                    "x": row["state"].x_mm,  # type: ignore[union-attr]
                    "y": row["state"].y_mm,  # type: ignore[union-attr]
                    "z": row["state"].z_mm,  # type: ignore[union-attr]
                    "r": row["state"].r_deg,  # type: ignore[union-attr]
                    "j1": row["state"].j1_deg,  # type: ignore[union-attr]
                    "j2": row["state"].j2_deg,  # type: ignore[union-attr]
                    "j3": row["state"].j3_deg,  # type: ignore[union-attr]
                    "j4": row["state"].j4_deg,  # type: ignore[union-attr]
                }
                for row in observations
            ],
            "camera_from_board": {
                role: [row["cameras"][role][0].tolist() for row in observations]  # type: ignore[index]
                for role in cameras
            },
        }
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {args.dump}")

    saved = False
    for role in cameras:
        target_poses = [row["cameras"][role][0] for row in observations]  # type: ignore[index]
        best = None
        for model in MODELS:
            base_poses = [
                tool_frame(params, row["state"], model)  # type: ignore[arg-type]
                for row in observations
            ]
            eye_to_hand = solve_eye_to_hand(base_poses, target_poses)
            estimates = list(eye_to_hand.estimates)
            translations = np.array([estimate[:3, 3] for estimate in estimates])
            rotations = np.array(
                [
                    Rotation.from_matrix(estimate[:3, :3]).as_rotvec()
                    for estimate in estimates
                ]
            )
            translation_spread = float(
                np.linalg.norm(np.std(translations, axis=0)) * 1000.0
            )
            rotation_spread = float(
                np.degrees(np.linalg.norm(np.std(rotations, axis=0)))
            )
            print(
                f"[{role}] {model}: translation spread "
                f"{translation_spread:.2f} mm, rotation spread {rotation_spread:.2f} deg"
            )
            score = translation_spread + 10.0 * rotation_spread
            if best is None or score < best[0]:
                best = (
                    score,
                    model,
                    eye_to_hand,
                    translation_spread,
                    rotation_spread,
                )
        assert best is not None
        _, model, eye_to_hand, translation_spread, rotation_spread = best
        base_from_camera = eye_to_hand.base_from_camera
        print(f"[{role}] chose model {model}")
        print(
            f"[{role}] T_tool_from_board=\n",
            np.array2string(eye_to_hand.tool_from_board, precision=6),
        )
        print(
            f"[{role}] eye-to-hand spread: translation {translation_spread:.2f} mm, "
            f"rotation {rotation_spread:.2f} deg"
        )
        if not np.isfinite(base_from_camera).all() or np.linalg.norm(
            base_from_camera[:3, 3]
        ) > 2.0:
            print(f"[{role}] implausible T_base_from_camera translation; not saving")
            continue
        if translation_spread > 20.0 or rotation_spread > 5.0:
            print(f"[{role}] spread too large; not saving")
            continue
        print(f"[{role}] T_world_from_optical=\n", np.array2string(base_from_camera, precision=6))
        camera = config["cameras"][role]
        camera["T_world_from_optical"] = base_from_camera.tolist()
        camera["calibration_date"] = time.strftime("%Y-%m-%d")
        camera["calibration_mode"] = "eye_to_hand_held_board"
        camera["eye_to_hand_spread_mm"] = round(translation_spread, 3)
        camera["eye_to_hand_spread_deg"] = round(rotation_spread, 3)
        camera["eye_to_hand_samples"] = len(observations)
        camera["eye_to_hand_model"] = model
        saved = True
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
