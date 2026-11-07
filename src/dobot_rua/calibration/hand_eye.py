"""Hand-eye calibration over 4x4 transforms.

``gripper_poses`` are ``T_base_from_tool``; ``target_poses`` are
``T_camera_from_board``. The result is ``T_tool_from_camera``.

OpenCV 5's headless wheel does not expose ``calibrateHandEye``, so the
AX = XB system is solved directly with SciPy least squares over the camera
extrinsic and the fixed board pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class HandEyeResult:
    transform: np.ndarray
    method: str


def _pose_from_params(params: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    transform[:3, 3] = params[3:6]
    return transform


def solve_hand_eye(
    gripper_poses: list[np.ndarray],
    target_poses: list[np.ndarray],
    method: str = "LEAST_SQUARES",
) -> HandEyeResult:
    if len(gripper_poses) != len(target_poses):
        raise ValueError("gripper and target pose counts differ")
    if len(gripper_poses) < 3:
        raise ValueError("at least three pose pairs are required")

    gripper = [np.asarray(pose, dtype=float) for pose in gripper_poses]
    target = [np.asarray(pose, dtype=float) for pose in target_poses]

    def residual(params: np.ndarray) -> np.ndarray:
        camera_to_tool = _pose_from_params(params[:6])
        board_in_base = _pose_from_params(params[6:])
        rows: list[np.ndarray] = []
        for base_from_tool, camera_from_board in zip(gripper, target, strict=True):
            predicted = base_from_tool @ camera_to_tool @ camera_from_board
            rows.append(predicted[:3, 3] - board_in_base[:3, 3])
            rows.append(
                Rotation.from_matrix(
                    predicted[:3, :3] @ board_in_base[:3, :3].T
                ).as_rotvec()
            )
        return np.concatenate(rows)

    def make_x0(translation, rotation):
        tool_from_camera = np.eye(4)
        tool_from_camera[:3, :3] = rotation.as_matrix()
        tool_from_camera[:3, 3] = translation
        board_in_base = gripper[0] @ tool_from_camera @ target[0]
        return np.concatenate(
            [
                Rotation.from_matrix(tool_from_camera[:3, :3]).as_rotvec(),
                tool_from_camera[:3, 3],
                Rotation.from_matrix(board_in_base[:3, :3]).as_rotvec(),
                board_in_base[:3, 3],
            ]
        )

    rotations = [
        Rotation.identity(),
        Rotation.from_euler("x", 90, degrees=True),
        Rotation.from_euler("x", -90, degrees=True),
        Rotation.from_euler("y", 90, degrees=True),
        Rotation.from_euler("z", 90, degrees=True),
        Rotation.from_euler("x", 180, degrees=True),
    ]
    translations = [
        (0.0, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (-0.10, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, -0.10, 0.0),
        (0.0, 0.0, 0.10),
        (0.0, 0.0, -0.10),
    ]
    best = None
    for rotation in rotations:
        for translation in translations:
            candidate = least_squares(
                residual, make_x0(translation, rotation), method="lm", max_nfev=3000
            )
            if best is None or candidate.cost < best.cost:
                best = candidate
    result = best
    return HandEyeResult(
        transform=_pose_from_params(result.x[:6]),
        method=method,
    )
