"""Camera pose from a board lying level on the table plus touch points.

A board on the table is level, so its measured normal gives the camera's tilt
and its distance gives the camera height directly. What the board cannot tell
us is its own yaw about the vertical, which is why the camera's yaw (and its
horizontal position) comes from two or more touched points:

    camera_point = R_cam_base @ (base_point - t_cam)
    R_cam_base   = R_cam_board @ Rz(-yaw)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LevelBoardSolution:
    rotation_cam_base: np.ndarray
    camera_position: np.ndarray
    yaw_rad: float
    rms_mm: float
    residuals_mm: np.ndarray


def camera_height_mm(pose_cam_board: np.ndarray, board_z_mm: float = 1.0) -> float:
    """Height of the camera above the table for a level board plane."""
    transform = np.asarray(pose_cam_board, dtype=float)
    centre = transform[:3, 3]
    normal = transform[:3, 2]
    normal = normal / np.linalg.norm(normal)
    return abs(float(normal @ centre)) * 1000.0 + float(board_z_mm)


def _rot_z(angle_rad: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, 0.0, float(angle_rad)]).as_matrix()


def solve_from_touches(
    rotation_cam_board: np.ndarray,
    base_points: np.ndarray,
    camera_points: np.ndarray,
    camera_z_mm: float | None = None,
) -> LevelBoardSolution:
    """Solve the camera pose from the level board and >= 2 touched points.

    ``base_points`` are the robot tip positions (m) at each touch, in the same
    order as the touched features' ``camera_points`` (m).
    """
    rotation_cam_board = np.asarray(rotation_cam_board, dtype=float)
    base = np.asarray(base_points, dtype=float)
    camera = np.asarray(camera_points, dtype=float)
    if base.shape != camera.shape or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base and camera points must be matching Nx3 arrays")
    if len(base) < 2:
        raise ValueError("at least two touch points are required")

    # The board's own +z points away from the camera, i.e. down in the base
    # frame, so its normal enters the model with a flipped sign.
    flip = np.diag([1.0, 1.0, -1.0])
    v = np.array([flip @ (rotation_cam_board.T @ point) for point in camera])

    def cost(yaw: float) -> float:
        rotated = np.array([_rot_z(yaw) @ vector for vector in v])
        translation_xy = (base[:, :2] - rotated[:, :2]).mean(axis=0)
        residual = base[:, :2] - translation_xy - rotated[:, :2]
        return float(np.sum(residual**2))

    search = minimize_scalar(cost, bounds=(-np.pi, np.pi), method="bounded")
    yaw = float(search.x)
    rotated = np.array([_rot_z(yaw) @ vector for vector in v])
    translation_xy = (base[:, :2] - rotated[:, :2]).mean(axis=0)
    # Rz preserves z, so the height is shared by every touch. When the level
    # board already fixed the camera height, use that instead of the touches.
    if camera_z_mm is not None:
        translation_z = float(camera_z_mm) / 1000.0
    else:
        translation_z = float(np.mean(base[:, 2] - v[:, 2]))
    camera_position = np.array(
        [translation_xy[0], translation_xy[1], translation_z], dtype=float
    )
    rotation = rotation_cam_board @ flip @ _rot_z(-yaw)

    predicted = np.array([rotation @ (point - camera_position) for point in base])
    residuals = np.linalg.norm(predicted - camera, axis=1) * 1000.0
    return LevelBoardSolution(
        rotation_cam_base=rotation,
        camera_position=camera_position,
        yaw_rad=yaw,
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        residuals_mm=residuals,
    )
