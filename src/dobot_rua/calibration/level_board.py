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


def _proper_rotation(matrix: np.ndarray, name: str) -> np.ndarray:
    """Return ``matrix`` if it is a rotation; reject reflections."""
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-6
    ):
        raise ValueError(f"{name} must be a proper rotation")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"{name} must be a proper rotation (det <= 0)")
    return rotation


def solve_from_touches(
    rotation_cam_board: np.ndarray,
    base_points: np.ndarray,
    camera_points: np.ndarray,
    camera_z_mm: float | None = None,
) -> LevelBoardSolution:
    """Solve the camera pose from the level board and >= 2 touched points.

    ``base_points`` are the robot tip positions (m) at each touch, in the same
    order as the touched features' ``camera_points`` (m).

    When ``camera_z_mm`` is given, the returned camera z is measured from the
    board plane (world z = 0), not from the robot base frame. Add the board
    plane's z in the robot frame before writing ``cameras.yaml``; this path
    cannot recover that offset without a TCP model, which is why the front
    calibration now uses ``tcp_extrinsic.solve_camera_and_tcp``.
    """
    rotation_cam_board = _proper_rotation(rotation_cam_board, "rotation_cam_board")
    base = np.asarray(base_points, dtype=float)
    camera = np.asarray(camera_points, dtype=float)
    if base.shape != camera.shape or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base and camera points must be matching Nx3 arrays")
    if len(base) < 2:
        raise ValueError("at least two touch points are required")

    # The board's own +z points away from the camera, i.e. down in the base
    # frame. Turning it back up is a rigid 180 deg rotation about x (flip y
    # and z). Flipping only z would be a reflection and would produce a
    # det(R) = -1 "camera pose" that mirrors every projected point.
    flip = np.diag([1.0, -1.0, -1.0])
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


def predict_base_points(
    solution: LevelBoardSolution, camera_points_m: np.ndarray
) -> np.ndarray:
    """Map camera-frame points to the base frame with a solved camera pose."""
    camera = np.asarray(camera_points_m, dtype=float)
    if camera.ndim != 2 or camera.shape[1] != 3:
        raise ValueError("camera points must be an Nx3 array")
    return camera @ solution.rotation_cam_base + solution.camera_position


def held_out_errors_mm(
    solution: LevelBoardSolution,
    camera_points_m: np.ndarray,
    base_points_m: np.ndarray,
) -> np.ndarray:
    """Per-point held-out error (mm) between predicted and touched base points."""
    predicted = predict_base_points(solution, camera_points_m)
    base = np.asarray(base_points_m, dtype=float)
    if predicted.shape != base.shape:
        raise ValueError("camera and base points must agree in count")
    return np.linalg.norm(predicted - base, axis=1) * 1000.0


def holdout_indices(count: int, holdout: int) -> list[int]:
    """Choose ``holdout`` interior indices, spread evenly over ``count`` points.

    The indices are deterministic and unique, so a calibration run is
    reproducible. The first and last touches are kept for the fit when
    possible, which preserves the workspace coverage of the solved pose.
    """
    count = int(count)
    holdout = int(holdout)
    if holdout < 0:
        raise ValueError("holdout must be non-negative")
    if holdout == 0:
        return []
    if holdout >= count:
        raise ValueError("holdout must be smaller than the touch count")
    raw = np.linspace(0, count - 1, holdout + 2)[1:-1]
    selected = sorted({int(round(value)) for value in raw})
    if len(selected) < holdout:
        for value in range(1, count - 1):
            if value not in selected:
                selected.append(value)
                if len(selected) == holdout:
                    break
        selected = sorted(selected)
    if len(selected) < holdout:
        raise ValueError("could not select that many distinct holdout indices")
    return selected[:holdout]
