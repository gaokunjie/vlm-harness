"""Joint solution for the fixed camera pose and a generic tool-frame offset.

NOTE (2026-11-09): the Magician's suction cup is vertical and centred on j4,
so the front calibration now uses the scalar ``tip_length_mm`` model
(``tip = flange - (0, 0, L)``) in ``level_board.py``. This module is kept for
reference and is no longer used by the front path.

Original description:

Joint solution for the fixed camera pose and the suction-cup tool offset.

The firmware reports the **bare flange** as its tool point (its end-effector
offset reads 0,0,0), while the suction cup extends past it by a fixed vector in
the flange frame. Touching marker centres with the cup therefore gives

    marker_in_base = P + Rz(yaw) @ marker_in_board
    marker_in_base = tip + R_tool @ tcp

with ``P`` the board origin in the base frame, ``yaw`` the board's rotation
about the vertical, and ``tcp`` the cup centre in the tool frame. Three or more
touches with differing tool orientations determine all seven parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TcpExtrinsic:
    rotation_world_optical: np.ndarray
    camera_position: np.ndarray
    yaw_rad: float
    board_origin: np.ndarray
    tcp_mm: np.ndarray
    residuals_mm: np.ndarray
    rms_mm: float


def _rot_z(angle_rad: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, 0.0, float(angle_rad)]).as_matrix()


def solve_camera_and_tcp(
    rotation_cam_board: np.ndarray,
    board_origin_camera_mm: np.ndarray,
    camera_points_mm: np.ndarray,
    touch_tips_mm: np.ndarray,
    tool_rotations: np.ndarray,
    *,
    board_normal_down: bool = True,
) -> TcpExtrinsic:
    """Fit the camera pose and the cup offset from >= 3 touched markers.

    ``rotation_cam_board`` is the pose returned by ``solvePnP``: the board's
    +z usually points away from the camera, i.e. down in the base frame.
    ``board_normal_down=True`` converts it to the level-board convention
    (board +z = world +z) with a rigid 180 deg rotation about x. Flipping only
    z would be a reflection and must never be used here.
    """
    rotation_cam_board = np.asarray(rotation_cam_board, dtype=float)
    if rotation_cam_board.shape != (3, 3) or not np.allclose(
        rotation_cam_board.T @ rotation_cam_board, np.eye(3), atol=1e-6
    ):
        raise ValueError("rotation_cam_board must be a proper rotation")
    if float(np.linalg.det(rotation_cam_board)) <= 0.0:
        raise ValueError("rotation_cam_board must be a proper rotation (det <= 0)")
    if board_normal_down:
        rotation_cam_board = rotation_cam_board @ np.diag([1.0, -1.0, -1.0])
    board_origin_camera = np.asarray(board_origin_camera_mm, dtype=float)
    camera_points = np.asarray(camera_points_mm, dtype=float)
    tips = np.asarray(touch_tips_mm, dtype=float)
    rotations = np.asarray(tool_rotations, dtype=float)
    if camera_points.ndim != 2 or camera_points.shape[1] != 3:
        raise ValueError("camera points must be an Nx3 array")
    if len(camera_points) < 3:
        raise ValueError("at least three touch points are required")
    if not (
        tips.shape == camera_points.shape
        and rotations.shape == (len(camera_points), 3, 3)
    ):
        raise ValueError("points, tips and rotations must agree in count")

    # Marker centres in the board frame, from the camera observation.
    markers = np.array(
        [
            rotation_cam_board.T @ (point - board_origin_camera)
            for point in camera_points
        ]
    )

    def unpack(vec: np.ndarray):
        return vec[:3], float(vec[3]), vec[4:]

    def residuals(vec: np.ndarray) -> np.ndarray:
        origin, yaw, tcp = unpack(vec)
        rotate = _rot_z(yaw)
        predicted = np.array([origin + rotate @ marker for marker in markers])
        observed = np.array([tip + tool @ tcp for tip, tool in zip(tips, rotations, strict=True)])
        return (predicted - observed).reshape(-1)

    yaw0 = _initial_yaw(markers, tips)
    origin0 = np.mean([tip - _rot_z(yaw0) @ marker for tip, marker in zip(tips, markers, strict=True)], axis=0)
    start = np.concatenate([origin0, [yaw0], np.zeros(3)])
    result = least_squares(residuals, start, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)

    origin, yaw, tcp = unpack(result.x)
    rotate = _rot_z(yaw)
    predicted = np.array([origin + rotate @ marker for marker in markers])
    observed = np.array([tip + tool @ tcp for tip, tool in zip(tips, rotations, strict=True)])
    per_touch = np.linalg.norm(predicted - observed, axis=1)

    rotation_world_optical = rotate @ rotation_cam_board.T
    camera_position = origin - rotation_world_optical @ board_origin_camera
    return TcpExtrinsic(
        rotation_world_optical=rotation_world_optical,
        camera_position=camera_position,
        yaw_rad=yaw,
        board_origin=origin,
        tcp_mm=tcp,
        residuals_mm=per_touch,
        rms_mm=float(np.sqrt(np.mean(per_touch**2))),
    )


def transform_from_solution(solution: TcpExtrinsic) -> np.ndarray:
    """Build a metre-scale ``T_world_from_optical`` from a millimetre solution.

    ``solve_camera_and_tcp`` works in whatever unit it is given. The front
    calibration session stores millimetres, while ``cameras.yaml`` and the
    perception stack expect metres; this is the single conversion point.
    """
    transform = np.eye(4)
    transform[:3, :3] = solution.rotation_world_optical
    transform[:3, 3] = np.asarray(solution.camera_position, dtype=float) / 1000.0
    return transform


def _initial_yaw(markers: np.ndarray, tips: np.ndarray) -> float:
    """Coarse yaw from the tip/marker horizontal pairs, ignoring the tool offset."""
    if len(markers) < 2:
        return 0.0
    best_yaw, best_cost = 0.0, np.inf
    for yaw in np.radians(np.arange(0.0, 360.0, 2.0)):
        rotate = _rot_z(yaw)
        residual = tips[:, :2] - np.array([rotate @ marker for marker in markers])[:, :2]
        cost = float(np.sum((residual - residual.mean(axis=0)) ** 2))
        if cost < best_cost:
            best_yaw, best_cost = float(yaw), cost
    return best_yaw
