"""Small geometry helpers shared by calibration, perception and evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

Point = Sequence[float]
QuaternionWXYZ = Sequence[float]
Line2D = np.ndarray


def pose_to_matrix(position: Point, quaternion_wxyz: QuaternionWXYZ) -> np.ndarray:
    """Build a 4x4 transform from a world position and a wxyz quaternion."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def mean_pose(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average 4x4 transforms; rotations use the chordal mean on SO(3)."""
    matrices = [np.asarray(transform, dtype=float) for transform in transforms]
    if not matrices:
        raise ValueError("at least one transform is required")
    if any(matrix.shape != (4, 4) for matrix in matrices):
        raise ValueError("every transform must be 4x4")
    rotations = Rotation.from_matrix([matrix[:3, :3] for matrix in matrices])
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotations.mean().as_matrix()
    result[:3, 3] = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    return result


def matrix_to_pose(
    transform: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Inverse of :func:`pose_to_matrix`; returns a canonical wxyz quaternion."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    position = tuple(float(value) for value in matrix[:3, 3])
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return position, (float(w), float(x), float(y), float(z))


def project_pixel_to_plane(
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    pixel_uv: Point,
    plane_z_m: float,
) -> np.ndarray:
    """Unproject one pixel onto a horizontal world plane ``z = plane_z_m``."""
    k = np.asarray(intrinsic_k, dtype=float)
    transform = np.asarray(t_world_from_optical, dtype=float)
    u, v = (float(value) for value in pixel_uv)
    ray_optical = np.linalg.inv(k) @ np.array([u, v, 1.0])
    ray_world = transform[:3, :3] @ ray_optical
    origin = transform[:3, 3]
    if abs(ray_world[2]) < 1e-12:
        raise ValueError("pixel ray is parallel to the target plane")
    scale = (float(plane_z_m) - origin[2]) / ray_world[2]
    if scale <= 0:
        raise ValueError("pixel ray does not reach the target plane in front of the camera")
    return origin + scale * ray_world


def project_point_to_pixel(
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    point_world: Point,
) -> tuple[float, float]:
    """Project a world point into the camera; raises if it is not in front."""
    k = np.asarray(intrinsic_k, dtype=float)
    transform = np.asarray(t_world_from_optical, dtype=float)
    point = np.asarray(point_world, dtype=float)
    point_optical = transform[:3, :3].T @ (point - transform[:3, 3])
    if point_optical[2] <= 1e-9:
        raise ValueError("point is behind the camera")
    projected = k @ point_optical
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def line_from_points(first: Point, second: Point) -> Line2D:
    """Return homogeneous line coefficients from two 2D points."""
    x1, y1 = (float(value) for value in first)
    x2, y2 = (float(value) for value in second)
    line = np.cross(np.array([x1, y1, 1.0]), np.array([x2, y2, 1.0]))
    norm = np.linalg.norm(line[:2])
    if norm < 1e-12:
        raise ValueError("cannot build a line from identical points")
    return line / norm


def intersect_two_lines(first: Line2D, second: Line2D) -> np.ndarray | None:
    """Return the 2D intersection, or ``None`` for parallel lines."""
    point = np.cross(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    if abs(point[2]) < 1e-12:
        return None
    return np.array([point[0] / point[2], point[1] / point[2]], dtype=float)
