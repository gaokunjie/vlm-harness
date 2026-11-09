"""Small geometry helpers shared by calibration, perception and evaluation."""

from __future__ import annotations

from typing import Sequence

import cv2
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


def validate_rigid_transform(matrix: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` unless ``matrix`` is a proper 4x4 rigid transform."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise ValueError(
            f"{name} is not a proper rigid transform (det={determinant:.6f})"
        )


def undistort_points(
    points_uv: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Undistort pixels, returning pixel coordinates under the same K."""
    points = np.asarray(points_uv, dtype=float)
    single = points.ndim == 1
    points = points.reshape(-1, 1, 2).astype(np.float32)
    k = np.asarray(intrinsic_k, dtype=float)
    dist = np.asarray(distortion, dtype=float).reshape(-1)
    result = cv2.undistortPoints(points, k, dist, None, k).reshape(-1, 2)
    return result[0] if single else result


def project_pixel_to_plane(
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    pixel_uv: Point,
    plane_z_m: float,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
    """Unproject one pixel onto a horizontal world plane ``z = plane_z_m``."""
    k = np.asarray(intrinsic_k, dtype=float)
    transform = np.asarray(t_world_from_optical, dtype=float)
    pixel = np.asarray(pixel_uv, dtype=float)
    if distortion is not None:
        pixel = undistort_points(pixel, k, distortion)
    u, v = (float(value) for value in pixel)
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


def solve_pnp_two_plane(
    world_points_m: np.ndarray,
    image_points_uv: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``T_world_from_optical`` from 3D-2D points spanning two planes.

    The object points are world metres and the image points are pixels.  When
    ``distortion`` is supplied the image points are undistorted first; PnP is
    then solved with the same K and zero distortion.
    """
    world = np.asarray(world_points_m, dtype=np.float64)
    image = np.asarray(image_points_uv, dtype=np.float64)
    k = np.asarray(intrinsic_k, dtype=float)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("world_points_m must have shape (N, 3)")
    if image.ndim != 2 or image.shape[1] != 2:
        raise ValueError("image_points_uv must have shape (N, 2)")
    if len(world) != len(image) or len(world) < 6:
        raise ValueError("PnP needs at least 6 matching points")
    camera_distortion: np.ndarray | None = None
    if distortion is not None:
        image = undistort_points(image, k, distortion)
        camera_distortion = np.zeros(5)
    success, rvec, tvec = cv2.solvePnP(
        world,
        image,
        k,
        camera_distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        success, rvec, tvec = cv2.solvePnP(
            world,
            image,
            k,
            camera_distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    if not success:
        raise ValueError("PnP failed")
    # cv2.solvePnP returns the world->camera pose.  This project's
    # T_world_from_optical is camera->world (project_point_to_pixel applies
    # its inverse), so invert before returning.
    rotation_camera_from_world = cv2.Rodrigues(rvec)[0]
    translation_camera_from_world = np.asarray(tvec, dtype=float).reshape(3)
    rotation_world_from_camera = rotation_camera_from_world.T
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = -rotation_world_from_camera @ translation_camera_from_world
    return transform


def derive_homography(
    t_world_from_optical: np.ndarray,
    intrinsic_k: np.ndarray,
    plane_z_m: float,
) -> np.ndarray:
    """Return the pixel-to-world-XY homography for ``z = plane_z_m``.

    The returned H applies to **undistorted** pixels and maps homogeneous
    ``[u, v, 1]`` to world ``[x, y, 1]``.
    """
    transform = np.asarray(t_world_from_optical, dtype=float)
    k = np.asarray(intrinsic_k, dtype=float)
    # project_point_to_pixel uses p_cam = R^T (p_world - t), so the plane
    # homography must use the transposed rotation and -R^T t.
    rotation_t = transform[:3, :3].T
    offset = -rotation_t @ transform[:3, 3]
    world_to_pixel = k @ np.column_stack(
        [
            rotation_t[:, 0],
            rotation_t[:, 1],
            rotation_t[:, 2] * float(plane_z_m) + offset,
        ]
    )
    homography = np.linalg.inv(world_to_pixel)
    return homography / homography[2, 2]


def apply_homography(h: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    """Map pixels through ``h``; supports (2,) and (N, 2) inputs."""
    points = np.asarray(points_uv, dtype=float)
    single = points.ndim == 1
    points = points.reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(h, dtype=float) @ homogeneous.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped[0] if single else mapped


def point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool | np.ndarray:
    """Ray-cast point-in-polygon; supports (2,) and (N, 2) points."""
    points = np.asarray(point_xy, dtype=float)
    single = points.ndim == 1
    points = points.reshape(-1, 2)
    polygon = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    inside = np.zeros(len(points), dtype=bool)
    for index, (x, y) in enumerate(points):
        count = 0
        for corner in range(len(polygon)):
            x1, y1 = polygon[corner]
            x2, y2 = polygon[(corner + 1) % len(polygon)]
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < crossing_x:
                    count += 1
        inside[index] = count % 2 == 1
    return bool(inside[0]) if single else inside
