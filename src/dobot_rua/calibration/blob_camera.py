"""Camera pose from the robot's own end-effector silhouette.

The level table board fixes the camera's tilt (the world vertical in the camera
frame) and its height. The camera's yaw about that vertical and its horizontal
position come from watching a point rigidly attached to the tool (the dark
end-effector blob) as the arm moves:

    camera_i = R_cam_base @ (t_i + R_tool_i @ q - t_cam)

with ``q`` the blob's position in the tool frame. For a fixed yaw the
cross-product constraint ``d_i x camera_i = 0`` is linear in ``(q, tx, ty)``,
so a one-dimensional search over the yaw solves the whole thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BlobCameraSolution:
    rotation_cam_base: np.ndarray
    camera_position_mm: np.ndarray
    tool_point_mm: np.ndarray
    rms_px: float


def _rotation_to_normal(normal: np.ndarray) -> np.ndarray:
    """Minimal rotation taking +z onto ``normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3) if n[2] > 0 else Rotation.from_euler("x", 180, degrees=True).as_matrix()
    angle = np.arctan2(sine, float(z @ n))
    return Rotation.from_rotvec(axis / sine * angle).as_matrix()


def _rot_z(angle: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, 0.0, float(angle)]).as_matrix()


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def solve_camera_from_blob(
    tool_poses: list[tuple[np.ndarray, np.ndarray]],
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    normal_cam: np.ndarray,
    camera_z_mm: float,
    yaw_samples: int = 360,
) -> BlobCameraSolution:
    """Solve for the camera pose given tool poses and the blob's pixels.

    ``tool_poses`` holds ``(position_mm, rotation_tool)`` per sample; the blob
    sits at ``q`` in the tool frame. ``normal_cam`` is the world up direction
    seen from the camera; ``camera_z_mm`` is the camera height above the table.
    """
    if len(tool_poses) != len(pixels):
        raise ValueError("one pixel per tool pose is required")
    if len(tool_poses) < 4:
        raise ValueError("at least four samples are required")
    k = np.asarray(intrinsic, dtype=float)
    k_inv = np.linalg.inv(k)
    normals = np.asarray(pixels, dtype=float)
    rays = np.array([k_inv @ np.array([u, v, 1.0]) for u, v in normals])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    base = _rotation_to_normal(normal_cam)

    def solve_at(yaw: float) -> tuple[np.ndarray, float, np.ndarray]:
        rotation = base @ _rot_z(yaw)
        rows = []
        targets = []
        for (position, tool_rotation), ray in zip(tool_poses, rays, strict=True):
            position = np.asarray(position, dtype=float)
            tool_rotation = np.asarray(tool_rotation, dtype=float)
            skew = _skew(ray)
            block = np.hstack([skew @ (rotation @ tool_rotation), -skew @ rotation[:, :2]])
            rows.append(block)
            targets.append(-skew @ (rotation @ position - rotation[:, 2] * camera_z_mm))
        matrix = np.vstack(rows)
        target = np.concatenate(targets)
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        q = solution[:3]
        t_xy = solution[3:]
        t = np.array([t_xy[0], t_xy[1], camera_z_mm])
        errors = []
        for (position, tool_rotation), pixel in zip(tool_poses, pixels, strict=True):
            point = np.asarray(position, dtype=float) + np.asarray(tool_rotation, dtype=float) @ q
            cam = rotation @ (point - t)
            uv = k @ cam
            errors.append(uv[:2] / uv[2] - np.asarray(pixel, dtype=float))
        rms = float(np.sqrt(np.mean(np.sum(np.array(errors) ** 2, axis=1))))
        return solution, rms, rotation

    best = None
    for index in range(yaw_samples):
        yaw = -np.pi + 2.0 * np.pi * index / yaw_samples
        solution, rms, rotation = solve_at(yaw)
        if best is None or rms < best[1]:
            best = (yaw, rms, solution, rotation)
    assert best is not None
    start_yaw, start_rms, solution, rotation = best
    refine = minimize_scalar(
        lambda yaw: solve_at(yaw)[1],
        bounds=(start_yaw - 2.0 * np.pi / yaw_samples, start_yaw + 2.0 * np.pi / yaw_samples),
        method="bounded",
    )
    yaw = float(refine.x)
    solution, rms, rotation = solve_at(yaw)
    return BlobCameraSolution(
        rotation_cam_base=rotation,
        camera_position_mm=np.array([solution[3], solution[4], camera_z_mm]),
        tool_point_mm=solution[:3],
        rms_px=rms,
    )
