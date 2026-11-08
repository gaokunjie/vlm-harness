"""Fixed-camera extrinsics from a table board plus robot touch points."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TouchFit:
    transform: np.ndarray
    residuals_mm: np.ndarray
    rms_mm: float
    max_mm: float


def fit_board_from_touches(
    board_points_m: np.ndarray, base_points_m: np.ndarray
) -> TouchFit:
    """Kabsch fit of ``T_base_from_board`` from paired board/base points."""
    board = np.asarray(board_points_m, dtype=float)
    base = np.asarray(base_points_m, dtype=float)
    if board.shape != base.shape or board.ndim != 2 or board.shape[1] != 3:
        raise ValueError("board and base points must be matching Nx3 arrays")
    if len(board) < 3:
        raise ValueError("at least three touch points are required")
    board_center = board.mean(axis=0)
    base_center = base.mean(axis=0)
    covariance = (board - board_center).T @ (base - base_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = base_center - rotation @ board_center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    predicted = board @ rotation.T + translation
    residuals_mm = np.linalg.norm(predicted - base, axis=1) * 1000.0
    return TouchFit(
        transform=transform,
        residuals_mm=residuals_mm,
        rms_mm=float(np.sqrt(np.mean(residuals_mm**2))),
        max_mm=float(residuals_mm.max()),
    )


def camera_extrinsic(
    t_base_board: np.ndarray, t_cam_board: np.ndarray
) -> np.ndarray:
    """Compose ``T_base_from_board @ inv(T_cam_from_board)``."""
    return np.asarray(t_base_board, dtype=float) @ np.linalg.inv(
        np.asarray(t_cam_board, dtype=float)
    )


def marker_center(board: cv2.aruco.CharucoBoard, marker_id: int) -> np.ndarray:
    ids = np.asarray(board.getIds()).reshape(-1)
    points = board.getObjPoints()
    index = {int(value): position for position, value in enumerate(ids)}
    if int(marker_id) not in index:
        raise ValueError(f"marker {marker_id} is not on this board")
    return np.asarray(points[index[int(marker_id)]], dtype=float).mean(axis=0)


def extreme_marker_ids(
    board: cv2.aruco.CharucoBoard, count: int = 4
) -> tuple[int, ...]:
    """Return the marker ids nearest the four board corners."""
    ids = np.asarray(board.getIds()).reshape(-1)
    centers = np.asarray([marker_center(board, int(value)) for value in ids])
    x_min, y_min = centers[:, 0].min(), centers[:, 1].min()
    x_max, y_max = centers[:, 0].max(), centers[:, 1].max()
    corners = np.asarray(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )
    selected: list[int] = []
    for corner in corners:
        distances = np.linalg.norm(centers[:, :2] - corner, axis=1)
        marker_id = int(ids[int(np.argmin(distances))])
        if marker_id not in selected:
            selected.append(marker_id)
    return tuple(selected[:count])


def camera_board_pose(
    board: cv2.aruco.CharucoBoard,
    corners,
    ids,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Solve ``T_camera_from_board`` from detected ArUco marker corners."""
    board_ids = np.asarray(board.getIds()).reshape(-1)
    points = board.getObjPoints()
    index = {int(value): position for position, value in enumerate(board_ids)}
    object_points = []
    image_points = []
    for corner_set, marker_id in zip(corners, np.asarray(ids).reshape(-1), strict=True):
        position = index.get(int(marker_id))
        if position is None:
            continue
        object_points.append(np.asarray(points[position], dtype=np.float32))
        image_points.append(np.asarray(corner_set).reshape(4, 2).astype(np.float32))
    if len(object_points) < 4:
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
    return transform
