"""Camera intrinsics from ChArUco detections."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .charuco import object_points_for


@dataclass(frozen=True)
class IntrinsicsResult:
    intrinsic_K: np.ndarray
    distortion: np.ndarray
    rms: float
    image_size_hw: tuple[int, int]


def calibrate_from_detections(
    board: cv2.aruco.CharucoBoard,
    detections: list[tuple[np.ndarray, np.ndarray]],
    image_size_hw: tuple[int, int],
) -> IntrinsicsResult:
    if len(detections) < 3:
        raise ValueError("at least three ChArUco views are required")
    height, width = image_size_hw
    object_points = []
    image_points = []
    for corners, ids in detections:
        object_points.append(object_points_for(board, ids))
        image_points.append(corners.astype(np.float32))
    rms, intrinsic, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        (int(width), int(height)),
        None,
        None,
    )
    return IntrinsicsResult(
        intrinsic_K=np.asarray(intrinsic, dtype=float),
        distortion=np.asarray(distortion, dtype=float).reshape(-1),
        rms=float(rms),
        image_size_hw=(int(height), int(width)),
    )


def calibrate_from_marker_detections(
    board: cv2.aruco.CharucoBoard,
    detections: list[tuple[np.ndarray, np.ndarray]],
    image_size_hw: tuple[int, int],
    *,
    initial_K: np.ndarray | None = None,
    fix_aspect: bool = False,
    zero_tangent: bool = False,
    fix_k3: bool = False,
    fix_principal: bool = False,
) -> IntrinsicsResult:
    """Calibrate from ArUco marker corners, useful for partial close-up views.

    ``detections`` are ``(corners, ids)`` from ``cv2.aruco.ArucoDetector``;
    corners have shape ``(N, 1, 4, 2)``. The board's marker object points are
    matched by marker id, so only visible markers contribute.
    """
    if len(detections) < 3:
        raise ValueError("at least three marker views are required")
    height, width = image_size_hw
    board_ids = np.asarray(board.getIds()).reshape(-1)
    board_points = board.getObjPoints()
    index = {int(marker_id): i for i, marker_id in enumerate(board_ids)}
    object_views = []
    image_views = []
    for corners, ids in detections:
        marker_corners = np.asarray(corners).reshape(-1, 4, 2)
        marker_ids = np.asarray(ids).reshape(-1)
        object_points = []
        image_points = []
        for corner_set, marker_id in zip(marker_corners, marker_ids, strict=True):
            index_in_board = index.get(int(marker_id))
            if index_in_board is None:
                continue
            object_points.append(
                np.asarray(board_points[index_in_board], dtype=np.float32)
            )
            image_points.append(corner_set.astype(np.float32))
        if object_points:
            object_views.append(np.concatenate(object_points, axis=0))
            image_views.append(np.concatenate(image_points, axis=0))
    if len(object_views) < 3:
        raise ValueError("not enough valid marker views")
    if fix_aspect and initial_K is None:
        raise ValueError("fix_aspect requires an initial_K")
    flags = 0
    guess = None
    if initial_K is not None:
        guess = np.asarray(initial_K, dtype=float).copy()
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    if fix_principal:
        if initial_K is None:
            raise ValueError("fix_principal requires an initial_K")
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    rms, intrinsic, distortion, _, _ = cv2.calibrateCamera(
        object_views,
        image_views,
        (int(width), int(height)),
        guess,
        None,
        flags=flags,
    )
    return IntrinsicsResult(
        intrinsic_K=np.asarray(intrinsic, dtype=float),
        distortion=np.asarray(distortion, dtype=float).reshape(-1),
        rms=float(rms),
        image_size_hw=(int(height), int(width)),
    )
