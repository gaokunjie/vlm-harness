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
