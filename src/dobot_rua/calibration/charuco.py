"""ChArUco board construction and detection."""

from __future__ import annotations

import cv2
import numpy as np


def make_board(
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    dictionary_name: str,
) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_mm) / 1000.0,
        float(marker_mm) / 1000.0,
        dictionary,
    )


def detect_board(
    board: cv2.aruco.CharucoBoard, image_bgr: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) == 0:
        return None, None
    return corners, ids


def object_points_for(
    board: cv2.aruco.CharucoBoard, charuco_ids: np.ndarray
) -> np.ndarray:
    points = board.getChessboardCorners()
    return points[charuco_ids.reshape(-1)]
