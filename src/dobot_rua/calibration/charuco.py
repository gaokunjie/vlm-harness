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


def detect_markers(
    board: cv2.aruco.CharucoBoard,
    image_bgr: np.ndarray,
    parameters: cv2.aruco.DetectorParameters | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect the board's ArUco markers without ChArUco corner interpolation.

    Partially occluded boards often fail ChArUco interpolation while still
    yielding enough decoded markers for a full PnP pose.  This is the
    fallback used when only part of the board is visible.
    """
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = 0.005
        parameters.adaptiveThreshWinSizeMax = 101
        parameters.errorCorrectionRate = 0.8
    detector = cv2.aruco.ArucoDetector(board.getDictionary(), parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if corners is None or ids is None or len(ids) == 0:
        return None, None
    return corners, ids


def _marker_correspondences(
    board: cv2.aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    board_ids = np.asarray(board.getIds(), dtype=int).reshape(-1)
    object_points = np.asarray(board.getObjPoints(), dtype=float)
    object_list = []
    image_list = []
    for corner, marker_id in zip(
        np.asarray(corners, dtype=float), np.asarray(ids, dtype=int).reshape(-1)
    ):
        matches = np.where(board_ids == int(marker_id))[0]
        if len(matches) == 0:
            continue
        object_list.append(object_points[matches[0]])
        image_list.append(np.asarray(corner, dtype=float).reshape(4, 2))
    if len(object_list) < 2:
        return None
    return (
        np.asarray(object_list, dtype=float).reshape(-1, 3),
        np.asarray(image_list, dtype=float).reshape(-1, 2),
    )


def board_pose_from_markers(
    board: cv2.aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Return ``T_cam_board`` from decoded markers only.

    Requires at least two decoded markers (eight corner correspondences).  The
    returned transform has the same board frame as :func:`board_pose`.
    """
    if corners is None or ids is None:
        return None
    correspondences = _marker_correspondences(board, corners, ids)
    if correspondences is None:
        return None
    object_points, image_points = correspondences
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(intrinsic_k, dtype=float),
        np.asarray(distortion, dtype=float).reshape(-1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


def marker_reprojection_px(
    board: cv2.aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    transform: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """RMS reprojection error of the decoded marker corners, in pixels."""
    correspondences = _marker_correspondences(board, corners, ids)
    if correspondences is None:
        return float("inf")
    object_points, image_points = correspondences
    rvec, _ = cv2.Rodrigues(np.asarray(transform, dtype=float)[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        np.asarray(transform, dtype=float)[:3, 3],
        np.asarray(intrinsic_k, dtype=float),
        np.asarray(distortion, dtype=float).reshape(-1),
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def cup_center_pixel_from_markers(
    board: cv2.aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
    cup_center_in_board_mm: tuple[float, float, float] = (25.0, 75.0, 0.0),
) -> tuple[tuple[float, float], np.ndarray] | None:
    """Marker-only variant of :func:`cup_center_pixel`."""
    transform = board_pose_from_markers(
        board, corners, ids, intrinsic_k, distortion
    )
    if transform is None:
        return None
    point_board_m = np.asarray(cup_center_in_board_mm, dtype=float) / 1000.0
    pixel = project_board_point(
        transform, point_board_m, intrinsic_k, distortion
    )
    return pixel, transform


def object_points_for(
    board: cv2.aruco.CharucoBoard, charuco_ids: np.ndarray
) -> np.ndarray:
    points = board.getChessboardCorners()
    return points[charuco_ids.reshape(-1)]


def board_pose(
    corners: np.ndarray,
    ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray | None:
    """Return the board-to-camera 4x4 transform (``T_cam_board``).

    The board frame is OpenCV's ChArUco frame: origin at the top-left corner,
    x right, y down, z out of the board, metres.  This transform maps board
    coordinates into camera coordinates, which is what ``cv2.projectPoints``
    expects.
    """
    if corners is None or ids is None or len(ids) < 6:
        return None
    object_points = object_points_for(board, ids).astype(np.float64)
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(intrinsic_k, dtype=float),
        np.asarray(distortion, dtype=float).reshape(-1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


def project_board_point(
    t_cam_board: np.ndarray,
    point_board_m: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, float]:
    """Project a point in board coordinates into the image."""
    transform = np.asarray(t_cam_board, dtype=float)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    point = np.asarray(point_board_m, dtype=float).reshape(1, 3)
    projected, _ = cv2.projectPoints(
        point,
        rvec,
        transform[:3, 3],
        np.asarray(intrinsic_k, dtype=float),
        np.asarray(distortion, dtype=float).reshape(-1),
    )
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def cup_center_pixel(
    corners: np.ndarray,
    ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
    cup_center_in_board_mm: tuple[float, float, float] = (25.0, 75.0, 0.0),
) -> tuple[tuple[float, float], np.ndarray] | None:
    """Return the suction-cup-center pixel and the board pose, or ``None``."""
    if corners is None or ids is None or len(ids) < 6:
        return None
    transform = board_pose(corners, ids, board, intrinsic_k, distortion)
    if transform is None:
        return None
    point_board_m = np.asarray(cup_center_in_board_mm, dtype=float) / 1000.0
    pixel = project_board_point(transform, point_board_m, intrinsic_k, distortion)
    return pixel, transform
