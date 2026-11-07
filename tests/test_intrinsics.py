import cv2
import numpy as np
import pytest

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.intrinsics import (
    calibrate_from_detections,
    calibrate_from_marker_detections,
)


def test_calibrate_from_synthetic_views_recovers_focal():
    board = make_board(8, 7, 25.0, 18.75, "DICT_5X5_1000")
    object_points = board.getChessboardCorners()
    ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
    true_k = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    rng = np.random.default_rng(0)
    detections = []
    for _ in range(15):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array(
            [
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                rng.uniform(0.45, 0.75),
            ]
        )
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, true_k, distortion
        )
        detections.append((image_points.astype(np.float32), ids.copy()))

    result = calibrate_from_detections(board, detections, (720, 1280))
    assert abs(result.intrinsic_K[0, 0] - 800.0) < 20.0
    assert np.allclose(result.intrinsic_K[0, 2], 640.0, atol=15.0)
    assert np.allclose(result.intrinsic_K[1, 2], 360.0, atol=15.0)
    assert result.rms < 0.1


def test_calibrate_from_synthetic_marker_views_recovers_focal():
    board = make_board(7, 8, 25.0, 18.75, "DICT_4X4_50")
    board_points = board.getObjPoints()
    board_ids = board.getIds().reshape(-1)
    true_k = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    rng = np.random.default_rng(1)
    detections = []
    for _ in range(15):
        chosen = rng.choice(len(board_points), 10, replace=False)
        object_points = np.concatenate(
            [board_points[index] for index in chosen], axis=0
        ).astype(np.float32)
        rvec = rng.uniform(-0.3, 0.3, 3)
        tvec = np.array(
            [
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                rng.uniform(0.35, 0.6),
            ]
        )
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, true_k, distortion
        )
        corners = image_points.reshape(-1, 1, 4, 2).astype(np.float32)
        ids = board_ids[chosen].reshape(-1, 1).astype(np.int32)
        # The real ArucoDetector returns one array per marker, i.e. a tuple.
        corner_tuple = tuple(corners[index] for index in range(corners.shape[0]))
        detections.append((corner_tuple, ids))

    result = calibrate_from_marker_detections(board, detections, (720, 1280))
    assert abs(result.intrinsic_K[0, 0] - 800.0) < 20.0
    assert np.allclose(result.intrinsic_K[0, 2], 640.0, atol=15.0)
    assert np.allclose(result.intrinsic_K[1, 2], 360.0, atol=15.0)
    assert result.rms < 0.1


def test_regularized_marker_calibration_keeps_square_pixels():
    board = make_board(7, 8, 25.0, 18.75, "DICT_4X4_50")
    board_points = board.getObjPoints()
    board_ids = board.getIds().reshape(-1)
    true_k = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    rng = np.random.default_rng(2)
    detections = []
    for _ in range(15):
        chosen = rng.choice(len(board_points), 8, replace=False)
        object_points = np.concatenate(
            [board_points[index] for index in chosen], axis=0
        ).astype(np.float32)
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array(
            [
                rng.uniform(-0.04, 0.04),
                rng.uniform(-0.04, 0.04),
                rng.uniform(0.35, 0.6),
            ]
        )
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, true_k, distortion
        )
        corners = image_points.reshape(-1, 1, 4, 2).astype(np.float32)
        ids = board_ids[chosen].reshape(-1, 1).astype(np.int32)
        corner_tuple = tuple(corners[index] for index in range(corners.shape[0]))
        detections.append((corner_tuple, ids))

    result = calibrate_from_marker_detections(
        board,
        detections,
        (720, 1280),
        initial_K=np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]),
        fix_aspect=True,
        zero_tangent=True,
        fix_k3=True,
    )
    assert result.intrinsic_K[0, 0] == pytest.approx(result.intrinsic_K[1, 1], rel=1e-9)
    assert abs(result.intrinsic_K[0, 0] - 800.0) < 80.0
    assert result.rms < 0.2
