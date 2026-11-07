import cv2
import numpy as np

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.intrinsics import calibrate_from_detections


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
