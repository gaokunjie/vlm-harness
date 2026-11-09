import numpy as np

from dobot_rua.calibration.charuco import (
    board_pose,
    cup_center_pixel,
    detect_board,
    make_board,
    object_points_for,
    project_board_point,
)


def test_rendered_board_detects_corners():
    board = make_board(8, 7, 25.0, 18.75, "DICT_5X5_1000")
    image = board.generateImage((1280, 720), marginSize=40)
    corners, ids = detect_board(board, image)
    assert corners is not None
    assert ids is not None
    assert len(ids) >= 30
    points = object_points_for(board, ids)
    assert points.shape == (len(ids), 3)
    assert np.allclose(points[:, 2], 0.0)


def test_board_pose_and_cup_center_projection():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    image = board.generateImage((640, 480), marginSize=40)
    corners, ids = detect_board(board, image)
    assert corners is not None and ids is not None
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    T = board_pose(corners, ids, board, K, np.zeros(5))
    assert T is not None
    center = project_board_point(T, np.array([0.025, 0.025, 0.0]), K, np.zeros(5))
    cup, T2 = cup_center_pixel(corners, ids, board, K, np.zeros(5))
    assert np.allclose(T, T2)
    assert cup[1] > center[1] + 5.0  # y is down; the cup is below the board


def test_cup_center_returns_none_with_too_few_corners():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = np.eye(3)
    assert (
        cup_center_pixel(
            np.zeros((2, 1, 2)), np.array([[1], [2]]), board, K, np.zeros(5)
        )
        is None
    )
