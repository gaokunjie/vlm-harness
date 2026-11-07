import numpy as np

from dobot_rua.calibration.charuco import detect_board, make_board, object_points_for


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
