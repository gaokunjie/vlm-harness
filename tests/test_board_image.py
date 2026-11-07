import numpy as np

from dobot_rua.calibration.board_image import render_charuco_a4
from dobot_rua.calibration.charuco import detect_board, make_board


def test_rendered_5x5_board_has_10mm_square_spacing():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    page = render_charuco_a4(board, dpi=300)
    gray = np.asarray(page.convert("L"))
    corners, ids = detect_board(board, gray)
    assert corners is not None
    assert ids is not None
    assert len(ids) >= 10

    points = corners.reshape(-1, 2)
    distances = []
    for index in range(len(points)):
        deltas = np.linalg.norm(points - points[index], axis=1)
        deltas[index] = np.inf
        distances.append(float(deltas.min()))
    median_spacing = float(np.median(distances))
    expected = 10.0 / 25.4 * 300.0
    assert abs(median_spacing - expected) < 3.0
