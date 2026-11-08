import numpy as np
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.charuco import make_board
from dobot_rua.calibration.touch import (
    camera_extrinsic,
    extreme_marker_ids,
    fit_board_from_touches,
    marker_center,
)


def test_fit_board_from_touches_recovers_known_transform():
    board_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler("xyz", [5.0, -10.0, 15.0], degrees=True).as_matrix()
    known[:3, 3] = [0.2, -0.1, 0.0]
    base_points = board_points @ known[:3, :3].T + known[:3, 3]
    result = fit_board_from_touches(board_points, base_points)
    assert np.allclose(result.transform, known, atol=1e-9)
    assert result.rms_mm < 1e-6


def test_camera_extrinsic_composition():
    t_base_board = np.eye(4)
    t_base_board[:3, 3] = [0.2, 0.0, 0.0]
    t_cam_board = np.eye(4)
    t_cam_board[:3, 3] = [0.0, 0.0, 0.5]
    t_base_camera = camera_extrinsic(t_base_board, t_cam_board)
    assert np.allclose(t_base_camera[:3, 3], [0.2, 0.0, -0.5])


def test_extreme_marker_ids_are_four_distinct_corners():
    board = make_board(7, 8, 25.0, 18.75, "DICT_4X4_50")
    ids = extreme_marker_ids(board, 4)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    centers = np.asarray([marker_center(board, value) for value in ids])
    assert len({tuple(np.round(center[:2], 6)) for center in centers}) == 4
