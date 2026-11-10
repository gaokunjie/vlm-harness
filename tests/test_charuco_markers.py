import cv2
import numpy as np

from dobot_rua.calibration.charuco import (
    board_pose_from_markers,
    cup_center_pixel_from_markers,
    detect_markers,
    make_board,
    marker_reprojection_px,
)


def _camera():
    return np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _synthetic_markers(board, K, T, subset=None):
    object_points = np.asarray(board.getObjPoints(), dtype=float).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points, rvec, T[:3, 3], K, np.zeros(5)
    )
    corners = projected.reshape(-1, 4, 2)[:, None, :, :]
    ids = board.getIds().reshape(-1)
    if subset is not None:
        corners = corners[:subset]
        ids = ids[:subset]
    return corners.astype(np.float32), ids.reshape(-1, 1)


def _pose():
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(
        np.array([0.25, -0.15, 0.1], dtype=float)
    )[0]
    T[:3, 3] = [0.0, 0.0, 0.30]
    return T


def test_marker_pose_recovers_known_transform():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = _camera()
    expected = _pose()
    corners, ids = _synthetic_markers(board, K, expected)
    result = board_pose_from_markers(board, corners, ids, K, np.zeros(5))
    assert result is not None
    assert np.allclose(result, expected, atol=1e-5)


def test_marker_pose_works_with_three_markers():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = _camera()
    expected = _pose()
    corners, ids = _synthetic_markers(board, K, expected, subset=3)
    result = board_pose_from_markers(board, corners, ids, K, np.zeros(5))
    assert result is not None
    assert np.allclose(result, expected, atol=1e-4)


def test_marker_pose_needs_two_markers():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = _camera()
    corners, ids = _synthetic_markers(board, K, _pose(), subset=1)
    assert board_pose_from_markers(board, corners, ids, K, np.zeros(5)) is None


def test_marker_reprojection_is_small_for_exact_points():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = _camera()
    expected = _pose()
    corners, ids = _synthetic_markers(board, K, expected)
    assert marker_reprojection_px(board, corners, ids, expected, K, np.zeros(5)) < 1e-3


def test_detect_markers_finds_rendered_board():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    image = board.generateImage((640, 480), marginSize=40)
    corners, ids = detect_markers(board, image)
    assert corners is not None and ids is not None
    assert len(ids) >= 6
    assert set(ids.reshape(-1).tolist()).issubset(
        set(board.getIds().reshape(-1).tolist())
    )


def test_cup_center_from_markers_matches_board_pose():
    board = make_board(5, 5, 10.0, 7.5, "DICT_4X4_50")
    K = _camera()
    expected = _pose()
    corners, ids = _synthetic_markers(board, K, expected)
    result = cup_center_pixel_from_markers(board, corners, ids, K, np.zeros(5))
    assert result is not None
    pixel, transform = result
    assert np.allclose(transform, expected, atol=1e-5)
    center = cv2.projectPoints(
        np.array([[[0.025, 0.025, 0.0]]]),
        cv2.Rodrigues(expected[:3, :3])[0],
        expected[:3, 3],
        K,
        np.zeros(5),
    )[0].reshape(2)
    assert pixel[1] > center[1] + 5.0
