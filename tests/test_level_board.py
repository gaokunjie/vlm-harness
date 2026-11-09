import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.level_board import (
    camera_height_mm,
    held_out_errors_mm,
    holdout_indices,
    predict_base_points,
    solve_from_touches,
)


def test_camera_height_from_level_board_plane():
    # board level on the table, camera 300 mm above it
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 0.300]
    pose[:3, :3] = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    assert abs(camera_height_mm(pose, board_z_mm=0.0) - 300.0) < 1e-6
    # tilted camera looking at a point 300 mm below it
    pose[:3, :3] = (
        Rotation.from_euler("x", 160.0, degrees=True)
        * Rotation.from_euler("z", 25.0, degrees=True)
    ).as_matrix()
    pose[:3, 3] = pose[:3, :3] @ np.array([0.0, 0.0, -0.300])
    assert abs(camera_height_mm(pose, board_z_mm=0.0) - 300.0) < 1e-6


def test_solve_from_touches_recovers_camera_pose():
    r_cam_base = Rotation.from_euler("xyz", [172.0, -4.0, 7.0], degrees=True).as_matrix()
    t_true = np.array([0.21, 0.03, 0.325])
    yaw = np.radians(18.0)
    # A rigid 180 deg flip about x, not the reflection diag(1, 1, -1).
    flip = np.diag([1.0, -1.0, -1.0])
    r_cam_board = (
        r_cam_base
        @ Rotation.from_euler("z", np.degrees(yaw), degrees=True).as_matrix()
        @ flip
    )
    base_points = np.array([[0.16, 0.02, 0.001], [0.185, 0.045, 0.001]])
    camera_points = np.array([r_cam_base @ (p - t_true) for p in base_points])

    solution = solve_from_touches(r_cam_board, base_points, camera_points)
    assert np.allclose(solution.camera_position, t_true, atol=1e-9)
    assert np.allclose(solution.rotation_cam_base, r_cam_base, atol=1e-9)
    assert np.linalg.det(solution.rotation_cam_base) == pytest.approx(1.0)
    assert solution.rms_mm < 1e-4


def _synthetic_board(n=8):
    r_cam_base = Rotation.from_euler("xyz", [172.0, -4.0, 7.0], degrees=True).as_matrix()
    t_true = np.array([0.21, 0.03, 0.325])
    yaw = np.radians(18.0)
    flip = np.diag([1.0, -1.0, -1.0])
    r_cam_board = (
        r_cam_base
        @ Rotation.from_euler("z", np.degrees(yaw), degrees=True).as_matrix()
        @ flip
    )
    base = np.array(
        [[0.15 + 0.01 * i, 0.02 + 0.008 * i, 0.001] for i in range(n)]
    )
    camera = np.array([r_cam_base @ (point - t_true) for point in base])
    return r_cam_board, base, camera


def test_holdout_indices_are_spread_and_unique():
    assert holdout_indices(10, 2) == [3, 6]
    assert holdout_indices(8, 2) == [2, 5]
    assert holdout_indices(5, 2) == [1, 3]
    assert holdout_indices(8, 0) == []
    with pytest.raises(ValueError, match="smaller"):
        holdout_indices(3, 3)
    with pytest.raises(ValueError, match="non-negative"):
        holdout_indices(3, -1)


def test_predict_base_points_recovers_held_out_points():
    r_cam_board, base, camera = _synthetic_board(8)
    fit = [0, 1, 2, 3, 4, 5]
    held = [6, 7]
    solution = solve_from_touches(r_cam_board, base[fit], camera[fit])

    predicted = predict_base_points(solution, camera[held])
    errors = held_out_errors_mm(solution, camera[held], base[held])

    assert np.allclose(predicted, base[held], atol=1e-9)
    assert errors.max() < 1e-6


def test_solve_from_touches_rejects_a_reflected_board_pose():
    reflected = np.diag([1.0, 1.0, -1.0])
    base_points = np.array([[0.16, 0.02, 0.001], [0.185, 0.045, 0.001]])
    camera_points = np.array([[0.10, -0.05, 0.32], [0.12, -0.02, 0.32]])
    with pytest.raises(ValueError, match="proper rotation"):
        solve_from_touches(reflected, base_points, camera_points)


def test_solve_from_touches_uses_both_points_for_yaw():
    r_cam_base = Rotation.from_euler("y", 8.0, degrees=True).as_matrix()
    t_true = np.array([0.20, -0.01, 0.33])
    # A rigid 180 deg flip about x, not the reflection diag(1, 1, -1).
    flip = np.diag([1.0, -1.0, -1.0])
    r_cam_board = (
        r_cam_base @ Rotation.from_euler("z", -35.0, degrees=True).as_matrix() @ flip
    )
    base_points = np.array([[0.15, 0.00, 0.001], [0.19, 0.03, 0.001]])
    camera_points = np.array([r_cam_base @ (p - t_true) for p in base_points])
    solution = solve_from_touches(r_cam_board, base_points, camera_points)
    assert abs(np.degrees(solution.yaw_rad) - (-35.0)) < 1e-3


def test_solve_from_touches_can_fix_the_camera_height():
    r_cam_base = Rotation.from_euler("x", 174.0, degrees=True).as_matrix()
    t_true = np.array([0.20, 0.0, 0.3312])
    # A rigid 180 deg flip about x, not the reflection diag(1, 1, -1).
    flip = np.diag([1.0, -1.0, -1.0])
    r_cam_board = (
        r_cam_base @ Rotation.from_euler("z", 30.0, degrees=True).as_matrix() @ flip
    )
    base_points = np.array([[0.15, 0.02, 0.004], [0.19, 0.06, 0.004]])
    camera_points = np.array([r_cam_base @ (p - t_true) for p in base_points])
    solution = solve_from_touches(
        r_cam_board, base_points, camera_points, camera_z_mm=331.2
    )
    assert abs(solution.camera_position[2] * 1000.0 - 331.2) < 1e-6
    assert solution.rms_mm < 1e-4
