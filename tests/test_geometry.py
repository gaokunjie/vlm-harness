import numpy as np
import pytest

from dobot_rua.geometry import (
    intersect_two_lines,
    line_from_points,
    matrix_to_pose,
    pose_to_matrix,
    project_pixel_to_plane,
    project_point_to_pixel,
)


def _downward_camera(f=500.0, cx=320.0, cy=240.0):
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4)
    # optical x = world x; optical y = -world y; optical z = -world z
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, 1.0]
    return K, T


def test_pose_roundtrip_identity():
    T = pose_to_matrix((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    position, quaternion = matrix_to_pose(T)
    assert np.allclose(position, [1.0, 2.0, 3.0])
    assert np.allclose(quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_project_principal_point_to_plane():
    K, T = _downward_camera()
    point = project_pixel_to_plane(K, T, (320.0, 240.0), 0.0)
    assert np.allclose(point, [0.0, 0.0, 0.0], atol=1e-9)


def test_project_offset_pixels_to_plane():
    K, T = _downward_camera()
    right = project_pixel_to_plane(K, T, (420.0, 240.0), 0.0)
    assert np.allclose(right, [0.2, 0.0, 0.0], atol=1e-6)
    down_in_image = project_pixel_to_plane(K, T, (320.0, 140.0), 0.0)
    assert np.allclose(down_in_image, [0.0, 0.2, 0.0], atol=1e-6)


def test_project_world_point_roundtrip():
    K, T = _downward_camera()
    uv = project_point_to_pixel(K, T, np.array([0.1, -0.05, 0.0]))
    point = project_pixel_to_plane(K, T, uv, 0.0)
    assert np.allclose(point, [0.1, -0.05, 0.0], atol=1e-6)


def test_line_intersection():
    line_a = line_from_points((0.0, 0.0), (10.0, 10.0))
    line_b = line_from_points((0.0, 10.0), (10.0, 0.0))
    intersection = intersect_two_lines(line_a, line_b)
    assert intersection is not None
    assert np.allclose(intersection, [5.0, 5.0])


def test_parallel_lines_have_no_intersection():
    line_a = line_from_points((0.0, 0.0), (10.0, 0.0))
    line_b = line_from_points((0.0, 5.0), (10.0, 5.0))
    assert intersect_two_lines(line_a, line_b) is None


def test_pixel_behind_camera_is_rejected():
    K, T = _downward_camera()
    with pytest.raises(ValueError):
        project_point_to_pixel(K, T, np.array([0.0, 0.0, 2.0]))
