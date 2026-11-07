import cv2
import numpy as np

from dobot_rua.perception.lines import detect_line_intersection, measure_intersection_gt


def _camera(f=500.0, cx=320.0, cy=240.0):
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, 1.0]
    return K, T


def _line_scene(color_bgr):
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.line(image, (100, 100), (540, 380), color_bgr, 3)
    cv2.line(image, (100, 380), (540, 100), color_bgr, 3)
    return image


def test_detect_black_line_intersection():
    image = _line_scene((0, 0, 0))
    K, T = _camera()
    result = detect_line_intersection(image, K, T, plane_z_m=0.0, line_color="black")
    assert result is not None
    assert result.line_color == "black"
    assert np.allclose(result.intersection_uv, (320.0, 240.0), atol=5.0)
    assert np.allclose(result.intersection_xyz_m, (0.0, 0.0, 0.0), atol=0.005)


def test_detect_red_line_intersection():
    image = _line_scene((0, 0, 255))
    K, T = _camera()
    result = detect_line_intersection(image, K, T, plane_z_m=0.0, line_color="red")
    assert result is not None
    assert result.line_color == "red"
    assert np.allclose(result.intersection_uv, (320.0, 240.0), atol=5.0)


def test_measure_intersection_gt_is_independent_and_accurate():
    image = _line_scene((0, 0, 0))
    K, T = _camera()
    result = measure_intersection_gt(image, K, T, plane_z_m=0.0, line_color="black")
    assert result is not None
    assert np.allclose(result.intersection_uv, (320.0, 240.0), atol=5.0)
    assert np.allclose(result.intersection_xyz_m, (0.0, 0.0, 0.0), atol=0.005)
