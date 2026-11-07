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


def test_black_detection_excludes_red_lines():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.line(image, (200, 300), (600, 300), (0, 0, 0), 3)
    cv2.line(image, (400, 100), (400, 500), (0, 0, 0), 3)
    cv2.line(image, (150, 350), (350, 350), (0, 0, 255), 3)
    cv2.line(image, (250, 250), (250, 450), (0, 0, 255), 3)
    K, T = _camera()
    black = detect_line_intersection(image, K, T, 0.0, "black")
    assert black is not None
    assert np.allclose(black.intersection_uv, (400.0, 300.0), atol=5.0)
    red = detect_line_intersection(image, K, T, 0.0, "red")
    assert red is not None
    assert np.allclose(red.intersection_uv, (250.0, 350.0), atol=5.0)


def test_detect_desaturated_red_lines():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    color = (100, 90, 120)  # low-saturation brown-red, BGR
    cv2.line(image, (200, 300), (440, 300), color, 3)
    cv2.line(image, (320, 180), (320, 420), color, 3)
    K, T = _camera()
    result = detect_line_intersection(image, K, T, 0.0, "red")
    assert result is not None
    assert np.allclose(result.intersection_uv, (320.0, 300.0), atol=6.0)


def test_detect_bright_dark_teal_lines():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    color = (110, 110, 80)  # desaturated dark teal, BGR
    cv2.line(image, (200, 300), (440, 300), color, 3)
    cv2.line(image, (320, 180), (320, 420), color, 3)
    K, T = _camera()
    result = detect_line_intersection(image, K, T, 0.0, "black")
    assert result is not None
    assert np.allclose(result.intersection_uv, (320.0, 300.0), atol=6.0)


def test_measure_intersection_gt_does_not_share_the_agent_mask(monkeypatch):
    image = _line_scene((0, 0, 0))
    K, T = _camera()
    import dobot_rua.perception.lines as lines

    monkeypatch.setattr(
        lines, "_line_mask", lambda image, color: np.zeros(image.shape[:2], np.uint8)
    )
    assert lines.detect_line_intersection(image, K, T, 0.0, "black") is None
    gt = lines.measure_intersection_gt(image, K, T, 0.0, "black")
    assert gt is not None
    assert np.allclose(gt.intersection_uv, (320.0, 240.0), atol=5.0)
