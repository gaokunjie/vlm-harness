import cv2
import numpy as np

from dobot_rua.perception.blocks import detect_blocks, measure_block_gt


def _camera(f=500.0, cx=320.0, cy=240.0):
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, 1.0]
    return K, T


def _scene():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (300, 220), (340, 260), (0, 255, 255), -1)  # yellow, BGR
    cv2.rectangle(image, (180, 180), (220, 220), (255, 0, 0), -1)  # blue, BGR
    return image


def test_detect_blocks_finds_yellow_and_blue_with_world_positions():
    image = _scene()
    K, T = _camera()
    blocks = detect_blocks(image, K, T, plane_z_m=0.0)
    by_color = {block.color: block for block in blocks}
    assert set(by_color) == {"yellow", "blue"}
    assert np.allclose(by_color["yellow"].center_uv, (320.0, 240.0), atol=2.0)
    assert np.allclose(by_color["blue"].center_uv, (200.0, 200.0), atol=2.0)
    assert np.allclose(by_color["yellow"].center_xyz_m, (0.0, 0.0, 0.0), atol=0.002)
    assert np.allclose(by_color["blue"].center_xyz_m, (-0.24, 0.08, 0.0), atol=0.002)


def test_measure_block_gt_is_independent_and_accurate():
    image = _scene()
    K, T = _camera()
    yellow = measure_block_gt(image, K, T, plane_z_m=0.0, color="yellow")
    assert yellow is not None
    assert yellow.color == "yellow"
    assert np.allclose(yellow.center_uv, (320.0, 240.0), atol=2.0)
    assert np.allclose(yellow.center_xyz_m, (0.0, 0.0, 0.0), atol=0.002)


def test_measure_block_gt_ignores_a_long_line_component():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (200, 200), (220, 220), (0, 255, 255), -1)
    cv2.line(image, (100, 100), (540, 380), (0, 0, 255), 3)
    K, T = _camera()
    yellow = measure_block_gt(image, K, T, plane_z_m=0.0, color="yellow")
    assert yellow is not None
    assert np.allclose(yellow.center_uv, (210.0, 210.0), atol=3.0)
