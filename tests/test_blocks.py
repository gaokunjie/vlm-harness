import cv2
import numpy as np

from dobot_rua.geometry import project_point_to_pixel
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


def test_block_projection_uses_block_top_plane():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0]) @ np.array(
        [
            [np.cos(0.35), 0, np.sin(0.35)],
            [0, 1, 0],
            [-np.sin(0.35), 0, np.cos(0.35)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.0, 0.30]
    true_xy = (0.14, 0.07)
    top_z = 0.025
    px = project_point_to_pixel(K, T, [true_xy[0], true_xy[1], top_z])
    image = np.full((480, 640, 3), 255, np.uint8)
    cv2.rectangle(
        image,
        (int(px[0]) - 20, int(px[1]) - 20),
        (int(px[0]) + 20, int(px[1]) + 20),
        (0, 255, 255),
        -1,
    )
    top = detect_blocks(image, K, T, top_z)
    table = detect_blocks(image, K, T, 0.0)
    assert top and table
    err_top = np.hypot(
        top[0].center_xyz_m[0] - true_xy[0], top[0].center_xyz_m[1] - true_xy[1]
    )
    err_table = np.hypot(
        table[0].center_xyz_m[0] - true_xy[0],
        table[0].center_xyz_m[1] - true_xy[1],
    )
    assert err_top < 0.004
    assert err_table > 0.010


def test_detect_blocks_uses_distortion():
    image = _scene()
    K, T = _camera()
    distortion = np.array([0.1, -0.05, 0, 0, 0.0])
    plain = {block.color: block for block in detect_blocks(image, K, T, plane_z_m=0.0)}
    distorted = {
        block.color: block
        for block in detect_blocks(
            image, K, T, plane_z_m=0.0, distortion=distortion
        )
    }
    assert not np.allclose(
        plain["blue"].center_xyz_m,
        distorted["blue"].center_xyz_m,
        atol=1e-6,
    )
