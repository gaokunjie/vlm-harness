import cv2
import numpy as np

from dobot_rua.calibration.visual_servo import (
    board_center_pixel,
    delta_mm,
    jacobian_from_probes,
    largest_dark_blob,
)


def test_board_center_pixel_averages_all_corners():
    corners = [
        np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]]),
        np.array([[[30.0, 10.0], [40.0, 10.0], [40.0, 20.0], [30.0, 20.0]]]),
    ]
    u, v = board_center_pixel(corners)
    assert u == 25.0
    assert v == 15.0


def test_jacobian_from_probes_recovers_known_affine():
    jacobian = np.array([[0.0, -2.0], [2.0, 0.0]])
    p0 = np.array([100.0, 200.0])
    step = 10.0
    px = p0 + jacobian[:, 0] * step
    py = p0 + jacobian[:, 1] * step
    estimate = jacobian_from_probes(p0, px, py, step)
    assert np.allclose(estimate, jacobian)


def test_delta_mm_solves_and_clamps_to_max_step():
    jacobian = np.eye(2)
    assert np.allclose(delta_mm(jacobian, (5.0, -3.0), max_mm=10.0), [5.0, -3.0])
    clamped = delta_mm(jacobian, (100.0, 0.0), max_mm=10.0)
    assert np.isclose(np.linalg.norm(clamped), 10.0)


def test_largest_dark_blob_finds_end_effector_reference():
    frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (600, 400), (900, 560), (20, 20, 20), -1)
    blob = largest_dark_blob(frame)
    assert blob is not None
    assert blob["area"] > 20000
    u, v = blob["reference"]
    assert 590.0 <= u <= 615.0
    assert 430.0 <= v <= 530.0


def test_largest_dark_blob_ignores_border_and_small_markers():
    frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (1279, 719), (10, 10, 10), 8)  # whiteboard frame
    for x in range(300, 360, 20):
        cv2.rectangle(frame, (x, 300), (x + 15, 315), (10, 10, 10), -1)  # markers
    assert largest_dark_blob(frame) is None
