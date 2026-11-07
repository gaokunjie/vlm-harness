import numpy as np
import pytest

from dobot_rua.geometry import matrix_to_pose
from dobot_rua.kinematics import forward_kinematics, load_fk, load_limits
from dobot_rua.workspace import (
    PointMask,
    build_fov_mask,
    build_reachability_mask,
    sample_layout,
)


def _downward_camera(f=500.0, cx=320.0, cy=240.0):
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, 1.0]
    return K, T


def _rectangle_mask():
    points = np.array(
        [[x, y] for x in range(0, 201, 10) for y in range(0, 201, 10)],
        dtype=float,
    )
    return PointMask.from_points(points, grid_mm=10.0)


def test_sample_layout_is_reproducible_with_seed():
    mask = _rectangle_mask()
    layout_a = sample_layout(
        np.random.default_rng(7), mask, target_xy_mm=(500.0, 500.0), n_blocks=2
    )
    layout_b = sample_layout(
        np.random.default_rng(7), mask, target_xy_mm=(500.0, 500.0), n_blocks=2
    )
    assert layout_a.block_xy_mm == layout_b.block_xy_mm
    assert layout_a.distractor_xy_mm == layout_b.distractor_xy_mm


def test_sample_layout_respects_separations():
    mask = _rectangle_mask()
    for seed in range(20):
        layout = sample_layout(
            np.random.default_rng(seed),
            mask,
            target_xy_mm=(500.0, 500.0),
            n_blocks=2,
            min_block_sep_mm=60.0,
            min_target_sep_mm=80.0,
        )
        assert mask.contains(*layout.block_xy_mm)
        assert layout.distractor_xy_mm is not None
        assert mask.contains(*layout.distractor_xy_mm)
        block = np.asarray(layout.block_xy_mm)
        distractor = np.asarray(layout.distractor_xy_mm)
        assert np.linalg.norm(block - distractor) >= 60.0 - 1e-9
        assert np.linalg.norm(block - np.array([500.0, 500.0])) >= 80.0 - 1e-9


def test_sample_layout_excludes_points_near_target():
    mask = _rectangle_mask()
    with pytest.raises(ValueError):
        sample_layout(
            np.random.default_rng(0),
            mask,
            target_xy_mm=(100.0, 100.0),
            n_blocks=1,
            min_target_sep_mm=500.0,
        )


def test_fov_mask_keeps_points_inside_image():
    K, T = _downward_camera()
    points = np.array(
        [[0.0, 0.0], [500.0, 0.0], [-500.0, 0.0], [0.0, 300.0], [700.0, 0.0]],
        dtype=float,
    )
    mask = build_fov_mask(K, T, (480, 640), points, margin_px=30)
    assert mask.contains(0.0, 0.0)
    assert mask.contains(500.0, 0.0)
    assert mask.contains(-500.0, 0.0)
    assert mask.contains(0.0, 300.0)
    assert not mask.contains(700.0, 0.0)


def test_reachability_mask_contains_a_known_fk_point():
    params = load_fk("configs/fk_magician.yaml")
    limits = load_limits("configs/robot_limits.yaml")
    xyz = forward_kinematics(params, 0.0, 45.0, 45.0)
    mask = build_reachability_mask(params, limits, z_mm=float(xyz[2]), grid_mm=5.0)
    assert mask.contains(float(xyz[0]), float(xyz[1]))
    assert not mask.contains(5000.0, 5000.0)
