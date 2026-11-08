import numpy as np
import pytest

from dobot_rua.kinematics import (
    forward_kinematics,
    three_axis_frame,
    inverse_kinematics,
    load_fk,
    load_limits,
    within_limits,
)


def test_fk_matches_ready_pose():
    # The ready pose is an extrapolation outside the sampled calibration
    # envelope; the FK file itself reports ~15 mm error there. This is a
    # structural sanity check, not a calibration-accuracy claim.
    params = load_fk("configs/fk_magician.yaml")
    xyz = forward_kinematics(params, 0.0, 10.88, 49.98)
    assert np.allclose(xyz, [120.0, 0.0, 20.0], atol=20.0)


def test_fk_rotates_about_z():
    params = load_fk("configs/fk_magician.yaml")
    xyz0 = forward_kinematics(params, 0.0, 45.0, 45.0)
    xyz90 = forward_kinematics(params, 90.0, 45.0, 45.0)
    assert np.allclose(xyz90[:2], [0.0, xyz0[0]], atol=1e-6)
    assert np.isclose(xyz90[2], xyz0[2])


def test_ik_roundtrip_within_limits():
    params = load_fk("configs/fk_magician.yaml")
    limits = load_limits("configs/robot_limits.yaml")
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        j1 = rng.uniform(-60.0, 60.0)
        j2 = rng.uniform(5.0, 80.0)
        j3 = rng.uniform(0.0, 85.0)
        target = forward_kinematics(params, j1, j2, j3)
        solutions = inverse_kinematics(params, *target, limits)
        assert solutions, f"no IK for {target}"
        checked += 1
        assert any(
            np.linalg.norm(forward_kinematics(params, *solution) - target) < 0.5
            for solution in solutions
        )
    assert checked == 60


def test_within_limits_margin():
    limits = load_limits("configs/robot_limits.yaml")
    assert within_limits((0.0, 45.0, 45.0, 0.0), limits)
    assert not within_limits((89.0, 45.0, 45.0, 0.0), limits, margin_deg=2.0)
    assert not within_limits((0.0, -1.0, 45.0, 0.0), limits)


def test_three_axis_frame_uses_j1_j2_j3_and_ignores_j4():
    params = load_fk("configs/fk_magician.yaml")
    frame_a = three_axis_frame(params, 0.0, 45.0, 45.0, 100.0, 0.0, 50.0)
    assert np.allclose(frame_a[:3, 3], [0.1, 0.0, 0.05])
    frame_b = three_axis_frame(params, 0.0, 45.0, 60.0, 100.0, 0.0, 50.0)
    assert not np.allclose(frame_a[:3, :3], frame_b[:3, :3])
