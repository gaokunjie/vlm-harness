import numpy as np

from dobot_rua.calibration.front_planar import (
    CalibrationRejected,
    PlanarSample,
    fit_two_plane,
    load_planar_calibration,
    nearest_solution,
    plan_grid,
    save_planar_calibration,
)

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
DIST = np.zeros(5)


def _project(point, transform, intrinsic):
    from dobot_rua.geometry import project_point_to_pixel

    return project_point_to_pixel(intrinsic, transform, point)


def _samples(transform, table_z=-0.02, top_z=0.005):
    out = []
    for plane, z in (("table", table_z), ("top", top_z)):
        for x in (-0.05, 0.0, 0.05):
            for y in (-0.04, 0.0, 0.04):
                point = np.array([x, y, z])
                pixel = _project(point, transform, K)
                out.append(
                    PlanarSample(
                        tip_xyz_m=tuple(point),
                        cup_pixel_uv=tuple(pixel),
                        plane=plane,
                        marker_count=12,
                        board_reprojection_px=0.6,
                    )
                )
    return out


def test_fit_two_plane_recovers_synthetic_pose():
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_gt[:3, 3] = [0.01, 0.02, 0.30]
    cal, diag = fit_two_plane(
        _samples(T_gt),
        K,
        DIST,
        -0.02,
        0.005,
        (25.0, 75.0, 0.0),
        0.025,
        0.002,
        {"focus": 350},
        holdout=2,
    )
    assert np.allclose(cal.T_world_from_optical[:3, :3], T_gt[:3, :3], atol=0.02)
    assert np.allclose(cal.T_world_from_optical[:3, 3], T_gt[:3, 3], atol=0.002)
    assert cal.fit_rmse_px < 1.0
    assert cal.H_table.shape == (3, 3) and cal.H_top.shape == (3, 3)


def test_fit_rejects_too_few_samples():
    try:
        fit_two_plane([], K, DIST, -0.02, 0.005, (25, 75, 0), 0.025, 0.002, {})
    except CalibrationRejected:
        pass
    else:
        raise AssertionError("empty samples accepted")


def test_save_load_round_trip(tmp_path):
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_gt[:3, 3] = [0.01, 0.02, 0.30]
    cal, _ = fit_two_plane(
        _samples(T_gt),
        K,
        DIST,
        -0.02,
        0.005,
        (25.0, 75.0, 0.0),
        0.025,
        0.002,
        {"focus": 350},
        holdout=2,
    )
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n  front:\n    device: /dev/video2\n", encoding="utf-8")
    save_planar_calibration(path, "front", cal)
    loaded = load_planar_calibration(path, "front")
    assert loaded is not None
    assert np.allclose(loaded.T_world_from_optical, cal.T_world_from_optical)
    assert loaded.calibration_hash == cal.calibration_hash


def test_load_rejects_hash_mismatch(tmp_path):
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_gt[:3, 3] = [0.01, 0.02, 0.30]
    cal, _ = fit_two_plane(
        _samples(T_gt),
        K,
        DIST,
        -0.02,
        0.005,
        (25.0, 75.0, 0.0),
        0.025,
        0.002,
        {"focus": 350},
        holdout=2,
    )
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n  front:\n    device: /dev/video2\n", encoding="utf-8")
    save_planar_calibration(path, "front", cal)
    text = path.read_text(encoding="utf-8").replace(cal.calibration_hash, "0" * 64)
    path.write_text(text, encoding="utf-8")
    assert load_planar_calibration(path, "front") is None


def test_plan_grid_and_nearest_solution():
    grid = plan_grid((0.0, 0.0), 20.0, 3, 2)
    assert len(grid) == 6
    assert grid[0] == (-20.0, -10.0)
    solutions = [(0.0, 10.0, 50.0), (1.0, 11.0, 49.0)]
    assert nearest_solution(solutions, (1.2, 11.2, 49.2)) == (1.0, 11.0, 49.0)
