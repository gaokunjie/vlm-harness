import cv2
import numpy as np

from dobot_rua.calibration.front_planar import (
    CalibrationRejected,
    PlanarSample,
    camera_controls_from_config,
    fit_two_plane,
    load_planar_calibration,
    nearest_solution,
    plan_grid,
    save_planar_calibration,
)
from dobot_rua.geometry import point_in_polygon

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
DIST = np.zeros(5)
REAL_DIST = np.array([0.16743173086769353, -0.2430665205621797, 0.002663547506527537, -9.843352823648358e-05, 0.11775657927190833])


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
    assert cal.calibrated_region.shape[1] == 2
    assert len(cal.calibrated_region) >= 3
    assert point_in_polygon(np.array([0.0, 0.0]), cal.calibrated_region)


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
    assert (tmp_path / "cameras.yaml.bak").exists()
    assert loaded is not None
    assert np.allclose(loaded.T_world_from_optical, cal.T_world_from_optical)
    assert np.allclose(loaded.calibrated_region, cal.calibrated_region)
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


def _distorted_samples(transform, distortion, table_z=-0.02, top_z=0.005):
    camera_from_world = np.linalg.inv(transform)
    rvec, _ = cv2.Rodrigues(camera_from_world[:3, :3])
    tvec = camera_from_world[:3, 3]
    out = []
    for plane, z in (("table", table_z), ("top", top_z)):
        for x in (-0.12, -0.06, 0.0, 0.06, 0.12):
            for y in (-0.08, 0.0, 0.08):
                point = np.array([x, y, z])
                pixel, _ = cv2.projectPoints(
                    point.reshape(1, 3), rvec, tvec, K, distortion
                )
                out.append(
                    PlanarSample(
                        tip_xyz_m=tuple(point),
                        cup_pixel_uv=(
                            float(pixel[0, 0, 0]),
                            float(pixel[0, 0, 1]),
                        ),
                        plane=plane,
                        marker_count=12,
                        board_reprojection_px=0.6,
                    )
                )
    return out


def test_fit_two_plane_reprojects_undistorted_pixels():
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_gt[:3, 3] = [0.01, 0.02, 0.30]
    cal, _ = fit_two_plane(
        _distorted_samples(T_gt, REAL_DIST),
        K,
        REAL_DIST,
        -0.02,
        0.005,
        (25.0, 75.0, 0.0),
        0.025,
        0.002,
        {"focus": 350},
        holdout=2,
    )
    assert cal.fit_rmse_px < 1.0
    assert cal.holdout_max_px < 1.0


def test_camera_controls_require_lock_keys():
    try:
        camera_controls_from_config({"focus_absolute": 350})
    except ValueError:
        pass
    else:
        raise AssertionError("missing camera controls accepted")


def test_fit_rejects_too_close_planes():
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_gt[:3, 3] = [0.01, 0.02, 0.30]
    try:
        fit_two_plane(
            _samples(T_gt),
            K,
            DIST,
            0.0,
            0.01,
            (25.0, 75.0, 0.0),
            0.025,
            0.002,
            {"focus": 350},
            holdout=2,
        )
    except CalibrationRejected:
        pass
    else:
        raise AssertionError("too-close planes accepted")


def test_provisional_calibration_round_trip(tmp_path):
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
        calibration_status="PROVISIONAL",
    )
    assert cal.calibration_status == "PROVISIONAL"
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n  front:\n    device: /dev/video2\n", encoding="utf-8")
    save_planar_calibration(path, "front", cal)
    loaded = load_planar_calibration(path, "front")
    assert loaded is not None
    assert loaded.calibration_status == "PROVISIONAL"
    assert loaded.calibration_hash == cal.calibration_hash
