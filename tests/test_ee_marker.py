import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.ee_marker import solve_camera_from_ee_marker


def _scene(n=12, seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    intrinsic = np.array([[1001.0, 0.0, 637.0], [0.0, 994.0, 372.0], [0.0, 0.0, 1.0]])
    # world -> camera
    rvec = Rotation.from_euler("xyz", [150.0, 4.0, 18.0], degrees=True).as_rotvec()
    tvec = np.array([180.0, -90.0, 320.0])
    offset = np.array([0.5, -0.3, 0.0])  # marker centre in the flange frame

    flange = []
    j4 = []
    for _ in range(n):
        flange.append([rng.uniform(120, 260), rng.uniform(-90, 90), rng.uniform(20, 160)])
        j4.append(rng.uniform(-70, 70))
    flange = np.array(flange)
    j4 = np.array(j4)

    markers = np.array(
        [f + Rotation.from_euler("z", j, degrees=True).apply(offset) for f, j in zip(flange, j4)]
    )
    pixels, _ = _project(markers, rvec, tvec, intrinsic)
    pixels = pixels + rng.normal(scale=noise_px, size=pixels.shape)
    return intrinsic, rvec, tvec, offset, pixels, flange, j4


def _project(points, rvec, tvec, intrinsic):
    rotation = Rotation.from_rotvec(rvec).as_matrix()
    camera = points @ rotation.T + tvec
    uv = camera @ intrinsic.T
    return uv[:, :2] / uv[:, 2:3], camera


def test_recovers_the_camera_pose_and_the_marker_offset():
    intrinsic, rvec, tvec, offset, pixels, flange, j4 = _scene()
    solution = solve_camera_from_ee_marker(pixels, flange, j4, intrinsic)

    expected_rotation = Rotation.from_rotvec(rvec).as_matrix().T
    assert np.allclose(solution.rotation_world_optical, expected_rotation, atol=1e-4)
    assert np.allclose(solution.camera_position_mm, -expected_rotation @ tvec, atol=1e-2)
    assert np.allclose(solution.marker_offset_mm, offset, atol=1e-2)
    assert solution.rms_px < 1e-4


def test_reports_the_reprojection_error_under_noise():
    intrinsic, _, _, _, pixels, flange, j4 = _scene(seed=1, noise_px=0.5)
    solution = solve_camera_from_ee_marker(pixels, flange, j4, intrinsic)

    assert solution.rms_px < 1.0
    assert len(solution.residuals_px) == len(pixels)


def test_needs_at_least_four_poses():
    intrinsic, _, _, _, pixels, flange, j4 = _scene(n=3)
    with pytest.raises(ValueError, match="at least four"):
        solve_camera_from_ee_marker(pixels[:3], flange[:3], j4[:3], intrinsic)


def test_rejects_coplanar_poses():
    intrinsic = np.eye(3)
    pixels = np.array([[0.0, 0.0]] * 4)
    flange = np.array([[100.0, 0.0, 50.0]] * 4)
    j4 = np.zeros(4)
    with pytest.raises(ValueError, match="not degenerate"):
        solve_camera_from_ee_marker(pixels, flange, j4, intrinsic)


def _corner_scene(n=8, seed=0, side=18.0, yaw_offset=0.4, noise_px=0.0):
    rng = np.random.default_rng(seed)
    intrinsic = np.array([[1001.0, 0.0, 637.0], [0.0, 994.0, 372.0], [0.0, 0.0, 1.0]])
    rvec = Rotation.from_euler("xyz", [150.0, 4.0, 18.0], degrees=True).as_rotvec()
    tvec = np.array([180.0, -90.0, 320.0])
    offset = np.array([0.6, -0.4, 0.0])
    half = side / 2.0
    local = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])

    flange, j1, j4, corners = [], [], [], []
    for _ in range(n):
        f = np.array([rng.uniform(130, 240), rng.uniform(-40, 0), rng.uniform(0, 40)])
        a1 = np.degrees(np.arctan2(f[1], f[0]))
        a4 = rng.uniform(-30, 30)
        centre = f + Rotation.from_euler("z", a4, degrees=True).apply(offset)
        yaw = np.radians(a1 + a4) + yaw_offset
        pts = centre + local @ Rotation.from_euler("z", yaw).as_matrix().T
        rotation = Rotation.from_rotvec(rvec).as_matrix()
        camera = pts @ rotation.T + tvec
        uv = camera @ intrinsic.T
        uv = uv[:, :2] / uv[:, 2:3]
        corners.append(uv + rng.normal(scale=noise_px, size=uv.shape))
        flange.append(f); j1.append(a1); j4.append(a4)
    return intrinsic, rvec, tvec, np.array(corners), np.array(flange), np.array(j1), np.array(j4)


def test_corner_solver_recovers_the_camera_pose():
    from dobot_rua.calibration.ee_marker import solve_camera_from_marker_corners

    intrinsic, rvec, tvec, corners, flange, j1, j4 = _corner_scene()
    solution = solve_camera_from_marker_corners(corners, flange, j4, j1, 18.0, intrinsic)

    expected = Rotation.from_rotvec(rvec).as_matrix().T
    assert np.allclose(solution.rotation_world_optical, expected, atol=1e-3)
    assert np.allclose(solution.camera_position_mm, -expected @ tvec, atol=0.5)
    assert solution.rms_px < 1e-3


def test_corner_solver_tolerates_corner_noise():
    from dobot_rua.calibration.ee_marker import solve_camera_from_marker_corners

    intrinsic, rvec, tvec, corners, flange, j1, j4 = _corner_scene(seed=2, noise_px=0.4)
    solution = solve_camera_from_marker_corners(corners, flange, j4, j1, 18.0, intrinsic)

    assert solution.rms_px < 1.0
    assert np.allclose(solution.camera_position_mm, -Rotation.from_rotvec(rvec).as_matrix().T @ tvec, atol=3.0)
