import numpy as np
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.blob_camera import solve_camera_from_blob

K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])


def _synthetic():
    r_cam_base = (
        Rotation.from_euler("x", 172.0, degrees=True)
        * Rotation.from_euler("z", 23.0, degrees=True)
    ).as_matrix()
    t = np.array([210.0, -35.0, 331.0])
    q = np.array([-55.0, 5.0, 25.0])
    rng = np.random.default_rng(2)
    poses, pixels = [], []
    for _ in range(9):
        j1 = rng.uniform(-20.0, 20.0)
        r_tool = Rotation.from_euler("z", j1, degrees=True).as_matrix()
        pos = np.array([rng.uniform(140.0, 175.0), rng.uniform(-35.0, 35.0), rng.uniform(8.0, 30.0)])
        point = pos + r_tool @ q
        cam = r_cam_base @ (point - t)
        uv = K @ cam
        poses.append((pos, r_tool))
        pixels.append(uv[:2] / uv[2])
    normal = r_cam_base @ np.array([0.0, 0.0, 1.0])
    return r_cam_base, t, q, poses, np.array(pixels), normal


def test_solve_camera_from_blob_recovers_pose():
    r_truth, t_truth, q_truth, poses, pixels, normal = _synthetic()
    result = solve_camera_from_blob(poses, pixels, K, normal, camera_z_mm=t_truth[2])
    assert np.allclose(result.rotation_cam_base, r_truth, atol=1e-4)
    assert np.allclose(result.camera_position_mm[:2], t_truth[:2], atol=1e-3)
    assert np.allclose(result.tool_point_mm, q_truth, atol=1e-3)
    assert result.rms_px < 1e-3
