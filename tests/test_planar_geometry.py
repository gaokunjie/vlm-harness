import numpy as np

from dobot_rua.geometry import (
    apply_homography,
    derive_homography,
    point_in_polygon,
    project_point_to_pixel,
    project_pixel_to_plane,
    solve_pnp_two_plane,
    undistort_points,
    validate_rigid_transform,
)

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])


def test_undistort_identity_is_noop():
    pts = np.array([[320.0, 240.0], [400.0, 300.0]])
    out = undistort_points(pts, K, np.zeros(5))
    assert np.allclose(out, pts, atol=1e-6)


def test_undistort_moves_edge_point():
    pts = np.array([[600.0, 400.0]])
    out = undistort_points(pts, K, np.array([0.1, -0.05, 0, 0, 0.0]))
    assert np.linalg.norm(out - pts) > 1.0


def test_project_pixel_to_plane_uses_distortion():
    T = np.eye(4)
    T[2, 3] = 1.0
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    dist = np.array([0.1, -0.05, 0, 0, 0.0])
    px = np.array([600.0, 400.0])
    expected = project_pixel_to_plane(
        K, T, undistort_points(px[None], K, dist)[0], 0.0
    )
    got = project_pixel_to_plane(K, T, px, 0.0, distortion=dist)
    assert np.allclose(got, expected, atol=1e-9)


def test_solve_pnp_two_plane_recovers_pose():
    R = np.diag([1.0, -1.0, -1.0])
    T_gt = np.eye(4)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = [0.02, -0.01, 0.30]
    world = []
    for z in (-0.02, 0.005):
        for x in (-0.05, 0.0, 0.05):
            for y in (-0.04, 0.0, 0.04):
                world.append([x, y, z])
    world = np.array(world)
    image = np.array([project_point_to_pixel(K, T_gt, p) for p in world])
    T = solve_pnp_two_plane(world, image, K, np.zeros(5))
    assert np.allclose(T[:3, :3], T_gt[:3, :3], atol=0.02)
    assert np.allclose(T[:3, 3], T_gt[:3, 3], atol=0.002)


def test_derive_homography_matches_projection():
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.01, 0.02, 0.25]
    z = -0.02
    world = np.array([[-0.05, -0.03], [0.02, 0.04], [0.06, -0.01]])
    pixels = np.array([project_point_to_pixel(K, T, [x, y, z]) for x, y in world])
    H = derive_homography(T, K, z)
    got = apply_homography(H, pixels)
    assert np.allclose(got, world, atol=1e-6)


def test_point_in_polygon():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert point_in_polygon(np.array([0.5, 0.5]), square)
    assert not point_in_polygon(np.array([1.5, 0.5]), square)


def test_validate_rigid_rejects_reflection():
    T = np.eye(4)
    T[2, 2] = -1.0
    try:
        validate_rigid_transform(T, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("reflection accepted")
