import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.tcp_extrinsic import solve_camera_and_tcp


def _rot_z(angle):
    return Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()


def _synthetic(n=6, seed=0):
    rng = np.random.default_rng(seed)
    yaw = np.radians(35.0)
    board_origin = np.array([150.0, 0.0, -21.0])
    tcp = np.array([12.0, -7.0, 58.0])
    # The physical marker coordinates live in the level-board frame (z up).
    # solvePnP returns the PnP board frame with +z pointing away from the
    # camera (down in the base frame), so the PnP pose carries an extra 180 deg
    # rotation about x relative to the level-board frame.
    rotation_cam_board_level = Rotation.from_euler(
        "xyz", [4.0, -3.0, 168.0], degrees=True
    ).as_matrix()
    rotation_cam_board = (
        rotation_cam_board_level
        @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    )
    board_origin_camera = np.array([120.0, -80.0, 300.0])

    markers = []
    camera_points = []
    for _ in range(n):
        marker = np.array([rng.uniform(-60, 60), rng.uniform(-90, 90), 0.0])
        markers.append(marker)
        camera_points.append(
            rotation_cam_board_level @ marker + board_origin_camera
        )

    tips = []
    rotations = []
    for marker in markers:
        tool = Rotation.from_euler(
            "zy", [rng.uniform(-60, 60), rng.uniform(-70, 70)], degrees=True
        ).as_matrix()
        rotations.append(tool)
        tips.append(board_origin + _rot_z(yaw) @ marker - tool @ tcp)

    truth = {
        "yaw": yaw,
        "board_origin": board_origin,
        "tcp": tcp,
        "rotation_cam_board": rotation_cam_board,
        "board_origin_camera": board_origin_camera,
    }
    return truth, np.array(camera_points), np.array(tips), np.array(rotations)


def test_solve_recovers_the_camera_and_the_tool_offset():
    truth, camera_points, tips, rotations = _synthetic()
    solution = solve_camera_and_tcp(
        truth["rotation_cam_board"],
        truth["board_origin_camera"],
        camera_points,
        tips,
        rotations,
    )

    assert solution.rms_mm < 0.05
    assert np.isclose(np.degrees(solution.yaw_rad), 35.0, atol=0.1)
    assert np.allclose(solution.tcp_mm, [12.0, -7.0, 58.0], atol=0.2)
    assert np.linalg.det(solution.rotation_world_optical) == pytest.approx(1.0)


def test_solve_applies_the_pnp_board_normal_down_convention():
    truth, camera_points, tips, rotations = _synthetic()
    solution = solve_camera_and_tcp(
        truth["rotation_cam_board"],
        truth["board_origin_camera"],
        camera_points,
        tips,
        rotations,
        board_normal_down=True,
    )
    rx180 = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    expected = (
        Rotation.from_euler("z", np.degrees(truth["yaw"]), degrees=True).as_matrix()
        @ rx180
        @ truth["rotation_cam_board"].T
    )
    assert np.allclose(solution.rotation_world_optical, expected, atol=1e-6)


def test_transform_from_solution_converts_millimetres_to_metres():
    from dobot_rua.calibration.tcp_extrinsic import transform_from_solution

    truth, camera_points, tips, rotations = _synthetic()
    solution = solve_camera_and_tcp(
        truth["rotation_cam_board"],
        truth["board_origin_camera"],
        camera_points,
        tips,
        rotations,
    )

    transform = transform_from_solution(solution)

    assert transform[:3, 3] == pytest.approx(
        np.asarray(solution.camera_position) / 1000.0
    )
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)


def test_solve_rejects_an_improper_board_rotation():
    truth, camera_points, tips, rotations = _synthetic()
    reflected = truth["rotation_cam_board"] @ np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        solve_camera_and_tcp(
            reflected,
            truth["board_origin_camera"],
            camera_points,
            tips,
            rotations,
        )


def test_solved_extrinsic_maps_camera_points_to_the_markers():
    truth, camera_points, tips, rotations = _synthetic()
    solution = solve_camera_and_tcp(
        truth["rotation_cam_board"],
        truth["board_origin_camera"],
        camera_points,
        tips,
        rotations,
    )

    for point, tip, tool in zip(camera_points, tips, rotations, strict=True):
        world = solution.rotation_world_optical @ point + solution.camera_position
        assert np.linalg.norm(world - (tip + tool @ solution.tcp_mm)) < 0.05


def test_solve_needs_three_touches():
    truth, camera_points, tips, rotations = _synthetic(n=3)
    with pytest.raises(ValueError, match="at least three"):
        solve_camera_and_tcp(
            truth["rotation_cam_board"],
            truth["board_origin_camera"],
            camera_points[:2],
            tips[:2],
            rotations[:2],
        )
