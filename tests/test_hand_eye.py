import numpy as np
from scipy.spatial.transform import Rotation

from dobot_rua.calibration.hand_eye import solve_hand_eye


def test_solve_hand_eye_recovers_camera_to_tool():
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler("xyz", [10.0, -5.0, 20.0], degrees=True).as_matrix()
    known[:3, 3] = [0.02, -0.01, 0.05]

    board_in_base = np.eye(4)
    board_in_base[:3, 3] = [0.2, 0.0, 0.0]

    rng = np.random.default_rng(0)
    gripper_poses = []
    target_poses = []
    for _ in range(20):
        base_from_tool = np.eye(4)
        base_from_tool[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        base_from_tool[:3, 3] = rng.uniform(-0.1, 0.1, 3)
        base_from_camera = base_from_tool @ known
        camera_from_board = np.linalg.inv(base_from_camera) @ board_in_base
        gripper_poses.append(base_from_tool)
        target_poses.append(camera_from_board)

    result = solve_hand_eye(gripper_poses, target_poses)
    rotation_error = Rotation.from_matrix(
        result.transform[:3, :3] @ known[:3, :3].T
    ).magnitude()
    translation_error = np.linalg.norm(result.transform[:3, 3] - known[:3, 3])
    assert rotation_error < 0.02
    assert translation_error < 0.003


def test_solve_eye_to_hand_recovers_fixed_camera_extrinsic():
    from dobot_rua.calibration.hand_eye import solve_eye_to_hand

    base_from_camera_truth = np.eye(4)
    base_from_camera_truth[:3, :3] = Rotation.from_euler(
        "xyz", [175.0, 2.0, -8.0], degrees=True
    ).as_matrix()
    base_from_camera_truth[:3, 3] = [0.58, 0.02, 0.71]

    tool_from_board = np.eye(4)
    tool_from_board[:3, :3] = Rotation.from_euler(
        "xyz", [90.0, 0.0, 15.0], degrees=True
    ).as_matrix()
    tool_from_board[:3, 3] = [0.004, -0.002, -0.17]

    rng = np.random.default_rng(3)
    base_poses = []
    camera_from_board_poses = []
    for _ in range(18):
        base_from_tool = np.eye(4)
        base_from_tool[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        base_from_tool[:3, 3] = rng.uniform(-0.15, 0.15, 3)
        camera_from_board = (
            np.linalg.inv(base_from_camera_truth) @ base_from_tool @ tool_from_board
        )
        base_poses.append(base_from_tool)
        camera_from_board_poses.append(camera_from_board)

    result = solve_eye_to_hand(base_poses, camera_from_board_poses)
    rotation_error = Rotation.from_matrix(
        result.base_from_camera[:3, :3] @ base_from_camera_truth[:3, :3].T
    ).magnitude()
    translation_error = np.linalg.norm(
        result.base_from_camera[:3, 3] - base_from_camera_truth[:3, 3]
    )
    assert rotation_error < 0.02
    assert translation_error < 0.003
    assert len(result.estimates) == len(base_poses)


def test_solve_eye_to_hand_spread_is_small_on_consistent_data():
    from dobot_rua.calibration.hand_eye import solve_eye_to_hand

    base_from_camera_truth = np.eye(4)
    base_from_camera_truth[:3, 3] = [0.5, 0.0, 0.6]
    tool_from_board = np.eye(4)
    tool_from_board[:3, 3] = [0.0, 0.0, -0.1]

    rng = np.random.default_rng(11)
    base_poses = []
    targets = []
    for _ in range(12):
        base = np.eye(4)
        base[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        base[:3, 3] = rng.uniform(-0.1, 0.1, 3)
        base_poses.append(base)
        targets.append(np.linalg.inv(base_from_camera_truth) @ base @ tool_from_board)

    result = solve_eye_to_hand(base_poses, targets)
    translations = np.array([estimate[:3, 3] for estimate in result.estimates])
    assert np.linalg.norm(np.std(translations, axis=0)) < 1e-4
