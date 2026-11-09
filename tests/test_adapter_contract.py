import numpy as np
import pytest

from dobot_rua.hardware.interface import FakeRobot
from dobot_rua.hardware.safety import SafetyLimits
from dobot_rua.integration.adapter import AdapterConfig, DobotAdapter
from dobot_rua.kinematics import load_fk

from uni_robot_agent.services.environment.contracts import (
    ArmCommand,
    Pose,
    PoseCommand,
    RequestRejected,
)
from uni_robot_agent.services.environment.snapshots import CameraFrame


class FakeCameraSource:
    def __init__(self):
        self.K = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        self.T = np.eye(4)
        self.T[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        self.T[:3, 3] = [0.0, 0.0, 1.0]
        self.last_tool_pose = None

    def capture(self, cameras=None, tool_pose=None):
        self.last_tool_pose = tool_pose
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        names = cameras or ["front", "side", "wrist"]
        return {
            name: CameraFrame(rgb=image, depth_m=None, intrinsic_K=self.K.copy(),
                              T_world_from_optical=self.T.copy())
            for name in names
        }


def _adapter():
    return DobotAdapter(
        robot=FakeRobot(),
        camera_source=FakeCameraSource(),
        config=AdapterConfig(),
        episode_id="trial-1",
        instruction="Place the yellow block at the black lines intersection.",
        metadata={"task_id": "line_intersection", "seed": 1},
    )


def test_description_declares_fixed_rotation_and_binary_gripper():
    adapter = _adapter()
    description = adapter.description
    arm = description.arms[0]
    assert arm.name == "arm"
    assert arm.control_frame == "suction_tip"
    assert arm.control.rotation == "fixed"
    assert arm.control.gripper == "binary"
    assert {camera.name for camera in description.cameras} == {"front", "side", "wrist"}


def test_read_state_is_a_contract_state():
    adapter = _adapter()
    state = adapter.read_state()
    assert "arm" in state.arms
    assert state.arms["arm"].gripper in (0.0, 1.0)
    assert state.native_steps == 0


def test_episode_identity_returns_instruction_and_metadata():
    adapter = _adapter()
    instruction, metadata = adapter.episode_identity()
    assert instruction == adapter.instruction
    assert metadata["task_id"] == "line_intersection"


def test_description_reflects_an_uncalibrated_camera_source():
    class UncalibratedSource(FakeCameraSource):
        has_calibration = False

    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=UncalibratedSource(),
        config=AdapterConfig(),
        episode_id="trial-uc",
        instruction="test",
        metadata={},
    )

    assert all(camera.has_calibration is False for camera in adapter.description.cameras)


def test_description_can_declare_uncalibrated_cameras():
    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=FakeCameraSource(),
        config=AdapterConfig(has_calibration=False),
        episode_id="trial-u",
        instruction="test",
        metadata={},
    )
    assert all(camera.has_calibration is False for camera in adapter.description.cameras)


def test_close_releases_camera_source():
    class ClosingSource(FakeCameraSource):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    source = ClosingSource()
    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=source,
        config=AdapterConfig(),
        episode_id="trial-c",
        instruction="test",
        metadata={},
    )
    adapter.close()
    assert source.closed is True


def test_step_pose_updates_state_and_returns_receipt():
    adapter = _adapter()
    command = PoseCommand(
        arms={
            "arm": ArmCommand(
                pose=Pose(position=(0.15, 0.0, 0.05), quaternion=(0.0, 1.0, 0.0, 0.0)),
                gripper=0.0,
            )
        }
    )
    receipt = adapter.step_pose(command)
    assert receipt.accepted
    position = receipt.state.arms["arm"].control_pose.position
    assert position[0] == pytest.approx(0.15, abs=1e-6)
    assert receipt.diagnostics["stop_reason"] in {"arrived", "stalled", "limit"}


def test_capture_freezes_three_calibrated_cameras():
    adapter = _adapter()
    observation = adapter.capture()
    assert set(observation.cameras) == {"front", "side", "wrist"}
    for frame in observation.cameras.values():
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.intrinsic_K.shape == (3, 3)
        assert frame.T_world_from_optical.shape == (4, 4)


def test_capture_passes_current_tool_pose():
    source = FakeCameraSource()
    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=source,
        config=AdapterConfig(),
        episode_id="trial-pose",
        instruction="test",
        metadata={},
    )
    adapter.capture()
    assert source.last_tool_pose is not None
    assert source.last_tool_pose.shape == (4, 4)


def test_step_joints_is_rejected_for_fixed_pose_control():
    adapter = _adapter()
    from uni_robot_agent.services.environment.contracts import JointCommand, JointTarget

    command = JointCommand(arms={"arm": JointTarget(joints=[0.0, 45.0, 45.0, 0.0], gripper=0.0)})
    with pytest.raises(RequestRejected):
        adapter.step_joints(command)


def test_finish_returns_offline_outcome():
    adapter = _adapter()
    result = adapter.finish("done", status="success")
    assert "task_success" in result
    assert result["reason"] == "done"


def test_read_state_reports_tool_down_orientation():
    adapter = _adapter()
    quaternion = adapter.read_state().arms["arm"].control_pose.quaternion
    assert quaternion == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-6)


def test_step_pose_rejects_orientation_change():
    adapter = _adapter()
    command = PoseCommand(
        arms={
            "arm": ArmCommand(
                pose=Pose(position=(0.12, 0.0, 0.02), quaternion=(1.0, 0.0, 0.0, 0.0)),
                gripper=1.0,
            )
        }
    )
    with pytest.raises(RequestRejected):
        adapter.step_pose(command)


def test_adapter_rejects_unreachable_pose_when_kinematics_supplied():
    from dobot_rua.hardware.safety import SafetyLimits
    from dobot_rua.kinematics import load_fk

    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=FakeCameraSource(),
        config=AdapterConfig(
            x_range_mm=(-1000.0, 1000.0),
            y_range_mm=(-1000.0, 1000.0),
            z_range_mm=(-1000.0, 1000.0),
            max_step_mm=5000.0,
        ),
        episode_id="trial-2",
        instruction="Place the yellow block at the black lines intersection.",
        metadata={},
        fk_params=load_fk("configs/fk_magician.yaml"),
        safety_limits=SafetyLimits.from_yaml("configs/robot_limits.yaml"),
    )
    command = PoseCommand(
        arms={
            "arm": ArmCommand(
                pose=Pose(position=(0.9, 0.9, 0.02), quaternion=(0.0, 1.0, 0.0, 0.0)),
                gripper=1.0,
            )
        }
    )
    with pytest.raises(RequestRejected):
        adapter.step_pose(command)


def test_capture_uses_three_axis_frame_when_fk_supplied():
    from dobot_rua.kinematics import load_fk

    source = FakeCameraSource()
    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=source,
        config=AdapterConfig(),
        episode_id="trial-three-axis",
        instruction="test",
        metadata={},
        fk_params=load_fk("configs/fk_magician.yaml"),
    )
    adapter.capture()
    assert source.last_tool_pose is not None
    assert np.allclose(source.last_tool_pose[:3, 3], [0.12, 0.0, 0.02], atol=1e-6)


class StickyRobot(FakeRobot):
    """FakeRobot that ignores motion commands."""

    def move_pose(self, x_mm, y_mm, z_mm, r_deg, timeout_s=5.0):
        return None


def _adapter_with_tip(tip_length_mm=66.6, robot=None):
    if robot is None:
        robot = FakeRobot()
    adapter = DobotAdapter(
        robot=robot,
        camera_source=FakeCameraSource(),
        config=AdapterConfig(tip_length_mm=tip_length_mm),
        episode_id="trial-tip",
        instruction="test",
        metadata={},
    )
    return adapter, robot


def _tip_command(x_m, y_m, z_m):
    return PoseCommand(
        arms={
            "arm": ArmCommand(
                pose=Pose(
                    position=(x_m, y_m, z_m),
                    quaternion=(0.0, 1.0, 0.0, 0.0),
                ),
                gripper=0.0,
            )
        }
    )


def test_read_state_reports_the_suction_tip_below_the_flange():
    adapter, _robot = _adapter_with_tip(66.6)

    state = adapter.read_state()
    position = state.arms["arm"].control_pose.position

    assert position == pytest.approx((0.12, 0.0, -0.0466), abs=1e-6)
    assert state.arms["arm"].metadata["flange_xyz_mm"] == pytest.approx(
        [120.0, 0.0, 20.0]
    )


def test_step_pose_commands_the_flange_above_the_tip():
    adapter, robot = _adapter_with_tip(66.6)

    receipt = adapter.step_pose(_tip_command(0.13, 0.0, 0.05))

    assert robot.read_state().z_mm == pytest.approx(116.6, abs=1e-6)
    tip = receipt.state.arms["arm"].control_pose.position
    assert tip == pytest.approx((0.13, 0.0, 0.05), abs=1e-6)
    assert receipt.diagnostics["stop_reason"] == "arrived"


def test_step_pose_reports_the_tip_error():
    robot = StickyRobot()
    robot.move_pose(120.0, 0.0, 20.0, 0.0)
    adapter, _robot = _adapter_with_tip(66.6, robot)

    receipt = adapter.step_pose(_tip_command(0.13, 0.0, 0.05))

    # target tip (130, 0, 50); measured tip (120, 0, 20 - 66.6)
    assert receipt.diagnostics["position_error_mm"] == pytest.approx(
        float(np.linalg.norm([10.0, 0.0, 96.6])), abs=1e-6
    )
    assert receipt.diagnostics["stop_reason"] == "stalled"


def test_step_pose_accepts_a_table_level_tip_target():
    adapter, robot = _adapter_with_tip(66.6)

    receipt = adapter.step_pose(_tip_command(0.15, 0.0, -0.0876))

    assert receipt.accepted
    assert robot.read_state().z_mm == pytest.approx(-21.0, abs=1e-6)


def test_tip_length_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        DobotAdapter(
            robot=FakeRobot(),
            camera_source=FakeCameraSource(),
            config=AdapterConfig(tip_length_mm=-1.0),
            episode_id="trial-bad-tip",
            instruction="test",
            metadata={},
        )


def test_capture_keeps_the_flange_frame_for_the_wrist_camera():
    source = FakeCameraSource()
    adapter = DobotAdapter(
        robot=FakeRobot(),
        camera_source=source,
        config=AdapterConfig(tip_length_mm=66.6),
        episode_id="trial-wrist-frame",
        instruction="test",
        metadata={},
        fk_params=load_fk("configs/fk_magician.yaml"),
    )

    adapter.capture()

    assert np.allclose(source.last_tool_pose[:3, 3], [0.12, 0.0, 0.02], atol=1e-6)
