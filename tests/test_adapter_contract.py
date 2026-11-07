import numpy as np
import pytest

from dobot_rua.hardware.interface import FakeRobot
from dobot_rua.integration.adapter import AdapterConfig, DobotAdapter

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
