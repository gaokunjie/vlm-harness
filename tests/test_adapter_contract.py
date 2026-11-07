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

    def capture(self, cameras=None):
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
        instruction="Place the red block at the black lines intersection.",
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


def test_step_pose_updates_state_and_returns_receipt():
    adapter = _adapter()
    command = PoseCommand(
        arms={
            "arm": ArmCommand(
                pose=Pose(position=(0.15, 0.0, 0.05), quaternion=(1.0, 0.0, 0.0, 0.0)),
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
