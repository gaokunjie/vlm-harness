"""EnvironmentBackend for the Dobot Magician + binary suction setup.

The adapter is deliberately thin: it converts world-metre poses and a scalar
gripper command into Dobot native calls, validates safety, and freezes
observations. Multi-step motion programs stay in the shared gripper tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..hardware.interface import RobotInterface, RobotState
from ..hardware.safety import SafetyError, SafetyLimits

from uni_robot_agent.services.environment.contracts import (
    ArmCommand,
    ArmControlCapabilities,
    ArmDescription,
    ArmState,
    CameraDescription,
    EnvironmentDescription,
    EnvironmentState,
    JointCommand,
    NativeCommand,
    Pose,
    PoseCommand,
    RequestRejected,
    StepReceipt,
)
from uni_robot_agent.services.environment.snapshots import CameraFrame, CapturedObservation


class CameraSource(Protocol):
    def capture(self, cameras: list[str] | None = None) -> dict[str, CameraFrame]: ...


@dataclass(frozen=True)
class AdapterConfig:
    control_frame: str = "suction_tip"
    control_hz: float = 10.0
    max_episode_steps: int = 200
    max_step_mm: float = 100.0
    position_tolerance_mm: float = 5.0
    x_range_mm: tuple[float, float] = (-300.0, 300.0)
    y_range_mm: tuple[float, float] = (-300.0, 300.0)
    z_range_mm: tuple[float, float] = (-30.0, 300.0)
    cameras: tuple[str, ...] = ("front", "side", "wrist")


class DobotAdapter:
    def __init__(
        self,
        robot: RobotInterface,
        camera_source: CameraSource,
        config: AdapterConfig,
        *,
        episode_id: str,
        instruction: str,
        metadata: dict[str, Any] | None = None,
        safety_limits: SafetyLimits | None = None,
        success_fn: Any | None = None,
    ) -> None:
        self.robot = robot
        self.camera_source = camera_source
        self.config = config
        self.episode_id = episode_id
        self.instruction = instruction
        self.metadata = dict(metadata or {})
        self.safety_limits = safety_limits
        self.success_fn = success_fn
        self._state_version = 0
        self._native_steps = 0
        self._finished = False
        self._outcome: dict[str, Any] | None = None

    @property
    def description(self) -> EnvironmentDescription:
        return EnvironmentDescription(
            arms=[
                ArmDescription(
                    name="arm",
                    control_frame=self.config.control_frame,
                    grasp_site_description=(
                        "Suction-cup tip; normal points down. Binary vacuum: "
                        "grasp=ON, release=OFF. Rotation is fixed."
                    ),
                    rotation_radius_m=0.0,
                    joint_names=["j1", "j2", "j3", "j4"],
                    control=ArmControlCapabilities(
                        commands=["pose"],
                        pose_solver="native",
                        rotation="fixed",
                        rotation_reference="measured",
                        gripper="binary",
                    ),
                )
            ],
            cameras=[
                CameraDescription(name=name, role=("wrist" if name == "wrist" else "scene"),
                                  has_depth=False, has_calibration=True)
                for name in self.config.cameras
            ],
            control_frequency_hz=self.config.control_hz,
            max_episode_steps=self.config.max_episode_steps,
            observation_interval_steps=1,
            metadata={
                "task_id": self.metadata.get("task_id", "line_intersection"),
                "seed": self.metadata.get("seed"),
                "instruction": self.instruction,
                "gripper_semantics": "grasp=suction ON (0), release=suction OFF (1)",
                "coordinate_frame": "world metres; x forward, y left, z up; wxyz quaternions",
                **self.metadata,
            },
        )

    def read_state(self) -> EnvironmentState:
        robot = self.robot.read_state()
        return EnvironmentState(
            arms={"arm": self._arm_state(robot)},
            state_version=self._state_version,
            native_steps=self._native_steps,
            ended=self._finished or self._outcome is not None,
        )

    def _arm_state(self, robot: RobotState) -> ArmState:
        gripper = 0.0 if robot.suction_on else 1.0
        return ArmState(
            control_pose=Pose(
                position=(robot.x_mm / 1000.0, robot.y_mm / 1000.0, robot.z_mm / 1000.0),
                quaternion=(1.0, 0.0, 0.0, 0.0),
            ),
            joints=[
                float(np.radians(value))
                for value in (robot.j1_deg, robot.j2_deg, robot.j3_deg, robot.j4_deg)
            ],
            gripper=gripper,
            gripper_command=gripper,
            metadata={"measured_xyz_mm": [robot.x_mm, robot.y_mm, robot.z_mm]},
        )

    def _check_pose(self, position_m: tuple[float, float, float]) -> None:
        x_mm, y_mm, z_mm = (float(value) * 1000.0 for value in position_m)
        for value, (low, high), axis in (
            (x_mm, self.config.x_range_mm, "x"),
            (y_mm, self.config.y_range_mm, "y"),
            (z_mm, self.config.z_range_mm, "z"),
        ):
            if not (low <= value <= high):
                raise RequestRejected(f"{axis}={value:.1f} mm outside workspace [{low}, {high}]")
        current = self.robot.read_state()
        distance = float(
            np.linalg.norm(
                np.array([x_mm, y_mm, z_mm]) - np.array([current.x_mm, current.y_mm, current.z_mm])
            )
        )
        if distance > self.config.max_step_mm:
            raise RequestRejected(
                f"step {distance:.1f} mm exceeds max {self.config.max_step_mm:.1f} mm"
            )

    def validate_commands(self, commands: list[Any], mode: str) -> None:
        if mode != "pose":
            raise RequestRejected("Dobot Magician accepts pose commands only")
        for command in commands:
            if not isinstance(command, PoseCommand):
                raise RequestRejected("expected a PoseCommand")
            if set(command.arms) != {"arm"}:
                raise RequestRejected("exactly one arm target is required")
            target = command.arms["arm"]
            if not isinstance(target, ArmCommand):
                raise RequestRejected("joint targets are not supported")
            self._check_pose(target.pose.position)

    def step_pose(self, command: PoseCommand) -> StepReceipt:
        self.validate_commands([command], "pose")
        if self._finished or self._outcome is not None:
            raise RequestRejected("episode has ended")
        target = command.arms["arm"]
        x_mm, y_mm, z_mm = (float(value) * 1000.0 for value in target.pose.position)
        suction_on = float(target.gripper) <= 0.5
        self.robot.set_suction(suction_on)
        self.robot.move_pose(x_mm, y_mm, z_mm, r_deg=0.0)
        self._state_version += 1
        self._native_steps += 1
        measured = self.robot.read_state()
        error_mm = float(
            np.linalg.norm(
                np.array([x_mm, y_mm, z_mm])
                - np.array([measured.x_mm, measured.y_mm, measured.z_mm])
            )
        )
        stop_reason = "arrived" if error_mm <= self.config.position_tolerance_mm else "stalled"
        return StepReceipt(
            accepted=True,
            state=self.read_state(),
            diagnostics={
                "stop_reason": stop_reason,
                "position_error_mm": error_mm,
                "executed_steps": 1,
                "target_xyz_mm": [x_mm, y_mm, z_mm],
                "measured_xyz_mm": [measured.x_mm, measured.y_mm, measured.z_mm],
            },
        )

    def step_joints(self, command: JointCommand) -> StepReceipt:
        raise RequestRejected("fixed-pose suction control does not expose joint commands")

    def step_native(self, command: NativeCommand) -> StepReceipt:
        values = list(command.values)
        if len(values) != 1:
            raise RequestRejected("native control expects one binary suction value")
        self.robot.set_suction(float(values[0]) <= 0.5)
        self._state_version += 1
        self._native_steps += 1
        return StepReceipt(
            accepted=True,
            state=self.read_state(),
            diagnostics={"stop_reason": "arrived", "executed_steps": 1},
        )

    def preview_joints(self, commands: list[JointCommand]) -> dict[str, Any]:
        raise RequestRejected("joint previews are unavailable for fixed-pose control")

    def capture(
        self, cameras: list[str] | None = None, *, high_resolution: bool = False
    ) -> CapturedObservation:
        frames = self.camera_source.capture(cameras)
        return CapturedObservation(
            episode_id=self.episode_id,
            state_version=self._state_version,
            captured_at=time.time(),
            state=self.read_state().model_dump(mode="json"),
            cameras=frames,
            terminated=self._finished or self._outcome is not None,
        )

    def finish(self, reason: str, status: str | None = None) -> dict[str, Any]:
        if self._outcome is None:
            success = None if self.success_fn is None else bool(self.success_fn())
            self._outcome = {
                "task_success": success,
                "reason": reason,
                "status": status,
                "native_steps": self._native_steps,
            }
        self._finished = True
        return dict(self._outcome)

    def close(self) -> None:
        self._finished = True
        self.robot.close()

    def set_success_fn(self, fn: Any) -> None:
        self.success_fn = fn
