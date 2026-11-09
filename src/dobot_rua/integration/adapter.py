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
from scipy.spatial.transform import Rotation

from ..hardware.interface import RobotInterface, RobotState
from ..hardware.safety import SafetyError, SafetyLimits
from ..geometry import pose_to_matrix
from ..kinematics import FKParams, inverse_kinematics, three_axis_frame

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
    def capture(
        self, cameras: list[str] | None = None, tool_pose: np.ndarray | None = None
    ) -> dict[str, CameraFrame]: ...


@dataclass(frozen=True)
class AdapterConfig:
    control_frame: str = "suction_tip"
    control_hz: float = 10.0
    max_episode_steps: int = 200
    max_step_mm: float = 100.0
    position_tolerance_mm: float = 5.0
    orientation_tolerance_rad: float = 0.35
    x_range_mm: tuple[float, float] = (-300.0, 300.0)
    y_range_mm: tuple[float, float] = (-300.0, 300.0)
    z_range_mm: tuple[float, float] = (-200.0, 300.0)
    cameras: tuple[str, ...] = ("front", "side", "wrist")
    has_calibration: bool = True
    # Vertical distance from the flange reference plane to the suction-cup
    # lip. The cup is vertical and centred on j4, so the tip xy equals the
    # flange xy and only z changes: tip = flange - (0, 0, tip_length_mm).
    # This is a scalar, not a 3-D tool-frame offset.
    tip_length_mm: float = 0.0


def _tool_quaternion(r_deg: float) -> tuple[float, float, float, float]:
    """Tool-down orientation: 180 deg about x, then the end rotation about z."""
    rotation = Rotation.from_euler("x", 180.0, degrees=True) * Rotation.from_euler(
        "z", float(r_deg), degrees=True
    )
    x, y, z, w = rotation.as_quat()
    return (float(w), float(x), float(y), float(z))


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
        fk_params: FKParams | None = None,
        success_fn: Any | None = None,
    ) -> None:
        self.robot = robot
        self.camera_source = camera_source
        self.config = config
        source_calibration = getattr(camera_source, "has_calibration", None)
        self._has_calibration = (
            bool(config.has_calibration) and source_calibration is not False
        )
        self.episode_id = episode_id
        self.instruction = instruction
        self.metadata = dict(metadata or {})
        self.safety_limits = safety_limits
        self.fk_params = fk_params
        self._tip_length_mm = float(self.config.tip_length_mm)
        if self._tip_length_mm < 0.0:
            raise ValueError("tip_length_mm must be non-negative")
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
                                  has_depth=False, has_calibration=self._has_calibration)
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
                "tip_length_mm": self._tip_length_mm,
                **self.metadata,
            },
        )

    def episode_identity(self) -> tuple[str, dict[str, Any]]:
        """Prepared-episode identity consumed by ``serve_episode``."""
        return self.instruction, dict(self.metadata)

    def read_state(self) -> EnvironmentState:
        robot = self.robot.read_state()
        return EnvironmentState(
            arms={"arm": self._arm_state(robot)},
            state_version=self._state_version,
            native_steps=self._native_steps,
            ended=self._finished or self._outcome is not None,
        )

    def _tip_from_robot(self, robot: RobotState) -> np.ndarray:
        """Map the firmware's flange position to the vertical suction-cup tip."""
        flange = np.array([robot.x_mm, robot.y_mm, robot.z_mm], dtype=float)
        return flange - np.array([0.0, 0.0, self._tip_length_mm])

    def _flange_for_tip(self, tip_mm: np.ndarray) -> np.ndarray:
        """Map a desired suction-tip position to the flange command.

        The cup is vertical and centred on j4, so this is only a vertical
        translation: flange = tip + (0, 0, tip_length_mm).
        """
        return np.asarray(tip_mm, dtype=float) + np.array(
            [0.0, 0.0, self._tip_length_mm]
        )

    def _arm_state(self, robot: RobotState) -> ArmState:
        gripper = 0.0 if robot.suction_on else 1.0
        tip = self._tip_from_robot(robot)
        flange = [robot.x_mm, robot.y_mm, robot.z_mm]
        return ArmState(
            control_pose=Pose(
                position=(tip[0] / 1000.0, tip[1] / 1000.0, tip[2] / 1000.0),
                quaternion=_tool_quaternion(robot.r_deg),
            ),
            joints=[
                float(np.radians(value))
                for value in (robot.j1_deg, robot.j2_deg, robot.j3_deg, robot.j4_deg)
            ],
            gripper=gripper,
            gripper_command=gripper,
            metadata={
                "measured_xyz_mm": flange,
                "flange_xyz_mm": flange,
                "tip_xyz_mm": tip.tolist(),
            },
        )

    def _check_pose(
        self,
        position_m: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Validate a suction-tip target and return the flange command."""
        tip_mm = np.array([float(value) * 1000.0 for value in position_m], dtype=float)
        for value, (low, high), axis in (
            (tip_mm[0], self.config.x_range_mm, "x"),
            (tip_mm[1], self.config.y_range_mm, "y"),
            (tip_mm[2], self.config.z_range_mm, "z"),
        ):
            if not (low <= value <= high):
                raise RequestRejected(f"{axis}={value:.1f} mm outside workspace [{low}, {high}]")

        current = self.robot.read_state()
        flange_mm = self._flange_for_tip(tip_mm)
        target_flange = (
            float(flange_mm[0]),
            float(flange_mm[1]),
            float(flange_mm[2]),
        )
        previous = (current.x_mm, current.y_mm, current.z_mm)
        if self.safety_limits is not None:
            try:
                self.safety_limits.check_step(
                    previous, target_flange, max_step_mm=self.config.max_step_mm
                )
            except SafetyError as error:
                raise RequestRejected(str(error)) from error
        else:
            distance = float(
                np.linalg.norm(np.array(target_flange) - np.array(previous))
            )
            if distance > self.config.max_step_mm:
                raise RequestRejected(
                    f"step {distance:.1f} mm exceeds max {self.config.max_step_mm:.1f} mm"
                )

        if self.fk_params is not None and self.safety_limits is not None:
            solutions = inverse_kinematics(
                self.fk_params,
                target_flange[0],
                target_flange[1],
                target_flange[2],
                self.safety_limits.joint_limits_deg,
                margin_deg=2.0,
            )
            if not solutions:
                raise RequestRejected(
                    f"target tip ({tip_mm[0]:.1f}, {tip_mm[1]:.1f}, "
                    f"{tip_mm[2]:.1f}) is not reachable within the joint limits"
                )

        w, x, y, z = (float(value) for value in quaternion_wxyz)
        target_rotation = Rotation.from_quat([x, y, z, w])
        w, x, y, z = _tool_quaternion(current.r_deg)
        current_rotation = Rotation.from_quat([x, y, z, w])
        angle = float(np.linalg.norm((current_rotation.inv() * target_rotation).as_rotvec()))
        if angle > self.config.orientation_tolerance_rad:
            raise RequestRejected(
                f"orientation change {angle:.3f} rad exceeds the fixed-rotation tolerance "
                f"{self.config.orientation_tolerance_rad:.3f} rad"
            )
        return flange_mm

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
            self._check_pose(target.pose.position, target.pose.quaternion)

    def step_pose(self, command: PoseCommand) -> StepReceipt:
        self.validate_commands([command], "pose")
        if self._finished or self._outcome is not None:
            raise RequestRejected("episode has ended")
        target = command.arms["arm"]
        tip_mm = np.array(
            [float(value) * 1000.0 for value in target.pose.position], dtype=float
        )
        flange_mm = self._flange_for_tip(tip_mm)
        suction_on = float(target.gripper) <= 0.5
        self.robot.set_suction(suction_on)
        self.robot.move_pose(
            float(flange_mm[0]), float(flange_mm[1]), float(flange_mm[2]), r_deg=0.0
        )
        self._state_version += 1
        self._native_steps += 1
        measured = self.robot.read_state()
        measured_tip = self._tip_from_robot(measured)
        error_mm = float(np.linalg.norm(tip_mm - measured_tip))
        stop_reason = "arrived" if error_mm <= self.config.position_tolerance_mm else "stalled"
        return StepReceipt(
            accepted=True,
            state=self.read_state(),
            diagnostics={
                "stop_reason": stop_reason,
                "position_error_mm": error_mm,
                "executed_steps": 1,
                "target_xyz_mm": tip_mm.tolist(),
                "flange_command_mm": flange_mm.tolist(),
                "measured_xyz_mm": [measured.x_mm, measured.y_mm, measured.z_mm],
                "measured_tip_mm": measured_tip.tolist(),
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
        robot = self.robot.read_state()
        if self.fk_params is not None:
            # The wrist camera is rigidly mounted on the three-axis part of the
            # arm; j4 does not rotate it. Its hand-eye calibration is expressed
            # in this flange frame, not in the suction-tip frame.
            tool_pose = three_axis_frame(
                self.fk_params,
                robot.j1_deg,
                robot.j2_deg,
                robot.j3_deg,
                robot.x_mm,
                robot.y_mm,
                robot.z_mm,
            )
        else:
            tool_pose = pose_to_matrix(
                (robot.x_mm / 1000.0, robot.y_mm / 1000.0, robot.z_mm / 1000.0),
                _tool_quaternion(robot.r_deg),
            )
        frames = self.camera_source.capture(cameras, tool_pose=tool_pose)
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
        close_cameras = getattr(self.camera_source, "close", None)
        if close_cameras is not None:
            close_cameras()
        self.robot.close()

    def set_success_fn(self, fn: Any) -> None:
        self.success_fn = fn
