"""Robot boundary used by the environment adapter and offline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RobotState:
    x_mm: float
    y_mm: float
    z_mm: float
    r_deg: float
    j1_deg: float
    j2_deg: float
    j3_deg: float
    j4_deg: float
    suction_on: bool


class RobotInterface(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def read_state(self) -> RobotState: ...
    def move_pose(
        self, x_mm: float, y_mm: float, z_mm: float, r_deg: float, timeout_s: float = 5.0
    ) -> None: ...
    def move_joints(
        self,
        j1_deg: float,
        j2_deg: float,
        j3_deg: float,
        j4_deg: float,
        timeout_s: float = 5.0,
    ) -> None: ...
    def set_suction(self, on: bool) -> None: ...
    def stop(self) -> None: ...


class FakeRobot:
    """Deterministic in-memory robot for adapter/contract tests."""

    def __init__(self) -> None:
        self._state = RobotState(
            x_mm=120.0,
            y_mm=0.0,
            z_mm=20.0,
            r_deg=0.0,
            j1_deg=0.0,
            j2_deg=10.88,
            j3_deg=49.98,
            j4_deg=0.0,
            suction_on=False,
        )
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False
        self._state = RobotState(**{**self._state.__dict__, "suction_on": False})

    def read_state(self) -> RobotState:
        return self._state

    def move_pose(
        self, x_mm: float, y_mm: float, z_mm: float, r_deg: float, timeout_s: float = 5.0
    ) -> None:
        self._state = RobotState(
            x_mm=float(x_mm),
            y_mm=float(y_mm),
            z_mm=float(z_mm),
            r_deg=float(r_deg),
            j1_deg=self._state.j1_deg,
            j2_deg=self._state.j2_deg,
            j3_deg=self._state.j3_deg,
            j4_deg=self._state.j4_deg,
            suction_on=self._state.suction_on,
        )

    def move_joints(
        self,
        j1_deg: float,
        j2_deg: float,
        j3_deg: float,
        j4_deg: float,
        timeout_s: float = 5.0,
    ) -> None:
        self._state = RobotState(
            x_mm=self._state.x_mm,
            y_mm=self._state.y_mm,
            z_mm=self._state.z_mm,
            r_deg=self._state.r_deg,
            j1_deg=float(j1_deg),
            j2_deg=float(j2_deg),
            j3_deg=float(j3_deg),
            j4_deg=float(j4_deg),
            suction_on=self._state.suction_on,
        )

    def set_suction(self, on: bool) -> None:
        self._state = RobotState(**{**self._state.__dict__, "suction_on": bool(on)})

    def stop(self) -> None:
        self.set_suction(False)
