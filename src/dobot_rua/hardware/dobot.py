"""Thin Dobot Magician wrapper; the pydobot driver is injectable for tests."""

from __future__ import annotations

from typing import Any

from .interface import RobotState


class DobotRobot:
    def __init__(self, port: str = "/dev/dobot_follower", driver: Any | None = None) -> None:
        self.port = port
        self._driver = driver
        self._suction_on = False

    def connect(self) -> None:
        if self._driver is None:
            from pydobot import Dobot

            self._driver = Dobot(port=self.port, verbose=False)

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.suck(False)
            finally:
                self._driver.close()
                self._driver = None
        self._suction_on = False

    def read_state(self) -> RobotState:
        if self._driver is None:
            raise RuntimeError("robot is not connected")
        x, y, z, r, j1, j2, j3, j4 = self._driver.pose()
        return RobotState(
            x_mm=float(x),
            y_mm=float(y),
            z_mm=float(z),
            r_deg=float(r),
            j1_deg=float(j1),
            j2_deg=float(j2),
            j3_deg=float(j3),
            j4_deg=float(j4),
            suction_on=self._suction_on,
        )

    def move_pose(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        r_deg: float,
        timeout_s: float = 5.0,
    ) -> None:
        if self._driver is None:
            raise RuntimeError("robot is not connected")
        self._driver.move_to(float(x_mm), float(y_mm), float(z_mm), float(r_deg))

    def move_joints(
        self,
        j1_deg: float,
        j2_deg: float,
        j3_deg: float,
        j4_deg: float,
        timeout_s: float = 5.0,
    ) -> None:
        if self._driver is None:
            raise RuntimeError("robot is not connected")
        move_by_angle = getattr(self._driver, "move_by_angle", None)
        if move_by_angle is None:
            raise NotImplementedError("driver does not expose joint commands")
        move_by_angle(float(j1_deg), float(j2_deg), float(j3_deg), float(j4_deg))

    def set_suction(self, on: bool) -> None:
        if self._driver is None:
            raise RuntimeError("robot is not connected")
        self._driver.suck(bool(on))
        self._suction_on = bool(on)

    def stop(self) -> None:
        if self._driver is not None:
            stop = getattr(self._driver, "stop", None)
            if stop is not None:
                stop()
            self._driver.suck(False)
        self._suction_on = False
