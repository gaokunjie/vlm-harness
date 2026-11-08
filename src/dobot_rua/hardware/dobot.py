"""Thin Dobot Magician wrapper; the pydobot driver is injectable for tests."""

from __future__ import annotations

import math
import struct
import time
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

    def set_speed(self, velocity: int, acceleration: int) -> None:
        if self._driver is None:
            raise RuntimeError("robot is not connected")
        self._driver.speed(int(velocity), int(acceleration))

    def close(self, release: bool = True) -> None:
        """Close the driver; ``release=False`` leaves the vacuum state as-is."""
        if self._driver is not None:
            try:
                if release:
                    self._driver.suck(False)
            finally:
                self._driver.close()
                self._driver = None
        if release:
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
        target = (float(x_mm), float(y_mm), float(z_mm), float(r_deg))
        if not (
            hasattr(self._driver, "_set_ptp_cmd")
            and hasattr(self._driver, "_get_queued_cmd_current_index")
        ):
            self._driver.move_to(*target, wait=True)
            return
        from pydobot.enums import PTPMode

        response = self._driver._set_ptp_cmd(
            *target, mode=PTPMode.MOVL_XYZ, wait=False
        )
        queued_index = None
        if response is not None and len(response.params) >= 4:
            queued_index = int(struct.unpack_from("L", response.params, 0)[0])
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                current = self._driver._get_queued_cmd_current_index()
            except Exception:
                break
            if queued_index is None or current >= queued_index:
                break
            time.sleep(0.05)
        if queued_index is None:
            time.sleep(0.4)
        # Wait until the measured pose stops changing; the queue index alone
        # does not prove the physical move has settled.
        last = None
        stable = 0
        while time.monotonic() < deadline + 2.0:
            state = self.read_state()
            position = (state.x_mm, state.y_mm, state.z_mm)
            if last is not None and math.dist(position, last) < 0.3:
                stable += 1
                if stable >= 4:
                    break
            else:
                stable = 0
            last = position
            time.sleep(0.1)

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
