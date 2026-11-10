import math

import pytest

from dobot_rua.hardware.dobot import (
    DobotRobot,
    firmware_from_physical,
    physical_from_firmware,
)

SCALE = 1.26932


def test_physical_from_firmware_applies_base_scale():
    radius = 200.0
    j1_fw = 16.7
    x = radius * math.cos(math.radians(j1_fw))
    y = radius * math.sin(math.radians(j1_fw))
    x_phys, y_phys, j1_phys = physical_from_firmware(x, y, SCALE)
    assert j1_phys == pytest.approx(j1_fw * SCALE)
    assert x_phys == pytest.approx(radius * math.cos(math.radians(j1_phys)))
    assert y_phys == pytest.approx(radius * math.sin(math.radians(j1_phys)))


def test_firmware_from_physical_round_trip():
    x, y = 200.0, 60.0
    x_fw, y_fw = firmware_from_physical(x, y, SCALE)
    radius = math.hypot(x, y)
    j1_phys = math.degrees(math.atan2(y, x))
    j1_fw = j1_phys / SCALE
    x_phys, y_phys, j1 = physical_from_firmware(x_fw, y_fw, SCALE)
    assert x_phys == pytest.approx(x, abs=1e-9)
    assert y_phys == pytest.approx(y, abs=1e-9)
    assert j1 == pytest.approx(j1_phys)


def test_move_joints_sends_scaled_j1_and_reports_physical_state():
    class FakeDriver:
        def __init__(self):
            self.pose_value = (120.0, 0.0, 20.0, 0.0, 0.0, 10.88, 49.98, 0.0)
            self.calls = []

        def pose(self):
            return self.pose_value

        def speed(self, velocity, acceleration):
            pass

        def suck(self, on):
            pass

        def close(self):
            pass

        def _set_ptp_cmd(self, x, y, z, r, mode, wait):
            self.calls.append((x, y, z, r))
            radius = 120.0
            self.pose_value = (
                radius * math.cos(math.radians(x)),
                radius * math.sin(math.radians(x)),
                z,
                r,
                x,
                y,
                z,
                r,
            )

    driver = FakeDriver()
    robot = DobotRobot(driver=driver, base_j1_scale=SCALE)
    robot.connect()
    physical_j1 = 16.69924545288086
    robot.move_joints(physical_j1, 35.71, 27.82, 88.0, timeout_s=1.0)
    assert driver.calls[-1][0] == pytest.approx(physical_j1 / SCALE, abs=1e-3)
    state = robot.read_state()
    assert state.j1_deg == pytest.approx(physical_j1, abs=1e-3)
