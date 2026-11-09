import pytest

from dobot_rua.hardware.interface import FakeRobot, RobotState
from dobot_rua.hardware.safety import SafetyError, SafetyLimits


def test_fake_robot_moves_and_controls_suction():
    robot = FakeRobot()
    robot.connect()
    state = robot.read_state()
    assert isinstance(state, RobotState)
    assert state.suction_on is False

    robot.move_pose(150.0, 10.0, 20.0, 0.0, timeout_s=1.0)
    moved = robot.read_state()
    assert moved.x_mm == pytest.approx(150.0)
    assert moved.y_mm == pytest.approx(10.0)
    assert moved.z_mm == pytest.approx(20.0)

    robot.set_suction(True)
    assert robot.read_state().suction_on is True
    robot.set_suction(False)
    assert robot.read_state().suction_on is False
    robot.close()


def test_safety_rejects_out_of_range_joints():
    limits = SafetyLimits.from_yaml("configs/robot_limits.yaml")
    limits.check_joints((0.0, 45.0, 45.0, 0.0))
    with pytest.raises(SafetyError):
        limits.check_joints((0.0, -5.0, 45.0, 0.0))


def test_safety_rejects_large_step():
    limits = SafetyLimits.from_yaml("configs/robot_limits.yaml")
    with pytest.raises(SafetyError):
        limits.check_step((0.0, 0.0, 20.0), (100.0, 0.0, 20.0), max_step_mm=20.0)
    limits.check_step((0.0, 0.0, 20.0), (10.0, 0.0, 20.0), max_step_mm=20.0)


def test_dobot_robot_wraps_injected_driver():
    from dobot_rua.hardware.dobot import DobotRobot

    class FakeDriver:
        def __init__(self):
            self.pose_value = (120.0, 0.0, 20.0, 0.0, 0.0, 10.88, 49.98, 0.0)
            self.suction = None

        def pose(self):
            return self.pose_value

        def move_to(self, x, y, z, r, wait=True):
            self.pose_value = (x, y, z, r, *self.pose_value[4:])

        def suck(self, on):
            self.suction = on

        def close(self):
            pass

    driver = FakeDriver()
    robot = DobotRobot(driver=driver)
    robot.connect()
    robot.move_pose(150.0, 10.0, 30.0, 0.0, timeout_s=1.0)
    state = robot.read_state()
    assert state.x_mm == pytest.approx(150.0)
    assert state.z_mm == pytest.approx(30.0)
    robot.set_suction(True)
    assert driver.suction is True


def test_dobot_home_applies_post_home_j4_offset():
    from dobot_rua.hardware.dobot import DobotRobot

    class FakeDriver:
        def __init__(self):
            self.pose_value = (120.0, 0.0, 20.0, 0.0, 0.0, 10.88, 49.98, 0.0)
            self.home_sent = False
            self.ptp_calls = []
            self.speed_args = None

        def pose(self):
            return self.pose_value

        def _send_command(self, message):
            self.home_sent = True
            # Emulate the firmware homing to (0, 45, 45, 0).
            self.pose_value = (199.4, 0.0, -8.49, 0.0, 0.0, 45.0, 45.0, 0.0)
            return None

        def speed(self, velocity, acceleration):
            self.speed_args = (velocity, acceleration)

        def _set_ptp_cmd(self, x, y, z, r, mode, wait):
            self.ptp_calls.append((x, y, z, r, mode, wait))
            # MOVJ_ANGLE: the four values are j1..j4.
            self.pose_value = (
                self.pose_value[0],
                self.pose_value[1],
                self.pose_value[2],
                r,
                x,
                y,
                z,
                r,
            )

        def suck(self, on):
            pass

        def close(self):
            pass

    driver = FakeDriver()
    robot = DobotRobot(driver=driver)
    robot.connect()
    state = robot.home(
        home_deg=(0.0, 45.0, 45.0, 0.0),
        post_home_j4_offset_deg=88.0,
        timeout_s=1.0,
        tolerance_deg=0.5,
    )
    assert driver.home_sent is True
    assert driver.speed_args == (20.0, 20.0)
    assert driver.ptp_calls
    assert driver.ptp_calls[-1][3] == pytest.approx(88.0)
    assert state.j4_deg == pytest.approx(88.0)
    assert state.j1_deg == pytest.approx(0.0)
    assert state.j2_deg == pytest.approx(45.0)
    assert state.j3_deg == pytest.approx(45.0)


def test_dobot_close_can_keep_suction_on():
    from dobot_rua.hardware.dobot import DobotRobot

    class FakeDriver:
        def __init__(self):
            self.pose_value = (120.0, 0.0, 20.0, 0.0, 0.0, 10.88, 49.98, 0.0)
            self.suction = None
            self.closed = False

        def pose(self):
            return self.pose_value

        def suck(self, on):
            self.suction = on

        def close(self):
            self.closed = True

    driver = FakeDriver()
    robot = DobotRobot(driver=driver)
    robot.connect()
    robot.set_suction(True)
    robot.close(release=False)
    assert driver.suction is True
    assert driver.closed is True


def test_dobot_move_joints_uses_movj_angle():
    from dobot_rua.hardware.dobot import DobotRobot

    class FakeDriver:
        def __init__(self):
            self.pose_value = (120.0, 0.0, 20.0, 0.0, 0.0, 10.88, 49.98, 0.0)
            self.calls = []

        def pose(self):
            return self.pose_value

        def speed(self, velocity, acceleration):
            pass

        def _set_ptp_cmd(self, x, y, z, r, mode, wait):
            self.calls.append((x, y, z, r, mode, wait))
            self.pose_value = (
                self.pose_value[0],
                self.pose_value[1],
                self.pose_value[2],
                r,
                x,
                y,
                z,
                r,
            )

        def suck(self, on):
            pass

        def close(self):
            pass

    driver = FakeDriver()
    robot = DobotRobot(driver=driver)
    robot.connect()
    robot.move_joints(0.0, 45.0, 45.0, 88.0, timeout_s=1.0)
    assert driver.calls
    assert driver.calls[-1][3] == 88.0
    assert driver.calls[-1][4].name == "MOVJ_ANGLE"
    assert driver.calls[-1][5] is False
