def test_mock_success_requires_target_xy_and_suction_off():
    from dobot_rua.hardware.interface import FakeRobot
    from dobot_rua.integration.mock_server import _mock_success

    robot = FakeRobot()
    robot.connect()
    robot.move_pose(160.0, -120.0, 90.0, 0.0)
    robot.set_suction(False)
    assert _mock_success(robot) is True
    robot.set_suction(True)
    assert _mock_success(robot) is False
    robot.move_pose(0.0, 0.0, 90.0, 0.0)
    assert _mock_success(robot) is False


def test_mock_camera_source_returns_rgb_frames():
    from dobot_rua.integration.mock_server import MockCameraSource

    frame = MockCameraSource().capture(["front"])["front"]
    # CameraFrame.rgb is RGB (the vendor snapshot store writes it with PIL).
    assert tuple(int(value) for value in frame.rgb[240, 320]) == (255, 255, 0)
    assert tuple(int(value) for value in frame.rgb[200, 200]) == (0, 0, 255)


def test_mock_server_prepare_returns_capturing_adapter(tmp_path):
    from dobot_rua.integration.mock_server import parser, prepare

    args = parser().parse_args(
        [
            "--run-id",
            "mock-1",
            "--instruction",
            "Place the red block at the black lines intersection.",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    adapter = prepare(args)
    observation = adapter.capture()
    assert set(observation.cameras) == {"front", "side", "wrist"}
    assert adapter.episode_identity()[0] == args.instruction
    assert adapter.description.metadata["mock"] is True
