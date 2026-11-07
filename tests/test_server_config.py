def test_server_parser_defaults():
    from dobot_rua.integration.server import parser

    args = parser().parse_args(
        ["--run-id", "r1", "--instruction", "test", "--output-dir", "/tmp/out"]
    )
    assert args.robot_port == "/dev/dobot_follower"
    assert args.host == "127.0.0.1"
    assert args.port == 18760
    assert args.mock is False


def test_server_prepare_mock_uses_synthetic_adapter(tmp_path):
    from dobot_rua.integration.server import parser, prepare

    args = parser().parse_args(
        [
            "--run-id",
            "r1",
            "--instruction",
            "test",
            "--output-dir",
            str(tmp_path / "out"),
            "--mock",
        ]
    )
    adapter = prepare(args)
    assert adapter.description.metadata["mock"] is True
    assert adapter.episode_identity()[0] == "test"
