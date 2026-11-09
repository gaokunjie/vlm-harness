import pytest


def test_task_metadata_uses_block_height(tmp_path):
    from dobot_rua.integration.server import _task_metadata

    path = tmp_path / "task.yaml"
    path.write_text(
        "table_z_m: -0.02\nblock_size_m: 0.03\nblock_height_m: 0.04\nplace_radius_m: 0.02\n",
        encoding="utf-8",
    )
    meta = _task_metadata(path)
    assert meta["block_top_z_m"] == pytest.approx(0.02)
    assert meta["block_height_m"] == pytest.approx(0.04)
    assert meta["place_radius_m"] == pytest.approx(0.02)


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
