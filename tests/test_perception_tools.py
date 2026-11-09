from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dobot_rua.integration.tools import PerceptionTools


def _scene(path):
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (300, 220), (340, 260), (0, 255, 255), -1)  # yellow
    cv2.rectangle(image, (180, 180), (220, 220), (255, 0, 0), -1)  # blue
    cv2.line(image, (200, 300), (600, 300), (0, 0, 0), 3)
    cv2.line(image, (400, 100), (400, 500), (0, 0, 0), 3)
    cv2.line(image, (150, 350), (350, 350), (0, 0, 255), 3)
    cv2.line(image, (250, 250), (250, 450), (0, 0, 255), 3)
    cv2.imwrite(str(path), image)


class _Mirror:
    def __init__(self, path):
        self.path = path

    def local_path(self, reference):
        return self.path


def _episode(path, *, z_m=0.05, gripper=1.0):
    intrinsic = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    extrinsic[:3, 3] = [0.0, 0.0, 1.0]
    snapshot = {
        "snapshot_id": "s1",
        "state_version": 0,
        "captured_at": 0.0,
        "cameras": {
            "front": {
                "rgb": {"artifact_id": "a1"},
                "intrinsic_K": intrinsic.tolist(),
                "T_world_from_optical": extrinsic.tolist(),
            }
        },
        "state": {},
    }
    state = SimpleNamespace(
        arms={
            "arm": SimpleNamespace(
                control_pose=SimpleNamespace(position=(0.0, 0.0, z_m)),
                gripper=gripper,
            )
        }
    )
    description = SimpleNamespace(
        environment=SimpleNamespace(
            metadata={"table_z_m": 0.0, "block_top_z_m": 0.025, "mock": True}
        )
    )
    return SimpleNamespace(
        latest_snapshot=snapshot,
        artifact_mirror=_Mirror(path),
        state=state,
        description=description,
    )


def test_list_objects_reads_snapshot_and_detects_blocks(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    tools = PerceptionTools(_episode(image_path))
    output = tools.list_objects()
    assert output.success
    colors = {block["color"] for block in output.extra["blocks"]}
    assert colors == {"yellow", "blue"}
    assert output.extra["mock"] is True


def test_find_line_intersection_reads_snapshot(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    tools = PerceptionTools(_episode(image_path))
    output = tools.find_line_intersection("black")
    assert output.success
    assert np.allclose(output.extra["intersection_uv"], (400.0, 300.0), atol=5.0)
    red = tools.find_line_intersection("red")
    assert red.success
    assert np.allclose(red.extra["intersection_uv"], (250.0, 350.0), atol=5.0)


def test_front_tools_fail_clearly_when_uncalibrated(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    episode = _episode(image_path)
    episode.latest_snapshot["cameras"]["front"]["T_world_from_optical"] = None
    tools = PerceptionTools(episode)

    output = tools.list_objects()

    assert output.success is False
    assert "calibrat" in (output.error or "").lower()


def test_find_line_intersection_fails_when_uncalibrated(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    episode = _episode(image_path)
    episode.latest_snapshot["cameras"]["front"]["T_world_from_optical"] = None
    tools = PerceptionTools(episode)

    output = tools.find_line_intersection("black")

    assert output.success is False
    assert "calibrat" in (output.error or "").lower()


def test_verify_place_fails_when_uncalibrated(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    episode = _episode(image_path)
    episode.latest_snapshot["cameras"]["front"]["T_world_from_optical"] = None
    episode.description.environment.metadata["mock"] = False
    tools = PerceptionTools(episode)

    output = tools.verify_place()

    assert output.success is False
    assert "calibrat" in (output.error or "").lower()


def test_front_tools_reject_a_reflected_snapshot_extrinsic(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    episode = _episode(image_path)
    episode.latest_snapshot["cameras"]["front"]["T_world_from_optical"] = np.diag(
        [1.0, 1.0, -1.0, 1.0]
    ).tolist()
    tools = PerceptionTools(episode)

    output = tools.list_objects()

    assert output.success is False
    assert "proper" in (output.error or "").lower()


def test_wrist_tools_use_measured_state(tmp_path):
    image_path = tmp_path / "front.png"
    _scene(image_path)
    tools = PerceptionTools(_episode(image_path, z_m=0.05, gripper=1.0))
    gap = tools.wrist_contact_gap()
    assert gap.extra["gap_mm"] == pytest.approx(25.0)
    assert gap.extra["contact"] is False
    grasp = tools.wrist_verify_grasp()
    assert grasp.extra["attached"] is False
