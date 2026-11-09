import numpy as np
import pytest
import yaml

from dobot_rua.hardware.cameras import (
    CameraSpec,
    RealCameraSource,
    load_camera_specs,
    load_tip_length_mm,
)


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False

    def read(self):
        return True, self.frame.copy()

    def release(self):
        self.released = True


def test_real_camera_source_converts_bgr_to_rgb():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # red in BGR
    source = RealCameraSource(
        {"front": CameraSpec(name="front", device="fake")},
        capture_factory=lambda spec: FakeCapture(bgr),
    )
    frames = source.capture(["front"])
    assert tuple(int(value) for value in frames["front"].rgb[0, 0]) == (255, 0, 0)
    assert frames["front"].intrinsic_K is None
    assert source.has_calibration is False
    source.close()


def test_real_camera_source_passes_calibration_through():
    K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    T = np.eye(4).tolist()
    source = RealCameraSource(
        {
            "front": CameraSpec(
                name="front",
                device="fake",
                intrinsic_K=K,
                T_world_from_optical=T,
                calibration_status="VALID",
                planar={
                    "status": "VALID",
                    "distortion": [0, 0, 0, 0, 0],
                    "T_world_from_optical": T,
                    "H_table": np.eye(3).tolist(),
                    "H_top": np.eye(3).tolist(),
                },
            )
        },
        capture_factory=lambda spec: FakeCapture(np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    frames = source.capture(["front"])
    assert source.has_calibration is True
    assert np.allclose(frames["front"].intrinsic_K, K)
    assert np.allclose(frames["front"].T_world_from_optical, T)


def test_wrist_tool_from_optical_composes_with_tool_pose():
    tool_from_camera = np.eye(4)
    tool_from_camera[:3, 3] = [0.01, 0.0, 0.05]
    source = RealCameraSource(
        {
            "wrist": CameraSpec(
                name="wrist",
                device="fake",
                intrinsic_K=[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                T_tool_from_optical=tool_from_camera.tolist(),
            )
        },
        capture_factory=lambda spec: FakeCapture(np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    tool_pose = np.eye(4)
    tool_pose[:3, 3] = [0.1, 0.2, 0.3]
    frames = source.capture(["wrist"], tool_pose=tool_pose)
    # Front is the task-critical camera; a wrist-only source is not a
    # calibrated episode source even though its transform is valid.
    assert source.has_calibration is False
    assert np.allclose(
        frames["wrist"].T_world_from_optical, tool_pose @ tool_from_camera
    )


def test_load_camera_specs_reads_roles(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text(
        """
cameras:
  wrist:
    device: /dev/video0
  front:
    device: /dev/video2
  side:
    device: /dev/video4
capture:
  format: mjpeg
  default_size: [1280, 720]
  fps: 30
""",
        encoding="utf-8",
    )
    specs = load_camera_specs(path)
    assert set(specs) == {"wrist", "front", "side"}
    assert specs["front"].device == "/dev/video2"
    assert specs["front"].width == 1280
    assert specs["front"].height == 720
    assert specs["front"].fourcc == "MJPG"


def _proper_transform() -> list[list[float]]:
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = [0.17, -0.01, 0.33]
    return transform.tolist()


def _write_calibration_config(path, matrix) -> None:
    config = {
        "cameras": {
            "front": {
                "device": "/dev/video2",
                "intrinsic_K": [
                    [1000.0, 0.0, 640.0],
                    [0.0, 1000.0, 360.0],
                    [0.0, 0.0, 1.0],
                ],
                "T_world_from_optical": matrix,
                "calibration_status": "VALID",
                "planar": {
                    "T_world_from_optical": matrix,
                    "H_table": np.eye(3).tolist(),
                    "H_top": np.eye(3).tolist(),
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_accepts_a_proper_world_from_optical(tmp_path):
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, _proper_transform())

    specs = load_camera_specs(path)

    assert specs["front"].T_world_from_optical is not None


def test_load_rejects_a_reflected_world_from_optical(tmp_path):
    matrix = np.array(_proper_transform())
    matrix[:3, 1] *= -1.0  # orthonormal, but det = -1: a reflection
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, matrix.tolist())

    with pytest.raises(ValueError, match="proper|reflection|det"):
        load_camera_specs(path)


def test_load_tip_length_mm_reads_the_front_value(tmp_path):
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, _proper_transform())
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["cameras"]["front"]["tip_length_mm"] = 66.6
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert load_tip_length_mm(path) == pytest.approx(66.6)


def test_load_tip_length_mm_defaults_to_zero(tmp_path):
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, _proper_transform())

    assert load_tip_length_mm(path) == pytest.approx(0.0)


def test_load_tip_length_mm_rejects_a_negative_value(tmp_path):
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, _proper_transform())
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["cameras"]["front"]["tip_length_mm"] = -1.0
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        load_tip_length_mm(path)


def test_load_rejects_a_non_orthonormal_rotation(tmp_path):
    matrix = np.array(_proper_transform())
    matrix[:3, :3] *= 1.1
    path = tmp_path / "cameras.yaml"
    _write_calibration_config(path, matrix.tolist())

    with pytest.raises(ValueError, match="orthonormal|proper"):
        load_camera_specs(path)


def test_uncalibrated_camera_drops_its_intrinsics():
    # A camera with intrinsics but no extrinsic cannot be projected from; the
    # frame must carry neither, otherwise the snapshot layer rejects the frame.
    K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    source = RealCameraSource(
        {"side": CameraSpec(name="side", device="fake", intrinsic_K=K)},
        capture_factory=lambda spec: FakeCapture(np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    frames = source.capture(["side"])
    assert frames["side"].intrinsic_K is None
    assert frames["side"].T_world_from_optical is None


def _planar_config(tmp_path, planar, status="VALID"):
    data = {
        "capture": {"default_size": [720, 1280]},
        "cameras": {
            "front": {
                "device": "/dev/video2",
                "intrinsic_K": np.eye(3).tolist(),
                "distortion": [0, 0, 0, 0, 0],
                "T_world_from_optical": np.eye(4).tolist(),
                "calibration_status": status,
                "planar": planar,
                "tip_length_mm": 66.6,
            }
        },
    }
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_valid_planar_T_overrides_stale_T(tmp_path):
    T = np.eye(4)
    T[0, 3] = 0.123
    path = _planar_config(
        tmp_path,
        {
            "T_world_from_optical": T.tolist(),
            "H_table": np.eye(3).tolist(),
            "H_top": np.eye(3).tolist(),
        },
    )
    spec = load_camera_specs(path)["front"]
    assert np.allclose(spec.T_world_from_optical, T)
    assert spec.calibration_status == "VALID"
    assert spec.planar is not None
    assert spec.planar["status"] == "VALID"
    assert spec.planar["distortion"] == [0, 0, 0, 0, 0]


def test_invalid_planar_leaves_front_uncalibrated(tmp_path):
    path = _planar_config(tmp_path, None, status="INVALID")
    spec = load_camera_specs(path)["front"]
    assert spec.T_world_from_optical is None
    assert spec.intrinsic_K is None
    assert RealCameraSource({"front": spec}).has_calibration is False


def test_planar_reflection_rejected(tmp_path):
    T = np.eye(4)
    T[2, 2] = -1.0
    path = _planar_config(
        tmp_path,
        {
            "T_world_from_optical": T.tolist(),
            "H_table": np.eye(3).tolist(),
            "H_top": np.eye(3).tolist(),
        },
    )
    with pytest.raises(ValueError):
        load_camera_specs(path)


def test_capture_sets_planar_calibration(tmp_path):
    T = np.eye(4)
    T[0, 3] = 0.123
    path = _planar_config(
        tmp_path,
        {
            "T_world_from_optical": T.tolist(),
            "H_table": np.eye(3).tolist(),
            "H_top": np.eye(3).tolist(),
        },
    )
    spec = load_camera_specs(path)["front"]
    source = RealCameraSource(
        {"front": spec}, capture_factory=lambda s: FakeCapture(np.zeros((4, 4, 3), dtype=np.uint8))
    )
    frame = source.capture(["front"])["front"]
    assert frame.calibration["status"] == "VALID"
    assert np.allclose(frame.T_world_from_optical, T)


def test_provisional_planar_loads(tmp_path):
    T = np.eye(4)
    T[0, 3] = 0.123
    path = _planar_config(
        tmp_path,
        {
            "T_world_from_optical": T.tolist(),
            "H_table": np.eye(3).tolist(),
            "H_top": np.eye(3).tolist(),
        },
        status="PROVISIONAL",
    )
    spec = load_camera_specs(path)["front"]
    assert spec.planar is not None
    assert spec.planar["status"] == "PROVISIONAL"
    assert RealCameraSource({"front": spec}).has_calibration is True
