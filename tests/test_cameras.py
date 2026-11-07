import numpy as np

from dobot_rua.hardware.cameras import CameraSpec, RealCameraSource, load_camera_specs


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
            )
        },
        capture_factory=lambda spec: FakeCapture(np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    frames = source.capture(["front"])
    assert source.has_calibration is True
    assert np.allclose(frames["front"].intrinsic_K, K)
    assert np.allclose(frames["front"].T_world_from_optical, T)


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
