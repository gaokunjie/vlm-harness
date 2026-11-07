"""V4L2 camera source returning the harness's RGB CameraFrame contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from uni_robot_agent.services.environment.snapshots import CameraFrame


@dataclass(frozen=True)
class CameraSpec:
    name: str
    device: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    intrinsic_K: list | None = None
    T_world_from_optical: list | None = None
    T_tool_from_optical: list | None = None


def _normalize_fourcc(value: str) -> str:
    code = str(value).upper()
    if code in {"MJPEG", "MJPG"}:
        return "MJPG"
    return code if len(code) == 4 else "MJPG"


def load_camera_specs(path: str | Path) -> dict[str, CameraSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    capture = data.get("capture", {})
    width, height = capture.get("default_size", [1280, 720])
    fps = int(capture.get("fps", 30))
    fourcc = _normalize_fourcc(capture.get("format", "mjpg"))
    specs: dict[str, CameraSpec] = {}
    for name, item in data["cameras"].items():
        specs[name] = CameraSpec(
            name=name,
            device=str(item["device"]),
            width=int(width),
            height=int(height),
            fps=fps,
            fourcc=fourcc,
            intrinsic_K=item.get("intrinsic_K"),
            T_world_from_optical=item.get("T_world_from_optical"),
            T_tool_from_optical=item.get("T_tool_from_optical"),
        )
    return specs


class RealCameraSource:
    """Open the configured V4L2 devices lazily and keep them open per episode."""

    def __init__(
        self,
        specs: dict[str, CameraSpec],
        capture_factory: Callable[[CameraSpec], Any] | None = None,
    ) -> None:
        self.specs = dict(specs)
        self._capture_factory = capture_factory or self._opencv_capture
        self._captures: dict[str, Any] = {}
        self.has_calibration = bool(self.specs) and all(
            spec.intrinsic_K is not None
            and (
                spec.T_world_from_optical is not None
                or spec.T_tool_from_optical is not None
            )
            for spec in self.specs.values()
        )

    def _opencv_capture(self, spec: CameraSpec) -> Any:
        cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {spec.name} at {spec.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_normalize_fourcc(spec.fourcc)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
        cap.set(cv2.CAP_PROP_FPS, spec.fps)
        return cap

    def _get_capture(self, name: str) -> Any:
        if name not in self._captures:
            self._captures[name] = self._capture_factory(self.specs[name])
        return self._captures[name]

    def capture(
        self,
        cameras: list[str] | None = None,
        tool_pose: np.ndarray | None = None,
    ) -> dict[str, CameraFrame]:
        names = list(cameras) if cameras else list(self.specs)
        frames: dict[str, CameraFrame] = {}
        for name in names:
            if name not in self.specs:
                raise ValueError(f"unknown camera: {name}")
            cap = self._get_capture(name)
            frame = None
            for _ in range(10):
                ok, candidate = cap.read()
                if ok:
                    frame = candidate
                    break
            if frame is None:
                raise RuntimeError(f"camera {name} returned no frame")
            spec = self.specs[name]
            if spec.T_tool_from_optical is not None:
                if tool_pose is None:
                    raise ValueError(
                        f"camera {name} is eye-in-hand; a current tool pose is required"
                    )
                world_from_optical = np.asarray(tool_pose, dtype=float) @ np.asarray(
                    spec.T_tool_from_optical, dtype=float
                )
            elif spec.T_world_from_optical is not None:
                world_from_optical = np.asarray(spec.T_world_from_optical, dtype=float)
            else:
                world_from_optical = None
            frames[name] = CameraFrame(
                rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                depth_m=None,
                intrinsic_K=(
                    None
                    if spec.intrinsic_K is None
                    else np.asarray(spec.intrinsic_K, dtype=float)
                ),
                T_world_from_optical=world_from_optical,
            )
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
