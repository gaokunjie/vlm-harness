"""V4L2 camera source returning the harness's RGB CameraFrame contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from uni_robot_agent.services.environment.snapshots import CameraFrame

from ..geometry import validate_rigid_transform


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
    distortion: list | None = None
    calibration_status: str | None = None
    planar: dict | None = None


def _normalize_fourcc(value: str) -> str:
    code = str(value).upper()
    if code in {"MJPEG", "MJPG"}:
        return "MJPG"
    return code if len(code) == 4 else "MJPG"


def _validate_rigid(name: str, matrix, field: str) -> None:
    """Reject a camera transform that is not a proper rigid transform.

    A reflection (``det(R) = -1``) is orthogonal but mirrors every projected
    point; it has silently corrupted a front-camera extrinsic before, so it is
    a hard load-time error rather than a warning.
    """
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9
    ):
        raise ValueError(f"camera {name}: {field} must be a 4x4 homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"camera {name}: {field} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise ValueError(
            f"camera {name}: {field} is not a proper rigid transform "
            f"(det={determinant:.6f}); a reflection is invalid"
        )


def load_tip_length_mm(path: str | Path) -> float:
    """Read the vertical flange-to-suction-cup-lip distance.

    The cup is vertical and centred on j4, so this is a scalar stored next to
    the front-camera calibration: ``tip_z = flange_z - tip_length_mm``.
    Zero means "not measured".
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    value = (data.get("cameras", {}).get("front", {}) or {}).get("tip_length_mm")
    if value is None:
        return 0.0
    length = float(value)
    if length < 0.0:
        raise ValueError("front.tip_length_mm must be non-negative")
    return length


def load_camera_specs(path: str | Path) -> dict[str, CameraSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    capture = data.get("capture", {})
    width, height = capture.get("default_size", [1280, 720])
    fps = int(capture.get("fps", 30))
    fourcc = _normalize_fourcc(capture.get("format", "mjpg"))
    specs: dict[str, CameraSpec] = {}
    for name, item in data["cameras"].items():
        status = item.get("calibration_status")
        planar_raw = item.get("planar")
        distortion = item.get("distortion")
        intrinsic = item.get("intrinsic_K")
        extrinsic = item.get("T_world_from_optical")
        tool_extrinsic = item.get("T_tool_from_optical")
        planar = None
        valid_planar = (
            name == "front" and status == "VALID" and isinstance(planar_raw, dict)
        )
        if valid_planar:
            validate_rigid_transform(
                planar_raw["T_world_from_optical"],
                f"camera {name}: planar.T_world_from_optical",
            )
            for homography_name in ("H_table", "H_top"):
                homography = np.asarray(planar_raw[homography_name], dtype=float)
                if homography.shape != (3, 3) or not np.isfinite(homography).all():
                    raise ValueError(
                        f"camera {name}: planar.{homography_name} must be finite 3x3"
                    )
                if abs(float(np.linalg.det(homography))) < 1e-12:
                    raise ValueError(
                        f"camera {name}: planar.{homography_name} is singular"
                    )
            extrinsic = planar_raw["T_world_from_optical"]
            planar = {
                **planar_raw,
                "status": status,
                "calibration_hash": item.get("calibration_hash"),
                "distortion": planar_raw.get("distortion", distortion),
            }
        elif name == "front":
            # A front camera without a valid planar calibration must fail
            # closed; the legacy top-level T is not a second source of truth.
            intrinsic = None
            extrinsic = None
            tool_extrinsic = None
            planar = None
        else:
            for field in ("T_world_from_optical", "T_tool_from_optical"):
                if item.get(field) is not None:
                    _validate_rigid(name, item[field], field)
        specs[name] = CameraSpec(
            name=name,
            device=str(item["device"]),
            width=int(width),
            height=int(height),
            fps=fps,
            fourcc=fourcc,
            intrinsic_K=intrinsic,
            T_world_from_optical=extrinsic,
            T_tool_from_optical=tool_extrinsic,
            distortion=distortion,
            calibration_status=status,
            planar=planar,
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
        front = self.specs.get("front")
        self.has_calibration = bool(
            front is not None
            and front.intrinsic_K is not None
            and (
                front.T_world_from_optical is not None
                or front.T_tool_from_optical is not None
            )
            and front.calibration_status == "VALID"
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
                    if spec.intrinsic_K is None or world_from_optical is None
                    else np.asarray(spec.intrinsic_K, dtype=float)
                ),
                T_world_from_optical=world_from_optical,
                calibration=spec.planar,
            )
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
