"""Two-plane PnP calibration for the fixed front camera.

Calibration samples pair an undistorted suction-cup-center pixel with the
robot suction-tip XYZ.  The two sample heights (table and block top) make the
3D points non-coplanar, so a full ``T_world_from_optical`` is observable
without rotating the robot.  H_table/H_top are derived convenience mappings.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from ..geometry import (
    derive_homography,
    project_point_to_pixel,
    solve_pnp_two_plane,
    undistort_points,
    validate_rigid_transform,
)

REQUIRED_CAMERA_CONTROLS = (
    "focus_automatic_continuous",
    "focus_absolute",
    "auto_exposure",
    "exposure",
    "white_balance_automatic",
    "white_balance_temperature",
)


class CalibrationRejected(RuntimeError):
    """Raised when samples or residuals fail the acceptance gates."""


def camera_controls_from_config(front: Mapping[str, Any]) -> dict[str, Any]:
    """Require the locked front-camera controls used in the calibration hash."""
    missing = [
        key
        for key in REQUIRED_CAMERA_CONTROLS
        if key not in front or front[key] is None
    ]
    if missing:
        raise ValueError(
            "front camera controls missing; lock focus/exposure/WB before "
            f"calibrating: {', '.join(missing)}"
        )
    return {key: front[key] for key in REQUIRED_CAMERA_CONTROLS}


def _region_from_samples(
    samples: Sequence[PlanarSample], margin_m: float = 0.01
) -> np.ndarray:
    points = np.asarray(
        [sample.tip_xyz_m[:2] for sample in samples], dtype=np.float32
    ).reshape(-1, 1, 2)
    hull = cv2.convexHull(points).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        raise CalibrationRejected(
            "calibration samples do not span a 2D region; widen the x/y grid"
        )
    centroid = hull.mean(axis=0)
    directions = hull - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return hull + directions / np.maximum(norms, 1e-9) * float(margin_m)


@dataclass(frozen=True)
class PlanarSample:
    tip_xyz_m: tuple[float, float, float]
    cup_pixel_uv: tuple[float, float]
    plane: str
    marker_count: int
    board_reprojection_px: float


@dataclass(frozen=True)
class PlanarCalibration:
    T_world_from_optical: np.ndarray
    H_table: np.ndarray
    H_top: np.ndarray
    intrinsic_K: np.ndarray
    distortion: np.ndarray
    p_cup_board_mm: tuple[float, float, float]
    table_z_m: float
    block_top_z_m: float
    block_height_m: float
    board_thickness_m: float
    calibrated_region: np.ndarray
    camera_controls: Mapping[str, Any]
    fit_rmse_px: float
    holdout_max_px: float
    samples: int
    calibration_hash: str
    measured_at: str


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def compute_calibration_hash(calibration: PlanarCalibration) -> str:
    """SHA-256 over every calibration field except the hash itself."""
    fields = {
        "T_world_from_optical": calibration.T_world_from_optical,
        "H_table": calibration.H_table,
        "H_top": calibration.H_top,
        "intrinsic_K": calibration.intrinsic_K,
        "distortion": calibration.distortion,
        "p_cup_board_mm": calibration.p_cup_board_mm,
        "table_z_m": calibration.table_z_m,
        "block_top_z_m": calibration.block_top_z_m,
        "block_height_m": calibration.block_height_m,
        "board_thickness_m": calibration.board_thickness_m,
        "calibrated_region": calibration.calibrated_region,
        "camera_controls": calibration.camera_controls,
        "fit_rmse_px": calibration.fit_rmse_px,
        "holdout_max_px": calibration.holdout_max_px,
        "samples": calibration.samples,
        "measured_at": calibration.measured_at,
    }
    encoded = json.dumps(fields, sort_keys=True, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reprojection_errors(
    samples: Sequence[PlanarSample],
    transform: np.ndarray,
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    errors = []
    for sample in samples:
        # The stored cup pixel is the raw distorted pixel; the pinhole
        # projection below is in the undistorted domain.
        undistorted = undistort_points(
            np.asarray(sample.cup_pixel_uv, dtype=float), intrinsic_k, distortion
        )
        projected = project_point_to_pixel(
            intrinsic_k, transform, sample.tip_xyz_m
        )
        errors.append(
            float(
                np.hypot(
                    projected[0] - undistorted[0],
                    projected[1] - undistorted[1],
                )
            )
        )
    return np.asarray(errors, dtype=float)


def fit_two_plane(
    samples: Sequence[PlanarSample],
    intrinsic_k: np.ndarray,
    distortion: np.ndarray,
    table_z_m: float,
    block_top_z_m: float,
    p_cup_board_mm: tuple[float, float, float],
    block_height_m: float,
    board_thickness_m: float,
    camera_controls: Mapping[str, Any],
    holdout: int = 2,
    measured_at: str | None = None,
    max_board_reprojection_px: float = 2.0,
    min_marker_count: int = 6,
    max_fit_rmse_px: float = 2.0,
    max_holdout_px: float = 3.0,
) -> tuple[PlanarCalibration, dict[str, Any]]:
    """Fit ``T_world_from_optical`` from samples on two parallel planes."""
    if abs(float(block_top_z_m) - float(table_z_m)) < 0.02:
        raise CalibrationRejected(
            "table and block-top planes are less than 20 mm apart; "
            "the PnP depth direction would be degenerate"
        )
    usable = [
        sample
        for sample in samples
        if sample.marker_count >= min_marker_count
        and sample.board_reprojection_px <= max_board_reprojection_px
    ]
    planes = {sample.plane for sample in usable}
    if "table" not in planes or "top" not in planes:
        raise CalibrationRejected(
            "samples must include both table and top planes"
        )
    if len(usable) < 6:
        raise CalibrationRejected(
            f"need at least 6 usable samples, got {len(usable)}"
        )
    calibrated_region = _region_from_samples(usable)

    ordered = sorted(
        usable, key=lambda s: (s.plane, s.tip_xyz_m[0], s.tip_xyz_m[1])
    )
    holdout_count = max(0, min(int(holdout), len(ordered) - 1))
    holdout_indices: set[int] = set()
    if holdout_count:
        holdout_indices = {
            int(index)
            for index in np.linspace(0, len(ordered) - 1, holdout_count, dtype=int)
        }
    fit_samples = [
        sample for index, sample in enumerate(ordered) if index not in holdout_indices
    ]
    holdout_samples = [
        sample for index, sample in enumerate(ordered) if index in holdout_indices
    ]
    if len(fit_samples) < 6:
        raise CalibrationRejected(
            f"need at least 6 fit samples, got {len(fit_samples)}"
        )

    world = np.asarray([sample.tip_xyz_m for sample in fit_samples], dtype=float)
    pixels = np.asarray([sample.cup_pixel_uv for sample in fit_samples], dtype=float)
    try:
        transform = solve_pnp_two_plane(world, pixels, intrinsic_k, distortion)
    except ValueError as error:
        raise CalibrationRejected(f"PnP failed: {error}") from error

    fit_errors = _reprojection_errors(
        fit_samples, transform, intrinsic_k, distortion
    )
    fit_rmse = float(np.sqrt(np.mean(fit_errors**2))) if len(fit_errors) else 0.0
    fit_max = float(np.max(fit_errors)) if len(fit_errors) else 0.0
    if holdout_samples:
        holdout_errors = _reprojection_errors(
            holdout_samples, transform, intrinsic_k, distortion
        )
        holdout_max = float(np.max(holdout_errors))
    else:
        holdout_errors = np.zeros(0)
        holdout_max = 0.0

    if fit_rmse > max_fit_rmse_px or holdout_max > max_holdout_px:
        raise CalibrationRejected(
            "calibration residuals exceed gates: "
            f"fit_rmse={fit_rmse:.3f}px, holdout_max={holdout_max:.3f}px"
        )

    homography_table = derive_homography(
        transform, intrinsic_k, float(table_z_m)
    )
    homography_top = derive_homography(transform, intrinsic_k, float(block_top_z_m))
    calibration = PlanarCalibration(
        T_world_from_optical=transform,
        H_table=homography_table,
        H_top=homography_top,
        intrinsic_K=np.asarray(intrinsic_k, dtype=float),
        distortion=np.asarray(distortion, dtype=float).reshape(-1),
        p_cup_board_mm=tuple(float(value) for value in p_cup_board_mm),
        table_z_m=float(table_z_m),
        block_top_z_m=float(block_top_z_m),
        block_height_m=float(block_height_m),
        board_thickness_m=float(board_thickness_m),
        calibrated_region=calibrated_region,
        camera_controls=dict(camera_controls),
        fit_rmse_px=fit_rmse,
        holdout_max_px=holdout_max,
        samples=len(usable),
        calibration_hash="",
        measured_at=measured_at or time.strftime("%Y-%m-%d"),
    )
    calibration = replace(
        calibration, calibration_hash=compute_calibration_hash(calibration)
    )
    diagnostics = {
        "usable_samples": len(usable),
        "fit_samples": len(fit_samples),
        "holdout_samples": len(holdout_samples),
        "fit_rmse_px": fit_rmse,
        "fit_max_px": fit_max,
        "holdout_max_px": holdout_max,
        "holdout_errors_px": holdout_errors.tolist(),
    }
    return calibration, diagnostics


def save_planar_calibration(
    path: Path, camera_name: str, calibration: PlanarCalibration
) -> None:
    """Write the planar calibration into ``cameras.<name>`` and mark it valid."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        warnings.warn(
            f"rewriting {path}; YAML comments are not preserved (backup: {backup})",
            stacklevel=2,
        )
    cameras = data.setdefault("cameras", {})
    camera = cameras.setdefault(camera_name, {})
    camera["intrinsic_K"] = calibration.intrinsic_K.tolist()
    camera["distortion"] = calibration.distortion.tolist()
    camera["calibration_status"] = "VALID"
    camera["calibration_mode"] = "two_plane_pnp"
    camera["calibration_date"] = calibration.measured_at
    camera["calibration_hash"] = calibration.calibration_hash
    camera["planar"] = {
        "T_world_from_optical": calibration.T_world_from_optical.tolist(),
        "H_table": calibration.H_table.tolist(),
        "H_top": calibration.H_top.tolist(),
        "p_cup_board_mm": list(calibration.p_cup_board_mm),
        "table_z_m": calibration.table_z_m,
        "block_top_z_m": calibration.block_top_z_m,
        "block_height_m": calibration.block_height_m,
        "board_thickness_m": calibration.board_thickness_m,
        "calibrated_region": calibration.calibrated_region.tolist(),
        "camera_controls": dict(calibration.camera_controls),
        "fit_rmse_px": calibration.fit_rmse_px,
        "holdout_max_px": calibration.holdout_max_px,
        "samples": calibration.samples,
        "measured_at": calibration.measured_at,
    }
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_planar_calibration(
    path: Path, camera_name: str
) -> PlanarCalibration | None:
    """Load and validate a planar calibration; ``None`` means unusable."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    camera = (data.get("cameras") or {}).get(camera_name) or {}
    planar = camera.get("planar")
    if camera.get("calibration_status") != "VALID" or not isinstance(planar, dict):
        return None
    intrinsic = camera.get("intrinsic_K")
    distortion = camera.get("distortion")
    if intrinsic is None or distortion is None:
        return None
    try:
        transform = np.asarray(planar["T_world_from_optical"], dtype=float)
        homography_table = np.asarray(planar["H_table"], dtype=float)
        homography_top = np.asarray(planar["H_top"], dtype=float)
        intrinsic_k = np.asarray(intrinsic, dtype=float)
        distortion_values = np.asarray(distortion, dtype=float).reshape(-1)
        calibrated_region = np.asarray(planar["calibrated_region"], dtype=float)
        if (
            calibrated_region.ndim != 2
            or calibrated_region.shape[1] != 2
            or len(calibrated_region) < 3
            or not np.isfinite(calibrated_region).all()
        ):
            return None
        validate_rigid_transform(transform, "front.planar.T_world_from_optical")
        for name, homography in (
            ("H_table", homography_table),
            ("H_top", homography_top),
        ):
            if homography.shape != (3, 3) or not np.isfinite(homography).all():
                return None
            if abs(float(np.linalg.det(homography))) < 1e-12:
                return None
        calibration = PlanarCalibration(
            T_world_from_optical=transform,
            H_table=homography_table,
            H_top=homography_top,
            intrinsic_K=intrinsic_k,
            distortion=distortion_values,
            p_cup_board_mm=tuple(
                float(value) for value in planar["p_cup_board_mm"]
            ),
            table_z_m=float(planar["table_z_m"]),
            block_top_z_m=float(planar["block_top_z_m"]),
            block_height_m=float(planar["block_height_m"]),
            board_thickness_m=float(planar["board_thickness_m"]),
            calibrated_region=calibrated_region,
            camera_controls=dict(planar.get("camera_controls") or {}),
            fit_rmse_px=float(planar["fit_rmse_px"]),
            holdout_max_px=float(planar["holdout_max_px"]),
            samples=int(planar["samples"]),
            calibration_hash=str(camera.get("calibration_hash", "")),
            measured_at=str(
                planar.get("measured_at", camera.get("calibration_date", ""))
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if compute_calibration_hash(calibration) != calibration.calibration_hash:
        return None
    return calibration


def plan_grid(
    center_xy_mm: tuple[float, float],
    spacing_mm: float,
    cols: int,
    rows: int,
) -> list[tuple[float, float]]:
    """Centered row-major grid: y rows first, then x columns."""
    center_x, center_y = (float(value) for value in center_xy_mm)
    x_offsets = (np.arange(cols) - (cols - 1) / 2.0) * float(spacing_mm)
    y_offsets = (np.arange(rows) - (rows - 1) / 2.0) * float(spacing_mm)
    return [
        (float(center_x + x), float(center_y + y))
        for y in y_offsets
        for x in x_offsets
    ]


def nearest_solution(
    solutions: Sequence[Sequence[float]],
    current_joints: Sequence[float],
) -> tuple[float, float, float] | None:
    """Return the joint solution closest to ``current_joints``."""
    if not solutions:
        return None
    current = np.asarray(current_joints, dtype=float)
    best = min(
        solutions,
        key=lambda solution: float(
            np.linalg.norm(np.asarray(solution, dtype=float) - current)
        ),
    )
    return tuple(float(value) for value in best)


@dataclass(frozen=True)
class PosePlan:
    x_mm: float
    y_mm: float
    z_mm: float
    plane: str


def build_pose_plan(
    center_xy_mm: tuple[float, float],
    spacing_mm: float,
    cols: int,
    rows: int,
    table_z_mm: float,
    block_height_mm: float,
    board_thickness_mm: float,
) -> list[PosePlan]:
    """Plan the two-height x/y sweep; ``z_mm`` is the suction-tip height."""
    grid = plan_grid(center_xy_mm, spacing_mm, cols, rows)
    plan: list[PosePlan] = []
    for plane, z_mm in (
        ("table", float(table_z_mm) + float(board_thickness_mm)),
        ("top", float(table_z_mm) + float(block_height_mm)),
    ):
        for x_mm, y_mm in grid:
            plan.append(PosePlan(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, plane=plane))
    return plan
