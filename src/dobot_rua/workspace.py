"""Reachable workspace, camera field of view, and randomized block layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import project_point_to_pixel
from .kinematics import FKParams, LimitsDeg, forward_kinematics


@dataclass(frozen=True)
class PointMask:
    """A finite set of millimetre XY points, addressed on a regular grid."""

    points_mm: np.ndarray
    grid_mm: float
    _keys: frozenset[tuple[int, int]]

    @classmethod
    def from_points(cls, points_mm: Sequence[Sequence[float]], grid_mm: float) -> "PointMask":
        points = np.asarray(points_mm, dtype=float).reshape(-1, 2)
        keys = frozenset(
            (int(round(x / grid_mm)), int(round(y / grid_mm)))
            for x, y in points
        )
        return cls(points_mm=points, grid_mm=float(grid_mm), _keys=keys)

    def _key(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(float(x_mm) / self.grid_mm)), int(round(float(y_mm) / self.grid_mm))

    def contains(self, x_mm: float, y_mm: float) -> bool:
        return self._key(x_mm, y_mm) in self._keys

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(frozen=True)
class Layout:
    block_xy_mm: tuple[float, float]
    distractor_xy_mm: tuple[float, float] | None
    seed: int


def build_fov_mask(
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    image_size_hw: tuple[int, int],
    points_mm: Sequence[Sequence[float]],
    margin_px: int = 30,
) -> PointMask:
    """Keep world-plane points that project inside the image with a margin."""
    height, width = image_size_hw
    inside: list[tuple[float, float]] = []
    for point in np.asarray(points_mm, dtype=float).reshape(-1, 2):
        try:
            u, v = project_point_to_pixel(
                intrinsic_k,
                t_world_from_optical,
                np.array([point[0] / 1000.0, point[1] / 1000.0, 0.0]),
            )
        except ValueError:
            continue
        if margin_px <= u <= width - margin_px and margin_px <= v <= height - margin_px:
            inside.append((float(point[0]), float(point[1])))
    return PointMask.from_points(inside, grid_mm=1.0)


def build_reachability_mask(
    params: FKParams,
    limits: LimitsDeg,
    z_mm: float,
    grid_mm: float = 5.0,
    joint_step_deg: float = 5.0,
) -> PointMask:
    """Sample joint space and keep FK points within ``grid_mm/2`` of ``z_mm``."""
    j1_values = np.arange(limits[0][0], limits[0][1] + 1e-9, joint_step_deg)
    j2_values = np.arange(limits[1][0], limits[1][1] + 1e-9, joint_step_deg)
    j3_values = np.arange(limits[2][0], limits[2][1] + 1e-9, joint_step_deg)
    points: list[tuple[float, float]] = []
    for j1 in j1_values:
        for j2 in j2_values:
            for j3 in j3_values:
                xyz = forward_kinematics(params, float(j1), float(j2), float(j3))
                if abs(xyz[2] - float(z_mm)) <= grid_mm / 2.0:
                    points.append((float(xyz[0]), float(xyz[1])))
    return PointMask.from_points(points, grid_mm=grid_mm)


def sample_layout(
    rng: np.random.Generator,
    mask: PointMask,
    target_xy_mm: tuple[float, float],
    n_blocks: int = 1,
    min_block_sep_mm: float = 60.0,
    min_target_sep_mm: float = 80.0,
    max_attempts: int = 10000,
) -> Layout:
    """Rejection-sample a target block and, when requested, a distractor."""
    if n_blocks not in (1, 2):
        raise ValueError("n_blocks must be 1 or 2")
    candidates = mask.points_mm
    if len(candidates) == 0:
        raise ValueError("workspace mask is empty")
    target = np.asarray(target_xy_mm, dtype=float)
    candidates = candidates[np.linalg.norm(candidates - target, axis=1) >= min_target_sep_mm]
    if len(candidates) == 0:
        raise ValueError("no reachable point is far enough from the target")

    block = candidates[int(rng.integers(len(candidates)))]
    distractor = None
    if n_blocks == 2:
        for _ in range(max_attempts):
            candidate = candidates[int(rng.integers(len(candidates)))]
            if np.linalg.norm(candidate - block) >= min_block_sep_mm:
                distractor = candidate
                break
        if distractor is None:
            raise ValueError("could not satisfy the minimum block separation")
    return Layout(
        block_xy_mm=(float(block[0]), float(block[1])),
        distractor_xy_mm=None if distractor is None else (float(distractor[0]), float(distractor[1])),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
