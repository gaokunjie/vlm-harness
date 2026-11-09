"""Two-link FK/IK for the Dobot Magician, in the firmware's own convention.

The controller reports **absolute** link angles with both joint angles
mirrored, which the measured data reproduces exactly:

    r = a + L1*cos(t2) + L2*cos(t3),  z = h + L1*sin(t2) + L2*sin(t3)
    t2 = c2_deg - j2,                t3 = c3_deg - j3

With the vendor's link lengths (135 mm, 147 mm, no offsets) this gives
``(199.4, 0, -8.49)`` at the home pose ``(45, 45)`` and ``(120, 0, 20)`` at the
documented ready pose ``(10.88, 49.98)``. The earlier "relative elbow" form
(``L2*cos(t2+t3)``) is a different robot and must not be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FKParams:
    a: float
    h: float
    l1: float
    l2: float
    c2_deg: float
    c3_deg: float


LimitsDeg = tuple[tuple[float, float], ...]


def load_fk(path: str | Path) -> FKParams:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return FKParams(
        a=float(data["a"]),
        h=float(data["h"]),
        l1=float(data["L1"]),
        l2=float(data["L2"]),
        c2_deg=float(data["c2_deg"]),
        c3_deg=float(data["c3_deg"]),
    )


def load_limits(path: str | Path) -> LimitsDeg:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = data["joint_limits_deg"]
    return tuple(
        (float(raw[key][0]), float(raw[key][1])) for key in ("j1", "j2", "j3", "j4")
    )


def forward_kinematics(
    params: FKParams, j1_deg: float, j2_deg: float, j3_deg: float
) -> np.ndarray:
    """Return the suction-tip position ``(x, y, z)`` in millimetres."""
    t2 = np.radians(params.c2_deg - float(j2_deg))
    t3 = np.radians(params.c3_deg - float(j3_deg))
    radius = params.a + params.l1 * np.cos(t2) + params.l2 * np.cos(t3)
    z = params.h + params.l1 * np.sin(t2) + params.l2 * np.sin(t3)
    j1 = np.radians(float(j1_deg))
    return np.array([radius * np.cos(j1), radius * np.sin(j1), z], dtype=float)


def _wrap_to_limits(value_deg: float, limit: tuple[float, float]) -> list[float]:
    """Return the equivalent angles (period 360) that fall inside the limit."""
    low, high = limit
    candidates = []
    for period in (-720.0, -360.0, 0.0, 360.0, 720.0):
        candidate = value_deg + period
        if low <= candidate <= high:
            candidates.append(candidate)
    return candidates


def within_limits(
    joints_deg: Sequence[float], limits: LimitsDeg, margin_deg: float = 2.0
) -> bool:
    if len(joints_deg) > len(limits):
        return False
    return all(
        low + margin_deg <= float(value) <= high - margin_deg
        for value, (low, high) in zip(joints_deg, limits, strict=False)
    )


def inverse_kinematics(
    params: FKParams,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    limits: LimitsDeg,
    margin_deg: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Return all ``(j1, j2, j3)`` solutions inside the joint limits.

    ``j4`` does not affect the suction-tip position. Both the positive-radius
    and negative-radius base-angle branches are considered.
    """
    radius_abs = float(np.hypot(x_mm, y_mm))
    base = float(np.arctan2(y_mm, x_mm))
    solutions: list[tuple[float, float, float]] = []

    for radius_sign in (1.0, -1.0):
        radius = radius_sign * radius_abs
        j1_rad = base if radius_sign > 0 else base + np.pi
        j1_deg = float(np.degrees(j1_rad))
        if j1_deg > 180.0:
            j1_deg -= 360.0
        if j1_deg < -180.0:
            j1_deg += 360.0

        a = radius - params.a
        b = float(z_mm) - params.h
        denom = 2.0 * params.l1 * params.l2
        if abs(denom) < 1e-12:
            continue
        cosine = (a * a + b * b - params.l1**2 - params.l2**2) / denom
        if cosine < -1.0 - 1e-9 or cosine > 1.0 + 1e-9:
            continue
        cosine = float(np.clip(cosine, -1.0, 1.0))
        # delta is the angle between the links (the elbow bend); the joints
        # themselves are the two absolute link angles t2 and t3 = t2 + delta.
        delta_abs = float(np.arccos(cosine))

        for delta in (delta_abs, -delta_abs):
            t2 = np.arctan2(b, a) - np.arctan2(
                params.l2 * np.sin(delta), params.l1 + params.l2 * np.cos(delta)
            )
            t3 = t2 + delta
            j2_deg = params.c2_deg - float(np.degrees(t2))
            j3_deg = params.c3_deg - float(np.degrees(t3))
            for j2_wrapped in _wrap_to_limits(j2_deg, limits[1]):
                for j3_wrapped in _wrap_to_limits(j3_deg, limits[2]):
                    joints = (j1_deg, j2_wrapped, j3_wrapped)
                    if within_limits(joints, limits, margin_deg=margin_deg):
                        solutions.append(joints)

    unique: list[tuple[float, float, float]] = []
    for solution in solutions:
        if not any(
            np.allclose(solution, existing, atol=1e-6) for existing in unique
        ):
            unique.append(solution)
    return unique


def three_axis_frame(
    params: FKParams,
    j1_deg: float,
    j2_deg: float,
    j3_deg: float,
    x_mm: float,
    y_mm: float,
    z_mm: float,
) -> np.ndarray:
    """Pose of the three-axis end frame (j1/j2/j3); j4 is deliberately absent.

    The wrist camera is mounted on the non-j4 part of the arm. The last-link
    angle is the absolute forearm angle ``t3`` in the arm plane; the base yaw is
    j1. The origin is the suction-tip position reported by the robot (j4 does
    not move it).
    """
    t3 = np.radians(params.c3_deg - float(j3_deg))
    rotation = Rotation.from_euler("z", float(j1_deg), degrees=True) * Rotation.from_euler(
        "y", -float(np.degrees(t3)), degrees=True
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return transform


