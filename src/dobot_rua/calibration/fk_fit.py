"""Least-squares fit of the two-link FK to measured joint/pose pairs.

The firmware's Cartesian pose is the reference: it is what ``move`` commands and
what the controller reports back. Fitting our chain to it makes ``forward_
kinematics``/``inverse_kinematics`` agree with the controller, which the
reachability masks depend on.

The chain is linear once written in the right basis (the firmware's absolute
link angles with mirrored joints)::

    r = a + l1*cos(c2 - j2) + l2*cos(c3 - j3)
      = a + m1*cos(j2) + m2*sin(j2) + m3*cos(j3) + m4*sin(j3)

so the six parameters come from two ordinary least-squares solves and a
closed-form unwrapping, with no initial guess and no local minima.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import FKParams

# "at least four samples" is kept for the caller's contract; the linear solve
# itself needs enough excitation to condition the five-column basis.


@dataclass(frozen=True)
class FKSample:
    j1_deg: float
    j2_deg: float
    j3_deg: float
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class FKFit:
    params: FKParams
    residuals_mm: tuple[float, ...]
    rms_mm: float
    max_mm: float
    azimuth_max_deg: float


def _design(j1_deg: np.ndarray, j2_deg: np.ndarray, j3_deg: np.ndarray) -> np.ndarray:
    j2 = np.radians(j2_deg)
    j3 = np.radians(j3_deg)
    return np.column_stack(
        [
            np.ones_like(j2),
            np.cos(j2),
            np.sin(j2),
            np.cos(j3),
            np.sin(j3),
        ]
    )


def fit_fk(samples: list[FKSample]) -> FKFit:
    """Fit ``(a, h, L1, L2, c2, c3)`` and report the residuals."""
    if len(samples) < 4:
        raise ValueError("need at least 4 samples to fit six parameters")

    j1 = np.radians(np.array([s.j1_deg for s in samples], dtype=float))
    j2_deg = np.array([s.j2_deg for s in samples], dtype=float)
    j3_deg = np.array([s.j3_deg for s in samples], dtype=float)
    x = np.array([s.x_mm for s in samples], dtype=float)
    y = np.array([s.y_mm for s in samples], dtype=float)
    z = np.array([s.z_mm for s in samples], dtype=float)

    # Signed radius along the j1 direction: this keeps the backward-reaching
    # branch (negative radius) distinguishable from the forward one.
    radius = x * np.cos(j1) + y * np.sin(j1)
    design = _design(j1, j2_deg, j3_deg)

    r_coef, *_ = np.linalg.lstsq(design, radius, rcond=None)
    z_coef, *_ = np.linalg.lstsq(design, z, rcond=None)

    # r: (cos j2, sin j2) coefficients are (l1 cos c2, l1 sin c2)
    l1 = float(np.hypot(r_coef[1], r_coef[2]))
    c2 = float(np.degrees(np.arctan2(r_coef[2], r_coef[1])))
    l2 = float(np.hypot(r_coef[3], r_coef[4]))
    c3 = float(np.degrees(np.arctan2(r_coef[4], r_coef[3])))

    params = FKParams(
        a=float(r_coef[0]),
        h=float(z_coef[0]),
        l1=l1,
        l2=l2,
        c2_deg=c2,
        c3_deg=c3,
    )

    predicted = _predict_columns(params, j1, j2_deg, j3_deg)
    per_sample = np.linalg.norm(predicted - np.column_stack([x, y, z]), axis=1)
    # Angle between the measured and predicted horizontal directions; a
    # backward-reaching branch (negative radius) flips both, so comparing the
    # raw azimuth against j1 would falsely flag it.
    cross = x * predicted[:, 1] - y * predicted[:, 0]
    dot = x * predicted[:, 0] + y * predicted[:, 1]
    azimuth = np.abs(np.degrees(np.arctan2(cross, dot)))

    return FKFit(
        params=params,
        residuals_mm=tuple(float(v) for v in per_sample),
        rms_mm=float(np.sqrt(np.mean(per_sample**2))),
        max_mm=float(np.max(per_sample)),
        azimuth_max_deg=float(np.max(np.abs(azimuth))),
    )


def _predict_columns(
    params: FKParams,
    j1_rad: np.ndarray,
    j2_deg: np.ndarray,
    j3_deg: np.ndarray,
) -> np.ndarray:
    t2 = np.radians(params.c2_deg - j2_deg)
    t3 = np.radians(params.c3_deg - j3_deg)
    radius = params.a + params.l1 * np.cos(t2) + params.l2 * np.cos(t3)
    z = params.h + params.l1 * np.sin(t2) + params.l2 * np.sin(t3)
    return np.column_stack([radius * np.cos(j1_rad), radius * np.sin(j1_rad), z])
