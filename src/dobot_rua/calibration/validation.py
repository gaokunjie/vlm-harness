"""Independent jog-based validation of a planar camera calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationReport:
    bias_xy_mm: tuple[float, float]
    bias_norm_mm: float
    random_mm: float
    passed: bool


def summarize_validation(
    errors_mm: np.ndarray,
    bias_gate_mm: float = 2.0,
    random_gate_mm: float = 3.0,
) -> ValidationReport:
    """Summarize estimated-minus-actual XY errors in millimetres."""
    errors = np.asarray(errors_mm, dtype=float).reshape(-1, 2)
    if len(errors) == 0:
        return ValidationReport((0.0, 0.0), 0.0, 0.0, False)
    bias = errors.mean(axis=0)
    bias_norm = float(np.linalg.norm(bias))
    random_mm = float(np.max(np.std(errors, axis=0)))
    return ValidationReport(
        bias_xy_mm=(float(bias[0]), float(bias[1])),
        bias_norm_mm=bias_norm,
        random_mm=random_mm,
        passed=bool(bias_norm < float(bias_gate_mm) and random_mm < float(random_gate_mm)),
    )
