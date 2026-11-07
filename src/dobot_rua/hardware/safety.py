"""Joint-limit, step-size, and workspace safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


class SafetyError(RuntimeError):
    """Raised before a command that could exceed the safe envelope."""


@dataclass(frozen=True)
class SafetyLimits:
    joint_limits_deg: tuple[tuple[float, float], ...]
    home_deg: tuple[float, float, float, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SafetyLimits":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw = data["joint_limits_deg"]
        limits = tuple(
            (float(raw[key][0]), float(raw[key][1])) for key in ("j1", "j2", "j3", "j4")
        )
        home = tuple(float(value) for value in data["home_deg"])
        return cls(joint_limits_deg=limits, home_deg=home)

    def check_joints(
        self, joints_deg: Sequence[float], margin_deg: float = 2.0
    ) -> None:
        if len(joints_deg) != 4:
            raise SafetyError(f"expected 4 joints, got {len(joints_deg)}")
        for index, (value, (low, high)) in enumerate(
            zip(joints_deg, self.joint_limits_deg, strict=True)
        ):
            if not (low + margin_deg <= float(value) <= high - margin_deg):
                raise SafetyError(
                    f"j{index + 1}={value:.2f} outside [{low + margin_deg:.2f}, "
                    f"{high - margin_deg:.2f}]"
                )

    def check_step(
        self,
        previous_xyz_mm: Sequence[float],
        next_xyz_mm: Sequence[float],
        max_step_mm: float = 20.0,
    ) -> None:
        distance = float(np.linalg.norm(np.asarray(next_xyz_mm) - np.asarray(previous_xyz_mm)))
        if distance > max_step_mm:
            raise SafetyError(
                f"step {distance:.2f} mm exceeds max {max_step_mm:.2f} mm"
            )
