"""Keyboard jogging: map a keypress to the next clamped Cartesian target."""

from __future__ import annotations

from dataclasses import dataclass

KEY_DELTAS: dict[str, tuple[float, float, float, float]] = {
    "w": (1.0, 0.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0, 0.0),
    "d": (0.0, -1.0, 0.0, 0.0),
    "r": (0.0, 0.0, 1.0, 0.0),
    "f": (0.0, 0.0, -1.0, 0.0),
    "q": (0.0, 0.0, 0.0, -1.0),
    "e": (0.0, 0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class JogBox:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    r: tuple[float, float]


@dataclass(frozen=True)
class JogStep:
    target: tuple[float, float, float, float]
    clamped: bool


def next_target(
    current: tuple[float, float, float, float],
    key: str,
    step_mm: float,
    step_deg: float,
    box: JogBox,
) -> JogStep | None:
    """Next pose for ``key``; ``None`` when the key is not a motion key."""
    delta = KEY_DELTAS.get(key.lower())
    if delta is None:
        return None
    raw = (
        current[0] + delta[0] * step_mm,
        current[1] + delta[1] * step_mm,
        current[2] + delta[2] * step_mm,
        current[3] + delta[3] * step_deg,
    )
    limits = (box.x, box.y, box.z, box.r)
    clamped = tuple(
        min(max(value, low), high)
        for value, (low, high) in zip(raw, limits, strict=True)
    )
    return JogStep(
        target=(float(clamped[0]), float(clamped[1]), float(clamped[2]), float(clamped[3])),
        clamped=bool(any(abs(a - b) > 1e-9 for a, b in zip(raw, clamped, strict=True))),
    )
