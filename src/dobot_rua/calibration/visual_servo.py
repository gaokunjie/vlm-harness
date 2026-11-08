"""Pixel-space helpers for moving the tool onto a target seen by a fixed camera.

The front camera has no extrinsic yet, but a plane-parallel tool motion maps
affinely to the image: probe two base axes, build the 2x2 Jacobian, and solve
for the base delta that moves the detected reference point onto the target.
"""

from __future__ import annotations

import cv2
import numpy as np


def board_center_pixel(corners) -> tuple[float, float]:
    """Mean of every detected marker corner: a stable board centre in pixels."""
    points = np.concatenate(
        [np.asarray(corner, dtype=float).reshape(4, 2) for corner in corners], axis=0
    )
    centre = points.mean(axis=0)
    return float(centre[0]), float(centre[1])


def jacobian_from_probes(
    p0, px, py, step_mm: float
) -> np.ndarray:
    """2x2 ``d(pixel)/d(base mm)`` from +x and +y probe moves."""
    p0 = np.asarray(p0, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if step_mm <= 0:
        raise ValueError("probe step must be positive")
    return np.column_stack([(px - p0) / step_mm, (py - p0) / step_mm])


def delta_mm(jacobian, error_px, max_mm: float) -> np.ndarray:
    """Base move (mm) that produces ``error_px``, clamped to ``max_mm``."""
    matrix = np.asarray(jacobian, dtype=float)
    error = np.asarray(error_px, dtype=float)
    try:
        delta = np.linalg.solve(matrix, error)
    except np.linalg.LinAlgError as exc:
        raise ValueError("probe Jacobian is singular") from exc
    norm = float(np.linalg.norm(delta))
    if norm > max_mm and norm > 0.0:
        delta = delta * (max_mm / norm)
    return delta


def largest_dark_blob(
    frame_bgr: np.ndarray,
    threshold: int = 70,
    margin: int = 60,
    min_area: float = 8000.0,
) -> dict | None:
    """Locate the end effector: largest dark blob away from the frame border.

    ``reference`` is the mean of the leftmost ten contour points, which sits on
    the tool tip (the suction-cup end) and barely moves when the arm yaws.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if best is None or area > best[0]:
            best = (area, contour)
    if best is None:
        return None
    area, contour = best
    points = contour.reshape(-1, 2).astype(float)
    if len(points) == 0:
        return None
    x_min = points[:, 0].min()
    leftmost = points[points[:, 0] <= x_min + 3.0].mean(axis=0)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        centroid = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    else:
        centroid = (float(leftmost[0]), float(leftmost[1]))
    return {
        "area": area,
        "reference": (float(leftmost[0]), float(leftmost[1])),
        "centroid": centroid,
    }
