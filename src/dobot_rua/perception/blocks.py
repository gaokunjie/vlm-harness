"""Colored block detection and an independent BGR/connected-component GT path."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import project_pixel_to_plane


@dataclass(frozen=True)
class BlockDetection:
    id: int
    color: str
    center_uv: tuple[float, float]
    center_xyz_m: tuple[float, float, float]
    confidence: float


_HSV_RANGES = {
    "red": ((0, 80, 60), (10, 255, 255)),
    "red_high": ((170, 80, 60), (180, 255, 255)),
    "blue": ((100, 80, 60), (130, 255, 255)),
    "green": ((40, 80, 60), (85, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
}


def _hsv_mask(image_bgr: np.ndarray, color: str) -> np.ndarray | None:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if color == "red":
        low = cv2.inRange(hsv, np.array(_HSV_RANGES["red"][0]), np.array(_HSV_RANGES["red"][1]))
        high = cv2.inRange(hsv, np.array(_HSV_RANGES["red_high"][0]), np.array(_HSV_RANGES["red_high"][1]))
        return cv2.bitwise_or(low, high)
    if color not in _HSV_RANGES:
        return None
    low, high = _HSV_RANGES[color]
    return cv2.inRange(hsv, np.array(low), np.array(high))


def detect_blocks(
    image_bgr: np.ndarray,
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    plane_z_m: float,
    colors: tuple[str, ...] = ("blue", "yellow"),
    distortion: np.ndarray | None = None,
) -> list[BlockDetection]:
    """Detect filled square blocks with HSV + contours + subpixel moments."""
    detections: list[BlockDetection] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for color in colors:
        mask = _hsv_mask(image_bgr, color)
        if mask is None:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 100.0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue
            # A square rotated 45 deg has an axis-aligned bbox fill of only
            # ~0.5, so use the minimum-area rectangle: it stays tight around
            # the block at any rotation and still rejects long thin lines.
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
            if rect_w <= 0.0 or rect_h <= 0.0:
                continue
            rect_aspect = min(rect_w, rect_h) / max(rect_w, rect_h)
            if rect_aspect < 0.6:
                continue
            fill = area / float(rect_w * rect_h)
            if fill < 0.6:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]
            world = project_pixel_to_plane(
                intrinsic_k,
                t_world_from_optical,
                (u, v),
                plane_z_m,
                distortion=distortion,
            )
            detections.append(
                BlockDetection(
                    id=len(detections),
                    color=color,
                    center_uv=(float(u), float(v)),
                    center_xyz_m=(float(world[0]), float(world[1]), float(world[2])),
                    confidence=float(min(1.0, fill)),
                )
            )
    return detections


def measure_block_gt(
    image_bgr: np.ndarray,
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    plane_z_m: float,
    color: str,
    distortion: np.ndarray | None = None,
) -> BlockDetection | None:
    """Independent GT path: BGR thresholds + connected components.

    Deliberately different from :func:`detect_blocks` so that GT and the
    agent-facing tool do not share the same failure modes.
    """
    b, g, r = cv2.split(image_bgr)
    if color == "red":
        mask = ((r > 150) & (g < 80) & (b < 80)).astype(np.uint8) * 255
    elif color == "blue":
        mask = ((b > 150) & (g < 80) & (r < 80)).astype(np.uint8) * 255
    elif color == "green":
        mask = ((g > 120) & (r < 100) & (b < 100)).astype(np.uint8) * 255
    elif color == "yellow":
        mask = ((r > 150) & (g > 150) & (b < 100)).astype(np.uint8) * 255
    else:
        return None
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: tuple[float, int] | None = None
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        width = float(stats[label, cv2.CC_STAT_WIDTH])
        height = float(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 100.0 or width <= 0 or height <= 0:
            continue
        if not 0.6 <= width / height <= 1.6:
            continue
        # A long thin line has a square-ish bbox but almost no fill; reject it
        # so the GT path cannot silently return a line as a block.
        if area / float(width * height) < 0.5:
            continue
        if best is None or area > best[0]:
            best = (area, label)
    if best is None:
        return None
    _, label = best
    u, v = centroids[label]
    world = project_pixel_to_plane(
        intrinsic_k,
        t_world_from_optical,
        (u, v),
        plane_z_m,
        distortion=distortion,
    )
    area = float(stats[label, cv2.CC_STAT_AREA])
    return BlockDetection(
        id=0,
        color=color,
        center_uv=(float(u), float(v)),
        center_xyz_m=(float(world[0]), float(world[1]), float(world[2])),
        confidence=float(min(1.0, area / 500.0)),
    )
