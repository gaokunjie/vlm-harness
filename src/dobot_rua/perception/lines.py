"""Line detection, intersection, and an independent Hough-based GT path."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import intersect_two_lines, project_pixel_to_plane


@dataclass(frozen=True)
class LineIntersection:
    line_color: str
    intersection_uv: tuple[float, float]
    intersection_xyz_m: tuple[float, float, float]
    line_ids: tuple[int, int]
    confidence: float


def _line_mask(image_bgr: np.ndarray, line_color: str) -> np.ndarray | None:
    if line_color == "black":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.inRange(gray, 0, 80)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if line_color == "red":
        low = cv2.inRange(hsv, np.array((0, 80, 50)), np.array((10, 255, 255)))
        high = cv2.inRange(hsv, np.array((170, 80, 50)), np.array((180, 255, 255)))
        return cv2.bitwise_or(low, high)
    if line_color == "blue":
        return cv2.inRange(hsv, np.array((100, 80, 50)), np.array((130, 255, 255)))
    if line_color == "green":
        return cv2.inRange(hsv, np.array((40, 80, 50)), np.array((85, 255, 255)))
    return None


def _fit_segments(segments: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Group Hough segments by orientation and fit one line per group."""
    angles = []
    for x1, y1, x2, y2 in segments:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
        angles.append(angle)
    order = np.argsort(angles)
    groups: list[list[int]] = []
    for index in order:
        if not groups or abs(angles[index] - angles[groups[-1][-1]]) > 12.0:
            groups.append([index])
        else:
            groups[-1].append(index)
    groups.sort(key=len, reverse=True)
    fitted: list[tuple[float, np.ndarray]] = []
    for group in groups[:4]:
        points = []
        for index in group:
            x1, y1, x2, y2 = segments[index]
            points.extend([(x1, y1), (x2, y2)])
        points_array = np.asarray(points, dtype=np.float32)
        if len(points_array) < 4:
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            points_array, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1)
        norm = float(np.hypot(vx, vy))
        if norm < 1e-9:
            continue
        normal = np.array([-vy / norm, vx / norm])
        line = np.array([normal[0], normal[1], -(normal @ np.array([x0, y0]))])
        fitted.append((float(np.mean([angles[index] for index in group])), line))
    return fitted


def detect_line_intersection(
    image_bgr: np.ndarray,
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    plane_z_m: float,
    line_color: str,
) -> LineIntersection | None:
    """Detect the intersection of the two dominant same-color lines."""
    mask = _line_mask(image_bgr, line_color)
    if mask is None:
        return None
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0, threshold=30, minLineLength=40, maxLineGap=10
    )
    if segments is None:
        return None
    segments = segments.reshape(-1, 4)
    fitted = _fit_segments(segments)
    if len(fitted) < 2:
        return None
    best = None
    for first in range(len(fitted)):
        for second in range(first + 1, len(fitted)):
            angle_gap = abs(fitted[first][0] - fitted[second][0])
            angle_gap = min(angle_gap, 180.0 - angle_gap)
            if angle_gap < 20.0:
                continue
            point = intersect_two_lines(fitted[first][1], fitted[second][1])
            if point is None:
                continue
            height, width = image_bgr.shape[:2]
            if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
                continue
            score = angle_gap
            if best is None or score > best[0]:
                best = (score, point, first, second)
    if best is None:
        return None
    _, point, first, second = best
    world = project_pixel_to_plane(intrinsic_k, t_world_from_optical, point, plane_z_m)
    return LineIntersection(
        line_color=line_color,
        intersection_uv=(float(point[0]), float(point[1])),
        intersection_xyz_m=(float(world[0]), float(world[1]), float(world[2])),
        line_ids=(int(first), int(second)),
        confidence=float(min(1.0, best[0] / 90.0)),
    )


def measure_intersection_gt(
    image_bgr: np.ndarray,
    intrinsic_k: np.ndarray,
    t_world_from_optical: np.ndarray,
    plane_z_m: float,
    line_color: str,
) -> LineIntersection | None:
    """Independent GT path: standard Hough transform + theta clustering.

    Deliberately different from :func:`detect_line_intersection`, which uses
    probabilistic Hough + per-orientation least-squares fitting.
    """
    mask = _line_mask(image_bgr, line_color)
    if mask is None:
        return None
    lines = cv2.HoughLines(mask, 1, np.pi / 180.0, threshold=80)
    if lines is None:
        return None
    entries = []
    for rho, theta in lines.reshape(-1, 2):
        angle = float(np.degrees(theta)) % 180.0
        entries.append((angle, float(rho), float(theta)))
    entries.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, float, float]]] = []
    for entry in entries:
        if not groups or abs(entry[0] - groups[-1][-1][0]) > 8.0:
            groups.append([entry])
        else:
            groups[-1].append(entry)
    groups.sort(key=len, reverse=True)
    if len(groups) < 2:
        return None
    best = None
    for first in range(min(4, len(groups))):
        for second in range(first + 1, min(4, len(groups))):
            mean_first = np.mean([item[0] for item in groups[first]])
            mean_second = np.mean([item[0] for item in groups[second]])
            gap = abs(mean_first - mean_second)
            gap = min(gap, 180.0 - gap)
            if gap < 20.0:
                continue
            line_first = _group_to_line(groups[first])
            line_second = _group_to_line(groups[second])
            point = intersect_two_lines(line_first, line_second)
            if point is None:
                continue
            height, width = image_bgr.shape[:2]
            if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
                continue
            if best is None or gap > best[0]:
                best = (gap, point, first, second)
    if best is None:
        return None
    _, point, first, second = best
    world = project_pixel_to_plane(intrinsic_k, t_world_from_optical, point, plane_z_m)
    return LineIntersection(
        line_color=line_color,
        intersection_uv=(float(point[0]), float(point[1])),
        intersection_xyz_m=(float(world[0]), float(world[1]), float(world[2])),
        line_ids=(int(first), int(second)),
        confidence=float(min(1.0, best[0] / 90.0)),
    )


def _group_to_line(group: list[tuple[float, float, float]]) -> np.ndarray:
    theta = float(np.mean([item[2] for item in group]))
    rho = float(np.mean([item[1] for item in group]))
    line = np.array([np.cos(theta), np.sin(theta), -rho], dtype=float)
    norm = np.linalg.norm(line[:2])
    return line / norm
