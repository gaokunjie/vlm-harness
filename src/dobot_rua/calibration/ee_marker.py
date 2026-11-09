"""Front-camera pose from an ArUco marker mounted on the end effector.

The marker sits flat on the end-effector face, centred on the suction-cup axis,
so its centre is the flange position plus a small constant offset expressed in
the flange frame (which rotates with j4 about the vertical):

    marker_base = flange + Rz(j4) @ offset

Projecting the marker centre through the camera gives two equations per pose,
so four or more poses that are not coplanar determine the camera rotation, its
horizontal offset and the marker's in-plane offset. The camera's height and the
marker plane's height trade off exactly (moving the whole scene down by the
same amount leaves every projection unchanged), so the marker's vertical offset
is fixed at ``marker_height_mm`` (0 by default: the marker lies on the flange
plane) and the camera height is recovered relative to that assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EEMarkerSolution:
    rotation_world_optical: np.ndarray
    camera_position_mm: np.ndarray
    marker_offset_mm: np.ndarray
    residuals_px: np.ndarray
    rms_px: float
    max_px: float


def _marker_points(
    flange_mm: np.ndarray, j4_deg: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            flange + Rotation.from_euler("z", float(angle), degrees=True).apply(offset)
            for flange, angle in zip(flange_mm, j4_deg, strict=True)
        ]
    )


def _project(
    points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray | None = None,
):
    """Project world points with the full OpenCV model, distortion included."""
    import cv2

    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=float).reshape(-1, 1, 3),
        np.asarray(rvec, dtype=float),
        np.asarray(tvec, dtype=float),
        intrinsic,
        None if distortion is None else np.asarray(distortion, dtype=float),
    )
    return projected.reshape(-1, 2)


def solve_camera_from_ee_marker(
    pixels: np.ndarray,
    flange_mm: np.ndarray,
    j4_deg: np.ndarray,
    intrinsic: np.ndarray,
    *,
    distortion: np.ndarray | None = None,
    marker_height_mm: float = 0.0,
) -> EEMarkerSolution:
    """Solve ``T_world_from_optical`` from >= 4 marker observations."""
    pixels = np.asarray(pixels, dtype=float)
    flange = np.asarray(flange_mm, dtype=float)
    j4 = np.asarray(j4_deg, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must be an Nx2 array")
    if len(pixels) < 4:
        raise ValueError("at least four poses are required")
    if flange.shape != (len(pixels), 3) or j4.shape != (len(pixels),):
        raise ValueError("flange and j4 arrays must match the pixel count")

    centred = flange - flange.mean(axis=0)
    if np.linalg.matrix_rank(centred, tol=1e-6) < 2:
        raise ValueError("the poses are degenerate (not degenerate: need variation)")

    import cv2

    # Initial pose from the marker centres assuming no offset.
    ok, rvec, tvec = cv2.solvePnP(
        flange.reshape(-1, 1, 3).astype(np.float64),
        pixels.reshape(-1, 1, 2).astype(np.float64),
        intrinsic,
        None if distortion is None else np.asarray(distortion, dtype=float),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed on the marker observations")
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)

    def residuals(vec: np.ndarray) -> np.ndarray:
        r = vec[:3]
        t = vec[3:6]
        offset = np.array([vec[6], vec[7], float(marker_height_mm)])
        points = _marker_points(flange, j4, offset)
        projected = _project(points, r, t, intrinsic)
        return (projected - pixels).reshape(-1)

    start = np.concatenate([rvec, tvec, np.zeros(2)])
    result = least_squares(
        residuals, start, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14
    )

    rvec, tvec = result.x[:3], result.x[3:6]
    offset = np.array([result.x[6], result.x[7], float(marker_height_mm)])
    points = _marker_points(flange, j4, offset)
    projected = _project(points, rvec, tvec, intrinsic)
    per_pose = np.linalg.norm(projected - pixels, axis=1)

    rotation_camera_world = Rotation.from_rotvec(rvec).as_matrix()
    rotation_world_optical = rotation_camera_world.T
    return EEMarkerSolution(
        rotation_world_optical=rotation_world_optical,
        camera_position_mm=-rotation_world_optical @ tvec,
        marker_offset_mm=offset,
        residuals_px=per_pose,
        rms_px=float(np.sqrt(np.mean(per_pose**2))),
        max_px=float(np.max(per_pose)),
    )


def solve_camera_from_marker_corners(
    corners_px: np.ndarray,
    flange_mm: np.ndarray,
    j4_deg: np.ndarray,
    j1_deg: np.ndarray,
    marker_side_mm: float,
    intrinsic: np.ndarray,
    *,
    distortion: np.ndarray | None = None,
    marker_height_mm: float = 0.0,
) -> EEMarkerSolution:
    """Solve the camera pose from the four corners of the end-effector marker.

    Unlike the centre-only model this also uses the marker's orientation, so it
    fixes the scale (given the printed side length) and removes the mirror
    ambiguity that a marker-centre-only fit leaves when every pose sits in one
    vertical plane.

    ``corners_px`` is ``Nx4x2`` in the detector's corner order. The marker lies
    flat on the end-effector face (its normal is world +z) and turns about the
    vertical with the flange (world yaw ``j1 + j4``); the fitted third component
    of the translation absorbs the tape's own rotation.
    """
    corners = np.asarray(corners_px, dtype=float)
    flange = np.asarray(flange_mm, dtype=float)
    j4 = np.asarray(j4_deg, dtype=float)
    j1 = np.asarray(j1_deg, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError("corners must be an Nx4x2 array")
    if len(corners) < 4:
        raise ValueError("at least four poses are required")
    if (
        flange.shape != (len(corners), 3)
        or j4.shape != (len(corners),)
        or j1.shape != (len(corners),)
    ):
        raise ValueError("poses must match the corner count")

    half = float(marker_side_mm) / 2.0
    # detector order: top-left, top-right, bottom-right, bottom-left
    local = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
    )

    def marker_corners(vec: np.ndarray) -> np.ndarray:
        offset = np.array([vec[6], vec[7], float(marker_height_mm)])
        yaw_offset = float(vec[8])
        out = []
        for flange_i, a4, a1 in zip(flange, j4, j1, strict=True):
            centre = flange_i + Rotation.from_euler("z", a4, degrees=True).apply(offset)
            yaw = np.radians(a1 + a4) + yaw_offset
            rotate = Rotation.from_euler("z", yaw).as_matrix()
            out.append(centre + local @ rotate.T)
        return np.array(out)

    import cv2

    first = marker_corners(np.concatenate([np.zeros(6), [0.0, 0.0], [0.0]]))
    ok, rvec, tvec = cv2.solvePnP(
        first.reshape(-1, 3).astype(np.float64),
        corners.reshape(-1, 2).astype(np.float64),
        intrinsic,
        None if distortion is None else np.asarray(distortion, dtype=float),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise ValueError("solvePnP failed on the marker corners")
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)

    def residuals(vec: np.ndarray) -> np.ndarray:
        points = marker_corners(vec)
        projected = _project(
            points.reshape(-1, 3), vec[:3], vec[3:6], intrinsic, distortion
        )
        return (projected - corners.reshape(-1, 2)).reshape(-1)

    start = np.concatenate([rvec, tvec, np.zeros(3)])
    result = least_squares(
        residuals, start, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    rvec, tvec = result.x[:3], result.x[3:6]
    offset = np.array([result.x[6], result.x[7], float(marker_height_mm)])

    points = marker_corners(result.x)
    projected = _project(points.reshape(-1, 3), rvec, tvec, intrinsic, distortion)
    per_pose = np.linalg.norm(
        projected.reshape(len(corners), 4, 2) - corners, axis=2
    ).mean(axis=1)

    rotation_camera_world = Rotation.from_rotvec(rvec).as_matrix()
    rotation_world_optical = rotation_camera_world.T
    return EEMarkerSolution(
        rotation_world_optical=rotation_world_optical,
        camera_position_mm=-rotation_world_optical @ tvec,
        marker_offset_mm=offset,
        residuals_px=per_pose,
        rms_px=float(np.sqrt(np.mean(per_pose**2))),
        max_px=float(np.max(per_pose)),
    )
