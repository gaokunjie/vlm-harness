#!/usr/bin/env python3
"""ChArUco / ArUco camera intrinsics for one role (default: front).

Use --marker-mode for close-up or heavily occluded views where only a few
ChArUco inner corners are visible: it calibrates from the four corners of each
detected ArUco marker instead.

Move/tilt the printed board in front of the camera. With --no-preview the
script captures automatically when the board has moved and enough markers /
corners are visible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import detect_board, make_board
from dobot_rua.calibration.intrinsics import (
    calibrate_from_detections,
    calibrate_from_marker_detections,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="front", help="role in cameras.yaml or a device path")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.75)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--marker-mode", action="store_true", help="calibrate from ArUco marker corners")
    p.add_argument("--min-markers", type=int, default=3, help="minimum visible markers in marker mode")
    p.add_argument("--min-corners", type=int, default=20, help="minimum ChArUco corners in normal mode")
    p.add_argument("--regularized", action="store_true", help="fix aspect ratio, zero tangential, estimate k1/k2 only")
    p.add_argument("--initial-focal", type=float, default=1000.0)
    p.add_argument("--initial-cx", type=float, default=640.0)
    p.add_argument("--initial-cy", type=float, default=360.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--timeout-s", type=float, default=300.0)
    return p


def _mean_corners(corners):
    """Mean of a marker-corner tuple or a single corner array."""
    if isinstance(corners, (tuple, list)):
        arrays = [np.asarray(item).reshape(-1, 2) for item in corners if item is not None]
        if not arrays:
            return None
        return np.concatenate(arrays, axis=0).mean(axis=0)
    return np.asarray(corners).reshape(-1, 2).mean(axis=0)


def _resolve(camera: str, config_path: Path):
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if camera in data.get("cameras", {}):
        return data["cameras"][camera]["device"], data
    return camera, data


def main() -> int:
    args = parser().parse_args()
    device, config = _resolve(args.camera, args.camera_config)
    board = make_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    marker_detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detections: list[tuple[np.ndarray, np.ndarray]] = []
    started = time.time()
    last_mean = None
    last_capture = 0.0
    print("SPACE = capture, q = finish")
    try:
        while (
            len(detections) < args.samples
            and time.time() - started < args.timeout_s
        ):
            ok, frame = cap.read()
            if not ok:
                continue
            preview = frame.copy()
            if args.marker_mode:
                corners, ids, _ = marker_detector.detectMarkers(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
                available = ids is not None and len(ids) >= args.min_markers
                mean = None if corners is None else _mean_corners(corners)
                if available:
                    try:
                        cv2.aruco.drawDetectedMarkers(preview, corners, ids)
                    except Exception:
                        pass
            else:
                corners, ids = detect_board(board, frame)
                available = (
                    corners is not None
                    and ids is not None
                    and len(ids) >= args.min_corners
                )
                mean = None if corners is None else corners.mean(axis=0)
                if corners is not None:
                    try:
                        cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids, (0, 255, 0))
                    except Exception:
                        pass
            if args.no_preview:
                key = 255
                if (
                    available
                    and time.time() - last_capture > 1.0
                    and (
                        last_mean is None
                        or np.linalg.norm(mean - last_mean) > 10.0
                    )
                ):
                    key = 32
            else:
                cv2.imshow("calibration", preview)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and available:
                detections.append((corners, ids))
                last_mean = mean
                last_capture = time.time()
                count = len(ids)
                label = "markers" if args.marker_mode else "corners"
                print(
                    f"captured {len(detections)}/{args.samples} {label}={count}",
                    flush=True,
                )
    finally:
        cap.release()
        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if len(detections) < 3:
        print("not enough views; need at least 3")
        return 1
    if args.marker_mode:
        initial_k = None
        if args.regularized:
            initial_k = np.array(
                [
                    [args.initial_focal, 0.0, args.initial_cx],
                    [0.0, args.initial_focal, args.initial_cy],
                    [0.0, 0.0, 1.0],
                ]
            )
        result = calibrate_from_marker_detections(
            board,
            detections,
            (args.height, args.width),
            initial_K=initial_k,
            fix_aspect=args.regularized,
            zero_tangent=args.regularized,
            fix_k3=args.regularized,
        )
    else:
        result = calibrate_from_detections(
            board, detections, (args.height, args.width)
        )
    print(f"RMS={result.rms:.4f}px")
    print("K=\n", np.array2string(result.intrinsic_K, precision=4))
    print("dist=", result.distortion)

    if args.camera in config.get("cameras", {}):
        camera = config["cameras"][args.camera]
        camera["intrinsic_K"] = result.intrinsic_K.tolist()
        camera["distortion"] = result.distortion.tolist()
        camera["image_size"] = [args.height, args.width]
        camera["calibration_date"] = time.strftime("%Y-%m-%d")
        camera["calibration_mode"] = "marker" if args.marker_mode else "charuco"
        args.output.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"wrote {args.output} [{args.camera}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
