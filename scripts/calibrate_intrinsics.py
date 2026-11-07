#!/usr/bin/env python3
"""ChArUco camera intrinsics for one role (default: front).

Move/tilt the 8x7 / 25 mm board in front of the camera, press SPACE to capture
at least 15-20 views, then q to finish. With --no-preview the script captures
automatically whenever the board has moved and enough corners are visible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dobot_rua.calibration.charuco import detect_board, make_board
from dobot_rua.calibration.intrinsics import calibrate_from_detections


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="front", help="role in cameras.yaml or a device path")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--squares-x", type=int, default=8)
    p.add_argument("--squares-y", type=int, default=7)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.75)
    p.add_argument("--dictionary", default="DICT_5X5_1000")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-preview", action="store_true")
    return p


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
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detections: list[tuple[np.ndarray, np.ndarray]] = []
    last_mean = None
    last_capture = 0.0
    print("SPACE = capture, q = finish")
    try:
        while len(detections) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
            corners, ids = detect_board(board, frame)
            preview = frame.copy()
            if corners is not None:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids, (0, 255, 0))
                except Exception:
                    pass
            if args.no_preview:
                key = 255
                if (
                    corners is not None
                    and len(ids) >= 20
                    and time.time() - last_capture > 1.0
                    and (
                        last_mean is None
                        or np.linalg.norm(corners.mean(axis=0) - last_mean) > 10.0
                    )
                ):
                    key = 32
            else:
                cv2.imshow("charuco", preview)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and corners is not None and len(ids) >= 6:
                detections.append((corners, ids))
                last_mean = corners.mean(axis=0)
                last_capture = time.time()
                print(f"captured {len(detections)}/{args.samples} corners={len(ids)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(detections) < 3:
        print("not enough views; need at least 3")
        return 1
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
        args.output.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"wrote {args.output} [{args.camera}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
