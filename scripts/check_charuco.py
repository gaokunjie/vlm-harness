#!/usr/bin/env python3
"""Try several ArUco dictionaries against the printed ChArUco board.

Hold/place the 8x7 board in the front camera view while this runs. It prints
the dictionary with the most detected corners and saves the best frame under
audit/charuco/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from dobot_rua.calibration.charuco import detect_board, make_board

DICTIONARIES = (
    "DICT_4X4_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="/dev/video2")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--marker-mm", type=float, default=18.75)
    p.add_argument("--duration", type=float, default=45.0)
    return p


def main() -> int:
    args = parser().parse_args()
    output = Path("audit/charuco")
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    boards = {}
    for name in DICTIONARIES:
        try:
            boards[name] = make_board(
                args.squares_x,
                args.squares_y,
                args.square_mm,
                args.marker_mm,
                name,
            )
        except Exception as error:  # noqa: BLE001 - invalid board for this dictionary
            print(f"skip {name}: {error}")
    best = (0, None)
    started = time.time()
    try:
        while time.time() - started < args.duration:
            ok, frame = cap.read()
            if not ok:
                continue
            for name, board in boards.items():
                corners, ids = detect_board(board, frame)
                if corners is not None and len(ids) > best[0]:
                    best = (len(ids), name)
                    print(f"best {len(ids)} corners with {name} at {time.time()-started:.1f}s", flush=True)
                    cv2.imwrite(str(output / f"best_{name}.png"), frame)
            time.sleep(0.05)
    finally:
        cap.release()
    print(f"RESULT best={best[0]} dictionary={best[1]}")
    if best[1]:
        print(f"debug image: {output / f'best_{best[1]}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
