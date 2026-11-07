#!/usr/bin/env python3
"""Generate a printable A4 ChArUco board at exact physical size."""

from __future__ import annotations

import argparse
from pathlib import Path

from dobot_rua.calibration.board_image import render_charuco_a4, save_board
from dobot_rua.calibration.charuco import make_board


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--squares-x", type=int, default=5)
    p.add_argument("--squares-y", type=int, default=5)
    p.add_argument("--square-mm", type=float, default=10.0)
    p.add_argument("--marker-mm", type=float, default=7.5)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-prefix", type=Path, default=Path("audit/boards/charuco_wrist_5x5_10mm"))
    return p


def main() -> int:
    args = parser().parse_args()
    board = make_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
    )
    page = render_charuco_a4(board, dpi=args.dpi)
    pdf = args.output_prefix.with_suffix(".pdf")
    png = args.output_prefix.with_suffix(".png")
    save_board(page, pdf, png, dpi=args.dpi)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(
        f"board {args.squares_x}x{args.squares_y}, "
        f"{args.square_mm} mm squares, {args.marker_mm} mm markers, "
        f"{args.dictionary}, A4, {args.dpi} DPI"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
