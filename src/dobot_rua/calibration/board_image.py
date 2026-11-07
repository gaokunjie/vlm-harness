"""Printable A4 ChArUco board with an exact 10 mm scale bar."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size_px: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def render_charuco_a4(
    board: cv2.aruco.CharucoBoard,
    *,
    dpi: int = 300,
    page_mm: tuple[float, float] = (210.0, 297.0),
    quiet_zone_mm: float = 5.0,
) -> Image.Image:
    """Render the board at its exact physical size on a white A4 page."""
    page_w = round(page_mm[0] / 25.4 * dpi)
    page_h = round(page_mm[1] / 25.4 * dpi)
    page = Image.new("RGB", (page_w, page_h), "white")

    squares_x, squares_y = board.getChessboardSize()
    square_px = board.getSquareLength() * 1000.0 / 25.4 * dpi
    board_width = round(square_px * squares_x)
    board_height = round(square_px * squares_y)
    quiet_px = round(quiet_zone_mm / 25.4 * dpi)
    board_image = board.generateImage(
        (board_width + 2 * quiet_px, board_height + 2 * quiet_px),
        marginSize=quiet_px,
    )
    board_rgb = Image.fromarray(cv2.cvtColor(board_image, cv2.COLOR_GRAY2RGB))
    x = (page_w - board_rgb.width) // 2
    y = max(quiet_px, (page_h - board_rgb.height) // 2 - round(20 / 25.4 * dpi))
    page.paste(board_rgb, (x, y))

    draw = ImageDraw.Draw(page)
    font = _font(max(12, round(4 / 25.4 * dpi)))
    bar_y = y + board_rgb.height + round(6 / 25.4 * dpi)
    bar_length = round(10.0 / 25.4 * dpi)
    bar_height = max(2, round(2 / 25.4 * dpi))
    draw.rectangle([x, bar_y, x + bar_length, bar_y + bar_height], fill="black")
    draw.text(
        (x + bar_length + round(3 / 25.4 * dpi), bar_y - round(2 / 25.4 * dpi)),
        "10 mm",
        fill="black",
        font=font,
    )
    draw.text(
        (x, bar_y + round(8 / 25.4 * dpi)),
        f"ChArUco {squares_x}x{squares_y}, "
        f"{board.getSquareLength()*1000:.1f} mm squares, "
        f"{board.getMarkerLength()*1000:.1f} mm markers. "
        "Print at 100% / actual size.",
        fill="black",
        font=font,
    )
    return page


def save_board(page: Image.Image, pdf_path: str | Path, png_path: str | Path, *, dpi: int = 300) -> None:
    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(png_path, dpi=(dpi, dpi))
    page.save(pdf_path, "PDF", resolution=float(dpi))
