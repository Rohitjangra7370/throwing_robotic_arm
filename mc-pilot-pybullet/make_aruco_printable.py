"""
Build print-ready A4 PDFs of the ArUco / ChArUco calibration targets.

WHY A PDF AND NOT THE PNGs
--------------------------
Physical size is the whole point. A calibration target whose printed marker is
79 mm when the code believes 80 mm puts a 1.25% systematic error into every
distance derived from it -- at our 0.6-0.8 m working range that is millimetres,
the same scale as the landing accuracy we are trying to measure.

PNGs carry size only as a DPI hint, which print pipelines are free to ignore;
"fit to page" in particular silently rescales. A PDF places each marker at an
explicit size in millimetres on a real A4 page, and CUPS is told
`print-scaling=none`, so there are two independent reasons for the geometry to
come out right instead of zero.

VERIFY, DO NOT TRUST
--------------------
Every page carries a 100 mm ruler with 10 mm ticks. After printing, measure it.
If it is not 100.0 mm, the printer scaled and every marker on that page is off
by the same factor -- either reprint, or measure a marker edge and pass the
MEASURED size to the calibrator. That ruler is the reason this script exists
rather than just sending the PNGs.

    python3 make_aruco_printable.py --out aruco_targets/print
"""

import argparse
import io
import os

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

DICT = cv2.aruco.DICT_4X4_50
RENDER_DPI = 600          # source resolution; Brother DCP-B7535DW does 1200


def _marker_image(marker_id, side_mm, quiet_mm):
    """Marker bitmap with its quiet zone, rendered at RENDER_DPI."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    side_px = int(round(side_mm / 25.4 * RENDER_DPI))
    quiet_px = int(round(quiet_mm / 25.4 * RENDER_DPI))
    img = cv2.aruco.generateImageMarker(d, marker_id, side_px)
    canvas = np.full((side_px + 2 * quiet_px, side_px + 2 * quiet_px), 255, np.uint8)
    canvas[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = img
    return Image.fromarray(canvas), side_mm + 2 * quiet_mm


def _charuco_image(sx, sy, square_mm, marker_ratio=0.75):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((sx, sy), square_mm / 1000.0,
                                   square_mm * marker_ratio / 1000.0, d)
    w = int(round(sx * square_mm / 25.4 * RENDER_DPI))
    h = int(round(sy * square_mm / 25.4 * RENDER_DPI))
    return Image.fromarray(board.generateImage((w, h), marginSize=0))


def _ruler(c, x_mm, y_mm, length_mm=100.0):
    """
    A printed 100 mm reference. The single most useful thing on the page:
    it converts "did the printer scale?" from a guess into a measurement.
    """
    c.setLineWidth(0.5)
    x0, y0 = x_mm * mm, y_mm * mm
    c.line(x0, y0, x0 + length_mm * mm, y0)
    for i in range(int(length_mm // 10) + 1):
        tick = 4.0 if i % 5 else 7.0
        c.line(x0 + i * 10 * mm, y0, x0 + i * 10 * mm, y0 + tick * mm)
    c.setFont("Helvetica", 7)
    c.drawString(x0, y0 - 3.5 * mm,
                 f"{length_mm:.0f} mm reference - MEASURE THIS. If it is not "
                 f"{length_mm:.0f}.0 mm the page was scaled; markers are off by the same factor.")


def build(out_dir, marker_mm, quiet_mm, ids, sx, sy, square_mm):
    os.makedirs(out_dir, exist_ok=True)
    W, H = A4
    paths = []

    # ---- pages 1..n: the loose markers, check points ---------------------- #
    # Lay out with an explicit fit check. The first version put 3 rows of 92 mm
    # tiles into ~257 mm of usable height and silently drew the bottom row at
    # y = -5 mm, straight over the ruler -- found by rasterising the PDF and
    # measuring, which is the same reason the ruler is on the page at all.
    tile = marker_mm + 2 * quiet_mm
    TOP, BOT = 18.0, 24.0                 # title band, ruler+caption band
    usable_h = H / mm - TOP - BOT
    cols = max(1, int((W / mm) // tile))
    rows = max(1, int(usable_h // tile))
    per_page = cols * rows
    assert cols * tile <= W / mm + 1e-6, "marker row wider than the page"
    assert rows * tile <= usable_h + 1e-6, "marker column taller than the usable height"

    gap_x = (W / mm - cols * tile) / (cols + 1)
    gap_y = (usable_h - rows * tile) / max(rows, 1)
    p1 = os.path.join(out_dir, "aruco_markers_A4.pdf")
    c = rl_canvas.Canvas(p1, pagesize=A4)
    c.setTitle("ArUco check-point markers")
    for k, mid in enumerate(ids):
        if k and k % per_page == 0:
            _ruler(c, 15.0, 14.0); c.showPage()
        idx = k % per_page
        r, col = divmod(idx, cols)
        x = gap_x + col * (tile + gap_x)
        y = H / mm - TOP - (r + 1) * tile - r * gap_y
        assert y >= BOT - 6.0, f"marker {mid} would overlap the ruler band (y={y:.1f})"
        img, _ = _marker_image(mid, marker_mm, quiet_mm)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(15 * mm, (H / mm - 12) * mm,
                     f"ArUco DICT_4X4_50 - {marker_mm:.0f} mm markers - PRINT AT 100%")
        c.drawImage(ImageReader(img), x * mm, y * mm,
                    width=tile * mm, height=tile * mm)
        c.setFont("Helvetica", 8)
        c.drawString(x * mm, (y - 3.5) * mm, f"id={mid}   {marker_mm:.0f} mm marker")
    _ruler(c, 15.0, 14.0)
    c.showPage(); c.save(); paths.append(p1)

    # ---- page 2: the ChArUco board, primary pose reference ---------------- #
    p2 = os.path.join(out_dir, "charuco_board_A4.pdf")
    c = rl_canvas.Canvas(p2, pagesize=A4)
    c.setTitle("ChArUco board")
    bw, bh = sx * square_mm, sy * square_mm
    x = (W / mm - bw) / 2.0
    y = (H / mm - bh) / 2.0 + 6.0
    c.drawImage(ImageReader(_charuco_image(sx, sy, square_mm)),
                x * mm, y * mm, width=bw * mm, height=bh * mm)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, (H / mm - 12) * mm,
                 f"ChArUco {sx}x{sy} - {square_mm:.0f} mm squares "
                 f"({square_mm*0.75:.1f} mm markers) - PRINT AT 100%")
    c.setFont("Helvetica", 8)
    c.drawString(15 * mm, (y - 5) * mm,
                 f"board {bw:.0f} x {bh:.0f} mm - origin corner is bottom-left of the "
                 f"chessboard; place flat on the landing plane.")
    _ruler(c, 15.0, 14.0)
    c.showPage(); c.save(); paths.append(p2)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="aruco_targets/print")
    ap.add_argument("--marker_mm", type=float, default=80.0)
    ap.add_argument("--quiet_mm", type=float, default=6.0)
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_mm", type=float, default=35.0)
    args = ap.parse_args()

    paths = build(args.out, args.marker_mm, args.quiet_mm, list(range(6)),
                  args.squares_x, args.squares_y, args.square_mm)
    for p in paths:
        print(f"wrote {p}")
    print("\nprint with scaling explicitly OFF:")
    print("  lp -d Brother_DCP_B7535DW_series -o media=A4 -o print-scaling=none \\")
    print("     -o cupsPrintQuality=High <file.pdf>")
    print("\nthen MEASURE the 100 mm ruler on each page before using the targets.")


if __name__ == "__main__":
    main()
