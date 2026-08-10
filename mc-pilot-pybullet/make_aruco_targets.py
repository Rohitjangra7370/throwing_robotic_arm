"""
Generate printable ArUco / ChArUco targets for calibrating the D435i to the robot base.

WHAT WE ACTUALLY NEED
---------------------
Not lens intrinsics -- the D435i reports factory intrinsics over control
transfers and its colour stream is factory-rectified (all distortion
coefficients read 0.0). What we need is the EXTRINSIC `T_B_C`: where the camera
sits in the robot's base frame, so a detected ball pixel can be turned into a
landing point by ray-plane intersection.

The camera faces the ground, and the landing plane IS the ground plane
(target_height = 0.0, i.e. the robot's own base plane). So flat markers lying on
that plane at measured positions are exactly the right target: they calibrate
the camera against the very surface we will measure landings on, rather than
against some other plane we then have to relate to it.

TWO TARGETS, DIFFERENT JOBS
---------------------------
* `charuco_board` -- one rigid board. Best pose accuracy from a single view
  because the chessboard corners localise to sub-pixel and the board's internal
  geometry is known exactly. Use this as the primary extrinsic reference: place
  it flat on the landing plane with its origin corner at a measured (x, y) in
  the base frame.

* `markers` -- individual ArUco squares. Use these as INDEPENDENT CHECK POINTS:
  lay several at tape-measured positions across the landing zone, then verify
  the calibrated ray-plane estimate reproduces each one. The board gives you the
  calibration; the loose markers tell you whether to believe it, which is the
  gate that matters (<1 cm).

PRINTING -- READ THIS
---------------------
Print at **100% / actual size**, NOT "fit to page". Fit-to-page silently rescales
and every distance you derive afterwards inherits the error. After printing,
MEASURE a marker edge with a ruler and pass the measured value to the calibrator
rather than the nominal one -- printers are routinely off by 1-3%, which at our
0.6-0.8 m working distance is millimetres of systematic error.

Mount flat. A curled sheet is a curved plane, and ray-plane assumes it is not.

    python3 make_aruco_targets.py --out aruco_targets/
"""

import argparse
import os

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50   # 4x4 is plenty for <50 ids and stays readable small


def _mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def make_markers(out_dir, ids, side_mm, dpi, margin_mm=8.0):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    side_px = _mm_to_px(side_mm, dpi)
    marg_px = _mm_to_px(margin_mm, dpi)
    paths = []
    for i in ids:
        img = cv2.aruco.generateImageMarker(d, i, side_px)
        # quiet zone: ArUco needs white border to detect reliably
        canvas = np.full((side_px + 2 * marg_px, side_px + 2 * marg_px), 255, np.uint8)
        canvas[marg_px:marg_px + side_px, marg_px:marg_px + side_px] = img
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        label = f"id={i}  {side_mm:.0f}mm  DICT_4X4_50"
        cv2.putText(canvas, label, (marg_px // 2, canvas.shape[0] - marg_px // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, dpi / 900.0, (0, 0, 0), max(1, dpi // 300))
        pth = os.path.join(out_dir, f"aruco_id{i:02d}_{side_mm:.0f}mm.png")
        cv2.imwrite(pth, canvas)
        paths.append(pth)
    return paths


def make_charuco(out_dir, squares_x, squares_y, square_mm, marker_mm, dpi):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y),
                                   square_mm / 1000.0, marker_mm / 1000.0, d)
    w = _mm_to_px(squares_x * square_mm, dpi)
    h = _mm_to_px(squares_y * square_mm, dpi)
    img = board.generateImage((w, h), marginSize=_mm_to_px(10, dpi))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    txt = (f"ChArUco {squares_x}x{squares_y}  square={square_mm:.0f}mm  "
           f"marker={marker_mm:.0f}mm  DICT_4X4_50  PRINT AT 100%")
    cv2.putText(img, txt, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                dpi / 1100.0, (0, 0, 0), max(1, dpi // 300))
    pth = os.path.join(out_dir, f"charuco_{squares_x}x{squares_y}_{square_mm:.0f}mm.png")
    cv2.imwrite(pth, img)
    return pth, board


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="aruco_targets")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--marker_mm", type=float, default=80.0)
    ap.add_argument("--n_markers", type=int, default=6)
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_mm", type=float, default=35.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = list(range(args.n_markers))
    mk = make_markers(args.out, ids, args.marker_mm, args.dpi)
    ch, board = make_charuco(args.out, args.squares_x, args.squares_y,
                             args.square_mm, args.square_mm * 0.75, args.dpi)

    # Detectability check against the ACTUAL camera intrinsics, so we size the
    # print for this camera and this working distance rather than by eye.
    fx_1080 = 1366.19
    print(f"\nwrote {len(mk)} markers + 1 ChArUco board to {args.out}/")
    print(f"  ChArUco: {args.squares_x}x{args.squares_y}, {args.square_mm:.0f} mm squares "
          f"-> {args.squares_x*args.square_mm:.0f} x {args.squares_y*args.square_mm:.0f} mm "
          f"({'fits A4' if args.squares_x*args.square_mm<=190 and args.squares_y*args.square_mm<=277 else 'NEEDS A3'})")
    print(f"\napparent size at 1920x1080 (fx={fx_1080:.0f}) for a "
          f"{args.marker_mm:.0f} mm marker:")
    for dist in (0.8, 1.2, 1.6, 2.0):
        px = args.marker_mm / 1000.0 * fx_1080 / dist
        verdict = "good" if px >= 60 else ("marginal" if px >= 35 else "TOO SMALL")
        print(f"    {dist:.1f} m -> {px:5.1f} px   {verdict}")
    print("\nPRINT AT 100% (not fit-to-page), mount FLAT, then MEASURE a printed")
    print("edge and pass --marker_mm / --square_mm with the measured value.")


if __name__ == "__main__":
    main()
