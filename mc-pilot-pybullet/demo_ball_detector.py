"""
Two-shot demo of perception/ball_detector.py against the live D435i.

Run once with the landing zone EMPTY (captures and caches "before"). Run
again with the ball placed (captures "after", runs detect_ball_bgsub,
draws the result, saves an annotated JPEG).

    python3 demo_ball_detector.py --before   # empty scene
    python3 demo_ball_detector.py --after    # ball placed
"""
import argparse
import pickle

import cv2
import numpy as np
import pyrealsense2 as rs

from perception.ball_detector import detect_ball_bgsub

CACHE = "/tmp/ball_detector_demo_before.pkl"


def _capture(width, height, settle=40):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(settle):
            pipe.wait_for_frames(2000)
        img = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data())
    finally:
        pipe.stop()
    return img


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--before", action="store_true")
    g.add_argument("--after", action="store_true")
    ap.add_argument("--out", default="/tmp/ball_detector_annotated.jpg")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    frame = _capture(args.width, args.height)

    if args.before:
        with open(CACHE, "wb") as f:
            pickle.dump(frame, f)
        print(f"cached 'before' frame ({frame.shape[1]}x{frame.shape[0]}) -> {CACHE}")
        print("now place the ball, then re-run with --after")
        return

    with open(CACHE, "rb") as f:
        before = pickle.load(f)
    if before.shape != frame.shape:
        raise SystemExit(f"cached before frame is {before.shape}, this capture is "
                         f"{frame.shape} -- re-run --before first")

    u, v, diag = detect_ball_bgsub(before, frame)
    print(f"detected ball at pixel ({u:.1f}, {v:.1f})")
    print(f"diagnostics: {diag}")

    annotated = frame.copy()
    x, y, w, h = diag["bbox"]
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawMarker(annotated, (int(round(u)), int(round(v))), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
    cv2.putText(annotated, f"({u:.0f},{v:.0f}) area={diag['area_px']:.0f}px",
               (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(args.out, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
