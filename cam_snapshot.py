"""
Grab one frame from the D435i, report what is in it, and annotate detections.

An aiming and sanity tool, not a calibration step. Point the camera, run this,
look at the JPEG, adjust, repeat. It answers the three things that actually
block calibration, in order:

  1. Is the camera seeing the right part of the room at all?
  2. Are the printed markers detected, and how big are they in pixels?
     (<35 px edge is unusable, 35-60 marginal, >60 robust.)
  3. Is the exposure good enough? A dim or blown frame kills marker detection
     long before geometry does, and auto-exposure on a mostly-white scene
     (whiteboard, paper targets) tends to underexpose the black squares.

    python3 cam_snapshot.py                     # 1920x1080 colour
    python3 cam_snapshot.py --out /tmp/aim.jpg  # iterate while aiming
"""

import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/cam_view.jpg")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--settle", type=int, default=40, help="frames to let AE settle")
    ap.add_argument("--exposure", type=float, default=None,
                    help="fix RGB exposure (units of 100 us); omit for auto")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    try:
        if args.exposure is not None:
            for s in prof.get_device().query_sensors():
                if s.supports(rs.option.exposure) and "RGB" in s.get_info(rs.camera_info.name):
                    s.set_option(rs.option.enable_auto_exposure, 0)
                    s.set_option(rs.option.exposure, args.exposure)
        for _ in range(args.settle):
            pipe.wait_for_frames(2000)
        img = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data())
    finally:
        pipe.stop()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, rejected = det.detectMarkers(gray)

    print(f"frame {img.shape[1]}x{img.shape[0]}")
    print(f"exposure: brightness mean {gray.mean():.0f}/255, "
          f"p1 {np.percentile(gray,1):.0f}, p99 {np.percentile(gray,99):.0f} "
          f"({'DARK - detection will suffer' if gray.mean() < 70 else 'ok'})")
    if ids is None or not len(ids):
        print(f"markers: NONE detected  ({len(rejected)} rejected candidates)")
    else:
        print(f"markers: {sorted(ids.flatten().tolist())}")
        for c, i in zip(corners, ids.flatten()):
            e = float(np.mean([np.linalg.norm(c[0][k] - c[0][(k + 1) % 4])
                               for k in range(4)]))
            ctr = c[0].mean(axis=0)
            verdict = "robust" if e >= 60 else ("marginal" if e >= 35 else "TOO SMALL")
            print(f"   id={i:2d}  edge {e:6.1f} px ({verdict})  "
                  f"centre ({ctr[0]:.0f}, {ctr[1]:.0f})")
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    cv2.imwrite(args.out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
