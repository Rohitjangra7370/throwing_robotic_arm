"""Live D435i feed that survives USB disconnects, with marker overlay for aiming.

Why this exists rather than realsense-viewer: aiming the camera is an iterative physical
task, and the two things you need while doing it -- "is the marker detected" and "how many
pixels is its edge" -- are exactly what decides whether calibration will work later. A
marker under ~35 px edge is unusable, 35-60 marginal, >60 robust.

SURVIVING DISCONNECTS. A USB3 camera on a long cable drops. When it does, librealsense
does not politely return -- frame waits time out, or the pipeline throws, and a naive
loop either hangs forever or dies. This does neither:

  * frames are pulled with try_wait_for_frames(), which returns False on timeout instead
    of raising, so a stall is a value to handle rather than an exception to catch;
  * after CONSECUTIVE_FAIL_LIMIT stalls the pipeline is torn down and the device is
    re-enumerated from a fresh context -- a stale context keeps handing back the device
    that just vanished;
  * reconnection retries forever with a bounded backoff, so unplugging the camera and
    plugging it back in resumes the feed instead of ending the session;
  * every reconnect is counted and shown on the overlay, because a feed that silently
    reconnects every few seconds looks fine and is telling you the cable is bad.

NOTE: USB access does not work from a sandboxed shell -- the device enumerates as zero
devices with no error. Run this outside the sandbox.

    python3 cam_feed.py                    # live window, 1280x720
    python3 cam_feed.py --res 1920 1080    # full res (intrinsics were measured here)
    python3 cam_feed.py --headless         # no window; writes --snapshot periodically

    q / ESC  quit        s  save a snapshot        f  toggle marker detection
"""
import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

CONSECUTIVE_FAIL_LIMIT = 5      # stalled frame waits before declaring the link dead
FRAME_TIMEOUT_MS = 1500
BACKOFF_START_S = 0.5
BACKOFF_MAX_S = 5.0

MARKER_MM = 80.0                # printed edge length of the loose ArUco targets


class ResilientCamera:
    """A colour stream that reconnects on its own."""

    def __init__(self, width, height, fps):
        self.w, self.h, self.fps = width, height, fps
        self.pipe = None
        self.reconnects = -1        # first connect increments to 0
        self.intrinsics = None

    def _teardown(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass            # already gone; nothing to salvage
            self.pipe = None

    @staticmethod
    def _supported_color(dev):
        out = set()
        for s in dev.query_sensors():
            for p in s.get_stream_profiles():
                v = p.as_video_stream_profile()
                if v and p.stream_type() == rs.stream.color \
                   and v.format() == rs.format.bgr8:
                    out.add((v.width(), v.height(), p.fps()))
        return out

    def _negotiate(self, dev):
        """Pick a profile this LINK actually offers.

        A long or slow cable negotiates USB2, on which whole modes vanish -- this
        camera offers 1280x720@30 on USB3 and not at all on USB2, where the same
        resolution caps at 15 fps. librealsense reports that as 'Couldn't resolve
        requests', which is NOT a transient condition: retrying the same request
        forever will never succeed. Distinguishing an unsatisfiable request from a
        missing device is the difference between adapting and hanging.
        """
        avail = self._supported_color(dev)
        if not avail:
            raise RuntimeError("device exposes no BGR8 colour profile")
        if (self.w, self.h, self.fps) in avail:
            return self.w, self.h, self.fps
        same_res = [c for c in avail if (c[0], c[1]) == (self.w, self.h)]
        if same_res:                      # keep resolution, drop frame rate
            best = max(same_res, key=lambda c: c[2])
        else:                             # keep the most pixels, then the most fps
            best = max(avail, key=lambda c: (c[0] * c[1], c[2]))
        print(f"[cam] {self.w}x{self.h}@{self.fps} unavailable on this link; "
              f"using {best[0]}x{best[1]}@{best[2]}", flush=True)
        return best

    def connect(self, quiet=False):
        """Block until a device is streaming. Retries forever with backoff."""
        self._teardown()
        self.reconnects += 1
        backoff = BACKOFF_START_S
        while True:
            try:
                # A fresh context each attempt: a stale one keeps returning the device
                # that just disappeared, and start() then fails on a dead handle.
                ctx = rs.context()
                devs = list(ctx.query_devices())
                if len(devs) == 0:
                    raise RuntimeError("no RealSense device enumerated")
                w, h, fps = self._negotiate(devs[0])
                pipe = rs.pipeline(ctx)
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                prof = pipe.start(cfg)
                self.w, self.h, self.fps = w, h, fps
                self.pipe = pipe
                self.intrinsics = (prof.get_stream(rs.stream.color)
                                   .as_video_stream_profile().get_intrinsics())
                dev = prof.get_device()
                name = dev.get_info(rs.camera_info.name)
                try:
                    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                except Exception:
                    usb = "?"
                if not quiet:
                    print(f"[cam] connected: {name}  USB {usb}  "
                          f"{self.w}x{self.h}@{self.fps}  "
                          f"fx={self.intrinsics.fx:.1f}  (reconnect #{self.reconnects})",
                          flush=True)
                    if str(usb).startswith("2"):
                        print("[cam] NOTE: USB2 link. Frame rate is capped, resolution "
                              "is not. This pipeline measures the ball AT REST, so "
                              "resolution is what matters and 1080p is still "
                              "available -- the link is adequate. In-flight tracking "
                              "would need USB3.", flush=True)
                    if (self.w, self.h) != (1920, 1080):
                        print(f"[cam] NOTE: intrinsics on record (fx=1366.19) are for "
                              f"1920x1080; at {self.w}x{self.h} they do not apply -- "
                              f"re-measure before using this stream for geometry.",
                              flush=True)
                return
            except Exception as e:
                print(f"[cam] waiting for device ({e}); retry in {backoff:.1f}s",
                      flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    def read(self):
        """Return a BGR frame, or None if the link stalled (caller should not exit)."""
        if self.pipe is None:
            self.connect()
        try:
            ok, frames = self.pipe.try_wait_for_frames(FRAME_TIMEOUT_MS)
        except Exception as e:
            print(f"[cam] stream error: {e}", flush=True)
            return None
        if not ok:
            return None
        c = frames.get_color_frame()
        if not c:
            return None
        return np.asanyarray(c.get_data())


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:                                  # OpenCV >= 4.7
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    except AttributeError:                # older API
        return None


def detect_and_draw(img, detector):
    """Outline markers and report the worst-case edge length in pixels."""
    if detector is None:
        return img, []
    corners, ids, _ = detector.detectMarkers(img)
    info = []
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        for c, i in zip(corners, ids.flatten()):
            p = c.reshape(4, 2)
            edge = float(np.mean([np.linalg.norm(p[k] - p[(k + 1) % 4])
                                  for k in range(4)]))
            info.append((int(i), edge))
            ctr = p.mean(axis=0).astype(int)
            quality = "OK" if edge > 60 else ("MARGINAL" if edge > 35 else "TOO SMALL")
            cv2.putText(img, f"id{int(i)} {edge:.0f}px {quality}",
                        (ctr[0] - 60, ctr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if edge > 60 else (0, 165, 255), 2)
    return img, info


def overlay(img, fps, cam, info, detecting):
    lines = [
        f"{img.shape[1]}x{img.shape[0]}  {fps:5.1f} fps",
        f"reconnects: {cam.reconnects}",
        f"markers: {len(info)}" + ("" if detecting else "  (detection off)"),
    ]
    if info:
        worst = min(e for _, e in info)
        # mm/px on the marker's own plane -- a direct read of achievable precision
        lines.append(f"worst edge {worst:.0f}px  ->  {MARKER_MM/worst:.2f} mm/px")
    y = 26
    for t in lines:
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 26
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--headless", action="store_true",
                    help="no window; periodically write --snapshot instead")
    ap.add_argument("--snapshot", default="/tmp/cam_latest.jpg")
    ap.add_argument("--snapshot_every", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until quit")
    args = ap.parse_args()

    cam = ResilientCamera(args.res[0], args.res[1], args.fps)
    cam.connect()

    detector, detecting = make_detector(), True
    if detector is None:
        print("[cam] this OpenCV lacks the ArucoDetector API; overlay disabled",
              flush=True)

    win = "D435i feed  [q]uit  [s]nap  [f]detect"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    t_start, last_snap = time.time(), 0.0
    fps, t_prev, stalls, frames_total = 0.0, time.time(), 0, 0

    while True:
        frame = cam.read()
        if frame is None:
            stalls += 1
            print(f"[cam] frame stall {stalls}/{CONSECUTIVE_FAIL_LIMIT}", flush=True)
            if stalls >= CONSECUTIVE_FAIL_LIMIT:
                # Reconnecting forever is right for a cable that comes and goes, and
                # wrong for one that never carried data. If the device enumerates,
                # negotiates a profile and reports intrinsics but has NEVER delivered a
                # single frame across several reconnects, then control transfers work
                # and streaming does not -- a physical-layer fault (charge-only or
                # marginal cable, voltage drop under streaming current, passive
                # extension past the USB2 5 m limit). No retry and no resolution
                # change fixes that, so say so and stop instead of looping quietly.
                if frames_total == 0 and cam.reconnects >= 3:
                    print("\n[cam] DIAGNOSIS: the camera enumerates and negotiates but "
                          "has never delivered a frame.\n"
                          "  Control transfers work; the data path does not. This is a "
                          "cable/link fault,\n"
                          "  not bandwidth -- it reproduces at the lowest mode on "
                          "offer. Try a different\n"
                          "  (short, data-rated) cable or port before changing any "
                          "setting.", flush=True)
                    break
                print("[cam] link considered down -- reconnecting", flush=True)
                cam.connect()
                stalls = 0
            continue
        stalls = 0
        frames_total += 1

        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        info = []
        if detecting:
            frame, info = detect_and_draw(frame, detector)
        frame = overlay(frame, fps, cam, info, detecting)

        if args.headless:
            if now - last_snap >= args.snapshot_every:
                tmp = args.snapshot + ".tmp.jpg"     # atomic: never serve a half file
                cv2.imwrite(tmp, frame)
                os.replace(tmp, args.snapshot)
                last_snap = now
        else:
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                path = f"/tmp/cam_snap_{int(now)}.jpg"
                cv2.imwrite(path, frame)
                print(f"[cam] saved {path}", flush=True)
            if k == ord("f"):
                detecting = not detecting

        if args.seconds and (now - t_start) > args.seconds:
            break

    cam._teardown()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"[cam] done. reconnects: {cam.reconnects}", flush=True)


if __name__ == "__main__":
    main()
