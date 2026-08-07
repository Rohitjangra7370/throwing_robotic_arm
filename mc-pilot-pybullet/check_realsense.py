"""
RealSense bench check — everything that does NOT depend on where the camera is mounted.

The camera is the ONLY exteroceptive sensor for this project, so its limits set
what the vision half of the experiment can and cannot claim. Those limits are
properties of the device and the link, not of the tripod, so they can all be
settled before it is bolted anywhere.

What this answers:
  1. Which device, and is it on a USB3 link? On USB2 the high-rate modes silently
     disappear and you find out mid-experiment.
  2. Which stream profiles actually exist on THIS unit -- enumerated, not quoted
     from a datasheet.
  3. Colour intrinsics (K + distortion), which is what ray-plane needs. Depth is
     never the position source here: ~2% of range is 2-4 cm at 1-2 m, the same
     size as the landing error being measured.
  4. Achieved frame rate and timestamp jitter, measured.
  5. Exposure range -- motion blur, not frame rate, is what usually destroys a
     fast-moving ball.

What it CANNOT answer until the camera is mounted and calibrated: T_B_C,
coverage of the landing zone, occlusion, and the <1 cm ray-plane accuracy gate.

    python3 check_realsense.py
    python3 check_realsense.py --seconds 5 --profile color
"""

import argparse
import sys

import numpy as np

OK, WARN, FAIL = "  OK  ", " CHECK", " FAIL "
_results = []


def _record(level, msg):
    _results.append((level, msg))
    print(f"[{level}] {msg}")


def section(t):
    print(f"\n--- {t} " + "-" * max(0, 62 - len(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--profile", choices=["color", "infrared", "depth"], default="color")
    args = ap.parse_args()

    import pyrealsense2 as rs

    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        _record(FAIL, "no RealSense device found")
        return 2
    dev = devs[0]

    # ------------------------------------------------------------------ #
    section("1. identity and link")
    def info(k):
        try:
            return dev.get_info(getattr(rs.camera_info, k))
        except Exception:
            return None
    for k in ("name", "serial_number", "firmware_version", "product_id",
              "product_line", "usb_type_descriptor", "physical_port"):
        v = info(k)
        if v is not None:
            _record(OK, f"{k}: {v}")
    usb = info("usb_type_descriptor") or "?"
    if usb.startswith("3"):
        _record(OK, f"USB {usb} -- full-rate modes available")
    else:
        _record(FAIL, f"USB {usb} -- NOT USB3. High-resolution / high-fps modes "
                      "will be missing or throttled. Reseat on a USB3 port "
                      "(blue) with the supplied cable before trusting anything below.")

    # ------------------------------------------------------------------ #
    section("2. stream profiles on THIS unit")
    by_stream = {}
    for s in dev.query_sensors():
        sname = s.get_info(rs.camera_info.name)
        for p in s.get_stream_profiles():
            if not p.is_video_stream_profile():
                by_stream.setdefault(("motion", str(p.stream_type())), set()).add(p.fps())
                continue
            v = p.as_video_stream_profile()
            key = (sname, str(p.stream_type()).split(".")[-1], p.format())
            by_stream.setdefault(key, set()).add((v.width(), v.height(), p.fps()))
    for key in sorted(by_stream, key=lambda k: str(k)):
        vals = by_stream[key]
        if key[0] == "motion":
            _record(OK, f"{key[1]}: {sorted(vals)} Hz")
            continue
        sensor, stream, fmt = key
        best_fps = max(v[2] for v in vals)
        top = sorted({v for v in vals if v[2] == best_fps}, reverse=True)[:2]
        biggest = sorted(vals, reverse=True)[0]
        _record(OK, f"{stream:9s} {str(fmt):12s} max-res {biggest[0]}x{biggest[1]}@{biggest[2]} "
                    f"| max-fps {best_fps} at {['%dx%d' % (w,h) for w,h,_ in top]}")

    # ------------------------------------------------------------------ #
    section("3. streaming, intrinsics, achieved rate")
    # Pick the best mode this unit ACTUALLY offers on the current link, rather
    # than a datasheet mode. On a USB2 link the high-rate modes are simply
    # absent, and hard-coding one turns a link problem into "Couldn't resolve
    # requests", which reads like a code bug.
    stream, fmt = {"color": (rs.stream.color, rs.format.bgr8),
                   "infrared": (rs.stream.infrared, rs.format.y8),
                   "depth": (rs.stream.depth, rs.format.z16)}[args.profile]
    avail = set()
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            if p.is_video_stream_profile() and p.stream_type() == stream and p.format() == fmt:
                v = p.as_video_stream_profile()
                avail.add((v.width(), v.height(), p.fps()))
    if not avail:
        _record(FAIL, f"no {args.profile} profiles in {fmt}")
        return 2
    # prefer >=720p if it can run at >=15 fps, else the highest-fps decent mode
    good = [m for m in avail if m[1] >= 720 and m[2] >= 15] or \
           [m for m in avail if m[2] >= 30] or list(avail)
    want = (stream,) + max(good, key=lambda m: (m[1], m[2])) + (fmt,)
    w, h, fps = want[1], want[2], want[3]
    _record(OK, f"selected best available {args.profile}: {w}x{h}@{fps}")

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(stream, w, h, fmt, fps)
    want = (stream, w, h, fmt, fps)
    try:
        prof = pipe.start(cfg)
    except Exception as e:
        _record(FAIL, f"could not start {args.profile} {w}x{h}@{fps}: {e}")
        return 2
    try:
        sp = prof.get_stream(want[0]).as_video_stream_profile()
        K = sp.get_intrinsics()
        _record(OK, f"{args.profile} {K.width}x{K.height}: fx={K.fx:.2f} fy={K.fy:.2f} "
                    f"ppx={K.ppx:.2f} ppy={K.ppy:.2f}")
        _record(OK, f"distortion {K.model}: {[round(c, 5) for c in K.coeffs]}")
        hfov = 2 * np.degrees(np.arctan(K.width / (2 * K.fx)))
        vfov = 2 * np.degrees(np.arctan(K.height / (2 * K.fy)))
        _record(OK, f"derived FOV: {hfov:.1f} deg H x {vfov:.1f} deg V")
        # what that means for the 0.90 m landing zone
        for h in (1.8, 2.0, 2.2):
            cov = 2 * h * np.tan(np.radians(hfov / 2))
            _record(OK, f"  overhead at {h:.1f} m -> covers {cov:.2f} m wide, "
                        f"{cov / K.width * 1e3:.2f} mm/px")

        dsensor = prof.get_device().first_depth_sensor() if args.profile != "color" else None
        if dsensor is not None:
            _record(OK, f"depth scale: {dsensor.get_depth_scale():.6f} m/unit")

        # exposure range: motion blur is the real limit on a moving ball
        for s in dev.query_sensors():
            try:
                if s.supports(rs.option.exposure):
                    r = s.get_option_range(rs.option.exposure)
                    nm = s.get_info(rs.camera_info.name)
                    _record(OK, f"{nm}: exposure {r.min:.0f}..{r.max:.0f} (step {r.step:.0f}), "
                                f"currently {s.get_option(rs.option.exposure):.0f}")
            except Exception:
                pass

        # measured rate + timestamp jitter
        import time
        ts, wall = [], []
        t0 = time.perf_counter()
        n_bad = 0
        while time.perf_counter() - t0 < args.seconds:
            try:
                f = pipe.wait_for_frames(2000)
            except Exception:
                n_bad += 1
                continue
            fr = f.get_color_frame() if args.profile == "color" else f.get_infrared_frame() \
                if args.profile == "infrared" else f.get_depth_frame()
            if not fr:
                n_bad += 1
                continue
            ts.append(fr.get_timestamp())
            wall.append(time.perf_counter() - t0)
        if len(ts) < 5:
            _record(FAIL, f"only {len(ts)} frames in {args.seconds}s -- stream is not healthy")
        else:
            wall = np.array(wall)
            gaps = np.diff(np.array(ts))
            # Rate from DEVICE timestamps, first frame to last. Wall-clock over
            # the whole window charges pipeline warm-up (~0.4 s) to the stream
            # and reported a healthy 30 fps feed as 27 fps -- a false alarm on
            # exactly the check meant to catch a starved link.
            rate = 1000.0 * (len(ts) - 1) / (ts[-1] - ts[0])
            _record(OK, f"achieved {rate:.1f} fps over {len(ts)} frames "
                        f"(requested {want[4]}; warm-up excluded)")
            _record(OK, f"frame-interval (device ts): mean {gaps.mean():.2f} ms  "
                        f"p99 {np.percentile(gaps, 99):.2f}  max {gaps.max():.2f}")
            if rate < 0.9 * want[4]:
                _record(WARN, f"achieved rate is >10% below requested -- USB bandwidth "
                              f"or exposure is limiting it")
            if n_bad:
                _record(WARN, f"{n_bad} dropped/incomplete frame waits")
            # what this rate means for a 1.5 m/s ball
            _record(OK, f"at {rate:.0f} fps a 1.5 m/s ball moves "
                        f"{1500 / rate:.1f} mm between frames")
    finally:
        pipe.stop()

    section("summary")
    nf = sum(1 for l, _ in _results if l == FAIL)
    nw = sum(1 for l, _ in _results if l == WARN)
    print(f"{len(_results)} checks: {nf} FAIL, {nw} CHECK, {len(_results)-nf-nw} OK")
    for l, m in _results:
        if l == FAIL:
            print("  FAIL:", m)
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
