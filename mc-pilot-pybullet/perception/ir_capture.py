"""
Record both D435i infrared imagers through a throw.

WHY BOTH IR STREAMS AND NOT COLOUR OR DEPTH
--------------------------------------------
Both IR imagers are global shutter (OV9282); the colour imager (OV2740) is
rolling shutter, which skews a fast ball. The IR field of view is also much
wider (89.7 x 58.8 deg vs 70.2 x 43.2), and that is what puts the flight in
frame from an overhead mount at all. The DEPTH stream is deliberately not
enabled: this project does not use the block-matching depth map as a position
source (see perception/stereo.py).

WHY RECORD-THEN-DUMP
--------------------
848x480 y8 is 407 kB per image, x2 cameras x 90 fps = 73 MB/s. A 2 s window is
146 MB, which sits in RAM comfortably. Writing during capture risks a disk stall
dropping frames in the middle of the flight, and there is no reason to accept
that when the whole window fits in memory.

TIMESTAMPS
----------
Frames are timestamped at MID-EXPOSURE, from the sensor timestamp, not at
arrival. At 5.8 m/s an 8.5 ms exposure is 5 cm of travel, so the choice of time
reference is a systematic error, not noise -- and it must agree with
ball_track.detect_candidates, whose intensity-weighted centroid is unbiased at
mid-exposure.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pyrealsense2 as rs

__all__ = ["IRRecorder", "save_recording", "load_recording"]


class IRRecorder:
    """Dual-IR capture into RAM. One throw per `record()` call."""

    def __init__(self, width=848, height=480, fps=90, exposure_us=2000,
                 gain=None, emitter=True):
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.exposure_us = int(exposure_us)
        self.gain = gain
        self.emitter = bool(emitter)
        self._pipe = None

    def __enter__(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                          rs.format.y8, self.fps)
        cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height,
                          rs.format.y8, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        sensor = profile.get_device().first_depth_sensor()
        # Manual exposure. Auto-exposure will happily pick 8.5 ms, which is 5 cm
        # of motion blur at impact speed, and will also change between frames --
        # both are fatal to a subpixel centroid.
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(self.exposure_us))
        if self.gain is not None:
            sensor.set_option(rs.option.gain, float(self.gain))
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1 if self.emitter else 0)
        self._profile = profile
        return self

    def __exit__(self, *exc):
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    def stream(self):
        """
        Yield `(timestamp, ir1, ir2)` once per captured frame pair, forever --
        until the caller stops iterating (e.g. breaks out of a `for`) or the
        pipeline is torn down.

        `timestamp` is the same mid-exposure, seconds, wall-clock expression
        `record()` has always used, NOT `time.time()`. It is left un-zeroed
        here on purpose: zeroing is each consumer's job at its own boundary --
        `record()` below does its own `ts - ts[0]`, `session_camera.RingBuffer
        .window()` does its own at the window edge. Verified on this camera
        2026-08-31: `get_timestamp()*1e-3` sat 11 ms from `time.time()`, so
        this value is directly comparable to a `time.time()` release instant.
        """
        if self._pipe is None:
            raise RuntimeError("use IRRecorder as a context manager")
        half_exp_s = 0.5 * self.exposure_us * 1e-6
        while True:
            fs = self._pipe.wait_for_frames(2000)
            f1 = fs.get_infrared_frame(1)
            f2 = fs.get_infrared_frame(2)
            if not f1 or not f2:
                continue
            ir1 = np.asanyarray(f1.get_data())
            ir2 = np.asanyarray(f2.get_data())
            ts = f1.get_timestamp() * 1e-3 + half_exp_s   # ms -> s, mid-exposure
            yield ts, ir1, ir2

    def record(self, seconds):
        """Capture for `seconds`, return the recording dict."""
        if self._pipe is None:
            raise RuntimeError("use IRRecorder as a context manager")
        n_expect = int(np.ceil(seconds * self.fps)) + 8
        ir1 = np.empty((n_expect, self.height, self.width), np.uint8)
        ir2 = np.empty((n_expect, self.height, self.width), np.uint8)
        ts = np.empty(n_expect, float)

        # Manual iteration (not `for ... in self.stream()`) so the time/count
        # budget is checked BEFORE blocking on the next frame, exactly as the
        # single `while` loop this replaced did -- a `for` loop would pull one
        # extra frame from the generator ahead of the break check.
        k, t_end = 0, time.monotonic() + float(seconds)
        frames = self.stream()
        while time.monotonic() < t_end and k < n_expect:
            frame_ts, f1, f2 = next(frames)
            ir1[k] = f1
            ir2[k] = f2
            ts[k] = frame_ts
            k += 1

        if k == 0:
            raise RuntimeError("captured zero frames -- is the camera streaming?")
        ts = ts[:k] - ts[0]
        meta = {"width": self.width, "height": self.height, "fps": self.fps,
                "exposure_us": self.exposure_us, "gain": self.gain,
                "emitter": self.emitter, "n_frames": int(k),
                "achieved_fps": float((k - 1) / max(ts[-1], 1e-9)) if k > 1 else 0.0}
        return {"t": ts, "ir1": ir1[:k], "ir2": ir2[:k], "meta": meta}


def save_recording(path, rec):
    """
    Uncompressed .npz on purpose -- savez_compressed spends ~30 s on 146 MB of
    uint8 for a modest saving, and the point of dumping after the throw is to be
    back to ready quickly.
    """
    np.savez(path, t=rec["t"], ir1=rec["ir1"], ir2=rec["ir2"],
             meta=json.dumps(rec["meta"]))


def load_recording(path):
    z = np.load(path, allow_pickle=False)
    return {"t": z["t"], "ir1": z["ir1"], "ir2": z["ir2"],
            "meta": json.loads(str(z["meta"]))}
