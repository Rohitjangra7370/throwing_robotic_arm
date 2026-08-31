"""
The session's camera thread: one owner of the D435i for the whole session.

WHY THE SESSION OWNS THE CAMERA
--------------------------------
The D435i can be opened by exactly one process, which is why
`throw_capture.py` and `run_closed_loop_throws.py` are decoupled through files
today. Owning it here buys correctness, not tidiness: the session ISSUES the
throw, so it knows the release instant exactly and records a window around it,
instead of inferring "that blob was probably a throw" from a disparity gate.
"""
from __future__ import annotations

import collections
import queue
import threading
import time

import numpy as np

PRE_S = 0.45          # kept before release
POST_S = 1.00         # kept after


class RingBuffer:
    """Fixed-duration dual-IR history. Not thread-safe; the owner locks."""

    def __init__(self, seconds=3.0, fps=90):
        self.capacity = max(1, int(round(seconds * fps)))
        self._ts = collections.deque(maxlen=self.capacity)
        self._ir1 = collections.deque(maxlen=self.capacity)
        self._ir2 = collections.deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._ts)

    def append(self, ts, ir1, ir2):
        # Copy, don't retain the frame as handed in. `IRRecorder.stream()`
        # yields a zero-copy numpy view over the RealSense SDK's own frame
        # buffer (np.asanyarray(f1.get_data())) -- fine for record(), which
        # copies each frame out (`ir1[k] = ...`) and drops the reference every
        # iteration, but fatal here: this buffer holds up to `capacity`
        # (~270 at the 3 s/90 fps default) frames *simultaneously*, which
        # vastly exceeds the SDK's internal frame pool. Verified on hardware
        # 2026-08-31: without the copy, streaming stalls and `wait_for_frames`
        # starts timing out after exactly 16 live frames, every time,
        # regardless of whether the caller's own lock is held; with the copy,
        # the identical loop holds 250+ frames with zero errors.
        self._ts.append(float(ts))
        self._ir1.append(ir1.copy())
        self._ir2.append(ir2.copy())

    def window(self, t_start, t_end):
        """
        Inclusive [t_start, t_end] slice, in exactly the shape
        `measure_landing.build_observations` consumes.

        Two conventions are copied from `IRRecorder.record()` rather than
        invented, because build_observations reads its output directly:
        the key is **`t`** (not `ts`), and `t` is zeroed to the window's first
        frame (record() does `ts = ts[:k] - ts[0]`). Getting either wrong is a
        KeyError or a silently shifted time origin.

        Slicing is by the frames' own timestamps, which on this camera are in
        the wall-clock domain -- measured 2026-08-31, `get_timestamp()*1e-3`
        sat 11 ms from `time.time()` -- so a `time.time()` release instant can
        be compared against them directly.
        """
        ts = np.asarray(self._ts, float)
        if ts.size == 0:
            raise ValueError("no frames: buffer is empty")
        keep = np.nonzero((ts >= t_start - 1e-9) & (ts <= t_end + 1e-9))[0]
        if keep.size == 0:
            raise ValueError(
                f"no frames in [{t_start:.3f}, {t_end:.3f}] -- buffer holds "
                f"{len(self)} frames spanning [{ts[0]:.3f}, {ts[-1]:.3f}]")
        ir1 = np.stack([self._ir1[i] for i in keep])
        ir2 = np.stack([self._ir2[i] for i in keep])
        t = ts[keep] - ts[keep][0]
        return {"t": t, "ir1": ir1, "ir2": ir2}


class CameraThread:
    """
    Continuous dual-IR capture into a ring buffer, plus a single-slot handoff of
    the newest frame for display.

    Single-slot is deliberate: a slow consumer drops frames rather than
    stalling capture, because a stalled capture loses the throw.

    FRAME OWNERSHIP CONTRACT
    -------------------------
    `IRRecorder.stream()` yields zero-copy numpy views over the RealSense
    SDK's own frame buffer -- cheap, but only valid for as long as the SDK
    frame itself is alive. Anything that outlives one capture-loop iteration
    must hold an owned copy, not the view:
      - `buf` (the `RingBuffer`) copies internally in `RingBuffer.append()`,
        because it retains up to `capacity` frames simultaneously -- holding
        that many live SDK frames at once exhausts the SDK's internal frame
        pool and stalls capture (found and fixed on real hardware
        2026-08-31; see `RingBuffer.append`'s docstring/comment).
      - `latest()` also copies, for the same reason at a smaller scale: it is
        read asynchronously by a display consumer with no guarantee about
        when, including after `stop()`, so the one slot it holds must not be
        a live SDK reference either.
      - `on_frame`, by contrast, is called SYNCHRONOUSLY, inline, once per
        captured frame, at capture rate (up to ~90 Hz) -- so it is handed the
        raw zero-copy view on purpose, to avoid paying a copy on every frame
        for callbacks that don't need one (e.g. blit-and-discard display).
        **The array is only valid for the duration of that call. A callback
        that wants to keep it past return must call `.copy()` itself.**
    """

    def __init__(self, seconds=3.0, fps=90, width=848, height=480,
                 exposure_us=2000, emitter=True, on_frame=None):
        """
        `on_frame(ts, ir1, ir2)`, if given, is called inline from the capture
        thread once per frame with the RAW zero-copy arrays -- see the class
        docstring's "FRAME OWNERSHIP CONTRACT". Copy inside the callback if
        you need to keep the data past the call.
        """
        self.buf = RingBuffer(seconds, fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._events = queue.Queue()
        self._latest = None
        self._cfg = dict(width=width, height=height, fps=fps,
                         exposure_us=exposure_us, emitter=emitter)
        self._on_frame = on_frame
        self._thread = None
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Signal the capture loop to end and wait up to 5 s for it to actually
        exit, then return True iff the camera is confirmed free (the worker
        thread is no longer alive) -- False if it is still running after the
        timeout.

        The D435i can be opened by exactly one process (see the module
        docstring), so a caller that calls `stop()` and then assumes it may
        reopen the camera without checking the return value has no positive
        confirmation of that -- `Thread.join(timeout=...)` returns silently
        on a timeout either way. Normal-path shutdown is well under 5 s, so
        False should be rare, but it is exactly the case worth not silently
        swallowing.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            return not self._thread.is_alive()
        return True

    def latest(self):
        with self._lock:
            return self._latest

    def mark_release(self, t_release, pre=PRE_S, post=POST_S):
        """
        Called by the worker thread the instant the throw fires. The window is
        cut once `post` seconds of frames past the release have actually
        arrived, then pushed to `pop_event`.
        """
        threading.Thread(target=self._cut, args=(t_release, pre, post),
                         daemon=True).start()

    def _cut(self, t_release, pre, post):
        deadline = t_release + post
        while time.time() < deadline + 0.05 and not self._stop.is_set():
            time.sleep(0.01)
        try:
            with self._lock:
                rec = self.buf.window(t_release - pre, t_release + post)
            self._events.put({"t_release": t_release, "rec": rec})
        except ValueError as e:
            self._events.put({"t_release": t_release, "error": str(e)})

    def pop_event(self, timeout=None):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        from perception.ir_capture import IRRecorder
        try:
            with IRRecorder(width=self._cfg["width"], height=self._cfg["height"],
                            fps=self._cfg["fps"],
                            exposure_us=self._cfg["exposure_us"],
                            emitter=self._cfg["emitter"]) as rec:
                for ts, ir1, ir2 in rec.stream():
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self.buf.append(ts, ir1, ir2)     # RingBuffer.append copies internally
                        self._latest = (ts, ir1.copy(), ir2.copy())   # see FRAME OWNERSHIP CONTRACT
                    if self._on_frame is not None:
                        self._on_frame(ts, ir1, ir2)       # RAW view -- see FRAME OWNERSHIP CONTRACT
        except Exception as e:                     # a camera fault ends the session
            self.error = e
