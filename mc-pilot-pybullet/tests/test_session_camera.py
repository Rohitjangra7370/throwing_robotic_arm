"""Ring buffer + camera-thread lifecycle. No camera is opened by these tests."""
import numpy as np
import pytest

import perception.ir_capture as ir_capture
from session_camera import CameraThread, RingBuffer


def _frame(v):
    return np.full((8, 8), v, np.uint8)


def test_ring_buffer_keeps_only_its_window():
    rb = RingBuffer(seconds=0.1, fps=100)      # capacity 10
    for k in range(25):
        rb.append(k * 0.01, _frame(k), _frame(k))
    assert len(rb) == 10
    # frames are filled with their own index, so the surviving frames identify
    # themselves -- the oldest kept must be frame 15, not frame 0
    assert rb.window(0.0, 1.0)["ir1"][0][0, 0] == 15


def test_window_extracts_the_requested_span_inclusive():
    rb = RingBuffer(seconds=10.0, fps=100)
    for k in range(100):
        rb.append(k * 0.01, _frame(k), _frame(k))
    w = rb.window(0.20, 0.30)
    assert w["ir1"][0][0, 0] == 20 and w["ir1"][-1][0, 0] == 30
    assert w["ir1"].shape[0] == w["t"].size == w["ir2"].shape[0]
    assert w["ir1"].shape[1:] == (8, 8)


def test_window_timestamps_are_relative_to_the_window_like_record_is():
    """IRRecorder.record() returns `t` zeroed to its first frame (`ts - ts[0]`).
    window() must use the same convention or measure_landing sees two different
    time origins."""
    rb = RingBuffer(seconds=10.0, fps=100)
    for k in range(100):
        rb.append(k * 0.01, _frame(k), _frame(k))
    w = rb.window(0.20, 0.30)
    assert w["t"][0] == pytest.approx(0.0)
    assert w["t"][-1] == pytest.approx(0.10)


def test_window_returns_stacked_arrays_under_the_key_build_observations_reads():
    """measure_landing.build_observations does `rec["ir1"], rec["ir2"], rec["t"]`
    -- the key is `t`, NOT `ts`, and ir1 must be a stacked (N, H, W) array."""
    rb = RingBuffer(seconds=10.0, fps=100)
    for k in range(10):
        rb.append(k * 0.01, _frame(k), _frame(k))
    w = rb.window(0.0, 0.09)
    assert set(w) >= {"t", "ir1", "ir2"}
    assert isinstance(w["ir1"], np.ndarray) and w["ir1"].ndim == 3


def test_window_raises_when_the_span_holds_no_frames():
    rb = RingBuffer(seconds=10.0, fps=100)
    rb.append(0.0, _frame(1), _frame(1))
    with pytest.raises(ValueError, match="no frames"):
        rb.window(5.0, 6.0)


def test_ring_buffer_append_copies_frames_not_views():
    """Real-hardware regression (2026-08-31): RingBuffer.append() originally
    stored the caller's array objects directly. IRRecorder.stream() yields
    zero-copy views over the RealSense SDK's own frame buffer, and the ring
    buffer retains up to `capacity` frames simultaneously (~270 at the
    default 3 s/90 fps) -- far more than the SDK's internal frame pool, which
    stalled capture after exactly 16 live frames on the real camera. Mutating
    the source array after append() must NOT change what's stored."""
    rb = RingBuffer(seconds=10.0, fps=100)
    src = _frame(1)
    rb.append(0.0, src, src.copy())
    src[:] = 99
    assert rb.window(0.0, 0.0)["ir1"][0][0, 0] == 1


def _fake_ir_recorder_factory(frames):
    """Stand-in for perception.ir_capture.IRRecorder: same constructor and
    context-manager shape (every kwarg CameraThread._run() passes is
    accepted and ignored), but `.stream()` replays a fixed, pre-built
    sequence instead of touching a camera. `frames` is an iterable (often a
    single-use generator) of `(ts, ir1, ir2)` tuples."""
    class _Fake:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self):
            yield from frames

    return _Fake


def test_camera_thread_latest_and_buffer_hold_independent_copies(monkeypatch):
    """Real-hardware regression follow-up (2026-08-31): the RingBuffer.append
    fix above left `_latest` in `CameraThread._run()` storing the same raw
    zero-copy view one line below the fix -- `latest()` exists precisely so a
    display consumer can hold a frame, which is the exact retain-beyond-the-
    iteration pattern that starved the SDK's frame pool in the first place.

    Drive the real capture loop (`_run()`, called synchronously since the
    fake stream is finite) with arrays that get mutated in place immediately
    after the loop body has consumed them -- emulating the SDK reusing that
    memory for the next frame -- and confirm both `latest()` and the ring
    buffer are unaffected, i.e. both hold genuine copies, not views."""
    ir1_a = np.full((4, 4), 1, np.uint8)
    ir2_a = np.full((4, 4), 2, np.uint8)

    def gen():
        yield 0.0, ir1_a, ir2_a
        # Mutate the SAME array objects right after the loop body has
        # consumed that frame (this runs when the `for` loop asks for the
        # next value, i.e. strictly after buf.append/_latest for frame 0).
        ir1_a[:] = 99
        ir2_a[:] = 99

    monkeypatch.setattr(ir_capture, "IRRecorder", _fake_ir_recorder_factory(gen()))

    ct = CameraThread()
    ct._run()
    assert ct.error is None

    ts, ir1, ir2 = ct.latest()
    assert ir1[0, 0] == 1 and ir2[0, 0] == 2            # not 99 -- latest() copied

    w = ct.buf.window(0.0, 0.0)
    assert w["ir1"][0][0, 0] == 1 and w["ir2"][0][0, 0] == 2   # buffer unaffected too


def test_on_frame_receives_the_raw_view_per_documented_contract(monkeypatch):
    """`on_frame` is deliberately NOT copied (see CameraThread's "FRAME
    OWNERSHIP CONTRACT" docstring): it is a synchronous, inline, per-frame
    callback at capture rate, so it is handed the raw zero-copy view on
    purpose to avoid paying a copy every frame for callbacks that don't need
    one. Pin the choice down: on_frame must see the identical array object
    stream() produced, not an independent copy -- and a consumer that needs
    to retain it is documented as responsible for copying it itself."""
    seen = {}

    def on_frame(ts, ir1, ir2):
        seen["ir1"], seen["ir2"] = ir1, ir2

    ir1_a = np.full((4, 4), 7, np.uint8)
    ir2_a = np.full((4, 4), 8, np.uint8)

    def gen():
        yield 0.0, ir1_a, ir2_a

    monkeypatch.setattr(ir_capture, "IRRecorder", _fake_ir_recorder_factory(gen()))

    ct = CameraThread(on_frame=on_frame)
    ct._run()
    assert ct.error is None
    assert seen["ir1"] is ir1_a and seen["ir2"] is ir2_a   # identity, not just equality


def test_stop_with_no_thread_started_returns_true():
    assert CameraThread().stop() is True


def test_stop_returns_true_once_the_worker_has_actually_exited(monkeypatch):
    """stop() must give a positive signal that the camera is actually free,
    not just return silently after its join timeout -- the D435i can only be
    opened by one process at a time (see the module docstring). Run a real
    background thread (not a synchronous call) through a small finite fake
    stream so it exits on its own, then confirm stop() reports True."""
    def gen():
        for k in range(5):
            yield k * 0.01, np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)

    monkeypatch.setattr(ir_capture, "IRRecorder", _fake_ir_recorder_factory(gen()))

    ct = CameraThread()
    ct.start()
    ct._thread.join(timeout=2.0)      # let the finite fake stream run to completion
    assert not ct._thread.is_alive()
    assert ct.stop() is True
    assert ct.error is None


def test_record_matches_documented_contract_with_injected_stream(monkeypatch):
    """No test anywhere in the repo calls IRRecorder.record() or .stream() --
    the broader suite passing elsewhere only shows the stream()/record()
    refactor broke nothing ELSE. record()'s output contract
    ({"t","ir1","ir2","meta"}, `t` zeroed to the first frame,
    meta["achieved_fps"], the k==0 error below) is what
    measure_landing.build_observations depends on, and was previously
    unguarded by any test.

    Exercise it here with an injected stream() so it needs no camera, and
    fake the clock so the time budget never expires -- the injected stream's
    own frame count is then the only thing deciding when record() stops,
    which makes this fully deterministic rather than racing real timing."""
    fps, seconds = 50, 0.1
    n_expect = int(np.ceil(seconds * fps)) + 8   # mirrors record()'s own formula: 13
    frames = [(10.0 + k * 0.02,
               np.full((4, 4), k, np.uint8),
               np.full((4, 4), k + 100, np.uint8))
              for k in range(n_expect)]

    def fake_stream(self):
        yield from frames
    monkeypatch.setattr(ir_capture.IRRecorder, "stream", fake_stream)
    monkeypatch.setattr(ir_capture.time, "monotonic", lambda: 0.0)   # never expires

    rec = ir_capture.IRRecorder(width=4, height=4, fps=fps, exposure_us=1000)
    rec._pipe = object()   # bypass the "use as context manager" guard -- record()
                            # itself never reads _pipe beyond checking it is set
    out = rec.record(seconds)

    assert set(out) == {"t", "ir1", "ir2", "meta"}
    assert out["t"].shape == (n_expect,)
    assert out["t"][0] == pytest.approx(0.0)           # zeroed to the first frame
    assert out["t"][-1] == pytest.approx(0.24)          # 12 * 0.02
    assert np.allclose(np.diff(out["t"]), 0.02)          # spacing preserved
    assert out["ir1"].shape == (n_expect, 4, 4)
    assert out["ir2"].shape == (n_expect, 4, 4)
    assert out["ir1"][3][0, 0] == 3 and out["ir2"][3][0, 0] == 103
    assert out["meta"]["n_frames"] == n_expect
    assert out["meta"]["achieved_fps"] == pytest.approx(50.0)
    assert out["meta"]["exposure_us"] == 1000


def test_record_raises_on_zero_frames(monkeypatch):
    """The k==0 guard protects a caller from silently getting an empty
    recording if the time budget expires before a single frame lands. Force
    that path deterministically by faking the clock rather than racing real
    timing -- the fake stream is never even invoked."""
    def fake_stream(self):
        if False:
            yield   # never reached: makes this a generator function, that's all
    monkeypatch.setattr(ir_capture.IRRecorder, "stream", fake_stream)

    clock = iter([1000.0, 2000.0, 2000.0, 2000.0])
    monkeypatch.setattr(ir_capture.time, "monotonic", lambda: next(clock))

    rec = ir_capture.IRRecorder(width=4, height=4, fps=10, exposure_us=1000)
    rec._pipe = object()
    with pytest.raises(RuntimeError, match="captured zero frames"):
        rec.record(seconds=1.0)
