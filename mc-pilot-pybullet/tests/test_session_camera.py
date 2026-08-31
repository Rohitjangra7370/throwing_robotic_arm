"""Ring buffer + camera-thread lifecycle. No camera is opened by these tests."""
import numpy as np
import pytest

from session_camera import RingBuffer


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
