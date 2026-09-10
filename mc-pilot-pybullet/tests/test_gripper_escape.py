"""
Bracket logic for the ball-escape search.

Percent-closed runs the opposite way to the usual search convention: the ball is
caged ABOVE the escape position and free below it, so `lo` is a position known to
RELEASE and `hi` one known to HOLD. Getting that backwards converges silently on
the wrong end of the bracket and would hand back a confidently wrong number, so
it is pinned here.
"""
import numpy as np
import pytest

from gripper_escape_test import (escape_time_from_traces, next_probe,
                                 update_bracket)


def test_caged_probe_moves_the_holding_end_down():
    lo, hi = update_bracket(0.0, 33.33, 16.7, caged=True)
    assert (lo, hi) == (0.0, 16.7)


def test_escaped_probe_moves_the_releasing_end_up():
    lo, hi = update_bracket(0.0, 33.33, 16.7, caged=False)
    assert (lo, hi) == (16.7, 33.33)


def test_search_converges_on_a_known_boundary():
    true_escape = 21.4
    lo, hi = 0.0, 33.33
    for _ in range(12):
        x = next_probe(lo, hi)
        lo, hi = update_bracket(lo, hi, x, caged=(x > true_escape))
    assert 0.5 * (lo + hi) == pytest.approx(true_escape, abs=0.05)
    assert lo <= true_escape <= hi


def _open_trace(p_start=33.33, p_end=0.4, onset_s=0.068, dur_s=0.42, hz=1000.0):
    """Synthetic open: flat through the onset delay, then linear travel."""
    n = int((onset_s + dur_s) * hz)
    t = np.arange(n) / hz
    p = np.where(t < onset_s, p_start,
                 p_start + (p_end - p_start) * np.clip((t - onset_s) / dur_s, 0, 1))
    v = np.gradient(p, t)
    return np.stack([t, p, v], axis=1)


def test_escape_time_lands_between_onset_and_settle():
    tr = np.array([_open_trace(), _open_trace()], dtype=object)
    t_esc, t_on = escape_time_from_traces(tr, x_escape=21.4)
    # Onset is detected by 0.5 percentage-points of travel, so it necessarily
    # fires slightly LATE -- here 32.93 pct-pts over 0.42 s = 78.4 pct-pt/s, so
    # 0.5 of them costs 6.4 ms. That bias is shared with
    # measure_gripper_latency.py, which is what makes the two comparable; it is
    # a property of the detector, not an error, and it is small against the
    # 50-140 ms remainder this whole test exists to resolve.
    lag = 0.5 / ((33.33 - 0.4) / 0.42)
    assert t_on == pytest.approx(0.068 + lag, abs=0.002)
    # 33.33 -> 21.4 is 36% of the travel, so ~36% of the 0.42 s stroke past onset
    assert t_esc == pytest.approx(0.068 + 0.36 * 0.42, abs=0.01)
    assert t_on < t_esc


def test_escape_time_refuses_a_position_never_reached():
    tr = np.array([_open_trace(p_end=25.0)], dtype=object)
    t_esc, t_on = escape_time_from_traces(tr, x_escape=10.0)
    assert t_esc is None                 # must not extrapolate
    lag = 0.5 / ((33.33 - 25.0) / 0.42)  # slower stroke -> later detection
    assert t_on == pytest.approx(0.068 + lag, abs=0.002)


def test_escape_time_refuses_a_position_above_the_grasp():
    """
    Found in the dry-run smoke test: with x_escape ABOVE the trace's starting
    position, `pos <= x` is true at sample 0, so the function returned t[0] --
    an escape 53 ms BEFORE the fingers moved, printed without complaint. The
    ball cannot leave before the gripper opens; refuse rather than report a
    negative remainder.
    """
    tr = np.array([_open_trace(p_start=33.33)], dtype=object)
    t_esc, t_on = escape_time_from_traces(tr, x_escape=93.75)
    assert t_esc is None
    assert t_on is not None and t_on > 0
