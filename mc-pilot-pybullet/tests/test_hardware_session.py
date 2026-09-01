"""State-machine and safety-gate tests. No Tk window, arm, or camera is created."""
import argparse
import threading
import time

import numpy as np
import pytest

from hardware_session import (SessionState, Stage, ThrowCycle, build_argparser,
                              build_cycle_args, build_stage_zero_args)
from robot_arm.kinova_hardware import HardwareThrowExecutor, SafetyLimits


def test_session_starts_cold_with_throwing_disabled():
    s = SessionState()
    assert s.stage is Stage.COLD
    assert not s.can_throw()


def test_no_go_from_stage_zero_blocks_throwing():
    s = SessionState()
    s.record_startup(go=False, failures=["FLOOR: -14.0 cm"])
    assert s.stage is Stage.BLOCKED
    assert not s.can_throw()
    assert "FLOOR" in s.blocked_reason


def test_go_then_camera_makes_the_session_ready():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    assert s.stage is Stage.CALIBRATED and not s.can_throw()
    s.camera_ready()
    assert s.stage is Stage.READY and s.can_throw()


def test_confirm_resets_after_every_throw():
    """Re-affirmed per throw, never once per session."""
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.confirmed = True
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.confirmed is False


def test_speed_scale_ladder_is_enforced_by_the_session():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    ok, why = s.check_scale(1.00)
    assert not ok and "0.15" in why
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.check_scale(0.30)[0]
    assert not s.check_scale(0.60)[0]


def test_a_refused_measurement_still_counts_as_a_logged_throw():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.record_throw({"speed_scale": 0.15, "landing_xy": None,
                    "refusal_reason": "inlier fraction 0.49"})
    assert s.n_throws == 1
    assert s.n_measured == 0


def test_model_update_needs_enough_measured_throws():
    # speed_scale=1.0 (not 0.15) since Defect 1 (2026-09-02): can_update_model
    # now gates on FULL-SPEED measured throws only -- see
    # test_model_update_ignores_rehearsal_only_throws below for the
    # rehearsal-exclusion case this test used to (incorrectly) exercise.
    s = SessionState(min_throws_for_update=3)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    for _ in range(2):
        s.record_throw({"speed_scale": 1.0, "landing_xy": [0.7, 0.0]})
    assert not s.can_update_model()
    s.record_throw({"speed_scale": 1.0, "landing_xy": [0.7, 0.0]})
    assert s.can_update_model()


def test_model_update_ignores_rehearsal_only_throws():
    """Defect 1 (2026-09-02): a session with plenty of landed ladder throws
    (speed_scale < 1.0) but zero full-speed throws must NOT unlock
    "Update model" -- rehearsals are evidence about the rig, not data for
    the release-model fit. n_throws/n_measured (the "how many landings"
    displays) still count them; only can_update_model must stay False."""
    s = SessionState(min_throws_for_update=3)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    for scale in (0.15, 0.30, 0.60, 0.60, 0.60):
        s.record_throw({"speed_scale": scale, "landing_xy": [0.7, 0.0]})
    assert s.n_measured == 5
    assert s.n_measured_full_speed == 0
    assert not s.can_update_model()


def test_policy_button_is_locked_until_the_model_is_updated():
    # speed_scale=1.0 -- see the comment on test_model_update_needs_enough_measured_throws.
    s = SessionState(min_throws_for_update=1)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.record_throw({"speed_scale": 1.0, "landing_xy": [0.7, 0.0]})
    assert not s.can_reoptimize_policy()
    s.record_model_update()
    assert s.can_reoptimize_policy()


# ---------------------------------------------------------------------------
# Regression: rehearse_or_throw's new on_release hook (kinova_hardware.py).
# ThrowCycle.step_throw_and_measure relies on this to tell the camera thread
# the exact release instant -- see the Task 8 implementer note. Dry-run only,
# a fake arm/coeffs pair, no pybullet/checkpoint needed, so this is fast and
# does not depend on the real hardware or a trained checkpoint being present.
# ---------------------------------------------------------------------------
class _FakeArm:
    """Two DOF, everything at rest -- only the release-callback wiring is under test."""

    def get_setpoint(self, coeffs, s, with_accel=True):
        z = np.zeros(2)
        return z, z, z


def _dry_run_executor():
    limits = SafetyLimits(
        qd_max=np.array([1.0, 1.0]), q_soft_lo=np.array([-6.1, -6.1]),
        q_soft_hi=np.array([6.1, 6.1]), speed_scale=1.0, positioning_scale=1.0,
        control_hz=40.0, gripper_lead_s=0.0,
    )
    return HardwareThrowExecutor(limits, dry_run=True)


def test_rehearse_or_throw_calls_on_release_exactly_once_dry_run():
    coeffs = {"t_w": 0.0, "t_r": 0.05, "T": 0.10}
    calls = []
    with _dry_run_executor() as ex:
        ex.rehearse_or_throw(coeffs, _FakeArm(), verbose=False,
                             on_release=lambda: calls.append(time.time()))
    assert len(calls) == 1


def test_rehearse_or_throw_on_release_none_is_a_noop():
    coeffs = {"t_w": 0.0, "t_r": 0.05, "T": 0.10}
    with _dry_run_executor() as ex:
        ex.rehearse_or_throw(coeffs, _FakeArm(), verbose=False, on_release=None)  # must not raise


# ---------------------------------------------------------------------------
# Regression (found 2026-09-01, Step 6 verification): the stage-zero and
# throw-cycle worker threads used to call self.ip_var.get() etc. directly.
# Tkinter variables may only be touched from the thread running mainloop --
# off that thread, .get() raises "RuntimeError: main thread is not in main
# loop", and because the error handler ALSO scheduled its recovery via
# root.after (another Tk call, from the same bad thread), that raised too and
# the whole failure died as an unhandled thread traceback: the operator
# clicks "Run start-of-day", the worker dies silently, and the window just
# sits there with no verdict and no error dialog.
#
# The fix moved every Tk variable read to the button handler (main thread)
# and turned what crosses into the worker into a plain dict -> these two
# module-level functions, which never import tkinter and never reference a
# SessionApp instance. That is provable directly, with no Tk display needed
# at all: call them from a background thread and confirm they do not raise.
# ---------------------------------------------------------------------------
def _run_in_thread(fn):
    """Call `fn` on a background thread; return (result, exception)."""
    result, error = {}, {}

    def worker():
        try:
            result["value"] = fn()
        except Exception as e:
            error["value"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "worker thread did not finish"
    return result.get("value"), error.get("value")


def test_stage_zero_arg_builder_is_safe_off_the_main_thread():
    base_args = build_argparser().parse_args(["--dry_run"])
    fields = {"ip": "10.0.0.5", "robot": "kinova_gen3_dyn",
             "log_path": "results_kinetic_chain_gen3_tcp/1",
             "opt_pose": "throw_pose_table_tcp.npy", "tool_offset_z": "0.12"}

    value, error = _run_in_thread(lambda: build_stage_zero_args(base_args, fields))

    assert error is None, f"build_stage_zero_args raised off the main thread: {error!r}"
    assert value.ip == "10.0.0.5"
    assert value.robot == "kinova_gen3_dyn"
    assert value.tool_offset_z == pytest.approx(0.12)
    assert value.floor_z == pytest.approx(-value.base_height)  # default fill-in still applies


def test_cycle_arg_builder_is_safe_off_the_main_thread():
    base_args = build_argparser().parse_args(["--dry_run"])
    fields = {"ip": "10.0.0.5", "robot": "kinova_gen3_dyn",
             "log_path": "results_kinetic_chain_gen3_tcp/1",
             "opt_pose": "throw_pose_table_tcp.npy", "tool_offset_z": "0.12"}

    value, error = _run_in_thread(lambda: build_cycle_args(base_args, fields))

    assert error is None, f"build_cycle_args raised off the main thread: {error!r}"
    assert value.ip == "10.0.0.5"
    assert value.tool_offset_z == pytest.approx(0.12)


def test_arg_builders_report_bad_numeric_fields_without_touching_tk():
    """
    A bad numeric field must raise ValueError (caught by the button handler,
    on the main thread, and turned into a messagebox) -- not a bare
    tkinter/float exception surfacing from a worker thread.
    """
    base_args = build_argparser().parse_args(["--dry_run"])
    fields = {"ip": "10.0.0.5", "robot": "kinova_gen3_dyn", "log_path": "x",
             "opt_pose": "y", "tool_offset_z": "not-a-number"}

    with pytest.raises(ValueError, match="tool_offset_z"):
        build_stage_zero_args(base_args, fields)
    with pytest.raises(ValueError, match="tool_offset_z"):
        build_cycle_args(base_args, fields)


# ---------------------------------------------------------------------------
# Regression (Task 8 review, IMPORTANT 1): a throw that has physically
# executed must never be dropped from the dataset. `load_extrinsic()` raises
# FileNotFoundError/ValueError, not RuntimeError -- catching only RuntimeError
# around the measurement step let those types escape past step_throw_and_measure
# and into _do_throw's generic handler, which does not log a record. By the
# time measurement runs, rehearse_or_throw has already executed (the ball has
# left the hand), so ANY exception from measure_landing must still produce a
# (None, {"refusal_reason": ...}, exec_stats) result, never propagate.
# ---------------------------------------------------------------------------
class _FakeArmForThrow:
    """Same 2-DOF fake as the rehearse_or_throw tests above, plus the extra
    attributes HardwareThrowExecutor.home() needs (arm._q_lo/_q_hi) so
    step_throw_and_measure can run its full dry-run path end to end."""

    def __init__(self):
        self._q_lo = np.array([-6.1, -6.1])
        self._q_hi = np.array([6.1, 6.1])

    def get_setpoint(self, coeffs, s, with_accel=True):
        z = np.zeros(2)
        return z, z, z


class _FakeProfile:
    q_neutral = [0.0, 0.0]


class _FakeCamera:
    """Stand-in for CameraThread -- step_throw_and_measure only calls
    mark_release/pop_event on it."""

    def __init__(self, event):
        self._event = event
        self.release_calls = []

    def mark_release(self, t):
        self.release_calls.append(t)

    def pop_event(self, timeout=None):
        return self._event


def test_measurement_failure_that_is_not_a_runtimeerror_still_logs_a_record(monkeypatch):
    def _raise_non_runtime_error(*a, **kw):
        raise ValueError("bogus calibration -- deliberately NOT a RuntimeError")

    monkeypatch.setattr("measure_landing.measure_landing", _raise_non_runtime_error)

    state = SessionState()
    camera = _FakeCamera({"t_release": 0.0, "rec": {
        "t": np.zeros(1), "ir1": np.zeros((1, 1, 1)), "ir2": np.zeros((1, 1, 1))}})
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmForThrow(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))   # already "loaded" by the caller, per the fix

    landing_xy, measurement, exec_stats = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=0.15, throw_index=0, extrinsic=extrinsic)

    # The throw physically executed (rehearse_or_throw ran, on_release fired) --
    # the ValueError from measurement must still come back as a refused-but-
    # logged result, not propagate out of step_throw_and_measure.
    assert camera.release_calls, "on_release must have fired -- the arm did move"
    assert landing_xy is None
    assert "bogus calibration" in measurement["refusal_reason"]
    assert isinstance(exec_stats, dict)


# ---------------------------------------------------------------------------
# Regression (Task 8 review, IMPORTANT 1's RESIDUAL note, closed 2026-09-02):
# an exception raised inside the execution block itself (set_gripper, home,
# rehearse_or_throw -- not just the measure_landing() call below it) must
# ALSO still produce a logged record once release has occurred. Simulated
# here by making the camera's mark_release() raise: _on_release sets its
# "release occurred" flag BEFORE calling mark_release, so this reproduces
# exactly "an exception raised from the execution path after the release
# callback has fired", the case the brief asks to test.
# ---------------------------------------------------------------------------
class _FakeCameraThatRaisesOnRelease:
    """mark_release raises -- simulating an execution-path failure that
    happens strictly after release. pop_event must never be reached: the
    exception should short-circuit step_throw_and_measure before measurement
    is attempted at all."""

    def mark_release(self, t):
        raise RuntimeError("camera thread died handling the release timestamp")

    def pop_event(self, timeout=None):
        raise AssertionError(
            "must not reach measurement -- the execution-path exception "
            "should have already produced a refusal result")


def test_exception_after_release_in_execution_block_still_logs_a_record():
    state = SessionState()
    camera = _FakeCameraThatRaisesOnRelease()
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmForThrow(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))

    landing_xy, measurement, exec_stats = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=1.0, throw_index=0, extrinsic=extrinsic)

    # This is the invariant the brief states directly: once the ball has
    # physically left the hand, a record is written no matter what fails
    # afterwards -- even when the failure is in the execution block, not
    # the measurement step.
    assert landing_xy is None
    assert "camera thread died" in measurement["refusal_reason"]
    assert isinstance(exec_stats, dict)


def test_exception_before_release_still_propagates():
    """The flip side, unchanged from before: nothing physical happened yet,
    so a pre-release failure is allowed to propagate to the caller (_do_throw
    turns it into _finish_throw_error) rather than being papered over as a
    refused-but-executed throw. get_setpoint() is called once per tick from
    the very start of rehearse_or_throw's loop (s=0), strictly before the
    s >= s_fire release check, so raising there is a clean pre-release
    failure -- release never fires and _FakeCameraThatRaisesOnRelease's
    mark_release is never reached."""
    class _FakeArmThatFailsBeforeRelease:
        _q_lo = np.array([-6.1, -6.1])
        _q_hi = np.array([6.1, 6.1])

        def get_setpoint(self, coeffs, s, with_accel=True):
            raise RuntimeError("trajectory evaluation failed before any release")

    state = SessionState()
    camera = _FakeCameraThatRaisesOnRelease()   # mark_release must never be called
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmThatFailsBeforeRelease(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))

    with pytest.raises(RuntimeError, match="trajectory evaluation failed"):
        cycle.step_throw_and_measure(
            plan, target=[0.7, 0.0], speed_scale=1.0, throw_index=0, extrinsic=extrinsic)
