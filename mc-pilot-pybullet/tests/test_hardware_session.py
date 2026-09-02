"""State-machine and safety-gate tests. No Tk window, arm, or camera is created."""
import argparse
import json
import os
import threading
import time

import numpy as np
import pytest

from hardware_session import (SessionState, Stage, ThrowCycle, build_argparser,
                              build_cycle_args, build_stage_zero_args,
                              finalize_throw_record)
from perception.ir_capture import load_recording
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


def test_measurement_failure_that_is_not_a_runtimeerror_still_logs_a_record(monkeypatch, tmp_path):
    def _raise_non_runtime_error(*a, **kw):
        raise ValueError("bogus calibration -- deliberately NOT a RuntimeError")

    monkeypatch.setattr("measure_landing.measure_landing", _raise_non_runtime_error)

    state = SessionState()
    camera = _FakeCamera({"t_release": 0.0, "rec": {
        "t": np.zeros(1), "ir1": np.zeros((1, 1, 1), np.uint8), "ir2": np.zeros((1, 1, 1), np.uint8)}})
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327,
                              throws_dir=str(tmp_path), camera_fps=90,
                              exposure_us=2000, no_emitter=False)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmForThrow(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))   # already "loaded" by the caller, per the fix

    landing_xy, measurement, exec_stats, capture_file = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=0.15, throw_index=0, extrinsic=extrinsic)

    # The throw physically executed (rehearse_or_throw ran, on_release fired) --
    # the ValueError from measurement must still come back as a refused-but-
    # logged result, not propagate out of step_throw_and_measure.
    assert camera.release_calls, "on_release must have fired -- the arm did move"
    assert landing_xy is None
    assert "bogus calibration" in measurement["refusal_reason"]
    assert isinstance(exec_stats, dict)
    # The raw recording is saved BEFORE measure_landing runs, so a measurement
    # failure must not erase it -- capture_file still points at a real file,
    # which is the whole point of saving first (see step_throw_and_measure's
    # capture-save comment): a future, improved fitter can be re-run against
    # this exact recording even though this fit failed.
    assert capture_file == os.path.join(str(tmp_path), "throw_000.npz")
    assert os.path.isfile(capture_file)


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

    landing_xy, measurement, exec_stats, capture_file = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=1.0, throw_index=0, extrinsic=extrinsic)

    # This is the invariant the brief states directly: once the ball has
    # physically left the hand, a record is written no matter what fails
    # afterwards -- even when the failure is in the execution block, not
    # the measurement step.
    assert landing_xy is None
    assert "camera thread died" in measurement["refusal_reason"]
    assert isinstance(exec_stats, dict)
    # mark_release raised BEFORE pop_event/the capture-save block was ever
    # reached (see _FakeCameraThatRaisesOnRelease's docstring) -- no window
    # was ever captured, so there is nothing to have saved.
    assert capture_file is None


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


# ---------------------------------------------------------------------------
# Regression: the raw dual-IR recording of every real throw was being thrown
# away (_do_throw always logged capture_file=None) -- Task 9's flight-GP
# ingestion (track_getter) degrades gracefully to "0 ingested" without one,
# but the bigger cost is that the raw frames a session's own throws produce
# can never be re-derived from later, contradicting HARDWARE_RUNBOOK.md's
# "keep every recording, it is a permanent regression fixture" rule. These
# tests cover: a successful cycle writes a real file and reports its path;
# that file round-trips through perception.ir_capture.load_recording with the
# same frame count/shapes (the property that makes it a usable fixture); and
# that a save failure still produces a logged (refused) record rather than
# losing the throw -- "saving must never cost a throw".
# ---------------------------------------------------------------------------
def test_successful_cycle_saves_recording_and_reports_capture_file(monkeypatch, tmp_path):
    def _fake_measure_landing(rec, R, t, z_floor, ball_radius):
        return {"x": 0.71, "y": 0.02, "sigma_xy_m": 0.01, "n_frames": 3,
               "n_inliers": 3, "rms_px": 0.2,
               "p0": [0.0, 0.0, 0.0], "v0": [1.0, 0.0, 0.0]}

    monkeypatch.setattr("measure_landing.measure_landing", _fake_measure_landing)

    state = SessionState()
    ir1 = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    ir2 = (ir1 + 1).astype(np.uint8)
    camera = _FakeCamera({"t_release": 1.23,
                          "rec": {"t": np.array([0.0, 0.01, 0.02]), "ir1": ir1, "ir2": ir2}})
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327,
                              throws_dir=str(tmp_path), camera_fps=90,
                              exposure_us=2000, no_emitter=False)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmForThrow(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))

    landing_xy, measurement, exec_stats, capture_file = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=1.0, throw_index=7, extrinsic=extrinsic)

    assert landing_xy == [pytest.approx(0.71), pytest.approx(0.02)]
    assert capture_file == os.path.join(str(tmp_path), "throw_007.npz")
    assert os.path.isfile(capture_file)

    loaded = load_recording(capture_file)
    assert loaded["ir1"].shape == ir1.shape
    assert loaded["ir2"].shape == ir2.shape
    assert loaded["t"].shape == (3,)
    np.testing.assert_array_equal(loaded["ir1"], ir1)
    np.testing.assert_array_equal(loaded["ir2"], ir2)
    assert loaded["meta"]["n_frames"] == 3
    assert loaded["meta"]["width"] == 5
    assert loaded["meta"]["height"] == 4
    assert loaded["meta"]["throw_index"] == 7


def test_capture_save_failure_still_logs_a_refused_record(monkeypatch, tmp_path):
    def _raise_on_save(path, rec):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("perception.ir_capture.save_recording", _raise_on_save)

    state = SessionState()
    camera = _FakeCamera({"t_release": 0.0, "rec": {
        "t": np.zeros(1), "ir1": np.zeros((1, 1, 1), np.uint8), "ir2": np.zeros((1, 1, 1), np.uint8)}})
    args = argparse.Namespace(duration=0.1, measure_timeout=1.0,
                              base_height=0.433, ball_radius=0.0327,
                              throws_dir=str(tmp_path), camera_fps=90,
                              exposure_us=2000, no_emitter=False)
    cycle = ThrowCycle(state, camera, args)

    plan = {"ex": _dry_run_executor(), "arm": _FakeArmForThrow(),
           "profile": _FakeProfile(),
           "coeffs": {"t_w": 0.0, "t_r": 0.05, "T": 0.10}}
    extrinsic = (np.eye(3), np.zeros(3))

    landing_xy, measurement, exec_stats, capture_file = cycle.step_throw_and_measure(
        plan, target=[0.7, 0.0], speed_scale=1.0, throw_index=0, extrinsic=extrinsic)

    # The ball physically left the hand (on_release fired) -- a disk write
    # failure past that point must still come back as a refused-but-logged
    # result, never a bare exception that would drop the throw.
    assert camera.release_calls, "on_release must have fired -- the arm did move"
    assert landing_xy is None
    assert capture_file is None
    assert "capture save failed" in measurement["refusal_reason"]
    assert "disk full" in measurement["refusal_reason"]
    assert isinstance(exec_stats, dict)


# -- Task 10: Button 2, policy re-optimization into a NEW checkpoint -------- #

def test_reoptimize_refuses_to_overwrite_an_existing_checkpoint(tmp_path):
    """Never overwrite a trained checkpoint -- it is the only copy."""
    from hardware_session import reoptimize_policy
    existing = tmp_path / "results_kinetic_chain_gen3_tcp" / "1"
    existing.mkdir(parents=True)
    (existing / "config_log.pkl").write_bytes(b"pretend checkpoint")

    class FakeMC:
        def reinforce_policy(self, **kw):
            raise AssertionError("must refuse BEFORE touching the model")

    with pytest.raises(FileExistsError, match="would overwrite"):
        reoptimize_policy(FakeMC(), str(existing), {"T_control": 1})


def test_reoptimize_passes_the_caller_s_kwargs_through_untouched(tmp_path):
    """
    reinforce_policy takes ~13 required arguments (T_control, num_particles,
    trial_index, particle init means/vars/bounds, opt_steps_list, lr_list,
    f_optimizer, ...). This function must NOT invent or reshape them -- it
    forwards exactly what the caller built, so there is one place that owns
    that argument set: adapt_policy_height.py's proven call.
    """
    from hardware_session import reoptimize_policy
    seen = {}

    class FakeMC:
        def reinforce_policy(self, **kw):
            seen.update(kw)
            return [0.1], None, None, None

    out = tmp_path / "new_ckpt"
    kwargs = {"T_control": 40, "num_particles": 200, "trial_index": 3,
              "opt_steps_list": [50], "lr_list": [0.01]}
    assert reoptimize_policy(FakeMC(), str(out), kwargs) == str(out)
    assert seen == kwargs
    assert out.is_dir()


def test_reoptimize_accepts_an_existing_but_empty_directory(tmp_path):
    """Pre-creating the output path is normal; only a POPULATED dir is refused."""
    from hardware_session import reoptimize_policy
    out = tmp_path / "empty_ckpt"
    out.mkdir()

    class FakeMC:
        def reinforce_policy(self, **kw):
            return [0.1], None, None, None

    assert reoptimize_policy(FakeMC(), str(out), {"T_control": 1}) == str(out)


# ---------------------------------------------------------------------------
# Regression (final whole-branch review, FIX 1): a throw that has ALREADY
# physically executed -- step_throw_and_measure returned successfully, so
# release has happened -- must never vanish from the dataset just because
# build_throw_record/append_log itself fails afterwards (disk-full,
# permission error, an unexpected bug in build_throw_record). Before this
# fix, _do_throw called build_throw_record/append_log inline, inside the
# SAME outer `try` that funnels any exception to `_finish_throw_error` --
# and that handler never calls `state.record_throw`, so a failure here
# silently dropped an already-executed throw from both the JSONL log and
# SessionState.throws, and with it the escalation ladder / update-model
# count.
#
# finalize_throw_record (module-level, Tk-free, same extraction pattern as
# build_stage_zero_args/build_cycle_args/reoptimize_policy above) is what
# _do_throw now calls right after step_throw_and_measure returns. These
# tests exercise it directly -- no Tk/SessionApp needed -- following this
# file's convention of stubbing at the ThrowCycle/step_throw_and_measure
# return-value level rather than spinning up a real GUI.
# ---------------------------------------------------------------------------
def _fake_plan(speed=1.63, q_rel=(0.1, 0.2), qd_rel=(0.3, 0.4),
              precheck_ok=True, release_box_ok=True):
    return {"speed": speed, "q_rel": list(q_rel), "qd_rel": list(qd_rel),
           "precheck_ok": precheck_ok, "release_box_ok": release_box_ok}


def test_finalize_throw_record_normal_path(tmp_path):
    """Happy path: no failure anywhere -- a real record comes back, no
    warning, and it lands on disk exactly once."""
    out_log = str(tmp_path / "hardware_session_log.jsonl")
    measurement = {"x": 0.71, "y": 0.02, "sigma_xy_m": 0.01, "n_frames": 3,
                   "n_inliers": 3, "rms_px": 0.2,
                   "p0": [0.0, 0.0, 0.0], "v0": [1.0, 0.0, 0.0]}

    record, warning = finalize_throw_record(
        landing_xy=[0.71, 0.02], measurement=measurement, exec_stats={"ticks": 40},
        capture_file=str(tmp_path / "throw_000.npz"),
        throw_index=0, target=[0.7, 0.0], plan=_fake_plan(), speed_scale=1.0,
        ball_id="ball-1", out_log=out_log)

    assert warning is None
    assert record["throw_index"] == 0
    assert record["landing_xy"] == [0.71, 0.02]
    assert record["refusal_reason"] is None

    with open(out_log) as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["throw_index"] == 0


def test_append_log_failure_after_a_successful_measurement_still_returns_a_record(
        monkeypatch, tmp_path):
    """
    The scenario the brief asks for directly: step_throw_and_measure already
    returned a clean, successful measurement (the ball landed and was
    tracked) -- but append_log then raises OSError (simulating disk-full).
    The throw must not be lost: finalize_throw_record must still return a
    record (so the caller's state.record_throw keeps the throw counted) and
    a non-None warning to surface in the GUI/log pane.
    """
    def _raise_disk_full(record, log_path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("run_closed_loop_throws.append_log", _raise_disk_full)

    state = SessionState()
    state.record_startup(go=True, failures=[])
    state.camera_ready()

    measurement = {"x": 0.71, "y": 0.02, "sigma_xy_m": 0.01, "n_frames": 3,
                   "n_inliers": 3, "rms_px": 0.2,
                   "p0": [0.0, 0.0, 0.0], "v0": [1.0, 0.0, 0.0]}
    out_log = str(tmp_path / "hardware_session_log.jsonl")

    record, warning = finalize_throw_record(
        landing_xy=[0.71, 0.02], measurement=measurement, exec_stats={"ticks": 40},
        capture_file=str(tmp_path / "throw_000.npz"),
        throw_index=0, target=[0.7, 0.0], plan=_fake_plan(), speed_scale=1.0,
        ball_id="ball-1", out_log=out_log)

    # This is the fix: a record always comes back, never a bare exception.
    assert record is not None
    assert warning is not None and "post-release logging failed" in warning
    assert "disk full" in record["refusal_reason"]
    assert record["throw_index"] == 0
    # The already-measured landing survives into the degraded record too --
    # the failure was in LOGGING, not in the throw/measurement itself.
    assert record["landing_xy"] == [0.71, 0.02]
    assert record["q_release"] == [0.1, 0.2]

    # What _do_throw's normal success path does next -- proving the throw
    # is not silently lost from the in-session dataset (escalation ladder /
    # update-model count) even though nothing could be written to disk.
    state.record_throw(record)
    assert len(state.throws) == 1
    assert state.throws[0] is record
    assert state.n_measured == 1   # landing_xy survived -> still counts as measured


def test_build_throw_record_failure_still_persists_a_degraded_record_to_disk(
        monkeypatch, tmp_path):
    """
    Same invariant, but the failure lives in build_throw_record itself (an
    unexpected bug, not a disk error) -- append_log is untouched, so the
    degraded fallback record's own re-log attempt succeeds and DOES make it
    to the JSONL file, not just SessionState's in-memory list.
    """
    def _raise_build_error(**kw):
        raise TypeError("simulated build_throw_record bug")

    monkeypatch.setattr("run_closed_loop_throws.build_throw_record", _raise_build_error)

    out_log = str(tmp_path / "hardware_session_log.jsonl")

    record, warning = finalize_throw_record(
        landing_xy=[0.71, 0.02], measurement={"x": 0.71, "y": 0.02},
        exec_stats={"ticks": 40}, capture_file=None,
        throw_index=3, target=[0.7, 0.0], plan=_fake_plan(), speed_scale=0.30,
        ball_id="ball-2", out_log=out_log)

    assert warning is not None
    assert "simulated build_throw_record bug" in record["refusal_reason"]
    assert record["throw_index"] == 3
    assert record["q_release"] == [0.1, 0.2]

    assert os.path.isfile(out_log)
    with open(out_log) as f:
        lines = f.readlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["throw_index"] == 3
    assert "simulated build_throw_record bug" in logged["refusal_reason"]


def test_finalize_throw_record_never_raises_even_when_everything_fails(monkeypatch, tmp_path):
    """Belt-and-suspenders: both build_throw_record AND every append_log call
    fail. finalize_throw_record must still return a usable (record, warning)
    pair rather than letting the second failure escape uncaught -- that
    would reopen exactly the hole this fix closes."""
    def _raise_build_error(**kw):
        raise TypeError("simulated build_throw_record bug")

    def _raise_disk_full(record, log_path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("run_closed_loop_throws.build_throw_record", _raise_build_error)
    monkeypatch.setattr("run_closed_loop_throws.append_log", _raise_disk_full)

    record, warning = finalize_throw_record(
        landing_xy=None, measurement={"refusal_reason": "no track"},
        exec_stats={}, capture_file=None,
        throw_index=5, target=[0.7, 0.0], plan=_fake_plan(), speed_scale=0.15,
        ball_id="ball-3", out_log=str(tmp_path / "hardware_session_log.jsonl"))

    assert record is not None
    assert record["throw_index"] == 5
    assert warning is not None
