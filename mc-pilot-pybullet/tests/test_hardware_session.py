"""State-machine and safety-gate tests. No Tk window, arm, or camera is created."""
import threading
import time

import numpy as np
import pytest

from hardware_session import (SessionState, Stage, build_argparser,
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
    s = SessionState(min_throws_for_update=3)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    for _ in range(2):
        s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert not s.can_update_model()
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.can_update_model()


def test_policy_button_is_locked_until_the_model_is_updated():
    s = SessionState(min_throws_for_update=1)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
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
