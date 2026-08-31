"""
Unit tests for the pure logic in run_closed_loop_throws.py -- the per-throw
status dashboard text and the structured log record it appends. The CLI glue
itself (argparse -> run_hardware_throw.plan_throw_for_target /
HardwareThrowExecutor) is not re-tested here: those primitives already have
their own coverage in test_hardware_planner.py, and this file only adds new
behavior (formatting + logging), per this project's own testing convention of
one regression per specific defect rather than re-deriving lower-level checks.
"""
import json
import time

import numpy as np
import pytest

import run_closed_loop_throws as C


def test_dashboard_shows_pass_and_release_box_ok_distinctly():
    """HARDWARE_RUNBOOK.md's own warning: PRECHECK PASS and 'release pos in
    safe box' are SEPARATE lines and must never be collapsed into one -- a
    plan can PASS precheck while the release point is outside the box."""
    text = C.format_status_dashboard(
        target=(0.75, 0.05), speed=1.498, speed_scale=1.0,
        precheck_ok=True, precheck_report="peak |tau| 41.4%",
        release_box_ok=True, release_pos=(0.04, 0.02, 1.52),
    )
    assert "PRECHECK: PASS" in text
    assert "release pos in safe box: True" in text
    assert "peak |tau| 41.4%" in text


def test_dashboard_flags_a_release_box_failure_even_when_precheck_passes():
    text = C.format_status_dashboard(
        target=(0.75, 0.05), speed=1.498, speed_scale=1.0,
        precheck_ok=True, precheck_report="peak |tau| 41.4%",
        release_box_ok=False, release_pos=(0.04, 0.02, 1.52),
    )
    assert "PRECHECK: PASS" in text
    assert "release pos in safe box: False" in text
    assert "REFUSE" in text.upper()


def test_dashboard_flags_precheck_failure_prominently():
    text = C.format_status_dashboard(
        target=(0.75, 0.05), speed=1.498, speed_scale=1.0,
        precheck_ok=False, precheck_report="TORQUE violation on joint 3",
        release_box_ok=True, release_pos=(0.04, 0.02, 1.52),
    )
    assert "PRECHECK: FAIL" in text
    assert "REFUSE" in text.upper()


def test_build_throw_record_has_every_field_the_runbook_requires():
    """HARDWARE_RUNBOOK.md Sec 4: 'Record per throw (this is the dataset, not
    a debug log)': target, commanded release speed, speed_scale, landing
    (x,y) once measured, exec stats (ticks/achieved Hz/worst_late_ms/
    release_wall_s), video/recording file id, ball ID."""
    rec = C.build_throw_record(
        throw_index=3, target=(0.75, 0.05), commanded_speed=1.498,
        speed_scale=1.0, q_release=[0.1] * 7, qd_release=[0.0] * 7,
        precheck_ok=True, exec_stats={"ticks": 342, "achieved_hz": 40.0,
                                      "worst_late_ms": 0.0, "release_wall_s": 4.93},
        ball_id="tennis-01", capture_file=None, landing_xy=None,
    )
    for key in ("throw_index", "timestamp", "target", "commanded_speed",
               "speed_scale", "precheck_ok", "exec_stats", "ball_id",
               "capture_file", "landing_xy"):
        assert key in rec, f"missing required field: {key}"
    assert rec["target"] == [0.75, 0.05]
    assert rec["landing_xy"] is None, (
        "must record landing_xy=None when unmeasured -- decoupled offline "
        "measurement (measure_landing.py) fills it in later, never guessed here"
    )


def test_append_log_writes_one_json_line_per_throw(tmp_path):
    log_path = tmp_path / "throw_log.jsonl"
    rec1 = C.build_throw_record(
        throw_index=0, target=(0.7, 0.0), commanded_speed=1.4, speed_scale=1.0,
        q_release=[0.0] * 7, qd_release=[0.0] * 7, precheck_ok=True,
        exec_stats={}, ball_id="b1", capture_file=None, landing_xy=None,
    )
    rec2 = C.build_throw_record(
        throw_index=1, target=(0.7, 0.1), commanded_speed=1.35, speed_scale=1.0,
        q_release=[0.0] * 7, qd_release=[0.0] * 7, precheck_ok=True,
        exec_stats={}, ball_id="b1", capture_file=None, landing_xy=None,
    )
    C.append_log(rec1, log_path)
    C.append_log(rec2, log_path)
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["throw_index"] == 0
    assert json.loads(lines[1])["throw_index"] == 1


def test_append_log_survives_numpy_types(tmp_path):
    """q_release/qd_release/exec_stats often carry numpy scalars/arrays --
    json.dump must not choke on them (a previous version of this project has
    a documented history of exactly this class of silent-crash-at-the-worst-
    moment bug)."""
    log_path = tmp_path / "throw_log.jsonl"
    rec = C.build_throw_record(
        throw_index=0, target=np.array([0.7, 0.0]), commanded_speed=np.float64(1.4),
        speed_scale=1.0, q_release=np.zeros(7), qd_release=np.zeros(7),
        precheck_ok=True, exec_stats={"worst_late_ms": np.float64(0.0)},
        ball_id="b1", capture_file=None, landing_xy=None,
    )
    C.append_log(rec, log_path)  # must not raise
    loaded = json.loads(log_path.read_text().strip())
    assert loaded["target"] == [0.7, 0.0]


# --------------------------------------------------------------------------- #
# Optional --measure: an OPT-IN convenience on top of this script's default
# decoupled design (landing_xy stays None unless asked). throw_capture.py is
# a separate, independently-armed process -- these helpers only poll the
# filesystem for what it already wrote, never start/stop/arm it.
# --------------------------------------------------------------------------- #
def test_newest_recording_after_ignores_stale_files(tmp_path):
    old = tmp_path / "throw_000.npz"
    old.write_bytes(b"x")
    t_cutoff = time.time()
    time.sleep(0.05)
    new = tmp_path / "throw_001.npz"
    new.write_bytes(b"y")
    assert C._newest_recording_after(str(tmp_path), t_cutoff) == str(new)


def test_newest_recording_after_returns_none_when_nothing_new(tmp_path):
    (tmp_path / "throw_000.npz").write_bytes(b"x")
    assert C._newest_recording_after(str(tmp_path), time.time() + 1.0) is None


def test_newest_recording_after_missing_dir_returns_none():
    assert C._newest_recording_after("/nonexistent/dir/xyz", 0.0) is None


def test_load_extrinsic_any_reads_calibration_json(tmp_path):
    """calibrate_camera_extrinsics.py writes R_B_C/t_B_C in a .json;
    measure_landing.py's own load_extrinsic wants R/t in a .npz. No
    converter existed anywhere in the repo before this."""
    R = np.eye(3).tolist()
    t = [1.1, 0.09, -0.43]
    p = tmp_path / "camera_extrinsics.json"
    p.write_text(json.dumps({"R_B_C": R, "t_B_C": t, "note": "laptop rig"}))
    got_R, got_t = C.load_extrinsic_any(str(p))
    assert np.allclose(got_R, R)
    assert np.allclose(got_t, t)


def test_load_extrinsic_any_refuses_a_non_orthonormal_rotation(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"R_B_C": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "t_B_C": [0, 0, 0]}))
    with pytest.raises(ValueError, match="orthonormal"):
        C.load_extrinsic_any(str(p))


def test_load_extrinsic_any_reads_npz(tmp_path):
    p = tmp_path / "extrinsic.npz"
    np.savez(p, R=np.eye(3), t=np.array([0.1, 0.2, 0.3]))
    got_R, got_t = C.load_extrinsic_any(str(p))
    assert np.allclose(got_R, np.eye(3))
    assert np.allclose(got_t, [0.1, 0.2, 0.3])


CKPT = "results_kinetic_chain_gen3/1"
TABLE = "throw_pose_table.npy"

pytestmark_ckpt = pytest.mark.skipif(
    not __import__("os").path.exists(TABLE), reason="trained pose table not present"
)


@pytestmark_ckpt
def test_wrist_roll_offset_deg_threads_through_the_cli(capsys, tmp_path):
    """--wrist_roll_offset_deg exists on throw_gui.py and run_hardware_throw.py;
    it must exist here too or a GUI/CLI call passing it fails at argparse."""
    rc = C.main([
        "--log_path", CKPT, "--opt_pose", TABLE,
        "--target", "0.72", "0.0", "--throw_index", "0",
        "--wrist_roll_offset_deg", "45",
        "--out_log", str(tmp_path / "log.jsonl"),
    ])
    out = capsys.readouterr().out
    assert rc in (0, 2)  # 0 = precheck passed dry-run; 2 = precheck refused -- either is a clean run, not a crash
    assert "CLOSED-LOOP THROW STATUS" in out


def test_build_throw_record_accepts_a_real_measurement():
    """--measure fills capture_file/landing_xy in; the default (unmeasured)
    case above must stay None -- this only proves a real value round-trips."""
    rec = C.build_throw_record(
        throw_index=0, target=(0.72, 0.0), commanded_speed=1.4, speed_scale=1.0,
        q_release=[0.0] * 7, qd_release=[0.0] * 7, precheck_ok=True, exec_stats={},
        ball_id="b1", capture_file="throws/throw_003.npz", landing_xy=(0.715, 0.008),
    )
    assert rec["capture_file"] == "throws/throw_003.npz"
    assert rec["landing_xy"] == (0.715, 0.008)


def test_build_throw_record_carries_the_measured_release_state():
    """measured_v0 vs commanded speed is the whole release model -- it must be logged."""
    meas = {"x": 0.71, "y": 0.02, "t_impact": 0.51, "sigma_xy_m": 0.018,
            "n_frames": 44, "n_inliers": 40, "rms_px": 0.42,
            "p0": np.array([0.30, 0.0, 0.02]), "v0": np.array([1.39, 0.0, 0.37]),
            "max_mask_frac": 0.03}
    r = C.build_throw_record(0, [0.71, 0.0], 1.44, 0.15, [0.0] * 7, [0.0] * 7,
                             True, {}, "tennis-01", "throws/throw_000.npz",
                             [0.71, 0.02], measurement=meas, release_in_box=True)
    assert r["landing_xy"] == [0.71, 0.02]
    assert r["measured_v0"] == [1.39, 0.0, 0.37]
    assert r["measured_p0"] == [0.30, 0.0, 0.02]
    assert r["sigma_xy_m"] == 0.018
    assert r["n_inliers"] == 40
    assert r["refusal_reason"] is None
    assert r["release_in_box"] is True


def test_build_throw_record_records_a_refusal_without_a_landing():
    """A refused track is still a logged throw -- it is not silently dropped."""
    r = C.build_throw_record(3, [0.71, 0.0], 1.44, 0.15, [0.0] * 7, [0.0] * 7,
                             True, {}, "tennis-01", "throws/throw_003.npz", None,
                             measurement={"refusal_reason": "inlier fraction 0.49"})
    assert r["landing_xy"] is None
    assert "0.49" in r["refusal_reason"]
    assert r["measured_v0"] is None


def test_build_throw_record_is_backward_compatible():
    """The old 11-positional-arg call must keep working unchanged."""
    r = C.build_throw_record(0, [0.75, 0.05], 1.5, 0.15, [0.0] * 7, [0.0] * 7,
                             True, {}, "b", "c.npz", None)
    assert r["landing_xy"] is None and r["measured_v0"] is None
