"""
The hardware throw session: cold rig -> calibrated -> N real throws -> a model
update on that data.

Spec: docs/superpowers/specs/2026-08-31-hardware-session-design.md

Stage 0 runs BEFORE the camera thread starts, because calibration needs colour
at 1920x1080 while tracking needs IR at 848x480/90fps. Sequential, so there is
no stream reconfiguration mid-session and start_of_day.py is reused exactly as
it is, opening and closing the camera itself.

    python3 hardware_session.py --dry_run

NO TASK IN THIS PLAN EVER EXECUTES A REAL THROW. This file wires the planner
and the safety gates -- `ThrowCycle` calls existing, already-gated code
(`run_hardware_throw.py`, `pickup_and_lift.py`, `HardwareThrowExecutor`), it
does not add a new way to move the arm.
"""
from __future__ import annotations

import argparse
import enum
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from hardware_learning import propose_targets, scale_allowed


class Stage(enum.Enum):
    COLD = "cold"
    BLOCKED = "blocked"
    CALIBRATED = "calibrated"
    READY = "ready"
    MODEL_UPDATED = "model_updated"


@dataclass
class SessionState:
    """
    Every safety rule the session enforces lives here, and nowhere else, so it
    can be tested without a window or an arm.
    """
    min_throws_for_update: int = 5
    stage: Stage = Stage.COLD
    blocked_reason: str = ""
    confirmed: bool = False
    throws: List[dict] = field(default_factory=list)
    model_updated: bool = False

    @property
    def n_throws(self):
        return len(self.throws)

    @property
    def n_measured(self):
        return sum(1 for t in self.throws if t.get("landing_xy") is not None)

    @property
    def n_measured_full_speed(self):
        """
        Measured throws that ALSO qualify as data for `fit_release_model`
        (hardware_learning.py): landed AND speed_scale == 1.0. `n_measured`
        keeps its existing meaning (any measured throw, rehearsal or not --
        it is what the throws table and "how many landings do we have"
        displays report, and several existing tests pin that meaning), so
        `can_update_model` gates on this separate property instead of
        redefining `n_measured`. A missing `speed_scale` key does not
        qualify either -- same non-qualifying-by-default rule as
        `fit_release_model` uses, kept consistent on purpose.
        """
        full_speed_tol = 1e-6
        out = 0
        for t in self.throws:
            if t.get("landing_xy") is None:
                continue
            scale = t.get("speed_scale")
            if scale is not None and abs(float(scale) - 1.0) < full_speed_tol:
                out += 1
        return out

    @property
    def logged_scales(self):
        # Built ONLY from this session's own record_throw history -- see the
        # module-level note above. Never seeded from a file on disk: a prior
        # (untrusted) JSONL could otherwise unlock a higher speed_scale than
        # was actually earned in THIS session, and there is deliberately no
        # session-resume path that would need one.
        return [t["speed_scale"] for t in self.throws
                if t.get("landing_xy") is not None]

    def record_startup(self, go, failures):
        self.stage = Stage.CALIBRATED if go else Stage.BLOCKED
        self.blocked_reason = "" if go else "; ".join(failures)

    def camera_ready(self):
        if self.stage is Stage.CALIBRATED:
            self.stage = Stage.READY

    def can_throw(self):
        return self.stage in (Stage.READY, Stage.MODEL_UPDATED)

    def check_scale(self, requested):
        return scale_allowed(requested, self.logged_scales)

    def record_throw(self, record):
        self.throws.append(record)
        self.confirmed = False        # re-affirm every single time

    def can_update_model(self):
        # Gated on full-speed measured throws only -- a session with plenty
        # of rehearsal-only landings must not unlock this button (Defect 1:
        # rehearsals are evidence about the rig, not data for the model).
        return self.n_measured_full_speed >= self.min_throws_for_update

    def record_model_update(self):
        self.model_updated = True
        self.stage = Stage.MODEL_UPDATED

    def can_reoptimize_policy(self):
        return self.model_updated


def run_stage_zero(args):
    """start_of_day.py's own stages, reused. Returns (go, failures, report)."""
    import start_of_day as sod
    rep = sod.Report()
    ok = sod.stage_env(rep)
    ok &= sod.stage_arm(rep, args)
    calib_ok, _ = sod.stage_calibrate(rep, args) if ok else (False, None)
    ok &= calib_ok
    ok &= sod.stage_throw(rep, args)
    return ok, [m for _, lvl, m in rep.rows if lvl == sod.FAIL], rep


def load_mc_model_for_update(log_path, opt_pose=None, seed=1):
    """
    Rebuild the MC_PILOT object a checkpoint was trained with and load its
    trained GP -- the SAME reuse-the-model pattern adapt_policy_height.py
    already establishes for height adaptation (mc.load_model_from_log(...),
    set_eval_mode(), then re-detach pretrain_gp so a later reinforce_model()
    call does not try to backprop through the cached solve a second time).

    Stops right there: no policy re-optimization, no target sampling, no
    particle rollout. `throwing_system` is still constructed, because
    MC_PILOT.__init__ requires one, but it is never rolled out here --
    PyBulletThrowingSystem only connects to PyBullet lazily inside
    .rollout(), which this path never calls, so this needs neither a
    display nor a real arm to run.

    `opt_pose` overrides the table path -- same escape hatch
    adapt_policy_height.py's own --opt_pose provides for a checkpoint that
    predates opt_pose being recorded in config_log. None uses the
    checkpoint's own recorded table. The table's own `tool_offset` stamp
    (not a GUI field) is used to reconstruct the throwing system, so this
    always matches what the checkpoint actually trained against.

    Returns (mc, cfg, model_optimization_opt_list) -- the last one is what
    mc.model_learning.reinforce_model(optimization_opt_list=...) needs; it
    is built with the same optimizer/epoch settings
    train_mc_pilot_pb_arm.py trains with.
    """
    import os
    import pickle as pkl

    import numpy as np
    import torch

    import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
    import model_learning.Model_learning as ML
    import policy_learning.Cost_function as Cost_function
    import policy_learning.MC_PILCO as MC_PILCO_module
    import policy_learning.Policy as Policy
    from robot_arm.robot_profiles import get_robot_profile
    from simulation_class.model_pybullet import PyBulletThrowingSystem

    cfg = pkl.load(open(os.path.join(log_path, "config_log.pkl"), "rb"))
    log = pkl.load(open(os.path.join(log_path, "log.pkl"), "rb"))
    num_trained = len(log["parameters_trial_list"])

    torch.manual_seed(seed)
    dtype, device = torch.float64, torch.device("cpu")

    STATE_DIM, INPUT_DIM, BALL_DIM, TARGET_DIM = 8, 1, 6, 2
    profile = get_robot_profile(cfg["robot_name"])
    RELEASE_POS = np.array(cfg["release_pos"], dtype=float)
    Ts, uM = cfg["Ts"], cfg["uM"]

    table_path = opt_pose or cfg.get("opt_pose")
    if table_path is None:
        raise ValueError(
            f"{log_path}'s config has no 'opt_pose' and none was supplied -- "
            f"cannot reconstruct the throwing system this checkpoint "
            f"trained against")
    table = list(np.load(table_path, allow_pickle=True))
    e0 = min(table, key=lambda e: abs(e["azimuth_deg"]))
    tool_offset = e0.get("tool_offset", [0.0, 0.0, 0.0])

    throwing_system = PyBulletThrowingSystem(
        mass=cfg["ball_mass"], radius=cfg["ball_radius"],
        launch_angle_deg=cfg.get("opt_launch_deg", float(e0["elev_deg"])),
        arm_noise=None, t_w=cfg["T_W"], t_r=cfg["T_R"],
        robot_name=profile.name, target_height=cfg.get("target_height", 0.0),
        base_height=float(cfg.get("base_height", 0.0)),
        opt_posture_table=table, opt_launch_deg=float(e0["elev_deg"]),
        tool_offset=tool_offset,
    )

    init_dict_RBF = {
        "active_dims": np.arange(0, BALL_DIM),
        "lengthscales_init": np.ones(BALL_DIM),
        "flg_train_lengthscales": True,
        "lambda_init": np.ones(1), "flg_train_lambda": False,
        "sigma_n_init": 1 * np.ones(1), "flg_train_sigma_n": True,
        "sigma_n_num": None, "dtype": dtype, "device": device,
    }
    model_learning_par = {
        "num_gp": 3, "T_sampling": Ts, "approximation_mode": "SOD",
        "approximation_dict": {"SOD_threshold_mode": "relative",
                               "SOD_threshold": 0.5,
                               "flg_SOD_permutation": False},
        "init_dict_list": [init_dict_RBF] * 3, "dtype": dtype, "device": device,
    }
    # Warm-start values only -- MC_PILOT.__init__ requires a control policy to
    # construct, but on_update_model never touches it (that is a separate,
    # later-gated button). Loading the trained weights instead of a random
    # init just avoids constructing something nonsensical for no reason.
    st = log["parameters_trial_list"][-1]
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "num_basis": st["centers"].shape[0], "u_max": uM,
        "lengthscales_init": st["log_lengthscales"].exp().numpy()[0],
        "centers_init": st["centers"].numpy(),
        "weight_init": st["f_linear.weight"].numpy(),
        "flg_drop": False, "dtype": dtype, "device": device,
    }
    rand_exploration_policy_par = {
        "full_state_dim": STATE_DIM, "u_max": uM, "u_min": cfg["uMin"],
        "n_strata": cfg["Nexp"], "dtype": dtype, "device": device,
    }
    cost_function_par = {
        "position_indices": [0, 1], "target_indices": [6, 7],
        "lengthscale": cfg["lc"], "dtype": dtype, "device": device,
    }

    mc = MC_PILCO_module.MC_PILOT(
        # Never called: this path runs neither exploration nor reinforce()'s
        # trial loop, the only two callers of target_sampler.
        target_sampler=lambda: np.zeros(TARGET_DIM),
        release_position=RELEASE_POS,
        throwing_system=throwing_system,
        T_sampling=Ts, state_dim=STATE_DIM, input_dim=INPUT_DIM,
        std_meas_noise=1e-3 * np.ones(STATE_DIM),
        f_model_learning=ML.Ballistic_Model_learning_RBF,
        model_learning_par=model_learning_par,
        f_rand_exploration_policy=Policy.Stratified_Throwing_Exploration,
        rand_exploration_policy_par=rand_exploration_policy_par,
        f_control_policy=Policy.Throwing_Policy,
        control_policy_par=control_policy_par,
        f_cost_function=Cost_function.Throwing_Cost,
        cost_function_par=cost_function_par,
        log_path=None, dtype=dtype, device=device,
        target_height=cfg.get("target_height", 0.0),
        Na=cfg.get("Na", 0),
    )

    mc.load_model_from_log(num_trial=num_trained - 1, folder=log_path.rstrip("/") + "/")
    mc.model_learning.set_eval_mode()
    with torch.no_grad():
        for k in range(mc.model_learning.num_gp):
            mc.model_learning.pretrain_gp(k)

    model_optimization_opt_dict = {
        "f_optimizer": "lambda p : torch.optim.Adam(p, lr = 0.01)",
        "criterion": Likelihood.Marginal_log_likelihood,
        "N_epoch": 1001, "N_epoch_print": 500,
    }
    model_optimization_opt_list = [model_optimization_opt_dict] * mc.model_learning.num_gp

    return mc, cfg, model_optimization_opt_list


def raw_ransac_points_from_capture(capture_file, R_bc, t_bc, rig=None):
    """
    Re-run the RAW triangulation + RANSAC stage (perception/trajectory.py) on
    one throw's saved dual-IR recording -- the track_getter callback
    hardware_learning.ingest_throws needs.

    Returns (points_base (N,3), times (N,)): the RAW RANSAC-inlier
    triangulated points, obtained the same way fit_ballistic's own init does
    (`rig.triangulate` per inlier, then `R_bc @ p_c + t_bc`) -- NEVER
    fit_ballistic's resampled (p0, v0) output. See
    hardware_learning.ingest_throws / track_to_state_samples for why feeding
    the fitted parabola back would be a tautology, not evidence.
    """
    import numpy as np

    from measure_landing import build_observations, default_rig
    from perception.ir_capture import load_recording
    from perception.trajectory import ransac_track

    rec = load_recording(capture_file)
    obs, _max_frac = build_observations(rec)
    rig = rig or default_rig()
    inliers, _fit = ransac_track(obs, rig, R_bc, t_bc)
    pts = np.array([R_bc @ rig.triangulate(*obs[i, 1:]) + t_bc for i in inliers])
    return pts, obs[inliers, 0]


def reoptimize_policy(mc, out_dir, reinforce_kwargs):
    """
    Re-optimize the policy against the updated model, reusing the GP -- the
    pattern adapt_policy_height.py already establishes.

    `reinforce_kwargs` is forwarded to `mc.reinforce_policy(**kwargs)`
    UNCHANGED. That call takes ~13 required arguments (T_control,
    num_particles, trial_index, particles_initial_state_mean/var, the three
    init flags and two bounds, opt_steps_list, lr_list, f_optimizer, ...), and
    this function deliberately does not invent, default, or reshape any of
    them: adapt_policy_height.py already owns how that set is built, and a
    second opinion about it here is exactly the kind of duplicated logic this
    repo has been bitten by. The caller assembles them the same way that script
    does.

    Writes a NEW checkpoint directory. The result is NOT thrown automatically:
    its release state may differ from the one whose finger clearance the
    operator visually verified, so it goes back through stage 0 and the
    escalation ladder like any other checkpoint.
    """
    import os
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(
            f"{out_dir} exists and is not empty -- this would overwrite a "
            f"trained checkpoint; pass a new out_dir")
    os.makedirs(out_dir, exist_ok=True)
    mc.reinforce_policy(**reinforce_kwargs)
    return out_dir


class ThrowCycle:
    """
    One throw, as the sequence of gates HARDWARE_RUNBOOK.md Sec 2 describes.
    Each step returns (ok, message) and refuses to advance past a failure.
    """

    def __init__(self, state, camera, args):
        self.state, self.camera, self.args = state, camera, args

    def step_pickup(self):
        from pickup_and_lift import pickup_and_lift
        grasped, pct = pickup_and_lift(self.args.ip, self.args.robot,
                                       self.args.pickup_pose, self.args.lift_z)
        return grasped, (f"grasped a ball ({pct:.1f}% closed)" if grasped else
                         f"CLOSED ON NOTHING ({pct:.1f}%) -- place a ball and retry")

    def step_plan(self, target, speed_scale):
        """
        Exactly the sequence run_closed_loop_throws.main() uses -- same calls,
        same order, so there is one planning path and not two.

        Two verdicts, deliberately kept apart: precheck can pass while the
        release position is outside the safe box, and the GUI must show both.
        """
        import numpy as np
        import run_hardware_throw as H
        from robot_arm.kinova_hardware import HardwareThrowExecutor

        a = self.args
        arm, profile, cid = H.build_arm(a.robot)
        pol, cfg = H.load_policy(a.log_path, None)
        coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = H.plan_throw_for_target(
            arm, profile, cfg, pol, target,
            opt_pose=a.opt_pose, u_cap=a.u_cap, tool_offset_z=a.tool_offset_z,
            wrist_roll_offset=np.deg2rad(a.wrist_roll_offset_deg))
        table = H.load_pose_table(cfg, a.opt_pose)
        box = H.release_box_from_table(
            arm, table, tool_offset=[0.0, 0.0, a.tool_offset_z]) if table else None
        limits = H.make_limits(profile, speed_scale, release_box=box, arm=arm,
                               positioning_scale=a.positioning_scale)
        ex = HardwareThrowExecutor(limits, dry_run=not a.arm, ip=a.ip)

        release_box_ok = ex.check_release_pos(rel)
        precheck_ok, report = ex.precheck(coeffs, arm,
                                          release_speed=float(np.linalg.norm(v_ach)))
        return {"arm": arm, "profile": profile, "cid": cid, "ex": ex,
                "coeffs": coeffs, "q_rel": q_rel, "qd_rel": qd_rel,
                "speed": speed, "rel": rel, "precheck_ok": precheck_ok,
                "report": report, "release_box_ok": release_box_ok}

    def step_throw_and_measure(self, plan, target, speed_scale, throw_index, extrinsic):
        """
        `extrinsic` is the (R, t) pair from `perception.base_frame.load_extrinsic()`,
        already loaded by the CALLER before any physical motion started this cycle
        (see `_do_throw`'s fail-fast load). Do not load it in here: that was the
        original bug (Task 8 review, IMPORTANT 1) -- `load_extrinsic()` raises
        `FileNotFoundError`/`ValueError`, and calling it down here, after the ball
        has already left the hand, meant a bad calibration file could raise past
        every handler and silently drop an already-executed throw from the
        dataset, contradicting `build_throw_record`'s own documented invariant
        that a refused throw is still logged.

        Also guarantees a record-worthy return -- never a bare exception --
        for ANY failure once the ball has physically left the hand, not just
        a measurement failure (Task 8 review IMPORTANT 1's residual note,
        closed here). Release is tracked two ways: `_on_release` (fired
        inside `rehearse_or_throw`'s streaming loop, right after the gripper
        OPEN command) sets a flag before doing anything else that could
        itself raise, and `rehearse_or_throw`'s own `released` return value
        is OR'd in as a second, independent confirmation on the normal-return
        path. If `set_gripper`, `home`, `rehearse_or_throw`, or
        `backend.close_realtime_feedback` raises AFTER that flag is set, this
        method still returns the same (None, {"refusal_reason": ...},
        exec_stats, None) shape the measurement except-clause below returns,
        instead of letting the exception escape to `_do_throw`'s outer
        handler and drop the throw. A failure BEFORE release (nothing
        physical lost yet) still propagates, unchanged from before.

        Returns (landing_xy, measurement, exec_stats, capture_file) -- a
        4-tuple, extended from the original 3-tuple to carry the path of the
        raw dual-IR recording saved to disk (or None if no window was ever
        captured, or the save itself failed). See the capture-save block
        below for why persistence happens BEFORE measure_landing runs.
        """
        import os
        import time

        import numpy as np
        from measure_landing import measure_landing
        from perception.ir_capture import save_recording

        ex, arm, profile = plan["ex"], plan["arm"], plan["profile"]

        released_occurred = False

        def _on_release():
            nonlocal released_occurred
            released_occurred = True   # set BEFORE anything that could itself raise
            self.camera.mark_release(time.time())

        try:
            with ex:
                ex.set_gripper(closed=True)
                ex.home(arm, np.array(profile.q_neutral, float), duration=self.args.duration)
                ex.backend.open_realtime_feedback()
                try:
                    released = ex.rehearse_or_throw(plan["coeffs"], arm, track=None,
                                                    on_release=_on_release)
                    released_occurred = released_occurred or bool(released)
                finally:
                    ex.backend.close_realtime_feedback()
        except Exception as e:
            if released_occurred:
                # The ball is already gone -- same situation the measurement
                # except-clause below handles, just triggered earlier in the
                # cycle. A refusal, never a dropped throw. No capture window
                # was ever obtained on this path, so there is nothing to save.
                exec_stats = dict(getattr(ex, "last_exec_stats", {}) or {})
                return None, {"refusal_reason": str(e)}, exec_stats, None
            raise   # nothing physical happened yet -- unchanged pre-release behavior

        exec_stats = dict(getattr(ex, "last_exec_stats", {}) or {})

        event = self.camera.pop_event(timeout=self.args.measure_timeout)
        if event is None or "error" in event:
            return (None, {"refusal_reason": (event or {}).get("error", "no capture window")},
                    exec_stats, None)

        # Persist the raw dual-IR window to disk BEFORE calling measure_landing,
        # not after. The entire reason measure_landing is split from capture is
        # so an improved fitter or a better camera calibration can be re-run
        # against the SAME raw frames weeks later (HARDWARE_RUNBOOK.md: "keep
        # every recording, it is a permanent regression fixture, not a scratch
        # file"). Saving first means the raw data survives even if
        # measure_landing itself crashes or refuses on this throw -- the
        # alternative (measure first, save after) would get the operator their
        # number a few seconds sooner but risks losing the only copy of a real
        # throw's raw frames to exactly the kind of failure this block exists to
        # survive. A few seconds' delay is cheap; that loss is not.
        #
        # `RingBuffer.window()` (session_camera.py) returns only {"t","ir1",
        # "ir2"} -- save_recording requires a "meta" key too, so it is built
        # here from the actual saved arrays (frame count/dimensions -- these
        # are what make the file self-describing on load) plus the camera's
        # REQUESTED config off `self.args` (fps/exposure/emitter -- "if
        # reachable": a bare test Namespace may omit them, hence getattr with a
        # default rather than a hard attribute error) and the throw index.
        capture_file = None
        try:
            rec = event["rec"]
            n_frames, height, width = rec["ir1"].shape
            meta = {
                "throw_index": int(throw_index),
                "n_frames": int(n_frames),
                "width": int(width),
                "height": int(height),
                "fps": int(getattr(self.args, "camera_fps", 0) or 0),
                "exposure_us": int(getattr(self.args, "exposure_us", 0) or 0),
                "emitter": not bool(getattr(self.args, "no_emitter", False)),
                "t_release": float(event.get("t_release", 0.0)),
            }
            os.makedirs(self.args.throws_dir, exist_ok=True)
            path = os.path.join(self.args.throws_dir, f"throw_{int(throw_index):03d}.npz")
            save_recording(path, {**rec, "meta": meta})
            capture_file = path
        except Exception as e:
            # Saving must never cost a throw. These are ~100 MB uint8 arrays --
            # if the write fails (disk full, permissions, whatever) or is slow
            # enough to raise, treat it exactly like a measurement refusal: a
            # record is still logged, with a refusal_reason naming the save
            # failure, rather than the throw silently vanishing. Measurement is
            # deliberately NOT attempted on data that could not be persisted --
            # this keeps the failure mode simple and matches "treat a save
            # failure like a measurement refusal" (i.e. landing_xy=None,
            # refusal_reason set) rather than a third, partially-successful
            # record shape.
            return None, {"refusal_reason": f"capture save failed: {e}"}, exec_stats, None

        R, t = extrinsic
        try:
            meas = measure_landing(event["rec"], R, t, z_floor=-self.args.base_height,
                                   ball_radius=self.args.ball_radius)
            return [float(meas["x"]), float(meas["y"])], meas, exec_stats, capture_file
        except Exception as e:
            # Broad on purpose, not just RuntimeError: the ball has ALREADY LEFT
            # THE HAND by this point (rehearse_or_throw already ran, above), so
            # any failure past this line -- whatever type it raises -- must still
            # yield a record with landing_xy=None and a refusal_reason, never
            # escape and drop the throw from the dataset. A refusal means
            # re-throw. Never loosen a threshold to force a number. The
            # recording is already safely on disk at this point (capture_file
            # is not None) even though the fit itself failed -- that is exactly
            # the case persistence-before-measurement exists for: a future,
            # improved fitter can still be re-run against this exact file.
            return None, {"refusal_reason": str(e)}, exec_stats, capture_file


# --------------------------------------------------------------------------- #
# Tk dashboard
#
# BINDING DECISIONS (from review) -- do not relitigate these:
#   1. The live view is drawn on the camera thread, never from Tk. Tkinter
#      owns the main thread; CameraThread's on_frame callback renders and
#      calls cv2.imshow/cv2.waitKey itself.
#   2. on_frame is handed ZERO-COPY views into the RealSense SDK's frame
#      buffer, valid only for the duration of the call -- see
#      session_camera.py's FRAME OWNERSHIP CONTRACT. _on_frame below renders
#      synchronously and never stores/queues ir1/ir2.
#   3. SessionState.logged_scales is built only from this session's own
#      record_throw history (enforced above) -- there is no session-resume
#      path, and that is deliberate.
# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--arm", action="store_true",
                    help="talk to the REAL arm during the throw cycle (default: dry-run "
                         "executor). Stage 0's own read-only checks/planner call are "
                         "unaffected by this flag either way.")
    ap.add_argument("--dry_run", action="store_true",
                    help="force args.arm=False -- cannot be overridden by also passing "
                         "--arm. This makes HardwareThrowExecutor use its dry-run "
                         "backend for the throw cycle's PLAN/EXECUTE path: no real "
                         "joint-speed streaming, no real gripper release. It does NOT "
                         "make the whole session inert: step_pickup() always calls "
                         "pickup_and_lift(), which is hardcoded dry_run=False and "
                         "always moves the real arm and grasps for real, independent "
                         "of this flag. Stage 0's own read-only checks/planner call "
                         "are unaffected by this flag either way.")

    g = ap.add_argument_group("board (must match what is physically on the floor)")
    g.add_argument("--squares_x", type=int, default=5)
    g.add_argument("--squares_y", type=int, default=7)
    g.add_argument("--square_mm", type=float, default=35.0,
                   help="MEASURED printed square, not nominal")
    g.add_argument("--marker_ratio", type=float, default=0.75)

    g = ap.add_argument_group("stage-0 cameras (1920x1080 colour, calibration only)")
    g.add_argument("--d435i_width", type=int, default=1920)
    g.add_argument("--d435i_height", type=int, default=1080)
    g.add_argument("--rtsp_url", default=None)
    g.add_argument("--n_frames", type=int, default=5)

    g = ap.add_argument_group("stage-0 gates")
    g.add_argument("--floor_z", type=float, default=None,
                   help="defaults to -base_height (base frame: base at 0, floor "
                        "at -base_height)")
    g.add_argument("--floor_tol_m", type=float, default=0.02)
    g.add_argument("--tilt_tol_deg", type=float, default=5.0)
    g.add_argument("--scale_tol_m", type=float, default=0.05)
    g.add_argument("--repeat_tol_m", type=float, default=0.02)
    g.add_argument("--drift_tol_m", type=float, default=0.03)
    g.add_argument("--drift_tol_deg", type=float, default=5.0)
    g.add_argument("--max_reproj_px", type=float, default=1.0)
    g.add_argument("--min_corners", type=int, default=8)
    g.add_argument("--no_write", action="store_true",
                   help="run stage-0's calibration gates but do not save the extrinsic")

    g = ap.add_argument_group("checkpoint / throw planning")
    g.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    g.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    g.add_argument("--tool_offset_z", type=float, default=0.12)
    g.add_argument("--base_height", type=float, default=0.433)
    g.add_argument("--u_cap", type=float, default=2.00)
    g.add_argument("--target", type=float, nargs=2, default=[0.71, 0.0],
                   help="stage 0's own fixed throw-readiness planner target; the throw "
                        "cycle's per-throw target comes from the GUI fields / the "
                        "auto-proposed spread, not this flag")
    g.add_argument("--plan_speed_scale", type=float, default=1.0)
    g.add_argument("--wrist_roll_offset_deg", type=float, default=90.0,
                   help="OPEN ITEM (see CLAUDE.md): tuned for the OLD checkpoint's 5 deg "
                        "release; the current checkpoint releases at 15 deg elevation and "
                        "this must be re-verified visually on the arm, not assumed safe "
                        "from the numeric precheck alone.")

    g = ap.add_argument_group("throw cycle")
    g.add_argument("--pickup_pose", default="pickup_pose.json")
    g.add_argument("--lift_z", type=float, default=0.10)
    g.add_argument("--duration", type=float, default=4.0)
    g.add_argument("--positioning_scale", type=float, default=1.0)
    g.add_argument("--speed_scale", type=float, default=0.15)
    g.add_argument("--ball_id", default="unassigned")
    g.add_argument("--ball_radius", type=float, default=0.0327)
    g.add_argument("--measure_timeout", type=float, default=30.0)
    g.add_argument("--min_throws_for_update", type=int, default=5)
    g.add_argument("--out_log", default="hardware_session_log.jsonl")
    g.add_argument("--throws_dir", default="throws/",
                   help="directory the raw dual-IR recording of every throw is saved "
                        "to (throw_<index:03d>.npz, same naming as throw_capture.py) "
                        "BEFORE measure_landing runs -- gitignored, ~100MB/throw. This "
                        "is the permanent regression fixture HARDWARE_RUNBOOK.md's rule "
                        "\"keep every recording\" refers to; capture_file in the session "
                        "log points here.")
    g.add_argument("--auto_targets", dest="auto_targets", action="store_true",
                   default=True,
                   help="auto-fill the per-throw target from hardware_learning."
                        "propose_targets, a stratified spread across the trained band "
                        "(default on)")
    g.add_argument("--no_auto_targets", dest="auto_targets", action="store_false",
                   help="use the Target X/Y fields for every throw instead")
    g.add_argument("--n_targets", type=int, default=10)
    g.add_argument("--target_seed", type=int, default=0)

    g = ap.add_argument_group("session camera (848x480 IR, tracking)")
    g.add_argument("--camera_fps", type=int, default=90)
    g.add_argument("--camera_width", type=int, default=848)
    g.add_argument("--camera_height", type=int, default=480)
    g.add_argument("--exposure_us", type=int, default=2000)
    g.add_argument("--no_emitter", action="store_true")
    g.add_argument("--camera_ready_timeout", type=float, default=8.0,
                   help="seconds to wait for the first confirmed frame before giving "
                        "up on the camera and staying CALIBRATED (throwing disabled)")

    return ap


# --------------------------------------------------------------------------- #
# Pure arg-snapshot builders -- deliberately module-level functions, not
# SessionApp methods.
#
# BUG (found 2026-09-01, Step 6 verification): the worker thread used to call
# `self.ip_var.get()` etc. directly. Tkinter variables may only be touched
# from the thread running mainloop -- off that thread `.get()` raises
# `RuntimeError: main thread is not in main loop`, and because the error
# handler itself scheduled a callback via `root.after` (another Tk call, from
# the same bad thread), THAT raised too and the whole failure died as an
# unhandled thread traceback. The operator-facing result: click "Run
# start-of-day", the worker dies silently, the window just sits there -- no
# verdict, no error dialog, for a safety-relevant app.
#
# The fix is a hard boundary: every `*_var.get()` read happens in the button
# handler, on the main thread, BEFORE the worker thread is started. What
# crosses into the worker is a plain dict of already-read strings. These two
# functions turn that dict (+ the base CLI args) into the Namespace each
# worker needs -- they never import tkinter and never reference `self`, so
# they cannot violate the single-thread rule by construction, and are
# unit-testable from any thread with no Tk display at all (see
# tests/test_hardware_session.py).
# --------------------------------------------------------------------------- #
def build_stage_zero_args(base_args, fields):
    """
    `fields` is the plain dict SessionApp._read_shared_fields() returns
    (ip/robot/log_path/opt_pose/tool_offset_z, already read from Tk on the
    main thread). Pure otherwise: touches no Tk object.
    """
    ns = argparse.Namespace(**vars(base_args))
    ns.ip = fields["ip"]
    ns.robot = fields["robot"]
    ns.log_path = fields["log_path"]
    ns.opt_pose = fields["opt_pose"]
    try:
        ns.tool_offset_z = float(fields["tool_offset_z"])
    except ValueError as e:
        raise ValueError(f"bad numeric field 'tool_offset_z': {e}") from e
    if ns.floor_z is None:
        ns.floor_z = -ns.base_height
    return ns


def build_cycle_args(base_args, fields):
    """Same contract as build_stage_zero_args -- see its docstring."""
    ns = argparse.Namespace(**vars(base_args))
    ns.ip = fields["ip"]
    ns.robot = fields["robot"]
    ns.log_path = fields["log_path"]
    ns.opt_pose = fields["opt_pose"]
    try:
        ns.tool_offset_z = float(fields["tool_offset_z"])
    except ValueError as e:
        raise ValueError(f"bad numeric field 'tool_offset_z': {e}") from e
    return ns


class SessionApp:
    """
    Thin Tk shell over SessionState / ThrowCycle -- same shape as
    closed_loop_gui.py (fields, a confirm checkbox that resets every run, a
    scrolling log, worker-thread orchestration). Not unit-tested, matching how
    the rest of this repo treats GUI and hardware code; every safety rule this
    depends on lives in SessionState, which IS tested.

    THREADING RULE (see the module-level note above `build_stage_zero_args`):
    every `self.*_var.get()`/`.set()`/widget `.configure()`/`.cget()` call
    happens on the Tk main thread -- either directly in a button handler, or
    in a `_finish_*` callback reached via `self._safe_after`. A worker thread
    method (`_do_stage_zero`, `_start_camera`, `_do_throw`) receives whatever
    plain values it needs as arguments and must never read `self.*_var` or
    touch a widget itself.
    """

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.state = SessionState(min_throws_for_update=args.min_throws_for_update)
        self.camera = None
        self.throw_index = 0
        self._busy = False
        # Set by on_update_model once "Update model" has run: the SAME
        # in-memory MC_PILOT object, with the real throws already appended
        # to its GP -- not a fresh reload of the checkpoint from disk. The
        # (later-gated) "Re-optimize policy" button needs exactly this
        # object so the newly ingested data actually feeds the new policy.
        self._mc = None
        self.log_q = queue.Queue()
        self.targets = propose_targets(int(args.n_targets), seed=int(args.target_seed))

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_log)
        self._refresh_buttons()
        self._set_status(f"COLD -- run start-of-day to begin", "gray")

    def _on_close(self):
        """
        Wired to WM_DELETE_WINDOW (Task 8 review, IMPORTANT 3). The D435i can
        be opened by exactly one process, and nothing previously ever called
        `CameraThread.stop()` on window close -- a leaked capture thread meant
        the operator had to kill the whole app to free the camera for the
        next run.
        """
        if self.camera is not None:
            print("[hardware_session] window closing -- stopping camera thread")
            stopped = self.camera.stop()
            if not stopped:
                print("[hardware_session] camera thread did not confirm stopped "
                     "within 5s on window close -- the D435i may still be held",
                     file=sys.stderr)
            self.camera = None
        self.root.destroy()

    # -- widget construction ------------------------------------------------ #
    def _field(self, frm, row, label, default):
        import tkinter as tk
        from tkinter import ttk
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(default))
        ttk.Entry(frm, textvariable=var, width=34).grid(row=row, column=1, sticky="w")
        return var

    def _build_widgets(self):
        import tkinter as tk
        from tkinter import ttk

        frm = ttk.Frame(self.root, padding=10)
        frm.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        r = 0
        self.ip_var = self._field(frm, r, "Arm IP", self.args.ip); r += 1
        self.robot_var = self._field(frm, r, "Robot profile", self.args.robot); r += 1
        self.log_path_var = self._field(frm, r, "Checkpoint log_path", self.args.log_path); r += 1
        self.opt_pose_var = self._field(frm, r, "Pose table", self.args.opt_pose); r += 1
        self.tool_offset_var = self._field(
            frm, r, "Tool offset z (m)", self.args.tool_offset_z); r += 1
        self.target_x_var = self._field(frm, r, "Target X (m)", self.args.target[0]); r += 1
        self.target_y_var = self._field(frm, r, "Target Y (m)", self.args.target[1]); r += 1
        self.ball_id_var = self._field(frm, r, "Ball ID", self.args.ball_id); r += 1

        ttk.Label(frm, text="speed_scale").grid(row=r, column=0, sticky="w", pady=2)
        self.speed_scale_var = tk.StringVar(value=str(self.args.speed_scale))
        speeds = ttk.Frame(frm)
        speeds.grid(row=r, column=1, sticky="w")
        for s in ("0.15", "0.30", "0.60", "1.00"):
            ttk.Radiobutton(speeds, text=s, variable=self.speed_scale_var, value=s).pack(side="left")
        r += 1

        self.auto_targets_var = tk.BooleanVar(value=self.args.auto_targets)
        ttk.Checkbutton(
            frm, text=f"Auto-cycle {len(self.targets)} stratified proposed targets "
                     f"(uncheck to use the Target X/Y fields instead)",
            variable=self.auto_targets_var).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        self.confirm_var = tk.BooleanVar(value=False)
        self.confirm_cb = ttk.Checkbutton(
            frm, text="Ball loaded, workspace clear, E-stop in hand",
            variable=self.confirm_var, command=self._on_confirm_toggled)
        self.confirm_cb.grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
        r += 1

        self.stage0_btn = ttk.Button(frm, text="Run start-of-day", command=self.on_run_stage_zero)
        self.stage0_btn.grid(row=r, column=0, sticky="ew", pady=4, padx=(0, 4))
        self.throw_btn = ttk.Button(frm, text="Pick up & throw", command=self.on_throw,
                                    state="disabled")
        self.throw_btn.grid(row=r, column=1, sticky="ew", pady=4)
        r += 1
        self.update_btn = ttk.Button(frm, text="Update model", command=self.on_update_model,
                                     state="disabled")
        self.update_btn.grid(row=r, column=0, sticky="ew", pady=4, padx=(0, 4))
        self.reopt_btn = ttk.Button(frm, text="Re-optimize policy", command=self.on_reoptimize_policy,
                                    state="disabled")
        self.reopt_btn.grid(row=r, column=1, sticky="ew", pady=4)
        r += 1

        self.status = ttk.Label(frm, text="idle", foreground="gray")
        self.status.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        columns = ("idx", "target", "speed_scale", "landing")
        self.table = ttk.Treeview(frm, columns=columns, show="headings", height=6)
        widths = {"idx": 40, "target": 140, "speed_scale": 80, "landing": 260}
        for c in columns:
            self.table.heading(c, text=c)
            self.table.column(c, width=widths[c])
        self.table.grid(row=r, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        r += 1

        self.log = tk.Text(frm, width=112, height=22, state="disabled",
                           bg="black", fg="#c0ffc0", font=("Courier", 10))
        self.log.grid(row=r, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        frm.rowconfigure(r, weight=1)

    # -- log / status helpers ------------------------------------------------ #
    def _append(self, text):
        self.log_q.put(text)

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", line)
                self.log.see("end")
                self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _set_status(self, text, color="black"):
        self.status.configure(text=text, foreground=color)

    def _on_confirm_toggled(self):
        """
        Checkbutton `command=` callback -- fires on the main thread whenever
        the operator clicks the confirm box.

        IMPORTANT 2 fix (Task 8 review): `SessionState.confirmed` is the
        actual safety-gate field (checked in `on_throw`, reset every throw by
        `record_throw` -- see the re-affirm-every-cycle rule). Before this
        fix the checkbox was a self-contained Tk widget `on_throw` read
        directly via `confirm_var.get()`, and `state.confirmed` silently
        tracked its own value with nothing ever consulting it -- the existing
        regression test (`test_confirm_resets_after_every_throw`) exercised a
        field production never read. This callback makes the checkbox the
        write side of `state.confirmed`, so `on_throw`'s check against
        `state.confirmed` (below) is checking the real thing.
        """
        self.state.confirmed = self.confirm_var.get()

    def _set_confirmed(self, value):
        """
        The one place that writes BOTH the visual checkbox and
        `state.confirmed` together -- use this instead of touching
        `confirm_var`/`state.confirmed` separately so the two can never
        drift apart (`confirm_var.set()` does NOT fire `_on_confirm_toggled`,
        since that command only runs for a real user click).
        """
        self.confirm_var.set(value)
        self.state.confirmed = value

    def _read_shared_fields(self):
        """
        Read the ip/robot/checkpoint fields shared between stage 0 and the
        throw cycle. MUST be called on the Tk main thread -- feeds
        build_stage_zero_args / build_cycle_args, which is where the
        corresponding worker thread actually runs. See the threading note on
        the class docstring.
        """
        return {
            "ip": self.ip_var.get(),
            "robot": self.robot_var.get(),
            "log_path": self.log_path_var.get(),
            "opt_pose": self.opt_pose_var.get(),
            "tool_offset_z": self.tool_offset_var.get(),
        }

    def _safe_after(self, fn, context):
        """
        Cross-thread hand-off onto the Tk main thread. `root.after` IS the
        sanctioned way to reach the GUI from a worker thread while mainloop
        is running -- the 2026-09-01 bug was Tk variable reads happening
        directly ON the worker thread (fixed by moving those to the
        main-thread callers, see the module-level note above
        build_stage_zero_args), not this hand-off itself.

        Still wrapped: if scheduling ever fails (window torn down mid-run,
        interpreter not in a state to accept the call), the failure must stay
        visible. Silently losing it would leave an operator staring at an
        idle window with no verdict and no error -- worse than a traceback,
        for a safety-relevant app. Falls back to stderr, which needs no Tk
        object to be reachable.
        """
        try:
            self.root.after(0, fn)
        except Exception as e:
            print(f"[hardware_session] could not schedule GUI update ({context}): {e!r}",
                 file=sys.stderr)

    def _refresh_buttons(self):
        self.throw_btn.configure(state=("normal" if (self.state.can_throw() and not self._busy)
                                        else "disabled"))
        self.update_btn.configure(state=("normal" if self.state.can_update_model() else "disabled"))
        self.reopt_btn.configure(state=("normal" if self.state.can_reoptimize_policy() else "disabled"))

    def _append_throw_row(self, record):
        landing = record.get("landing_xy")
        landing_txt = (f"{landing[0]:+.3f}, {landing[1]:+.3f}" if landing is not None
                       else f"REFUSED: {record.get('refusal_reason')}")
        tgt = record["target"]
        self.table.insert("", "end", values=(
            record["throw_index"], f"{tgt[0]:+.3f}, {tgt[1]:+.3f}",
            record["speed_scale"], landing_txt))

    # -- Run start-of-day ---------------------------------------------------- #
    def on_run_stage_zero(self):
        if self._busy:
            return
        from tkinter import messagebox
        fields = self._read_shared_fields()      # Tk reads happen HERE, main thread only
        try:
            sz_args = build_stage_zero_args(self.args, fields)   # pure -- see its docstring
        except ValueError as e:
            messagebox.showerror("Bad input", str(e))
            return
        self._busy = True
        self.stage0_btn.configure(state="disabled")
        self._refresh_buttons()
        self._set_status("running start-of-day checks (arm read-only, planner only) ...",
                         "orange")
        self._append("\n$ start-of-day\n")
        threading.Thread(target=self._do_stage_zero, args=(sz_args,), daemon=True).start()

    def _do_stage_zero(self, sz_args):
        """
        Worker thread. `sz_args` was already built on the main thread by
        on_run_stage_zero -- this function must never read `self.*_var` or
        touch a widget; only self._safe_after() may reach back into the GUI.
        """
        try:
            go, failures, rep = run_stage_zero(sz_args)
        except Exception as e:                     # never leave the session hung
            self._safe_after(lambda: self._finish_stage_zero_error(e), "stage-zero error result")
            return
        self._safe_after(lambda: self._finish_stage_zero(go, failures, rep), "stage-zero result")

    def _finish_stage_zero(self, go, failures, rep):
        for stage, level, msg in rep.rows:
            self._append(f"[{level:5s}][{stage}] {msg}\n")
        self.state.record_startup(go=go, failures=failures)
        if go:
            self._append("\n=== GO -- calibrated, planned, and gated ===\n")
            if self.camera is not None:
                # A camera thread from an earlier "Run start-of-day" is still
                # live -- the D435i can be opened by exactly one process, so
                # starting a second CameraThread here would race it for the
                # device (Task 8 review, IMPORTANT 3). record_startup() above
                # just reset stage back to CALIBRATED unconditionally; restore
                # READY immediately since this camera was already confirmed.
                self._append("=== camera already live from a previous run -- "
                             "not starting a second CameraThread ===\n")
                self.state.camera_ready()
                self._busy = False
                self.stage0_btn.configure(state="normal")
                self._set_status("READY -- calibrated + camera live (from earlier run)",
                                 "green")
            else:
                self._set_status("GO -- starting session camera ...", "orange")
                # _busy stays True (and stage0_btn stays disabled) through camera
                # startup -- only cleared in _finish_camera now, not here. This is
                # the IMPORTANT 3 fix: _busy used to clear right here, before the
                # camera thread even started, so a second click on "Run
                # start-of-day" could open a second CameraThread while the first
                # still held the only D435i the process may open.
                threading.Thread(target=self._start_camera, daemon=True).start()
        else:
            self._busy = False
            self.stage0_btn.configure(state="normal")
            self._append(f"\n=== NO-GO -- BLOCKED: {self.state.blocked_reason} ===\n")
            self._set_status(f"NO-GO / BLOCKED: {self.state.blocked_reason}", "red")
        self._refresh_buttons()

    def _finish_stage_zero_error(self, exc):
        self.state.record_startup(go=False, failures=[str(exc)])
        self._busy = False
        self.stage0_btn.configure(state="normal")
        self._append(f"\n=== start-of-day raised: {exc!r} ===\n")
        self._set_status(f"NO-GO / BLOCKED: {exc}", "red")
        self._refresh_buttons()

    # -- camera --------------------------------------------------------------- #
    def _start_camera(self):
        import cv2
        from session_camera import CameraThread
        from session_overlay import render_overlay
        # Imported ONCE here, not per-frame in _on_frame (which runs at up to
        # ~90 Hz) -- stashed as instance attributes for _on_frame to use. See
        # the Task 8 review's MINOR note.
        self._cv2 = cv2
        self._render_overlay = render_overlay

        cam = CameraThread(seconds=3.0, fps=self.args.camera_fps,
                           width=self.args.camera_width, height=self.args.camera_height,
                           exposure_us=self.args.exposure_us,
                           emitter=not self.args.no_emitter, on_frame=self._on_frame)
        cam.start()
        # Visible immediately (not only once confirmed) so a window-close or
        # the second-thread guard in _finish_stage_zero can find and stop this
        # thread even while it is still waiting to confirm its first frame.
        self.camera = cam
        deadline = time.time() + self.args.camera_ready_timeout
        confirmed = False
        while time.time() < deadline:
            if cam.error is not None:
                break
            if len(cam.buf) > 0:
                confirmed = True
                break
            time.sleep(0.05)
        if not confirmed:
            # Startup failed (timeout or camera fault) -- the D435i can only be
            # opened by one process at a time, so a failed CameraThread MUST be
            # stopped here, before the operator is allowed to retry "Run
            # start-of-day" (Task 8 review, IMPORTANT 3): otherwise the retry's
            # new CameraThread would race this one for the device. stop()'s
            # return value is the actual confirmation the camera is free --
            # never discard it.
            stopped = cam.stop()
            self.camera = None
            if not stopped:
                print("[hardware_session] camera thread did not confirm stopped "
                     "within 5s after a failed startup -- the D435i may still be "
                     "held; restarting the app may be required", file=sys.stderr)
        self._safe_after(lambda: self._finish_camera(confirmed, cam.error), "camera-ready result")

    def _finish_camera(self, confirmed, error):
        self._busy = False
        self.stage0_btn.configure(state="normal")
        if confirmed:
            self.state.camera_ready()
            self._append("\n=== camera confirmed live -- session READY ===\n")
            self._set_status("READY -- calibrated + camera live", "green")
        else:
            self._append(f"\n=== camera did not confirm a frame within "
                         f"{self.args.camera_ready_timeout:.0f}s"
                         f"{': ' + repr(error) if error else ''} -- staying CALIBRATED, "
                         f"throwing stays disabled ===\n")
            self._set_status("CALIBRATED but camera not confirmed -- throwing disabled", "red")
        self._refresh_buttons()

    def _on_frame(self, ts, ir1, ir2):
        """
        Runs INLINE on the camera thread, at capture rate (up to ~90 Hz).
        `ir1`/`ir2` are ZERO-COPY views into the RealSense SDK's own frame
        buffer, valid only for this call (see session_camera.py's FRAME
        OWNERSHIP CONTRACT) -- render synchronously, never store or queue
        them. Tk is never touched from here. `self._cv2`/`self._render_overlay`
        are imported once in `_start_camera`, not per-call here -- see that
        method's comment.

        Reads self.state.stage/n_throws below -- on the camera thread, while
        the main thread concurrently mutates SessionState (record_throw() etc).
        GIL-safe (no torn reads of these plain attributes/properties), but a
        real cross-thread access nonetheless; this codebase documents such
        races explicitly elsewhere rather than leaving them implicit -- see
        session_camera.py's RingBuffer: "Not thread-safe; the owner locks".
        """
        try:
            frame = self._render_overlay(
                ir1, ir2, status=f"stage={self.state.stage.value} throws={self.state.n_throws}")
            self._cv2.imshow("session -- live IR", frame)
            self._cv2.waitKey(1)
        except Exception:
            pass   # a display hiccup must never kill the capture thread

    # -- Pick up & throw ------------------------------------------------------ #
    def on_throw(self):
        if self._busy:
            return
        from tkinter import messagebox
        if not self.state.can_throw():
            messagebox.showwarning("Not ready", "Session is not READY -- run start-of-day "
                                                "(and confirm the camera came up) first.")
            return
        if not self.state.confirmed:
            messagebox.showwarning(
                "Not confirmed",
                "Check \"Ball loaded, workspace clear, E-stop in hand\" first -- "
                "this resets after every throw on purpose.")
            return
        try:
            speed_scale = float(self.speed_scale_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "speed_scale must be a number.")
            return
        ok, why = self.state.check_scale(speed_scale)
        if not ok:
            messagebox.showwarning("Escalation ladder", why)
            return

        if self.auto_targets_var.get():
            target = [float(x) for x in self.targets[self.throw_index % len(self.targets)]]
        else:
            try:
                target = [float(self.target_x_var.get()), float(self.target_y_var.get())]
            except ValueError:
                messagebox.showerror("Bad input", "Target X/Y must be numbers.")
                return

        fields = self._read_shared_fields()      # Tk reads happen HERE, main thread only
        ball_id = self.ball_id_var.get()          # ditto
        try:
            cycle_args = build_cycle_args(self.args, fields)   # pure -- see its docstring
        except ValueError as e:
            messagebox.showerror("Bad input", str(e))
            return

        self._set_confirmed(False)   # re-affirm required every cycle, not just once --
                                      # resets the checkbox AND state.confirmed together
        self._busy = True
        self.throw_btn.configure(state="disabled")
        self._set_status(f"pickup -> plan -> throw (target={tuple(round(t,3) for t in target)}, "
                         f"speed_scale={speed_scale}) ...", "orange")
        self._append(f"\n$ throw {self.throw_index}  target={target}  "
                     f"speed_scale={speed_scale}\n")
        throw_index = self.throw_index   # snapshot -- the worker must not depend on
                                         # self.throw_index still meaning the same thing
                                         # if this method runs again before it finishes
        threading.Thread(target=self._do_throw,
                         args=(cycle_args, ball_id, target, speed_scale, throw_index),
                         daemon=True).start()

    def _do_throw(self, cycle_args, ball_id, target, speed_scale, throw_index):
        """
        Worker thread. `cycle_args`/`ball_id` were already read from Tk on the
        main thread by on_throw -- this function, and everything it calls
        (ThrowCycle, run_closed_loop_throws, pybullet), must never read
        `self.*_var` or touch a widget; only self._safe_after() may reach
        back into the GUI.
        """
        plan = None
        try:
            cycle = ThrowCycle(self.state, self.camera, cycle_args)

            # Fail fast, BEFORE any physical motion: a missing/invalid
            # calibration file must refuse HERE, where refusing costs nothing,
            # not after the ball has left the hand (Task 8 review, IMPORTANT 1).
            # Loading it once here and threading it through step_throw_and_measure
            # is the fix -- see that method's docstring for why it must never
            # load its own extrinsic again.
            from perception import base_frame
            extrinsic = base_frame.load_extrinsic()

            grasped, msg = cycle.step_pickup()
            self._safe_after(lambda: self._append(f"[pickup] {msg}\n"), "pickup log line")
            if not grasped:
                self._safe_after(lambda: self._finish_throw_refused(msg), "pickup refusal")
                return

            plan = cycle.step_plan(target, speed_scale)
            self._safe_after(lambda: self._append(
                f"[plan] release pos in safe box: {plan['release_box_ok']}\n"
                f"{plan['report']}\nPRECHECK: {'PASS' if plan['precheck_ok'] else 'FAIL'}\n"),
                "plan log line")
            if not plan["precheck_ok"] or not plan["release_box_ok"]:
                why = ("precheck failed" if not plan["precheck_ok"] else "") + \
                      (" and " if not plan["precheck_ok"] and not plan["release_box_ok"] else "") + \
                      ("release position outside the safe box" if not plan["release_box_ok"] else "")
                self._safe_after(lambda: self._finish_throw_refused(f"REFUSE: {why}"),
                                 "plan refusal")
                return

            landing_xy, measurement, exec_stats, capture_file = cycle.step_throw_and_measure(
                plan, target, speed_scale, throw_index, extrinsic)

            from run_closed_loop_throws import append_log, build_throw_record
            record = build_throw_record(
                throw_index=throw_index, target=target,
                commanded_speed=plan["speed"], speed_scale=speed_scale,
                q_release=plan["q_rel"], qd_release=plan["qd_rel"],
                precheck_ok=plan["precheck_ok"], exec_stats=exec_stats,
                ball_id=ball_id, capture_file=capture_file,
                landing_xy=landing_xy, measurement=measurement,
                release_in_box=plan["release_box_ok"])
            append_log(record, cycle_args.out_log)
        except Exception as e:
            self._safe_after(lambda: self._finish_throw_error(e), "throw error result")
            return
        finally:
            if plan is not None:
                try:
                    import pybullet as p
                    p.disconnect(plan["cid"])
                except Exception:
                    pass
        self._safe_after(lambda: self._finish_throw_ok(record), "throw result")

    def _finish_throw_refused(self, why):
        self._set_confirmed(False)   # belt-and-suspenders: on_throw already reset this
                                      # before dispatching, but re-assert it here in case
                                      # the operator re-checked the box while the worker
                                      # was running (state.confirmed is the authority the
                                      # NEXT on_throw() call reads -- see IMPORTANT 2).
        self._busy = False
        self._append(f"\n=== throw refused: {why} ===\n")
        self._set_status(f"REFUSED: {why}", "red")
        self._refresh_buttons()

    def _finish_throw_error(self, exc):
        self._set_confirmed(False)   # see _finish_throw_refused
        self._busy = False
        self._append(f"\n=== throw cycle raised: {exc!r} ===\n")
        self._set_status(f"ERROR: {exc}", "red")
        self._refresh_buttons()

    def _finish_throw_ok(self, record):
        from tkinter import messagebox
        self._set_confirmed(False)   # see _finish_throw_refused; record_throw() below
                                      # also resets state.confirmed, this keeps the
                                      # visible checkbox in sync with it too
        self.state.record_throw(record)
        self.throw_index += 1
        self._busy = False
        self._append_throw_row(record)
        landing = record.get("landing_xy")
        if landing is None:
            self._append(f"\n=== throw {record['throw_index']} logged -- MEASUREMENT "
                         f"REFUSED: {record.get('refusal_reason')} ===\n")
            self._set_status("logged -- measurement refused, re-throw", "orange")
        else:
            self._append(f"\n=== throw {record['throw_index']} logged -- landing "
                         f"{tuple(round(v, 3) for v in landing)} ===\n")
            self._set_status("logged -- clean run", "green")
        self._refresh_buttons()
        messagebox.showinfo("Reload", "Place the next ball at the pickup pose, then "
                                      "re-check confirm before the next throw.")

    # -- Update model (Re-optimize policy is gated here, wired in a later task) #
    def on_update_model(self):
        if self._busy:
            return
        from tkinter import messagebox
        if not self.state.can_update_model():
            messagebox.showwarning(
                "Not enough data",
                f"Need {self.state.min_throws_for_update} measured throws logged "
                "at FULL SPEED (speed_scale == 1.0) before the model can be "
                "updated -- see the gate in SessionState.can_update_model. "
                "Rehearsal-speed landings do not count (see "
                "n_measured_full_speed).")
            return

        fields = self._read_shared_fields()      # Tk reads happen HERE, main thread only
        throws = list(self.state.throws)         # snapshot -- the worker must not
                                                  # depend on self.state.throws still
                                                  # meaning the same thing if the
                                                  # operator logs another throw
                                                  # before this finishes
        self._busy = True
        self.update_btn.configure(state="disabled")
        self._set_status("loading checkpoint and re-analyzing real throws ...", "orange")
        self._append("\n$ update model\n")
        threading.Thread(target=self._do_update_model, args=(fields, throws), daemon=True).start()

    def _do_update_model(self, fields, throws):
        """
        Worker thread. `fields`/`throws` were already read/snapshotted on the
        main thread by on_update_model -- this function, and everything it
        calls, must never touch `self.*_var` or a widget; only
        `self._safe_after()` may reach back into the GUI.

        Loads the trained checkpoint's GP (load_mc_model_for_update, the
        adapt_policy_height.py pattern), re-triangulates each qualifying real
        throw's RAW RANSAC-inlier points from its stored recording, appends
        them to the GP via hardware_learning.ingest_throws, reinforces the
        GP on the combined data, and reports. Deliberately never touches
        mc.control_policy or calls reinforce_policy -- policy re-optimization
        is a separate, later-gated button (on_reoptimize_policy), so the
        operator sees what the real throws did to the MODEL before the
        policy moves.

        A throw's recording can only be re-triangulated if it was saved to
        disk (`record["capture_file"]` set) AND a camera extrinsic exists on
        disk right now. As of the capture-persistence fix in
        step_throw_and_measure, a normal live throw DOES get a real
        capture_file (saved before measure_landing runs, in --throws_dir) --
        the remaining gap is just the extrinsic: `capture_file` can still be
        None for a throw whose own capture save failed (see that method's
        "saving must never cost a throw" comment), and no camera extrinsic
        exists on disk yet as of this writing (see CLAUDE.md). Both failure
        modes are refusals, not crashes: ingest_throws reports the affected
        throws as skipped rather than raising, and the release-model fit
        (built from each throw's already-computed measured_v0, not a
        re-triangulation) is unaffected either way.
        """
        try:
            from hardware_learning import ingest_throws
            from perception import base_frame

            mc, cfg, opt_list = load_mc_model_for_update(
                fields["log_path"], opt_pose=(fields["opt_pose"] or None))

            try:
                R_bc, t_bc = base_frame.load_extrinsic()
                extrinsic_note = ""
            except (FileNotFoundError, ValueError) as e:
                R_bc = t_bc = None
                extrinsic_note = (
                    f"\nNOTE: no camera extrinsic on disk ({e}) -- the flight "
                    f"GP cannot re-triangulate any recording without one, so "
                    f"every throw with a capture_file is reported as skipped "
                    f"above; the release model (fit from each throw's "
                    f"already-measured v0, not a re-triangulation) is "
                    f"unaffected.")

            def track_getter(record):
                path = record.get("capture_file")
                if not path or R_bc is None:
                    return None
                try:
                    return raw_ransac_points_from_capture(path, R_bc, t_bc)
                except (RuntimeError, ValueError, OSError) as e:
                    return None

            out = ingest_throws(mc, throws, track_getter, na=int(cfg.get("Na", 0)))
            mc.model_learning.reinforce_model(optimization_opt_list=opt_list)
        except Exception as e:
            self._safe_after(lambda: self._finish_update_model_error(e),
                             "update-model error result")
            return
        self._safe_after(lambda: self._finish_update_model_ok(mc, out, extrinsic_note),
                         "update-model result")

    def _finish_update_model_ok(self, mc, out, extrinsic_note):
        self._busy = False
        self._mc = mc   # for the (later-gated) Re-optimize policy button
        self._append(f"\n=== model update ===\n{out['text']}{extrinsic_note}\n")
        self.state.record_model_update()
        self._set_status("MODEL UPDATED -- see report above", "green")
        self._refresh_buttons()

    def _finish_update_model_error(self, exc):
        self._busy = False
        self._append(f"\n=== model update raised: {exc!r} ===\n")
        self._set_status(f"ERROR: {exc}", "red")
        self._refresh_buttons()

    def on_reoptimize_policy(self):
        """
        Button 2 (Task 10). Re-optimizes the SAME in-memory `self._mc` that
        `on_update_model` left behind -- its GP already has the real throws
        folded in, which is the entire point of gating this button on
        `can_reoptimize_policy()` (== `model_updated`): the operator sees what
        the real throws did to the model before the policy moves.

        `_do_reoptimize_policy` does the heavy work (torch, particle rollout)
        off the main thread, same discipline as `on_update_model`/`on_throw`:
        Tk reads happen here, only `_safe_after` may reach back into the GUI
        from the worker.
        """
        if self._busy:
            return
        from tkinter import messagebox
        if not self.state.can_reoptimize_policy():
            messagebox.showwarning(
                "Model not updated yet",
                "Run 'Update model' first -- SessionState.can_reoptimize_policy() "
                "is false until the model has been updated against real "
                "throws (see the gate in on_update_model).")
            return
        if self._mc is None:
            # Should be unreachable if can_reoptimize_policy() is true (both
            # are set together in _finish_update_model_ok), but a stashed
            # object going missing must not silently no-op or crash the
            # worker thread with a confusing AttributeError.
            messagebox.showerror(
                "No updated model in memory",
                "can_reoptimize_policy() is true but self._mc is None -- this "
                "should not happen. Re-run 'Update model' before retrying.")
            return

        fields = self._read_shared_fields()      # Tk reads happen HERE, main thread only
        self._busy = True
        self.reopt_btn.configure(state="disabled")
        self._set_status("re-optimizing policy against the updated model ...", "orange")
        self._append("\n$ re-optimize policy\n")
        threading.Thread(target=self._do_reoptimize_policy, args=(fields,),
                         daemon=True).start()

    def _do_reoptimize_policy(self, fields):
        """
        Worker thread. `fields` was already read on the main thread by
        `on_reoptimize_policy` -- this function must never touch `self.*_var`
        or a widget; only `self._safe_after()` may reach back into the GUI.

        Re-optimizes `self._mc.control_policy` in place against the model
        `on_update_model` already updated with real throws -- NOT a fresh
        reload of the checkpoint from disk, so the newly ingested data
        actually feeds the new policy (see the comment on `self._mc` in
        `__init__`).

        Builds the `reinforce_policy()` argument set the same way
        `adapt_policy_height.py` does (T_control, num_particles, trial_index,
        particle-init mean/var over a target domain, opt_steps_list/lr_list
        indexed by trial, the dropout/convergence knobs, `policy_reinit_dict`)
        -- see that script around line 223 for the proven call this mirrors.
        Unlike that script this is not a height adaptation: there is no new
        basket height, so the target domain (`lm`/`lM`/`gM`) and control
        horizon (`T`) are read back from the checkpoint's OWN config_log.pkl
        unchanged, not re-derived from a new flight band.

        `reoptimize_policy()` (module-level, tested in
        tests/test_hardware_session.py) owns the one load-bearing safety
        property -- refuse to write into an existing, non-empty directory --
        so a trained checkpoint can never be overwritten. This function then
        persists config_log.pkl/log.pkl into that same new directory the way
        adapt_policy_height.py does at its own tail: without that, "a NEW
        checkpoint directory" would be an empty folder nothing downstream
        (run_hardware_throw.py, load_mc_model_for_update, a future session)
        could actually load.
        """
        import os
        import pickle as pkl

        import numpy as np
        import torch

        try:
            mc = self._mc
            log_path_in = fields["log_path"]
            cfg = pkl.load(open(os.path.join(log_path_in, "config_log.pkl"), "rb"))
            log = pkl.load(open(os.path.join(log_path_in, "log.pkl"), "rb"))
            num_trained = len(log["parameters_trial_list"])

            dtype, device = torch.float64, torch.device("cpu")
            RELEASE_POS = np.array(cfg["release_pos"], dtype=float)
            release_xy = RELEASE_POS[:2]
            Ts, T, M, uM = cfg["Ts"], cfg["T"], cfg["M"], cfg["uM"]
            lm, lM = cfg["lm"], cfg["lM"]

            centre = np.array([release_xy[0] + 0.5 * (lm + lM), release_xy[1]])
            initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre])
            initial_state_var = np.concatenate(
                [1e-4 * np.ones(6), (0.5 * (lM - lm)) ** 2 * np.ones(2)])

            reinforce_kwargs = dict(
                T_control=int(round(T / Ts)),
                num_particles=M,
                trial_index=num_trained - 1,
                particles_initial_state_mean=torch.tensor(
                    initial_state, dtype=dtype, device=device),
                particles_initial_state_var=torch.tensor(
                    initial_state_var, dtype=dtype, device=device),
                flg_particles_init_uniform=False,
                particles_init_up_bound=None, particles_init_low_bound=None,
                flg_particles_init_multi_gauss=False,
                # reinforce_policy indexes these by trial_index
                opt_steps_list=[cfg.get("Nopt", 1500)] * num_trained,
                lr_list=[0.01] * num_trained,
                f_optimizer="lambda p, lr : torch.optim.Adam(p, lr)",
                num_step_print=100, p_dropout_list=[0.25] * num_trained,
                p_drop_reduction=0.25 / 2,
                alpha_diff_cost=0.99, min_diff_cost=0.02, num_min_diff_cost=400,
                min_step=400, lr_min=0.0025,
                policy_reinit_dict={
                    "lenghtscales_par": np.array(cfg["lengthscales_init"]),
                    "centers_par": np.array([1.0, 1.0]), "weight_par": uM},
            )

            # A fresh, timestamped sibling of the checkpoint's own
            # results_root -- reoptimize_policy() below is what actually
            # refuses a collision, this just makes one vanishingly unlikely
            # in the first place.
            results_root = f"{cfg['results_root']}_reopt_{time.strftime('%Y%m%d_%H%M%S')}"
            out_dir = os.path.join(results_root, str(cfg.get("seed", 1)))

            reoptimize_policy(mc, out_dir, reinforce_kwargs)

            adapted = dict(cfg)
            adapted.update({"results_root": results_root,
                            "reoptimized_from": log_path_in, "new_trials_used": 0})
            pkl.dump(adapted, open(os.path.join(out_dir, "config_log.pkl"), "wb"))
            out_log = {"parameters_trial_list": [mc.control_policy.state_dict()],
                       "cost_trial_list": [],
                       "reoptimized_from": log_path_in, "new_trials_used": 0}
            pkl.dump(out_log, open(os.path.join(out_dir, "log.pkl"), "wb"))
        except Exception as e:
            self._safe_after(lambda: self._finish_reoptimize_policy_error(e),
                             "reoptimize-policy error result")
            return
        self._safe_after(lambda: self._finish_reoptimize_policy_ok(out_dir),
                         "reoptimize-policy result")

    def _finish_reoptimize_policy_ok(self, out_dir):
        self._busy = False
        text = (f"New checkpoint written to `{out_dir}`. It has NOT been "
               f"validated on hardware — re-run start-of-day against it "
               f"and restart the escalation ladder at 0.15.")
        self._append(f"\n=== policy re-optimization ===\n{text}\n")
        self._set_status(f"NEW CHECKPOINT -- unvalidated, restart at 0.15: {out_dir}", "green")
        self._refresh_buttons()

    def _finish_reoptimize_policy_error(self, exc):
        self._busy = False
        self._append(f"\n=== policy re-optimization raised: {exc!r} ===\n")
        self._set_status(f"ERROR: {exc}", "red")
        self._refresh_buttons()


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.floor_z is None:
        args.floor_z = -args.base_height
    if args.dry_run:
        args.arm = False   # explicit and cannot be overridden by also passing --arm

    import tkinter as tk
    root = tk.Tk()
    root.title("Kinova Gen3 -- hardware throw session")
    SessionApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
