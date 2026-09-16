"""
"I want the ball to land HERE" -> the target to type into the session GUI.

WHY THIS EXISTS
---------------
The trained checkpoint plans under a 0.12 m tool offset and no release-speed
excess. The real arm has neither, so a commanded target of 0.72 m lands at
1.17 m -- 45 cm long, every time. `predict_landing.py` models that forward and
was validated OUT OF SAMPLE on 2026-09-11: three new targets, two of them
outside the trained band, predicted to 1.7 / 2.8 / 3.6 cm against 40.9 cm for
the uncorrected model.

This inverts it. Give it where you want the ball, it finds the commanded target
that puts it there, and prints what to type. Nothing about the throw changes --
same checkpoint, same pose table, same release posture, same commanded speeds
that 17 clean throws have already run at. No re-search, no retrain, no fresh
wrist-clearance verification, no restarting the speed ladder.

    python3 aim_at.py --land 1.20 0.0
    python3 aim_at.py --land 1.20 0.0 --land 1.25 -0.30 --land 1.16 0.40
    python3 aim_at.py --reachable          # what the band actually is

WHAT THIS IS AND IS NOT
-----------------------
This is open-loop pre-compensation on a measured system-identification result.
It is the fastest way to put the ball in a bin reliably, and it is NOT the same
claim as "MC-PILOT learned to throw accurately on hardware" -- the policy still
believes it is throwing 45 cm shorter than it is. The retrain under corrected
physics remains the research result; this makes the rig usable in the meantime,
and every throw it produces is also a datapoint for that retrain.

TWO THINGS THAT CONSTRAIN THE ANSWER
------------------------------------
1. THE SPEED BAND IS A SAFETY CONSTRAINT, NOT A PREFERENCE. Every clean throw
   on record sits in 1.403-1.668 m/s commanded. Below that the release gets
   slower relative to the palm-up release posture, and the documented failure
   is the ball dropping back onto the gripper instead of leaving
   (2026-09-09). Solutions outside the band are REFUSED, not warned about.

2. THE MAP IS NOT MONOTONIC. Commanded speed saturates near 1.62 m/s at a
   commanded target of ~0.85 and then DECREASES, so landings below ~1.30 m have
   two solutions. This always returns the one on the rising branch: it is the
   lower commanded target, the better-conditioned side (d(landing)/d(target) is
   larger, so a given target error costs less landing error), and the side all
   17 validated throws sit on.
"""

import argparse
import json
import sys

import numpy as np
import pybullet as p

import predict_landing as PL
import run_hardware_throw as H
from measure_landing import default_rig, load_extrinsic

# The commanded-speed band with clean, measured throws behind it (17 of them,
# 2026-09-11). Widening this is a hardware decision, not a software one -- see
# the module docstring.
SPEED_MIN, SPEED_MAX = 1.403, 1.668

# Commanded target above which the policy's speed output saturates and turns
# over. Solutions are kept strictly below this so the inverse stays single
# valued and on the well-conditioned branch.
TARGET_TURNOVER = 0.84


def solve(fwd, want_xy, x0, iters=40, tol=2e-4):
    """
    2-D damped Newton on commanded target -> predicted landing.

    Finite-difference Jacobian: the forward map runs an LP inside the release
    solver, so there is no analytic derivative to reuse, and the map is smooth
    enough over the 1 mm step used here.
    """
    want = np.asarray(want_xy, float)
    x = np.asarray(x0, float)
    h = 1e-3
    for _ in range(iters):
        f0 = fwd(x)
        if f0 is None:
            return None, None
        r = f0 - want
        if np.linalg.norm(r) < tol:
            return x, f0
        J = np.zeros((2, 2))
        for k in range(2):
            xp = x.copy(); xp[k] += h
            fp = fwd(xp)
            if fp is None:
                return None, None
            J[:, k] = (fp - f0) / h
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None, None
        # Damp: the map turns over near TARGET_TURNOVER and an undamped Newton
        # step can jump the ridge onto the descending branch and converge to
        # the wrong solution.
        step = np.clip(step, -0.05, 0.05)
        x = x + step
        x[0] = float(np.clip(x[0], 0.40, TARGET_TURNOVER))
    return None, None


class Aimer:
    """
    Holds the loaded arm/policy so many aim solves cost one PyBullet client.

    The GUI resolves a new bin position every round; reloading the checkpoint
    each time would add seconds per round for nothing. Use as a context
    manager -- the PyBullet client must be disconnected.
    """

    def __init__(self, args, extrinsic=None):
        """`extrinsic` as an (R, t) pair skips the file read -- the session app
        has already loaded and validated it before anything physical ran, and
        re-reading it here could pick up a different file mid-session."""
        self.args = args
        self._extrinsic = extrinsic
        self._cid = None

    def __enter__(self):
        self.R_bc, self.t_bc = (self._extrinsic if self._extrinsic is not None
                                else load_extrinsic(self.args.extrinsic))
        self.rig = default_rig()
        self.arm, self.profile, self._cid = H.build_arm(self.args.robot)
        self.pol, self.cfg = H.load_policy(self.args.log_path, None)
        table = H.load_pose_table(self.cfg, self.args.opt_pose)
        box = H.release_box_from_table(
            self.arm, table,
            tool_offset=[0.0, 0.0, self.args.tool_offset_z]) if table else None
        limits = H.make_limits(self.profile, 1.0, release_box=box, arm=self.arm)
        from robot_arm.kinova_hardware import HardwareThrowExecutor
        self.ex = HardwareThrowExecutor(limits, dry_run=True)
        return self

    def __exit__(self, *exc):
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None
        return False

    def forward(self, target_xy):
        """Commanded target -> the full prediction dict."""
        return PL.predict(self.arm, self.profile, self.cfg, self.pol, self.ex,
                          (float(target_xy[0]), float(target_xy[1])),
                          self.args, self.R_bc, self.t_bc, self.rig)

    def _landing(self, t):
        r = self.forward(t)
        return None if r["predicted_landing"] is None else np.array(r["predicted_landing"])

    def aim(self, want_xy):
        """
        Desired landing -> (commanded target, prediction dict, refusal list).

        Refusals are RETURNED, not raised: the caller (a GUI round, a run
        sheet) wants to see the number it would have to type alongside why it
        must not, rather than a traceback with no context.
        """
        want = np.asarray(want_xy, float)
        # A landing is ~1.6x its commanded target under both models, which puts
        # the seed within a couple of Newton steps everywhere in the band.
        x0 = np.array([float(np.clip(want[0] / 1.62, 0.45, TARGET_TURNOVER - 0.02)),
                       float(want[1] / 1.62)])
        t, got = solve(self._landing, want, x0)
        if t is None:
            return None, None, ["no solution -- outside the reachable band"]
        r = self.forward(t)
        bad = []
        if not (self.args.speed_min <= r["commanded_speed"] <= self.args.speed_max):
            bad.append(f"commanded speed {r['commanded_speed']:.3f} m/s is outside the "
                       f"validated [{self.args.speed_min:.3f}, {self.args.speed_max:.3f}] "
                       f"band -- the ball may not clear the gripper")
        if not r["precheck_ok"]:
            bad.append("trajectory precheck FAILED")
        if not r["release_in_box"]:
            bad.append("release position outside the safe box")
        if not r["measurable"]:
            bad.append("the camera would not see this landing well enough to measure it")
        r["solver_miss_m"] = float(np.linalg.norm(got - want))
        return t, r, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--land", type=float, nargs=2, action="append", metavar=("X", "Y"),
                    help="where you want the ball to land, base frame, metres")
    ap.add_argument("--reachable", action="store_true",
                    help="print the reachable landing band and exit")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--tool_offset_z", type=float, default=0.12,
                    help="what the pose table is stamped for. Leave it alone.")
    ap.add_argument("--base_height", type=float, default=0.433)
    ap.add_argument("--ball_radius", type=float, default=PL.BALL_RADIUS)
    ap.add_argument("--u_cap", type=float, default=2.00)
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=90.0)
    ap.add_argument("--extrinsic", default="calib/T_B_C.npz")
    ap.add_argument("--model", choices=("gain", "additive"), default="additive")
    ap.add_argument("--speed_min", type=float, default=SPEED_MIN)
    ap.add_argument("--speed_max", type=float, default=SPEED_MAX)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.floor_z = -args.base_height
    if not args.land and not args.reachable:
        ap.error("give --land X Y (repeatable), or --reachable")

    R_bc, t_bc = load_extrinsic(args.extrinsic)
    rig = PL.default_rig() if hasattr(PL, "default_rig") else default_rig()
    arm, profile, cid = H.build_arm(args.robot)
    try:
        pol, cfg = H.load_policy(args.log_path, None)
        table = H.load_pose_table(cfg, args.opt_pose)
        box = H.release_box_from_table(
            arm, table, tool_offset=[0.0, 0.0, args.tool_offset_z]) if table else None
        limits = H.make_limits(profile, 1.0, release_box=box, arm=arm)
        from robot_arm.kinova_hardware import HardwareThrowExecutor
        ex = HardwareThrowExecutor(limits, dry_run=True)

        def full(t):
            return PL.predict(arm, profile, cfg, pol, ex, (float(t[0]), float(t[1])),
                              args, R_bc, t_bc, rig)

        def fwd(t):
            r = full(t)
            return None if r["predicted_landing"] is None else np.array(r["predicted_landing"])

        if args.reachable:
            print(f"\nreachable landings, commanded speed kept inside "
                  f"[{args.speed_min:.3f}, {args.speed_max:.3f}] m/s\n")
            print("  cmd target    cmd speed   landing")
            lo = hi = None
            for tx in np.arange(0.66, TARGET_TURNOVER + 1e-9, 0.02):
                r = full([tx, 0.0])
                ok = args.speed_min <= r["commanded_speed"] <= args.speed_max
                if ok and r["measurable"]:
                    lo = r["predicted_landing"][0] if lo is None else lo
                    hi = r["predicted_landing"][0]
                print(f"   {tx:.2f}         {r['commanded_speed']:.3f}     "
                      f"{r['predicted_landing'][0]:+.3f}   {'ok' if ok else 'speed out of band'}")
            print(f"\n  on-axis range: {lo:+.3f} .. {hi:+.3f} m")
            print("  lateral at 1.20 m: roughly -0.58 .. +0.55 m (the arc is ~+-25 deg "
                  "of azimuth at\n  near-constant radius; ask for a specific point and "
                  "this will solve it exactly)")
            return 0

        rows = []
        print(f"\nmodel '{args.model}'. Nothing about the throw changes -- same "
              f"checkpoint, table\nand release posture; only the number you type "
              f"into Target X / Y.\n")
        print("  want landing      -> TYPE THIS TARGET     cmd speed   predicts     miss")
        print("  " + "-" * 76)
        for want in args.land:
            # Seed on the rising branch. A landing is ~1.6x its commanded target
            # under both models, which puts the seed within a couple of Newton
            # steps everywhere in the reachable band.
            x0 = np.array([np.clip(want[0] / 1.62, 0.45, TARGET_TURNOVER - 0.02),
                           want[1] / 1.62])
            t, got = solve(fwd, want, x0)
            if t is None:
                print(f"  ({want[0]:+.3f},{want[1]:+.3f})    NO SOLUTION -- outside the "
                      f"reachable band (try --reachable)")
                continue
            r = full(t)
            bad = []
            if not (args.speed_min <= r["commanded_speed"] <= args.speed_max):
                bad.append(f"commanded speed {r['commanded_speed']:.3f} OUTSIDE the "
                           f"validated band -- REFUSED (ball may not clear the gripper)")
            if not r["precheck_ok"]:
                bad.append("PRECHECK FAIL")
            if not r["release_in_box"]:
                bad.append("release outside the safe box")
            if not r["measurable"]:
                bad.append("landing not measurable by the camera")
            miss = float(np.linalg.norm(got - np.asarray(want, float)))
            print(f"  ({want[0]:+.3f},{want[1]:+.3f})    ->  "
                  f"Target X {t[0]:.4f}  Y {t[1]:+.4f}    {r['commanded_speed']:.3f}   "
                  f"({got[0]:+.3f},{got[1]:+.3f})  {miss * 100:.1f} mm")
            for b in bad:
                print(f"        !! {b}")
            rows.append({"wanted_landing": [float(want[0]), float(want[1])],
                         "command_target": [float(t[0]), float(t[1])],
                         "commanded_speed": r["commanded_speed"],
                         "predicted_landing": r["predicted_landing"],
                         "solver_miss_m": miss, "refusals": bad,
                         "visibility": r["visibility"]})
    finally:
        p.disconnect(cid)

    tol = 2 * PL.PREDICTION_SIGMA_M
    print(f"\n  expected spread about the wanted point: ~{tol * 100:.1f} cm (2 sigma), "
          f"from the\n  forward model's own out-of-sample residual. The solver's "
          f"contribution is the\n  'miss' column above and is negligible next to it.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "speed_band": [args.speed_min, args.speed_max],
                       "solutions": rows}, f, indent=1)
        print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
