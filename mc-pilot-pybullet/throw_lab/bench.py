"""
Benchmark: shipped piecewise-cubic throw vs jerk-limited / time-optimal /
whip-optimized alternatives, on the real Gen3 torque model.

    /usr/bin/python3 -m throw_lab.bench --quick
    /usr/bin/python3 -m throw_lab.bench --speeds 1.2 1.4 1.6 --azimuths -30 0 30
    /usr/bin/python3 -m throw_lab.bench --whip          # + the dynopt allocation
    /usr/bin/python3 -m throw_lab.bench --jitter        # + release-timing sweep

Everything is measured through `ThrowHarness`, i.e. through the same
computed-torque controller and the same 50 Hz PyBullet world the shipped
trainer uses, with the release state produced by the shipped
`OptimizedReleaseSolver` off the shipped pose table.  The ONLY thing that
varies between rows is the trajectory between neutral and the release state.

Metrics, and why each is here:
  speed%      |v_ball at release| / |J(q_rel) qd_rel|.  Velocity fidelity.
  dir         angle between the two.  A degree at 0.8 m is 1.4 cm of miss.
  |w|         EE angular velocity at release.  Not a trajectory-quality metric
              on its own -- it is the multiplier on the 162.8 mm Robotiq TCP
              offset (see CLAUDE.md's TCP blocker), so a profile that raises it
              makes that separate, larger error worse.
  landErr     |landing - landing the plan would have produced if tracked
              perfectly|, both integrated by the same PyBullet free flight.
  dv/step     change in release speed per one 20 ms shift of the release
              instant.  Directly buys down the real arm's 67.9 +- 6.4 ms
              gripper latency and its 25 ms command quantum.
  sat%        fraction of joint-steps whose commanded torque hit tau_max.
  t_r         time from rest to release.  Cycle time, and on hardware, the
              window over which open-loop drift accumulates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_class.release_solver import OptimizedReleaseSolver  # noqa: E402

from throw_lab import dynopt, shapes as sh  # noqa: E402
from throw_lab.harness import BaselinePlanAdapter, ThrowHarness  # noqa: E402
from throw_lab.planner import LabThrowPlanner  # noqa: E402

# (label, throw_shape_name, kwargs, windup_shape_name, windup_kwargs)
DEFAULT_PROFILES = [
    ("lab_const_accel", "const_accel", {}, "cubic", {}),
    ("lab_minjerk", "min_jerk", {}, "min_jerk", {}),
    ("lab_trapS_b25", "trap_accel", {"beta": 0.25}, "trap_vel", {"beta": 0.25}),
    ("lab_trapS_b10", "trap_accel", {"beta": 0.10}, "trap_vel", {"beta": 0.10}),
    ("lab_plateau_f15", "plateau", {"beta": 0.25, "frac": 0.15}, "trap_vel",
     {"beta": 0.10}),
]


def load_table(path):
    tbl = np.load(path, allow_pickle=True)
    return list(tbl)


def make_release(solver, arm, speed, azimuth_deg, dist):
    az = np.deg2rad(azimuth_deg)
    target_xy = np.array([dist * np.cos(az), dist * np.sin(az)])
    v_cmd = np.array([speed, 0.0, 0.0])
    rp, q_rel, qd_rel, v_rel = solver.solve(arm, v_cmd, target_xy=target_xy)
    return rp, np.asarray(q_rel), np.asarray(qd_rel), np.asarray(v_rel), target_xy


def legacy_release_state(H, v_rel, rp, args):
    """IK + pseudoinverse release state, via the shipped planner.

    `plan_throw` computes q_release/qd_release long before it plans the
    follow-through, but it RAISES on an infeasible follow-through, so the only
    way to read the release state back out is to give it a horizon it can
    actually finish.  Its follow-through scan tries the caller's own window
    first and then an ABSOLUTE ladder of candidates greater than it, so which
    rungs get tried depends on the horizon passed in -- the shipped code's own
    comment records T=2.2 succeeding where T=2.0 failed for the same release.
    Measured here on legacy states at 0.8-1.0 m/s: "velocity 1.00x of limit",
    i.e. it refuses by a rounding margin.  Walk a few horizons rather than
    re-deriving IK + pinv in a second place.

    Returns (q_release, qd_release, horizon_that_worked) or None.
    """
    base = args.t_w + args.dt_throw + args.brake + args.ret
    for extra in (0.0, 0.2, 0.45, 0.8, 1.4, 2.2, 3.5):
        try:
            _c, q_rel, qd_rel, _v = H.arm.plan_throw(
                v_rel, rp, args.t_w, args.t_w + args.dt_throw, base + extra,
                monotonic_windup=False,
            )
        except RuntimeError:
            continue
        return np.asarray(q_rel), np.asarray(qd_rel), base + extra
    return None


def evaluate(H, plan, label, jitter=False):
    """One plan -> ThrowResult + landing error against its own ideal flight."""
    res = H.run(plan, label=label)
    ideal_xy, _ = H.free_flight(res.release_pos_planned, res.v_planned)
    res.extra["ideal_xy"] = ideal_xy
    res.extra["land_err"] = (
        float(np.linalg.norm(res.land_xy - ideal_xy))
        if np.all(np.isfinite(res.land_xy)) and np.all(np.isfinite(ideal_xy))
        else float("nan")
    )
    if jitter:
        speeds = {}
        lands = {}
        for b in (-1, 1):
            rb = H.run(plan, label=label, release_bias_steps=b)
            speeds[b] = rb.speed_release
            lands[b] = rb.land_xy
        res.extra["dv_per_step"] = 0.5 * abs(speeds[1] - speeds[-1])
        if all(np.all(np.isfinite(lands[b])) for b in (-1, 1)):
            res.extra["dland_per_step"] = 0.5 * float(
                np.linalg.norm(lands[1] - lands[-1])
            )
        else:
            res.extra["dland_per_step"] = float("nan")
    return res


def row(res):
    e = res.extra
    return (
        f"  {res.label:<18} "
        f"{100 * res.speed_ratio:6.2f}%  "
        f"{res.dir_err_deg:5.2f}d  "
        f"{np.linalg.norm(res.omega_release):5.2f}  "
        f"{100 * e.get('land_err', float('nan')):6.2f}cm  "
        f"{100 * e.get('dv_per_step', float('nan')):6.2f}cm/s "
        f"{100 * e.get('dland_per_step', float('nan')):6.2f}cm "
        f"{1000 * float(np.max(np.abs(res.q_err_at_release))):6.1f}mrad "
        f"{100 * res.tau_sat_frac:5.1f}%  "
        f"{res.peak_tau_ratio:5.2f}  "
        f"{res.t_r:6.2f}s "
        f"{res.peak_jerk:7.1f} "
        f"{res.accel_step:6.2f}"
    )


HEADER = (
    "  profile             speed%   dir    |w|   landErr  dv/step  dLand/st"
    " qErrRel    sat%  tauPk    t_r    jerk  dQddJoin"
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--table", default="throw_pose_table.npy")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.2, 1.5])
    ap.add_argument("--azimuths", type=float, nargs="+", default=[-30.0, 0.0, 30.0])
    ap.add_argument("--dist", type=float, default=0.70,
                    help="target distance used only to set the aim azimuth")
    ap.add_argument("--t_w", type=float, default=0.5)
    ap.add_argument("--dt_throw", type=float, default=1.1)
    ap.add_argument("--brake", type=float, default=0.4)
    ap.add_argument("--ret", type=float, default=0.6)
    ap.add_argument("--release_mode", choices=("table", "legacy"), default="table",
                    help="table = opt_pose kinetic-chain release (hardware path); "
                         "legacy = IK + Jacobian-pseudoinverse release, the "
                         "torque-bound regime measure_tracking_error.py sampled")
    ap.add_argument("--ball_mass", type=float, default=0.0577)
    ap.add_argument("--ball_radius", type=float, default=0.0327)
    ap.add_argument("--base_height", type=float, default=0.0)
    ap.add_argument("--jerk_max", type=float, default=None,
                    help="per-joint jerk limit [rad/s^3]; off by default")
    ap.add_argument("--quick", action="store_true",
                    help="one release state only")
    ap.add_argument("--jitter", action="store_true",
                    help="also measure release-timing sensitivity (3x runtime)")
    ap.add_argument("--whip", action="store_true",
                    help="also run the dynopt time-allocation optimizer")
    ap.add_argument("--whip_min_steps", type=int, default=20,
                    help="control-bandwidth floor on the optimized throw window, "
                         "in 50 Hz control steps (see dynopt.min_feasible_duration)")
    ap.add_argument("--payload_in_check", type=int, default=1,
                    help="include the gripped ball in the planner's torque check")
    ap.add_argument("--out", default="throw_lab/results/bench.npz")
    args = ap.parse_args(argv)

    if args.quick:
        args.speeds = args.speeds[-1:]
        args.azimuths = [0.0]

    table = load_table(args.table)
    t0 = time.time()
    rows = []
    shipped_failures = []

    with ThrowHarness(
        robot_name=args.robot,
        base_height=args.base_height,
        ball_mass=args.ball_mass,
        ball_radius=args.ball_radius,
    ) as H:
        solver = OptimizedReleaseSolver(opt_posture_table=table)
        planner = LabThrowPlanner(
            H.arm,
            payload_mass=args.ball_mass if args.payload_in_check else 0.0,
            jerk_max=(None if args.jerk_max is None
                      else np.full(len(H.arm._qd_max), args.jerk_max)),
        )

        for speed in args.speeds:
            for az in args.azimuths:
                if args.release_mode == "table":
                    rp, q_rel, qd_rel, v_rel, tgt = make_release(
                        solver, H.arm, speed, az, args.dist
                    )
                    ovr = dict(q_release_override=q_rel,
                               qd_release_override=qd_rel,
                               monotonic_windup=True)
                    horizon = args.t_w + args.dt_throw + args.brake + args.ret
                else:
                    # Legacy IK + pinv release, exactly what
                    # measure_tracking_error.py exercises: the release state
                    # comes OUT of plan_throw rather than going in.
                    az_r = np.deg2rad(az)
                    rp = np.array(H.arm._profile.default_release_pos, dtype=float)
                    alpha = np.deg2rad(35.0)
                    v_rel = speed * np.array([
                        np.cos(alpha) * np.cos(az_r),
                        np.cos(alpha) * np.sin(az_r),
                        np.sin(alpha),
                    ])
                    ovr = dict(monotonic_windup=False)
                    got = legacy_release_state(H, v_rel, rp, args)
                    if got is None:
                        shipped_failures.append(
                            (speed, az, "plan_throw could not produce a release "
                                        "state at any horizon")
                        )
                        print("  SKIPPED: in legacy mode the release state itself "
                              "comes out of plan_throw, so a refusal skips the "
                              "whole state -- the lab profiles are NOT evaluated "
                              "here and no claim is made about them")
                        continue
                    q_rel, qd_rel, horizon = got
                print(
                    f"\n=== release state [{args.release_mode}]: speed "
                    f"{speed:.2f} m/s, azimuth {az:+.0f} deg  "
                    f"(|qd_rel| {np.linalg.norm(qd_rel):.3f} rad/s)"
                )
                print(HEADER)

                # ---- shipped baseline -----------------------------------
                # `plan_throw` can REFUSE a release state the lab planner
                # handles: its single-cubic follow-through has no a-priori
                # velocity bound, so the candidate-duration scan can come back
                # with "velocity 1.00x of limit" and raise.  Observed on legacy
                # release states at 0.8-1.0 m/s.  Record it instead of crashing
                # the sweep -- it is a result, not an accident.
                try:
                    coeffs, qr, qdr, _ = H.arm.plan_throw(
                        v_rel, rp, args.t_w, args.t_w + args.dt_throw,
                        horizon if args.release_mode == "legacy"
                        else args.t_w + args.dt_throw + args.brake + args.ret,
                        **ovr
                    )
                except RuntimeError as exc:
                    shipped_failures.append((speed, az, str(exc)))
                    print(f"  {'SHIPPED_cubic':<18} REFUSED BY plan_throw: {exc}")
                else:
                    base = BaselinePlanAdapter(H.arm, coeffs, qr, qdr)
                    res = evaluate(H, base, "SHIPPED_cubic", jitter=args.jitter)
                    res.extra.update(speed=speed, azimuth=az, kind="shipped")
                    rows.append(res)
                    print(row(res))

                # ---- lab profiles ---------------------------------------
                for label, tname, tkw, wname, wkw in DEFAULT_PROFILES:
                    try:
                        plan = planner.plan(
                            q_rel, qd_rel,
                            t_w=args.t_w, dt_throw=args.dt_throw,
                            brake_dur=args.brake, return_dur=args.ret,
                            throw_shape=tname, shape_kw=tkw,
                            windup_shape=wname, windup_kw=wkw,
                            label=label,
                        )
                    except RuntimeError as exc:
                        print(f"  {label:<18} INFEASIBLE: {exc}")
                        continue
                    res = evaluate(H, plan, label, jitter=args.jitter)
                    res.extra.update(speed=speed, azimuth=az, kind="lab",
                                     plan_summary=plan.summary())
                    rows.append(res)
                    print(row(res))

                # ---- whip: optimized per-joint time allocation ------------
                if args.whip:
                    shape = sh.get_velocity_shape("trap_accel", beta=0.25)
                    alloc = dynopt.optimize_allocation(
                        planner.chk, q_rel, qd_rel, shape, n_samples=48,
                        min_dur=args.whip_min_steps * 0.02,
                    )
                    print("  " + dynopt.report(alloc))
                    plan = planner.plan(
                        q_rel, qd_rel,
                        t_w=args.t_w, dt_throw=alloc["dt_throw"],
                        brake_dur=args.brake, return_dur=args.ret,
                        throw_shape=shape, windup_shape="trap_vel",
                        windup_kw={"beta": 0.10},
                        stagger=alloc["stagger"], local_dur=alloc["local_dur"],
                        label="lab_whip_opt",
                    )
                    res = evaluate(H, plan, "lab_whip_mintime", jitter=args.jitter)
                    res.extra.update(speed=speed, azimuth=az, kind="whip",
                                     frac=alloc["frac"])
                    rows.append(res)
                    print(row(res))

                    cl = dynopt.optimize_closed_loop(
                        H, planner, q_rel, qd_rel, shape,
                        throw_dur0=args.dt_throw,
                    )
                    print("  " + dynopt.report_closed_loop(cl))
                    res = evaluate(H, cl["plan"], "lab_whip_closedloop",
                                   jitter=args.jitter)
                    res.extra.update(speed=speed, azimuth=az, kind="whip_cl",
                                     frac=cl["frac"])
                    rows.append(res)
                    print(row(res))

    # ---- aggregate ---------------------------------------------------
    print(f"\n=== aggregate over {len(set((r.extra['speed'], r.extra['azimuth']) for r in rows))} "
          f"release states ({time.time() - t0:.1f}s) ===")
    print(HEADER)
    labels = []
    for r in rows:
        if r.label not in labels:
            labels.append(r.label)
    agg = {}
    for lab in labels:
        sub = [r for r in rows if r.label == lab]
        agg[lab] = dict(
            speed_ratio=float(np.mean([r.speed_ratio for r in sub])),
            dir_err=float(np.mean([r.dir_err_deg for r in sub])),
            omega=float(np.mean([np.linalg.norm(r.omega_release) for r in sub])),
            land_err=float(np.nanmean([r.extra.get("land_err", np.nan) for r in sub])),
            dv_step=float(np.nanmean([r.extra.get("dv_per_step", np.nan) for r in sub])),
            q_err=float(np.mean([np.max(np.abs(r.q_err_at_release)) for r in sub])),
            dland_step=float(np.nanmean(
                [r.extra.get("dland_per_step", np.nan) for r in sub])),
            sat=float(np.mean([r.tau_sat_frac for r in sub])),
            tau_pk=float(np.mean([r.peak_tau_ratio for r in sub])),
            t_r=float(np.mean([r.t_r for r in sub])),
            jerk=float(np.mean([r.peak_jerk for r in sub])),
            accel_step=float(np.mean([r.accel_step for r in sub])),
            n=len(sub),
        )
        a = agg[lab]
        print(
            f"  {lab:<18} {100 * a['speed_ratio']:6.2f}%  {a['dir_err']:5.2f}d  "
            f"{a['omega']:5.2f}  {100 * a['land_err']:6.2f}cm  "
            f"{100 * a['dv_step']:6.2f}cm/s {100 * a['dland_step']:6.2f}cm "
            f"{1000 * a['q_err']:6.1f}mrad "
            f"{100 * a['sat']:5.1f}%  {a['tau_pk']:5.2f}  {a['t_r']:6.2f}s "
            f"{a['jerk']:7.1f} {a['accel_step']:6.2f}"
        )

    if shipped_failures:
        print(f"\n  plan_throw REFUSED {len(shipped_failures)} release state(s):")
        for sp, az, exc in shipped_failures:
            print(f"    speed {sp:.2f} az {az:+.0f}: {exc.splitlines()[0]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        labels=np.array(labels, dtype=object),
        agg=np.array([agg[l] for l in labels], dtype=object),
        rows=np.array(
            [
                dict(
                    label=r.label,
                    speed=r.extra["speed"],
                    azimuth=r.extra["azimuth"],
                    speed_ratio=r.speed_ratio,
                    dir_err=r.dir_err_deg,
                    omega=float(np.linalg.norm(r.omega_release)),
                    land_xy=r.land_xy,
                    ideal_xy=r.extra.get("ideal_xy"),
                    land_err=r.extra.get("land_err"),
                    dv_per_step=r.extra.get("dv_per_step"),
                    peak_q_err=r.peak_q_err,
                    q_err_release=float(np.max(np.abs(r.q_err_at_release))),
                    dland_per_step=r.extra.get("dland_per_step"),
                    tau_sat_frac=r.tau_sat_frac,
                    peak_tau_ratio=r.peak_tau_ratio,
                    t_r=r.t_r,
                    dt_throw=r.dt_throw,
                    peak_jerk=r.peak_jerk,
                    accel_step=r.accel_step,
                )
                for r in rows
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )
    print(f"\nsaved -> {args.out}")
    return rows, agg


if __name__ == "__main__":
    main()
