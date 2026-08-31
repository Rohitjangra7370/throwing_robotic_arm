"""
Throw-window duration sweep: where does "faster" stop helping?

The time-optimal framing ("bang-bang acceleration extracts the maximum velocity
out of your joint limits") assumes the arm TRACKS what you command.  On a
torque-controlled arm at a finite control rate it does not, and the error grows
as the window shrinks.  This sweep measures both sides of that trade at once:

  * the planner's own feasibility ratios (continuous-time torque/velocity), and
  * what the closed loop actually produced (realized release speed, direction,
    landing error), through the same 50 Hz world the trainer uses.

Run:
    /usr/bin/python3 -m throw_lab.sweep
    /usr/bin/python3 -m throw_lab.sweep --shape min_jerk --durations 0.3 0.5 0.8 1.1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_class.release_solver import OptimizedReleaseSolver  # noqa: E402

from throw_lab import shapes as sh  # noqa: E402
from throw_lab.bench import make_release  # noqa: E402
from throw_lab.harness import ThrowHarness  # noqa: E402
from throw_lab.planner import LabThrowPlanner  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--table", default="throw_pose_table.npy")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--azimuths", type=float, nargs="+", default=[-20.0, 0.0, 20.0])
    ap.add_argument("--dist", type=float, default=0.70)
    ap.add_argument(
        "--durations", type=float, nargs="+",
        default=[0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.10, 1.40, 1.80, 2.40],
    )
    ap.add_argument("--shapes", nargs="+",
                    default=["const_accel", "trap_accel", "min_jerk"])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--ball_mass", type=float, default=0.0577)
    ap.add_argument("--out", default="throw_lab/results/duration_sweep.npz")
    args = ap.parse_args(argv)

    table = list(np.load(args.table, allow_pickle=True))
    recs = []

    with ThrowHarness(robot_name=args.robot, ball_mass=args.ball_mass) as H:
        solver = OptimizedReleaseSolver(opt_posture_table=table)
        planner = LabThrowPlanner(H.arm, payload_mass=args.ball_mass)

        for name in args.shapes:
            kw = {"beta": args.beta} if name == "trap_accel" else {}
            shape = sh.get_velocity_shape(name, **kw)
            print(f"\n=== throw shape: {shape.name}")
            print("  dt_throw  tauPk  qdPk |   speed%    dir   landErr  dv/step"
                  "   qErrRel   t_r     feasible")
            for dur in args.durations:
                agg = {k: [] for k in
                       ("speed", "dir", "land", "dv", "qerr", "t_r", "tau", "qd")}
                infeasible = 0
                for az in args.azimuths:
                    rp, q_rel, qd_rel, v_rel, _ = make_release(
                        solver, H.arm, args.speed, az, args.dist
                    )
                    # NOTE: the planner GROWS an infeasible window, so pin the
                    # requested duration by checking it directly first; a grown
                    # plan would silently report a different dt_throw.
                    st = planner._stagger_frac(0.5) * dur
                    lo = dur - st
                    q_w = planner._windup_pose(q_rel, qd_rel, lo, shape.S1)
                    chk = planner.chk.check_span(
                        planner._throw_sampler(q_w, q_rel, qd_rel, st, lo, shape),
                        0.0, dur,
                    )
                    if not chk.ok:
                        infeasible += 1
                    agg["tau"].append(chk.tau_ratio)
                    agg["qd"].append(chk.qd_ratio)
                    try:
                        plan = planner.plan(
                            q_rel, qd_rel, dt_throw=dur, throw_shape=shape,
                            windup_shape="trap_vel", windup_kw={"beta": 0.10},
                            label=f"{shape.name}@{dur:.2f}",
                        )
                    except RuntimeError:
                        continue
                    res = H.run(plan)
                    ideal, _ = H.free_flight(res.release_pos_planned, res.v_planned)
                    r_hi = H.run(plan, release_bias_steps=1)
                    r_lo = H.run(plan, release_bias_steps=-1)
                    agg["speed"].append(res.speed_ratio)
                    agg["dir"].append(res.dir_err_deg)
                    agg["land"].append(
                        np.linalg.norm(res.land_xy - ideal)
                        if np.all(np.isfinite(res.land_xy)) else np.nan
                    )
                    agg["dv"].append(0.5 * abs(r_hi.speed_release - r_lo.speed_release))
                    agg["qerr"].append(float(np.max(np.abs(res.q_err_at_release))))
                    agg["t_r"].append(res.t_r)
                m = {k: float(np.nanmean(v)) if v else float("nan")
                     for k, v in agg.items()}
                # the plan's realized dt_throw may exceed the request if the
                # planner had to grow it -- report the request and the flag
                print(
                    f"  {dur:7.2f}  {m['tau']:5.2f}  {m['qd']:4.2f} | "
                    f"{100 * m['speed']:7.2f}%  {m['dir']:5.2f}d  "
                    f"{100 * m['land']:6.2f}cm  {100 * m['dv']:6.2f}cm/s  "
                    f"{1000 * m['qerr']:6.1f}mrad  {m['t_r']:5.2f}s   "
                    f"{'GROWN' if infeasible else 'ok':>6}"
                )
                recs.append(dict(shape=shape.name, dt_throw=dur,
                                 infeasible=infeasible, **m))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, recs=np.array(recs, dtype=object), allow_pickle=True)
    print(f"\nsaved -> {args.out}")
    return recs


if __name__ == "__main__":
    main()
