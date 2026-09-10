"""
ICRA 2027 paper figure data: release speed -> landing range, at the winning
overhead release direction/posture from throw_pose_table.npy (azimuth=0,
5 deg elevation -- the same candidate behind Table I/II and the 0.82 m
safe-throw figure).

Read-only analysis script, same pattern as paper_ablation_feasibility.py.
Reuses find_throw_pose.py's own ballistic_range() (drag-augmented forward
integration) and _fkj() (real FK/Jacobian) -- no new physics, no manual
formula. The commanded speed is swept from near-zero to the table's own
kinematic ceiling (1.6281 m/s, the joint-velocity LP's max for this exact
direction) in the SAME direction throughout, which is how the deployed
policy actually operates (it commands a scalar speed in a fixed direction,
u in [0, uM]) -- not a hypothetical.

t_throw was hardcoded to 1.1s ("matches the trained/deployed windup-to-release
duration") until 2026-09-03. It does not: train_mc_pilot_pb_arm.py
unconditionally sets T_W,T_R=0.5,1.6 whenever --opt_pose is used (verified
against this checkpoint's own config_log.pkl, T_R=1.6), and T_R is passed
straight through to plan_throw's t_r arg -- the same duration
windup_path_feasible/throw_ramp_feasible call t_throw. Fixed to read T_R
from the checkpoint config instead of hardcoding it, so it cannot drift from
what was actually trained/deployed again. This raises safe_speed/safe_range
(a longer ramp lowers peak torque, admitting more speeds) -- both figures
and any prose reporting them need re-checking against the new safe_speed
value this now prints. paper_ablation_feasibility.py has the same t_throw=1.1
default and was very likely used to produce Table I/II with it -- NOT fixed
here (a 446k+152k-candidate full-grid rerun, hours-scale, out of scope for
this pass; flagged to the user separately).

Usage: python3 paper_range_speed_sweep.py [out.json]
Run from mc-pilot-pybullet/.
"""
import json
import os
import pickle as pkl
import sys

import numpy as np
import pybullet as p
import pybullet_data

import find_throw_pose as ftp


def main(out_path):
    ftp.set_robot("kinova_gen3_dyn")
    ftp.set_floor_z(-0.433)
    ftp.set_tool_offset([0.0, 0.0, 0.0])  # pre-TCP-fix table: zero offset

    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Go through find_throw_pose's own loader, not loadURDF directly: it is
    # the single place that honours REPAIR_INERTIALS. Loading the description
    # raw gives every bodyless frame the simulator's 1 kg default, which on the
    # Gen3 is +3.00 kg at the wrist and inflates every torque check in the
    # cascade below.
    arm = ftp.load_arm()
    ftp._set_n_full(arm)

    table = np.load("throw_pose_table.npy", allow_pickle=True)
    entry = next(e for e in table if float(e["azimuth_deg"]) == 0.0)
    q = np.asarray(entry["q"], dtype=float)
    qd_full = np.asarray(entry["qd"], dtype=float)
    v_dir = np.asarray(entry["v_dir"], dtype=float)
    speed_kin_max = float(entry["speed"])  # joint-velocity LP max, this direction

    pos, _ = ftp._fkj(arm, q)
    pos = np.asarray(pos, dtype=float)

    # J(q) is fixed at this posture and the LP is linear in speed for a fixed
    # direction, so qd scales exactly (not approximately) with commanded
    # speed: qd(s) = qd_full * (s / speed_kin_max).
    cfg = pkl.load(open(os.path.join("results_kinetic_chain_gen3", "1",
                                     "config_log.pkl"), "rb"))
    t_throw = float(cfg["T_R"])  # real deployed ramp duration -- was hardcoded 1.1
    speeds = np.linspace(0.05, speed_kin_max, 40)
    ranges, cascade_ok = [], []
    for s in speeds:
        vel = s * v_dir
        rng, _ = ftp.ballistic_range(pos, vel, mass=ftp.MASS, radius=ftp.RAD,
                                     floor_z=ftp.FLOOR_Z)
        ranges.append(rng)

        qd_s = qd_full * (s / speed_kin_max)
        ok = (ftp.windup_path_feasible(arm, q, qd_s, t_throw)
              and ftp.throw_ramp_feasible(arm, q, qd_s, t_throw)
              and ftp.follow_through_feasible(arm, q, qd_s))
        cascade_ok.append(bool(ok))
    ranges = np.array(ranges)
    cascade_ok = np.array(cascade_ok)

    # Safe-throw ceiling: the largest speed (and its range) that still
    # passes the full whole-trajectory cascade.
    passing = np.where(cascade_ok)[0]
    if len(passing) == 0:
        sys.exit("no swept speed passed the full cascade -- check t_throw/grid")
    i_safe = passing[-1]

    out = dict(
        speeds=speeds.tolist(),
        ranges=ranges.tolist(),
        cascade_ok=cascade_ok.tolist(),
        release_pos=pos.tolist(),
        v_dir=v_dir.tolist(),
        speed_kin_max=speed_kin_max,
        range_at_kin_max=float(ranges[-1]),
        safe_speed=float(speeds[i_safe]),
        safe_range=float(ranges[i_safe]),
        note=(f"Speed swept in the fixed azimuth=0, 5deg-elevation direction "
              f"from throw_pose_table.npy; qd(s) scaled exactly from the LP "
              f"solution (fixed J(q), linear in speed); ranges via "
              f"find_throw_pose.ballistic_range (drag-augmented forward "
              f"integration); cascade_ok is the real windup_path_feasible + "
              f"throw_ramp_feasible + follow_through_feasible check at each "
              f"speed, t_throw={t_throw:.2f}s (read from "
              f"results_kinetic_chain_gen3/1/config_log.pkl's T_R -- was "
              f"hardcoded 1.1s until 2026-09-03, see module docstring), "
              f"floor_z=-0.433, zero tool offset (pre-TCP-fix table, "
              f"matching Table I/II)."),
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"kinematic max: speed={speed_kin_max:.4f} m/s -> range={ranges[-1]:.4f} m")
    print(f"cascade-safe:  speed={speeds[i_safe]:.4f} m/s -> range={ranges[i_safe]:.4f} m")
    print(f"wrote {out_path}")
    p.disconnect(cid)


if __name__ == "__main__":
    if "--repair_inertials" in sys.argv:
        ftp.REPAIR_INERTIALS = True
        sys.argv = [a for a in sys.argv if a != "--repair_inertials"]
    main(sys.argv[1] if len(sys.argv) > 1 else "range_speed_sweep.json")
