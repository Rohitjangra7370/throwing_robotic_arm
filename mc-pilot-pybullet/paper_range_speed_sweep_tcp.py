"""
ICRA 2027 paper figure data (fig_range_ceiling), TCP-offset-corrected version.

Supersedes paper_range_speed_sweep.py, which swept throw_pose_table.npy (zero
tool offset, 5deg elevation) at t_throw=1.1s. Two things changed for the
results_kinetic_chain_gen3_tcp/1 checkpoint:

  1. Physics: tool_offset=[0,0,0.12], throw_pose_table_tcp.npy (azimuth=0 entry
     is 15deg elevation, kinematic max 2.0705 m/s -- a different release
     geometry, not just a rescaled one, per the gripper-TCP-offset fix).
  2. t_throw: the old script hardcoded 1.1s ("matches the trained/deployed
     windup-to-release duration") but train_mc_pilot_pb_arm.py unconditionally
     sets T_W,T_R=0.5,1.6 whenever --opt_pose is used (both old and tcp
     checkpoints confirm T_R=1.6 in their own config_log.pkl) -- 1.1 does not
     match either checkpoint's actual deployed ramp duration. This script
     reads T_R from the checkpoint's own config_log.pkl instead of
     hardcoding it, so it cannot drift from what was actually trained/deployed.

Reuses find_throw_pose.py's own ballistic_range() (drag-augmented forward
integration), _fkj() (real FK/Jacobian, TOOL_OFFSET-aware) and the three real
feasibility-cascade checks -- no new physics, no manual formula. The
commanded speed is swept from near-zero to the table's own kinematic ceiling
in the SAME direction throughout, matching how the deployed policy actually
operates (a scalar speed u in [0, uM] through a fixed release direction).

The max cascade-passing speed doubles as the eval-time follow-through-safe
speed cap (see eval_adapted_height.py's SAFE_U_CAP, itself derived this way
for the old table) -- re-derived here for the tcp table rather than reusing
the old table's 1.60 m/s figure, since the tcp table's higher-elevation
release has a materially different kinematic ceiling (2.07 vs 1.628 m/s).

Usage: python3 paper_range_speed_sweep_tcp.py [out.json]
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

CHECKPOINT = "results_kinetic_chain_gen3_tcp/1"
TABLE = "throw_pose_table_tcp.npy"


def main(out_path):
    cfg = pkl.load(open(os.path.join(CHECKPOINT, "config_log.pkl"), "rb"))
    assert cfg["opt_pose"] == TABLE, (cfg["opt_pose"], TABLE)
    t_throw = float(cfg["T_R"])  # real deployed ramp duration, not a guess

    ftp.set_robot("kinova_gen3_dyn")
    ftp.set_floor_z(-float(cfg["base_height"]))
    ftp.set_tool_offset([0.0, 0.0, 0.12])

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

    table = np.load(TABLE, allow_pickle=True)
    entry = next(e for e in table if float(e["azimuth_deg"]) == 0.0)
    assert entry["tool_offset"] == [0.0, 0.0, 0.12]
    assert entry["floor_z"] == -float(cfg["base_height"])
    q = np.asarray(entry["q"], dtype=float)
    qd_full = np.asarray(entry["qd"], dtype=float)
    v_dir = np.asarray(entry["v_dir"], dtype=float)
    speed_kin_max = float(entry["speed"])

    pos, _ = ftp._fkj(arm, q)
    pos = np.asarray(pos, dtype=float)

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

    passing = np.where(cascade_ok)[0]
    if len(passing) == 0:
        sys.exit("no swept speed passed the full cascade -- check t_throw/grid")
    i_safe = passing[-1]

    # arm's own kinematic reach, independent of the ballistic release model:
    # sweep the same fixed direction/posture family to its joint-velocity LP
    # ceiling and report the landing range at that ceiling. This is a
    # DIFFERENT quantity from safe_range (which is capped by follow-through
    # recoverability, not by reach) -- kept separate deliberately, the two
    # were conflated across the old PNG/prose/JSON and that was part of the
    # original discrepancy this script exists to resolve.
    reach_range = float(ranges[-1])

    out = dict(
        checkpoint=CHECKPOINT,
        table=TABLE,
        t_throw=t_throw,
        speeds=speeds.tolist(),
        ranges=ranges.tolist(),
        cascade_ok=cascade_ok.tolist(),
        release_pos=pos.tolist(),
        v_dir=v_dir.tolist(),
        speed_kin_max=speed_kin_max,
        range_at_kin_max=reach_range,
        safe_speed=float(speeds[i_safe]),
        safe_range=float(ranges[i_safe]),
        note=(f"Speed swept in the fixed azimuth=0, 15deg-elevation direction "
              f"from {TABLE}; qd(s) scaled exactly from the LP solution (fixed "
              f"J(q), linear in speed); ranges via find_throw_pose.ballistic_range "
              f"(drag-augmented forward integration); cascade_ok is the real "
              f"windup_path_feasible + throw_ramp_feasible + follow_through_feasible "
              f"check at each speed, t_throw={t_throw:.2f}s (read from "
              f"{CHECKPOINT}/config_log.pkl's T_R, the actual value plan_throw was "
              f"called with during training/eval -- not hardcoded), "
              f"floor_z={-float(cfg['base_height']):.3f}, tool_offset_z=0.12 "
              f"(TCP-corrected table, matching results_kinetic_chain_gen3_tcp/1)."),
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"t_throw (from checkpoint config): {t_throw:.3f}s")
    print(f"kinematic max: speed={speed_kin_max:.4f} m/s -> range={reach_range:.4f} m")
    print(f"cascade-safe:  speed={speeds[i_safe]:.4f} m/s -> range={ranges[i_safe]:.4f} m")
    print(f"wrote {out_path}")
    p.disconnect(cid)


if __name__ == "__main__":
    if "--repair_inertials" in sys.argv:
        ftp.REPAIR_INERTIALS = True
        sys.argv = [a for a in sys.argv if a != "--repair_inertials"]
    main(sys.argv[1] if len(sys.argv) > 1 else "range_speed_sweep_tcp.json")
