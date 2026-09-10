"""
ICRA 2027 paper ablation: release-instant-only feasibility checking vs. the
production whole-trajectory checking in find_throw_pose.py::search_release_state.

Read-only analysis script, not part of the production pipeline. Reuses the
exact grid enumeration and feasibility functions from find_throw_pose.py --
does not reimplement or approximate them -- so results are directly
comparable to what search_release_state() would find.

"Release-instant-only" here means the specific historical check this project
used before throw_ramp_feasible/windup_path_feasible/follow_through_feasible
were added (see status_update/HANDOFF.md, 2026-07-23 entry): static torque
feasibility at the release posture (gravity only, zero velocity) plus a
kinematic release-velocity check (windup_within_limits) plus torque
feasibility AT the release instant only (release_dynamics_feasible, qdd=0).
"Full whole-trajectory" adds windup_path_feasible (neutral->windup swing),
throw_ramp_feasible (the actual cubic ramp, 40 samples, ball wrench
included), and follow_through_feasible (post-release recovery, 8 candidate
durations x 30 samples, torque AND velocity).

Usage: python3 paper_ablation_feasibility.py <robot_name> [out.json]
Run from this directory (mc-pilot-pybullet/), matching every other script here.
"""
import json
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

import find_throw_pose as ftp


def run_ablation(robot_name, t_throw=1.1, out_path=None, repair_inertials=None):
    if robot_name != ftp._ROBOT_NAME:
        ftp.set_robot(robot_name)
    if repair_inertials is not None:
        ftp.REPAIR_INERTIALS = bool(repair_inertials)
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = ftp.load_arm()
    ftp._set_n_full(arm)
    lo, hi = ftp.joint_limits(arm)

    g2 = np.deg2rad(np.arange(-120, 121, 5))
    g4 = np.deg2rad(np.arange(-147, 148, 5))
    g6 = np.deg2rad(np.arange(-120, 121, 8))

    # The three grids sweep the arm's PITCH joints -- the ones whose axis is
    # perpendicular to the swing plane, i.e. everything the release LP does not
    # freeze -- while every roll/twist joint is held at zero. For the Gen3 and
    # Panda that is indices (1,3,5) with (0,2,4,6) held, which is exactly the
    # literal [0, s2, 0, s4, 0, s6, 0] vector this script used to build; for a
    # UR it is (1,2,3) with (0,4,5) held. Derived from the profile rather than
    # written out, so the SAME posture-and-direction grid is applied to every
    # arm -- the whole point of the ablation is that only the arm changes.
    pitch = ftp.pitch_indices()

    stats = dict(n_postures=0, n_static_feasible=0, n_lp_success=0,
                 n_windup_kin_ok=0, n_release_instant_ok=0,
                 n_windup_path_fail=0, n_ramp_fail=0, n_follow_fail=0,
                 n_full_ok=0)
    best_instant, best_full = None, None
    t0 = time.time()

    for s2 in g2:
        for s4 in g4:
            for s6 in g6:
                q = ftp.sagittal_q(pitch, s2, s4, s6)
                pos, J = ftp._fkj(arm, q)
                if pos[2] < 0.15:
                    continue
                # Swing-plane normal from the first two pitch columns. On the
                # Gen3 these are J[:,1] and J[:,3], as before.
                u1, u3 = J[:, pitch[0]], J[:, pitch[1]]
                nvec = np.cross(u1, u3)
                nn = np.linalg.norm(nvec)
                if nn < 1e-8:
                    continue
                nvec /= nn
                h_ip = np.cross(nvec, [0.0, 0.0, 1.0])
                hn = np.linalg.norm(h_ip)
                if hn < 1e-9:
                    continue
                h_ip /= hn
                v_ip = np.cross(nvec, h_ip)
                if v_ip[2] < 0:
                    v_ip = -v_ip
                stats["n_postures"] += 1
                if not ftp.static_feasible(arm, q):
                    continue
                stats["n_static_feasible"] += 1
                for elev_deg in range(5, 46, 5):
                    th = np.deg2rad(elev_deg)
                    if abs(v_ip[2]) < np.sin(th):
                        continue
                    spsi = np.sin(th) / v_ip[2]
                    cpsi = np.sqrt(max(0.0, 1.0 - spsi * spsi))
                    for sgn in (1.0, -1.0):
                        d = sgn * cpsi * h_ip + spsi * v_ip
                        s, qd = ftp.aimed_speed(J, d, ftp.QD, freeze_roll=True)
                        if qd is None or s < 0.5:
                            continue
                        stats["n_lp_success"] += 1
                        if not ftp.windup_within_limits(q, qd, t_throw, lo, hi):
                            continue
                        stats["n_windup_kin_ok"] += 1
                        if not ftp.release_dynamics_feasible(arm, q, qd):
                            continue
                        # -- release-instant-only criterion satisfied --
                        stats["n_release_instant_ok"] += 1
                        rng, landing_pos = ftp.ballistic_range(pos, s * d)
                        land = float(np.hypot(landing_pos[0], landing_pos[1]))
                        if best_instant is None or land > best_instant["land"]:
                            best_instant = {"land": land, "speed": float(s),
                                             "elev_deg": int(elev_deg)}
                        # -- now test whole-trajectory feasibility --
                        if not ftp.windup_path_feasible(arm, q, qd, t_throw):
                            stats["n_windup_path_fail"] += 1
                            continue
                        if not ftp.throw_ramp_feasible(arm, q, qd, t_throw):
                            stats["n_ramp_fail"] += 1
                            continue
                        if not ftp.follow_through_feasible(arm, q, qd):
                            stats["n_follow_fail"] += 1
                            continue
                        stats["n_full_ok"] += 1
                        if best_full is None or land > best_full["land"]:
                            best_full = {"land": land, "speed": float(s),
                                         "elev_deg": int(elev_deg)}
    elapsed = time.time() - t0
    p.disconnect(cid)
    result = {
        "robot": robot_name,
        "pitch_joints": list(pitch),
        "repair_inertials": bool(ftp.REPAIR_INERTIALS),
        # Recorded from 2026-09-10: the ramp duration is not a free knob, it is
        # what the deployed planner uses (train_mc_pilot_pb_arm.py forces
        # T_W,T_R = 0.5,1.6 whenever --opt_pose is given). Runs before this date
        # did not record it and used the 1.1 default, which is shorter than
        # deployed and therefore over-rejects on torque.
        "t_throw": float(t_throw),
        "elapsed_s": elapsed,
        "stats": stats,
        "best_release_instant_only": best_instant,
        "best_full_trajectory": best_full,
    }
    print(json.dumps(result, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    argv = sys.argv[1:]
    repair = "--repair_inertials" in argv
    t_throw = 1.1
    if "--t_throw" in argv:
        i = argv.index("--t_throw")
        t_throw = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if a != "--repair_inertials"]
    robot = args[0] if args else "kinova_gen3_dyn"
    out = args[1] if len(args) > 1 else (
        f"paper_ablation_{robot}{'_repaired' if repair else ''}.json")
    run_ablation(robot, t_throw=t_throw, out_path=out, repair_inertials=repair)
