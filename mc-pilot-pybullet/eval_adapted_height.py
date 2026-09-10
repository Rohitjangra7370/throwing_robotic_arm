"""
Real-physics evaluation of a policy adapted to a new basket height with ZERO
new robot trials (see adapt_policy_height.py / MC-PILOT Sec. 6.4).

The adapted policy is still 8-D (target = Px, Py): the basket height is a
property of the *task*, not a policy input -- which is exactly why the paper's
adaptation works without new data. So this evaluator samples targets from the
height's own reachable band (stored in the adapted config) and cuts the
trajectory at that height's plane.
"""
import argparse
import os
import pickle as pkl

import numpy as np
import torch

import policy_learning.Policy as Policy
from simulation_class.model_pybullet import PyBulletThrowingSystem
from run_hardware_throw import _check_tool_offset_matches_table

# Follow-through-safe release-speed ceiling. This is a PER-ARM, PER-TABLE
# property, not a constant: it used to be hardcoded to 1.60, which is the
# Gen3's own value (its table's kinematic max is not recoverable-from -- only
# 76.8% of it is). Applied to any other arm it silently clamps every throw to
# a speed that arm has no reason to respect -- measured on the UR7e, whose
# table max is 4.0 m/s: every one of 30 throws released at an identical
# 1.54 m/s, three independently-trained seeds evaluated to byte-identical
# numbers, and the reported error was ~99 cm of pure evaluator artifact.
# Now read from RobotProfile.safe_u_cap for the checkpoint's own arm, falling
# back to the checkpoint's trained uM when the arm needs no cap.
SAFE_U_CAP = 1.60          # kept as the Gen3 default; see robot_profiles


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--num_throws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--u_cap", type=float, default=None,
                    help="release-speed ceiling; default is the "
                         "checkpoint arm's RobotProfile.safe_u_cap, "
                         "or its trained uM when the arm needs none")
    ap.add_argument("--tool_offset_z", type=float, default=0.0,
                    help=("TCP offset the checkpoint's --opt_pose table was "
                          "searched at (e.g. 0.12). Must match the table's own "
                          "tool_offset stamp -- evaluating at the wrong offset "
                          "silently measures the wrong physics, exactly the "
                          "regression this flag exists to prevent."))
    ap.add_argument("--json_out", default=None)
    ap.add_argument(
        "--targets", default=None,
        help=(
            "path to an (N,2) .npy of absolute base-frame target positions to "
            "evaluate on, instead of drawing fresh ones from --seed. Every "
            "adapted checkpoint re-optimizes its own reachable band, so drawing "
            "per-checkpoint targets makes each condition a different test: the "
            "ground checkpoint's band is 7 cm wide and the h=0.10 band is 87 cm "
            "wide. Pass one shared file to compare conditions on identical "
            "targets. --num_throws is then taken from the file."
        ),
    )
    ap.add_argument(
        "--opt_pose", default=None,
        help=(
            "pose-table path override. Checkpoints trained before the trainer "
            "started recording 'opt_pose' in config_log.pkl have no way to "
            "report their own release geometry; supply it here to evaluate them."
        ),
    )
    args = ap.parse_args()

    cfg = pkl.load(open(os.path.join(args.log_path, "config_log.pkl"), "rb"))
    log = pkl.load(open(os.path.join(args.log_path, "log.pkl"), "rb"))
    st = log["parameters_trial_list"][-1]

    pol = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0],
        u_max=cfg["uM"], lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"))
    pol.load_state_dict(st)
    pol.eval()

    table_path = args.opt_pose or cfg.get("opt_pose")
    if table_path is None:
        raise SystemExit(
            "config_log.pkl has no 'opt_pose' key (checkpoint predates the "
            "trainer recording it) and --opt_pose was not given. Pass the pose "
            "table this policy was trained through, e.g. "
            "--opt_pose throw_pose_table.npy"
        )
    if args.u_cap is None:
        from robot_arm.robot_profiles import get_robot_profile
        cap = get_robot_profile(cfg["robot_name"]).safe_u_cap
        args.u_cap = float(cap) if cap is not None else float(cfg["uM"])
        print(f"u_cap = {args.u_cap:.3f} m/s "
              f"({'profile safe_u_cap' if cap is not None else 'trained uM, arm needs no cap'})")
    table = list(np.load(table_path, allow_pickle=True))
    _check_tool_offset_matches_table(table, args.tool_offset_z)
    launch = float(cfg.get("opt_launch_deg", table[0]["elev_deg"]))
    RP = np.array(cfg["release_pos"], dtype=float)
    h = float(cfg["target_height"])
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg["gM"]
    f_lo, f_hi = lm - RP[0], lM - RP[0]

    print(f"adapted policy: h={h:.2f}m  band {lm:.3f}-{lM:.3f}m  "
          f"T={cfg['T']:.3f}s  new_trials_used={cfg.get('new_trials_used','?')}")

    def policy(s, t):
        with torch.no_grad():
            u = float(pol(torch.tensor(np.asarray(s), dtype=torch.float64)
                          .unsqueeze(0), t=0, p_dropout=0.0).item())
        return np.array([min(u, args.u_cap)])

    rng = np.random.default_rng(args.seed)
    fixed = None
    if args.targets:
        fixed = np.load(args.targets)
        args.num_throws = len(fixed)
        print(f"shared target set: {args.targets}  n={len(fixed)}")
    errs, speeds, rows = [], [], []
    for i in range(args.num_throws):
        if fixed is not None:
            tgt = np.asarray(fixed[i], dtype=float)
        else:
            flight = rng.uniform(f_lo, f_hi)
            beta = rng.uniform(-gM, gM)
            tgt = RP[:2] + flight * np.array([np.cos(beta), np.sin(beta)])
        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=launch, arm_noise=None,
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=h, base_height=float(cfg.get("base_height", 0.0)),
            opt_posture_table=table, opt_launch_deg=launch,
            tool_offset=[0.0, 0.0, args.tool_offset_z])
        pos, _, _ = sysm.rollout(np.concatenate([RP, np.zeros(3), tgt]),
                                 policy, T=cfg["T"], dt=cfg["Ts"], noise=0.0)
        land = pos[-1][:2]
        e = float(np.linalg.norm(land - tgt))
        sp = float(np.linalg.norm(sysm.last_release_info["v_release"]))
        errs.append(e); speeds.append(sp)
        rows.append(dict(tx=float(tgt[0]), ty=float(tgt[1]),
                         lx=float(land[0]), ly=float(land[1]), err=e, speed=sp, h=h))
    e = np.array(errs); sp = np.array(speeds)
    print(f"n={len(e)}  mean {e.mean()*100:.2f}cm  median {np.median(e)*100:.2f}cm  "
          f"max {e.max()*100:.2f}cm  hit<10cm {100*np.mean(e<0.10):.0f}%  "
          f"speed {sp.min():.2f}-{sp.max():.2f} m/s")
    if args.json_out:
        import json
        json.dump(rows, open(args.json_out, "w"))


if __name__ == "__main__":
    main()
