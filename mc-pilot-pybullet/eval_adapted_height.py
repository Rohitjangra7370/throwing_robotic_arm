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

# Follow-through-safe speed ceiling, measured by bisecting plan_throw on the
# shipped table: the table's KINEMATIC max (1.628 m/s) is not recoverable-from.
SAFE_U_CAP = 1.60


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--num_throws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--u_cap", type=float, default=SAFE_U_CAP)
    ap.add_argument("--json_out", default=None)
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

    table = list(np.load(cfg["opt_pose"], allow_pickle=True))
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
    errs, speeds, rows = [], [], []
    for i in range(args.num_throws):
        flight = rng.uniform(f_lo, f_hi)
        beta = rng.uniform(-gM, gM)
        tgt = RP[:2] + flight * np.array([np.cos(beta), np.sin(beta)])
        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=launch, arm_noise=None,
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=h, opt_posture_table=table, opt_launch_deg=launch)
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
