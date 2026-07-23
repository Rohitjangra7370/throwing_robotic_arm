"""
Adapt a trained throwing policy to a NEW basket height with ZERO new robot
trials -- the paper's "changing task requirements" result (MC-PILOT,
arXiv:2502.05595, Sec. 6.4):

    "MC-PILOT can adapt the throwing policy to the new task specifications
     without performing a new exploration. Namely, it only requires
     re-executing the policy optimization step with the updated requirements."

Why this works: the GP models the BALL's flight dynamics (delta_v as a
function of position and velocity). That is independent of where the basket
is -- the target height enters only through the task requirements (which
targets are asked for, and when the flight ends). So the learned model is
reused verbatim and only the policy is re-optimised, entirely in simulation
through that model. No robot interaction.

What "updated requirements" means concretely here:
  * target domain  -- the reachable flight-distance band shrinks with basket
    height (measured slope 0.382 m per m for the overhead release), so
    targets are sampled from the band belonging to the new height.
  * control horizon -- the particle rollout is a fixed-length loop whose cost
    is taken at the final step, so the horizon must match the (shorter)
    flight time to the raised plane, else the policy optimises for a landing
    that happens after the ball has already passed the basket.

Usage:
    python adapt_policy_height.py --log_path results_kinetic_chain_gen3/1 \
        --height 0.20 --out results_kinetic_chain_gen3_h20
"""
import argparse
import os
import pickle as pkl

import numpy as np
import torch

import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


def flight_band(release_pos, v_dir, height, mass, radius, u_lo, u_hi):
    """Integrate the real drag ballistics to get (t_flight, f_lo, f_hi) for a
    landing plane at `height`. Measured rather than assumed -- the analytic
    no-drag form is close here but the whole point of this project is that we
    check."""
    from simulation_class.model import _ball_accel

    rel_xy = np.asarray(release_pos[:2], dtype=float)

    def integrate(spd):
        x = np.asarray(release_pos, dtype=float).copy()
        v = spd * np.asarray(v_dir, dtype=float)
        dt = 0.0005
        for i in range(20000):
            a = _ball_accel(x, v, mass, radius, np.zeros(3))
            v = v + a * dt
            xn = x + v * dt
            if xn[2] <= height and v[2] < 0 and i > 2:
                return (i + 1) * dt, float(np.linalg.norm(xn[:2] - rel_xy))
            x = xn
        raise RuntimeError(f"ball never reaches plane z={height}")

    t_lo, f_lo = integrate(u_lo)
    t_hi, f_hi = integrate(u_hi)
    return 0.5 * (t_lo + t_hi), f_lo, f_hi


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3/1",
                    help="trained run whose GP model is reused")
    ap.add_argument("--height", type=float, required=True,
                    help="new basket height in metres")
    ap.add_argument("--out", required=True, help="results root for the adapted policy")
    ap.add_argument("--opt_pose", default=None,
                    help="posture table used for training (only needed for checkpoints predating its recording in config_log)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--Nopt", type=int, default=1500)
    args = ap.parse_args()

    cfg = pkl.load(open(os.path.join(args.log_path, "config_log.pkl"), "rb"))
    log = pkl.load(open(os.path.join(args.log_path, "log.pkl"), "rb"))
    num_trained = len(log["parameters_trial_list"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_num_threads(1)

    STATE_DIM, INPUT_DIM, BALL_DIM, TARGET_DIM = 8, 1, 6, 2
    profile = get_robot_profile(cfg["robot_name"])
    RELEASE_POS = np.array(cfg["release_pos"], dtype=float)
    Ts, uM, uMin = cfg["Ts"], cfg["uM"], cfg["uMin"]
    M, Nb, lc, gM = cfg["M"], cfg["Nb"], cfg["lc"], cfg["gM"]

    # checkpoints trained before opt_pose was recorded in config_log need it
    # supplied explicitly -- refuse to guess silently.
    table_path = args.opt_pose or cfg.get("opt_pose")
    if table_path is None:
        raise SystemExit(
            "this checkpoint's config has no 'opt_pose'; pass --opt_pose <table.npy> "
            "(the table the policy was trained with -- a different table is a different throw)")
    table = list(np.load(table_path, allow_pickle=True))
    e0 = min(table, key=lambda e: abs(e["azimuth_deg"]))
    v_dir = np.array(e0["v_dir"], dtype=float)

    # --- updated requirements for the new height -------------------------
    t_flight, f_lo, f_hi = flight_band(RELEASE_POS, v_dir, args.height,
                                       cfg["ball_mass"], cfg["ball_radius"],
                                       uMin, uM)
    t_ground, g_lo, g_hi = flight_band(RELEASE_POS, v_dir, 0.0,
                                       cfg["ball_mass"], cfg["ball_radius"],
                                       uMin, uM)
    # Keep the horizon in the same proportion to flight time that the trained
    # (ground) policy used, so the landing convention is preserved.
    T_new = cfg["T"] * (t_flight / t_ground)
    print(f"height {args.height:.2f} m: flight {t_flight:.3f}s (ground {t_ground:.3f}s) "
          f"-> T {cfg['T']:.3f} -> {T_new:.3f}s")
    print(f"reachable flight band {f_lo:.3f}-{f_hi:.3f} m "
          f"(ground {g_lo:.3f}-{g_hi:.3f} m)")

    release_xy = RELEASE_POS[:2]

    def sample_target():
        flight = np.random.uniform(f_lo, f_hi)
        beta = np.random.uniform(-gM, gM)
        return release_xy + flight * np.array([np.cos(beta), np.sin(beta)])

    throwing_system = PyBulletThrowingSystem(
        mass=cfg["ball_mass"], radius=cfg["ball_radius"],
        launch_angle_deg=cfg["opt_launch_deg"] if "opt_launch_deg" in cfg
        else float(e0["elev_deg"]),
        arm_noise=None, t_w=cfg["T_W"], t_r=cfg["T_R"],
        robot_name=profile.name, target_height=args.height,
        opt_posture_table=table, opt_launch_deg=float(e0["elev_deg"]),
    )

    # --- model/policy/cost objects, identical to training ----------------
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

    # Warm-start the adapted policy from the trained one: same task, moved
    # plane, so the trained weights are a far better starting point than a
    # random init (and it is what "re-optimising" implies).
    st = log["parameters_trial_list"][-1]
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "num_basis": st["centers"].shape[0], "u_max": uM,
        "lengthscales_init": st["log_lengthscales"].exp().numpy()[0],
        "centers_init": st["centers"].numpy(),
        "weight_init": st["f_linear.weight"].numpy(),
        "flg_drop": True, "dtype": dtype, "device": device,
    }
    rand_exploration_policy_par = {
        "full_state_dim": STATE_DIM, "u_max": uM, "u_min": uMin,
        "n_strata": cfg["Nexp"], "dtype": dtype, "device": device,
    }
    cost_function_par = {
        "position_indices": [0, 1], "target_indices": [6, 7],
        "lengthscale": lc, "dtype": dtype, "device": device,
    }

    results_root = args.out
    log_path = os.path.join(results_root, str(args.seed))
    os.makedirs(log_path, exist_ok=True)

    mc = MC_PILCO_module.MC_PILOT(
        target_sampler=sample_target,
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
        log_path=log_path, dtype=dtype, device=device,
        target_height=args.height,
    )

    # === the whole point: reuse the learned model, no new exploration ===
    mc.load_model_from_log(num_trial=num_trained - 1,
                           folder=args.log_path.rstrip("/") + "/")
    # reinforce() does this before every policy-optimisation step.
    mc.model_learning.set_eval_mode()
    # load_model_from_log runs pretrain_gp WITH grad enabled, so the cached
    # solve tensors (alpha, K_X_inv, ...) keep autograd history; the policy
    # loop then backprops through them and the second iteration dies with
    # "backward through the graph a second time". Recompute them detached --
    # the GP is fixed data here, only the policy is being optimised.
    with torch.no_grad():
        for k in range(mc.model_learning.num_gp):
            mc.model_learning.pretrain_gp(k)

    centre = np.array([release_xy[0] + 0.5 * (f_lo + f_hi), release_xy[1]])
    initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre])
    initial_state_var = np.concatenate(
        [1e-4 * np.ones(6), (0.5 * (f_hi - f_lo)) ** 2 * np.ones(2)])

    print(f"\n=== re-optimising policy only (no new trials), {args.Nopt} steps ===")
    cost_list, _, _, _ = mc.reinforce_policy(
        T_control=int(T_new / Ts),
        num_particles=M,
        trial_index=num_trained - 1,
        particles_initial_state_mean=torch.tensor(initial_state, dtype=dtype, device=device),
        particles_initial_state_var=torch.tensor(initial_state_var, dtype=dtype, device=device),
        flg_particles_init_uniform=False,
        particles_init_up_bound=None, particles_init_low_bound=None,
        flg_particles_init_multi_gauss=False,
        # reinforce_policy indexes these by trial_index
        opt_steps_list=[args.Nopt] * num_trained, lr_list=[0.01] * num_trained,
        f_optimizer="lambda p, lr : torch.optim.Adam(p, lr)",
        num_step_print=100, p_dropout_list=[0.25] * num_trained,
        p_drop_reduction=0.25 / 2,
        alpha_diff_cost=0.99, min_diff_cost=0.02, num_min_diff_cost=400,
        min_step=400, lr_min=0.0025,
        policy_reinit_dict={"lenghtscales_par": np.array(control_policy_par["lengthscales_init"]),
                            "centers_par": np.array([1.0, 1.0]), "weight_par": uM},
    )

    adapted = dict(cfg)
    adapted.update({"opt_pose": table_path, "target_height": float(args.height), "T": float(T_new),
                    "lm": float(release_xy[0] + f_lo), "lM": float(release_xy[0] + f_hi),
                    "results_root": results_root,
                    "adapted_from": args.log_path, "new_trials_used": 0})
    pkl.dump(adapted, open(os.path.join(log_path, "config_log.pkl"), "wb"))
    out_log = {"parameters_trial_list": [mc.control_policy.state_dict()],
               "cost_trial_list": list(cost_list) if cost_list is not None else [],
               "adapted_from": args.log_path, "target_height": float(args.height),
               "new_trials_used": 0}
    pkl.dump(out_log, open(os.path.join(log_path, "log.pkl"), "wb"))
    print(f"\nAdapted policy saved to {log_path} (0 new robot trials)")


if __name__ == "__main__":
    main()
