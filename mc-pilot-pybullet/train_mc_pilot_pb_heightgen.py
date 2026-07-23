"""
Height-generalized MC-PILOT: ONE policy conditioned on basket height.
Target is (Px, Py, h) with h ~ U(0, 0.45) per throw; state is 9-D.
Based on Config PB-A.

This keeps the original zero-noise baseline structure while swapping in a
selected robot profile's release pose, speed bounds, and timing hints so each
supported arm can be trained into its own results directory.

Examples:
  python train_mc_pilot_pb_arm.py --robot franka_panda
  python train_mc_pilot_pb_arm.py --robot xarm6 --seed 2 --num_trials 12
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
from robot_arm.robot_profiles import available_robot_names, get_robot_profile, profile_to_dict
from simulation_class.model_pybullet import PyBulletThrowingSystem


DEFAULT_RANGE_BY_ROBOT = {
    "kuka_iiwa": (0.4, 1.1),
    "franka_panda": (0.6, 1.1),
    "xarm6": (0.6, 1.0),
    "kinova_gen3": (0.67, 0.74),
    "kinova_gen3_dyn": (0.67, 0.74),
}

# (H_MAX, flight-loss-per-metre-height slope k in f_hi(h) = f_hi(0) - k*h).
# kuka's original H_MAX=0.45 with the old polar lM_of_h slope (1.1, in
# distance-from-origin terms over its 0.4-1.1m range) is kept as-is for
# backward compatibility; kinova's numbers come from a direct flight-vs-height
# measurement (2026-07-20): flight 18.95cm -> 16.46cm over h=0-0.15m, slope
# ~0.17-0.18. H_MAX=0.10m chosen to leave >=5cm flight span at the top of the
# height range given kinova's already-tight ~7cm ground-level span.
HEIGHT_BUDGET_BY_ROBOT = {
    "kuka_iiwa": (0.45, None),         # None => use the legacy polar lM_of_h rule
    "kinova_gen3": (0.10, 0.18),
    "kinova_gen3_dyn": (0.10, 0.18),
}


def default_results_root(robot_name: str, target_height: float = 0.0) -> str:
    base = ("results_mc_pilot_pb_A" if robot_name == "kuka_iiwa"
            else f"results_mc_pilot_pb_A_{robot_name}")
    if target_height and target_height > 0.0:
        base += f"_h{int(round(target_height * 100)):02d}"
    return base


def build_parser():
    parser = argparse.ArgumentParser("Train MC-PILOT for a specific PyBullet robot arm")
    parser.add_argument(
        "--robot",
        type=str,
        default="kuka_iiwa",
        choices=available_robot_names(),
        help="robot arm profile to train; defaults to the original KUKA baseline arm",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=25)
    parser.add_argument("--results_root", type=str, default=None)

    parser.add_argument("--Nexp", type=int, default=10)
    parser.add_argument(
        "--Na", type=int, default=0,
        help=(
            "data-augmentation trajectories per real trial (paper Table 1: 0 sim, "
            "2 real) -- rotated copies about the vertical axis, physically valid "
            "for free-flight ballistics, no new real interaction needed"
        ),
    )
    parser.add_argument("--Nopt", type=int, default=1500)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--Nb", type=int, default=300)
    parser.add_argument("--Ts", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=0.60)
    parser.add_argument("--lc", type=float, default=0.5)
    parser.add_argument("--gM_deg", type=float, default=30.0)
    parser.add_argument(
        "--lm",
        type=float,
        default=None,
        help="minimum target distance; defaults to a robot-specific baseline range",
    )
    parser.add_argument(
        "--lM",
        type=float,
        default=None,
        help="maximum target distance; defaults to a robot-specific baseline range",
    )
    parser.add_argument(
        "--uMin",
        type=float,
        default=None,
        help="minimum release speed; defaults to the selected profile speed lower bound",
    )
    parser.add_argument(
        "--uM",
        type=float,
        default=None,
        help="maximum release speed; defaults to the selected profile speed upper bound",
    )
    parser.add_argument(
        "--lengthscale_xy",
        type=float,
        default=None,
        help="shared XY policy lengthscale; defaults to 0.15 times the target range",
    )
    parser.add_argument(
        "--target_height",
        type=float,
        default=0.0,
        help="landing-plane height in metres (elevated basket); 0.0 = ground",
    )
    parser.add_argument(
        "--opt_pose", type=str, default=None,
        help=("path to an azimuth->posture table from find_throw_pose.py. Throws "
              "from the searched release state (aimed by base rotation, whole "
              "trajectory torque/velocity validated) instead of IK+pinv."),
    )
    parser.add_argument(
        "--h_max", type=float, default=None,
        help="override the robot's basket-height range (m); default per HEIGHT_BUDGET_BY_ROBOT",
    )
    parser.add_argument(
        "--height_slope", type=float, default=None,
        help=("override flight-distance loss per metre of basket height. Measure it "
              "for the release state in use -- the overhead throw's is 0.382, vs "
              "0.18 for the old low-speed kinova release."),
    )
    parser.add_argument(
        "--flight_targets",
        action="store_true",
        help=(
            "sample targets by FLIGHT distance from the release point instead of "
            "the paper's polar-from-origin convention (see train_mc_pilot_pb_arm.py "
            "for why this matters for tight-speed-budget arms like kinova_gen3)."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    profile = get_robot_profile(args.robot)
    range_defaults = DEFAULT_RANGE_BY_ROBOT.get(profile.name, DEFAULT_RANGE_BY_ROBOT["kuka_iiwa"])

    seed = args.seed
    num_trials = args.num_trials
    Nexp = args.Nexp
    Nopt = args.Nopt
    M = args.M
    Nb = args.Nb
    Ts = args.Ts
    T = args.T
    lc = args.lc
    lm = args.lm if args.lm is not None else range_defaults[0]
    lM = args.lM if args.lM is not None else range_defaults[1]
    uMin = args.uMin if args.uMin is not None else float(profile.speed_bounds[0])
    uM = args.uM if args.uM is not None else float(profile.speed_bounds[1])
    gM = np.deg2rad(args.gM_deg)

    if not (0.0 < lm < lM):
        raise ValueError(f"Invalid target range: lm={lm}, lM={lM}")
    if not (0.0 < uMin <= uM):
        raise ValueError(f"Invalid speed bounds: uMin={uMin}, uM={uM}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.float64
    device = torch.device("cpu")
    torch.set_num_threads(1)

    STATE_DIM = 9
    INPUT_DIM = 1
    BALL_DIM = 6
    TARGET_DIM = 3

    RELEASE_POS = np.array(profile.default_release_pos, dtype=float)
    T_W, T_R, T_ARM = profile.timing

    # opt_pose (aimed release-state table) support, mirroring
    # train_mc_pilot_pb_arm.py: the overhead throw releases from a searched
    # posture, not the profile's static default_release_pos.
    opt_posture_table = None
    opt_launch_deg = 35.0
    if args.opt_pose is not None:
        _loaded = np.load(args.opt_pose, allow_pickle=True)
        if _loaded.ndim != 1:
            raise ValueError("--opt_pose here expects an azimuth->posture TABLE "
                             "(from find_throw_pose.py), not a single posture")
        opt_posture_table = list(_loaded)
        opt_launch_deg = float(opt_posture_table[0]["elev_deg"])
        T_W, T_R = 0.5, 1.6

    lengthscale_xy = (
        args.lengthscale_xy if args.lengthscale_xy is not None else 0.15 * (lM - lm)
    )

    H_MAX, height_slope = HEIGHT_BUDGET_BY_ROBOT.get(profile.name, (0.45, None))
    if args.h_max is not None:
        H_MAX = args.h_max
    if args.height_slope is not None:
        height_slope = args.height_slope
    lengthscale_h = 0.15 * H_MAX   # same 0.15-x-range rule, applied to the h axis
    lengthscales_init = np.array([lengthscale_xy, lengthscale_xy, lengthscale_h], dtype=float)

    release_xy = np.array(profile.default_release_pos[:2], dtype=float)
    if opt_posture_table is not None:
        # Real-FK release point of the table's az~=0 entry -- both the target
        # annulus and the particle model's launch point must be anchored here,
        # not on the profile default (a ~0.5m error for the overhead posture).
        import pybullet as _p
        import pybullet_data as _pd
        _az0 = min(opt_posture_table, key=lambda e: abs(e["azimuth_deg"]))
        _cid = _p.connect(_p.DIRECT)
        _tmp = _p.loadURDF(_pd.getDataPath() + "/" + profile.urdf_rel_path,
                           useFixedBase=True, physicsClientId=_cid)
        for _j in range(7):
            _p.resetJointState(_tmp, _j, _az0["q"][_j], physicsClientId=_cid)
        _real = _p.getLinkState(_tmp, profile.ee_link, computeForwardKinematics=True,
                                physicsClientId=_cid)[4]
        _p.disconnect(_cid)
        release_xy = np.array(_real[:2], dtype=float)
        RELEASE_POS = np.array(_real, dtype=float)
        print(f"opt_pose table: release anchored on real FK {np.round(RELEASE_POS,4)} "
              f"(profile default was {profile.default_release_pos})")

    if args.flight_targets:
        f_lo = lm - release_xy[0]
        f_hi0 = lM - release_xy[0]
        if not (0.0 < f_lo < f_hi0):
            raise ValueError(
                f"flight_targets: invalid flight range [{f_lo:.3f}, {f_hi0:.3f}] "
                f"from lm={lm}, lM={lM}, release_x={release_xy[0]}"
            )
        slope = height_slope if height_slope is not None else 0.0

        def f_hi_of_h(h):
            return max(f_lo + 0.02, f_hi0 - slope * h)

        def sample_target():
            h = np.random.uniform(0.0, H_MAX)
            flight = np.random.uniform(f_lo, f_hi_of_h(h))
            beta = np.random.uniform(-gM, gM)
            xy = release_xy + flight * np.array([np.cos(beta), np.sin(beta)])
            return np.array([xy[0], xy[1], h])
    else:
        def lM_of_h(h):
            # reachable max range shrinks with basket height (ballistics, uM=2.5)
            return max(lm + 0.1, lM - 1.1 * h)

        def sample_target():
            h = np.random.uniform(0.0, H_MAX)
            dist = np.random.uniform(lm, lM_of_h(h))
            angle = np.random.uniform(-gM, gM)
            return np.array([dist * np.cos(angle), dist * np.sin(angle), h])

    throwing_system = PyBulletThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=opt_launch_deg if opt_posture_table is not None else 35.0,
        arm_noise=None,
        t_w=T_W,
        t_r=T_R,
        robot_name=profile.name,
        target_height=args.target_height,
        opt_posture_table=opt_posture_table,
        opt_launch_deg=opt_launch_deg,
    )

    num_gp = 3
    gp_input_dim = BALL_DIM

    init_dict_RBF = {}
    init_dict_RBF["active_dims"] = np.arange(0, gp_input_dim)
    init_dict_RBF["lengthscales_init"] = np.ones(gp_input_dim)
    init_dict_RBF["flg_train_lengthscales"] = True
    init_dict_RBF["lambda_init"] = np.ones(1)
    init_dict_RBF["flg_train_lambda"] = False
    init_dict_RBF["sigma_n_init"] = 1 * np.ones(1)
    init_dict_RBF["flg_train_sigma_n"] = True
    init_dict_RBF["sigma_n_num"] = None
    init_dict_RBF["dtype"] = dtype
    init_dict_RBF["device"] = device

    model_learning_par = {}
    model_learning_par["num_gp"] = num_gp
    model_learning_par["T_sampling"] = Ts
    model_learning_par["approximation_mode"] = "SOD"
    model_learning_par["approximation_dict"] = {
        "SOD_threshold_mode": "relative",
        "SOD_threshold": 0.5,
        "flg_SOD_permutation": False,
    }
    model_learning_par["init_dict_list"] = [init_dict_RBF] * num_gp
    model_learning_par["dtype"] = dtype
    model_learning_par["device"] = device

    f_model_learning = ML.Ballistic_Model_learning_RBF

    rand_exploration_policy_par = {
        "full_state_dim": STATE_DIM,
        "u_max": uM,
        "u_min": uMin,
        "n_strata": Nexp,
        "dtype": dtype,
        "device": device,
    }
    f_rand_exploration_policy = Policy.Stratified_Throwing_Exploration

    if args.flight_targets:
        centers_init = np.column_stack(
            [
                np.random.uniform(release_xy[0], release_xy[0] + f_hi0, Nb),
                np.random.uniform(-f_hi0 * np.sin(gM), f_hi0 * np.sin(gM), Nb),
                np.random.uniform(0.0, H_MAX, Nb),   # basket height dimension
            ]
        )
    else:
        centers_init = np.column_stack(
            [
                np.random.uniform(lm * np.cos(-gM), lM, Nb),
                np.random.uniform(-lM * np.sin(gM), lM * np.sin(gM), Nb),
                np.random.uniform(0.0, H_MAX, Nb),   # basket height dimension
            ]
        )
    weight_init = uM * (np.random.rand(1, Nb) - 0.5)

    control_policy_par = {
        "full_state_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "num_basis": Nb,
        "u_max": uM,
        "lengthscales_init": lengthscales_init,
        "centers_init": centers_init,
        "weight_init": weight_init,
        "flg_drop": True,
        "dtype": dtype,
        "device": device,
    }
    f_control_policy = Policy.Throwing_Policy

    policy_reinit_dict = {
        "lenghtscales_par": lengthscales_init,
        "centers_par": np.array([1.0, 1.0, 1.0]),
        "weight_par": uM,
    }

    cost_function_par = {
        "position_indices": [0, 1],
        "target_indices": [6, 7],
        "lengthscale": lc,
        "dtype": dtype,
        "device": device,
    }
    f_cost_function = Cost_function.Throwing_Cost

    results_root = args.results_root or "results_mc_pilot_pb_A_hgen"
    log_path = os.path.join(results_root, str(seed))
    os.makedirs(log_path, exist_ok=True)

    mc_pilot_obj = MC_PILCO_module.MC_PILOT(
        target_sampler=sample_target,
        release_position=RELEASE_POS,
        throwing_system=throwing_system,
        T_sampling=Ts,
        state_dim=STATE_DIM,
        input_dim=INPUT_DIM,
        f_model_learning=f_model_learning,
        model_learning_par=model_learning_par,
        f_rand_exploration_policy=f_rand_exploration_policy,
        rand_exploration_policy_par=rand_exploration_policy_par,
        f_control_policy=f_control_policy,
        control_policy_par=control_policy_par,
        f_cost_function=f_cost_function,
        cost_function_par=cost_function_par,
        std_meas_noise=1e-3 * np.ones(STATE_DIM),
        log_path=log_path,
        dtype=dtype,
        device=device,
        arm_noise=None,
        target_height=args.target_height,
        Na=args.Na,
    )

    model_optimization_opt_dict = {}
    model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr = 0.01)"
    model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
    model_optimization_opt_dict["N_epoch"] = 1001
    model_optimization_opt_dict["N_epoch_print"] = 500
    model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

    policy_optimization_dict = {}
    policy_optimization_dict["num_particles"] = M
    n_list = Nexp + num_trials
    policy_optimization_dict["opt_steps_list"] = [Nopt] * n_list
    policy_optimization_dict["lr_list"] = [0.01] * n_list
    policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
    policy_optimization_dict["num_step_print"] = 100
    policy_optimization_dict["p_dropout_list"] = [0.25] * n_list
    policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
    policy_optimization_dict["alpha_diff_cost"] = 0.99
    policy_optimization_dict["min_diff_cost"] = 0.02
    policy_optimization_dict["num_min_diff_cost"] = 400
    policy_optimization_dict["min_step"] = 400
    policy_optimization_dict["lr_min"] = 0.0025
    policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict

    centre_target = np.array([lm + (lM - lm) / 2, 0.0])
    initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre_target])
    initial_state_var = np.concatenate(
        [
            1e-4 * np.ones(6),
            (0.5 * (lM - lm)) ** 2 * np.ones(2),
        ]
    )

    reinforce_param_dict = {
        "initial_state": initial_state,
        "initial_state_var": initial_state_var,
        "T_exploration": T,
        "T_control": T,
        "num_trials": num_trials,
        "num_explorations": Nexp,
        "model_optimization_opt_list": model_optimization_opt_list,
        "policy_optimization_dict": policy_optimization_dict,
    }

    config_log = {
        "seed": seed,
        "config": "PB_A_zero_noise_robot_specific",
        "robot_name": profile.name,
        "robot_profile": profile_to_dict(profile),
        "simulator": "PyBullet",
        "arm_noise": "None",
        "release_pos": RELEASE_POS.tolist(),
        "num_trials": num_trials,
        "Nexp": Nexp,
        "Na": args.Na,
        "Nopt": Nopt,
        "M": M,
        "Nb": Nb,
        "uM": uM,
        "uMin": uMin,
        "Ts": Ts,
        "T": T,
        "T_W": T_W,
        "T_R": T_R,
        "T_ARM": T_ARM,
        "lc": lc,
        "lm": lm,
        "lM": lM,
        "gM": float(gM),
        "flight_targets": bool(args.flight_targets),
        "H_MAX": float(H_MAX),
        "height_slope": height_slope,
        "lengthscales_init": list(lengthscales_init),
        "results_root": results_root,
        "opt_pose": args.opt_pose,
        "opt_launch_deg": float(opt_launch_deg),
        "release_pos": list(map(float, RELEASE_POS)),
    }
    pkl.dump(config_log, open(os.path.join(log_path, "config_log.pkl"), "wb"))

    print(
        f"\nTraining robot={profile.name}, seed={seed}, "
        f"uMin={uMin:.3f}, uM={uM:.3f}, range=[{lm:.3f}, {lM:.3f}]"
    )
    cost_trial_list, _, _ = mc_pilot_obj.reinforce(**reinforce_param_dict)

    print("\n\nTraining complete.")
    print(f"Results saved to: {log_path}")
    print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")


if __name__ == "__main__":
    main()
