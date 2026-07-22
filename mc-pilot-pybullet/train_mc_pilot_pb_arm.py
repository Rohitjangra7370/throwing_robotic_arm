"""
Arm-specific MC-PILOT PyBullet training based on Config PB-A.

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
    "kuka_iiwa": (0.6, 1.1),
    "franka_panda": (0.6, 1.1),
    "xarm6": (0.6, 1.0),
    # Recalibrated: (0.67, 0.87) assumed speed_bounds up to 1.0 m/s was fully
    # achievable; qd_max clip_scale showed clipping starts at u=0.625 and the
    # true zero-clipping ceiling across the full +-30-deg azimuth range is
    # u=0.61 (see robot_profiles.py's kinova_gen3 notes and speed_bounds).
    # 0.75 m was measured via a real rollout at u=0.60 — but a target AT that
    # distance needs u = uM exactly, which the (uM/2)(tanh+1) squash can only
    # approach asymptotically; 0.74 (needs u~0.57) leaves the policy headroom.
    # Use with --flight_targets so the whole domain is reachable off-axis too.
    "kinova_gen3": (0.67, 0.74),
    "kinova_gen3_dyn": (0.67, 0.74),
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
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--results_root", type=str, default=None)

    parser.add_argument("--Nexp", type=int, default=5)
    parser.add_argument(
        "--Na", type=int, default=0,
        help=(
            "data-augmentation trajectories per real trial (paper Table 1: 0 sim, "
            "2 real) -- each is a copy of the real trajectory rotated by a random "
            "angle about the vertical axis; free-flight ballistics are rotationally "
            "symmetric so these are physically valid, no new real interaction needed"
        ),
    )
    parser.add_argument("--Nopt", type=int, default=1500)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--Nb", type=int, default=250)
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
        "--device",
        type=str,
        default="cpu",
        help="torch device for GP/policy optimization (cpu or cuda); PyBullet stays CPU",
    )
    parser.add_argument(
        "--flight_targets",
        action="store_true",
        help=(
            "sample targets by FLIGHT distance from the release point instead of "
            "the paper's polar-from-origin convention. The polar convention "
            "ignores the release-position offset, so off-axis cells of the "
            "(lm,lM)x(+-gM) wedge can need far more flight than the arm's speed "
            "ceiling delivers (for kinova_gen3 at uM=0.6, nothing beyond ~15 deg "
            "azimuth is reachable). With this flag, lm/lM keep their on-axis "
            "landing-distance meaning and are converted to a flight-distance "
            "annulus around the release point, so every sampled target is "
            "reachable at every azimuth."
        ),
    )
    parser.add_argument(
        "--residual_physics",
        action="store_true",
        help=(
            "TossingBot-style residual policy (Zeng et al. 2019): release speed = "
            "clamp(v_hat(target) + learned_residual). v_hat is the analytical no-drag "
            "parabolic release speed (paper Eq. 13); the RBF learns only the residual. "
            "The policy starts AT the analytical baseline and refines it, so it is "
            ">= baseline by construction. Targets the arms where Eq.13 currently beats "
            "MC-PILOT (kuka/franka/kinova-dyn)."
        ),
    )
    parser.add_argument(
        "--delta_max_frac",
        type=float,
        default=0.5,
        help="residual bound as a fraction of uM (residual in +-frac*uM). Only used with --residual_physics.",
    )
    parser.add_argument(
        "--ball_mass", type=float, default=0.0577,
        help="thrown ball mass (kg). Default 0.0577 = tennis ball. Lower + larger radius => higher drag.",
    )
    parser.add_argument(
        "--ball_radius", type=float, default=0.0327,
        help="thrown ball radius (m). Default 0.0327 = tennis ball. e.g. 0.004kg/0.06m ~= whiffle (~19%%g drag).",
    )
    parser.add_argument(
        "--opt_pose", type=str, default=None,
        help=(
            "path to a throw_pose.npy (from find_throw_pose.py) = an AIMED release pose. "
            "Enables real-dynamics throwing from that posture with direction-constrained "
            "joint velocities (base=azimuth, velocity vector aimed at target). Requires a "
            "torque-mode robot (kinova_gen3_dyn)."
        ),
    )
    parser.add_argument(
        "--residual_dynamics",
        action="store_true",
        help=(
            "Residual physics at the MODEL level: the GP learns delta_v - gravity "
            "instead of the full delta_v, so the dynamics model is grounded in known "
            "gravity and extrapolates correctly into thinly-sampled regions. Attacks "
            "the model bias at its source (unlike --residual_physics, which only "
            "re-parameterises the policy and inherits the biased model)."
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
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    STATE_DIM = 8
    INPUT_DIM = 1
    BALL_DIM = 6
    TARGET_DIM = 2

    RELEASE_POS = np.array(profile.default_release_pos, dtype=float)
    T_W, T_R, T_ARM = profile.timing

    opt_posture = None
    opt_posture_table = None
    opt_launch_deg = 35.0
    if args.opt_pose is not None:
        _loaded = np.load(args.opt_pose, allow_pickle=True)
        if _loaded.ndim == 1:
            # azimuth->posture TABLE (array of dicts) -- the hardware-valid mode
            opt_posture_table = list(_loaded)
            opt_launch_deg = float(opt_posture_table[0]["elev_deg"])
        else:
            # legacy single-posture file (dict)
            _pose = _loaded.item()
            opt_posture = np.array(_pose["q"], dtype=float)
            opt_launch_deg = float(_pose["elev_deg"])
        # a long, gentle windup keeps joint-velocity overshoot in check for the aimed throw
        T_W, T_R = 0.5, 1.6
    lengthscale_xy = (
        args.lengthscale_xy if args.lengthscale_xy is not None else 0.15 * (lM - lm)
    )
    lengthscales_init = np.array([lengthscale_xy, lengthscale_xy], dtype=float)

    release_xy = np.array(profile.default_release_pos[:2], dtype=float)
    if opt_posture_table is not None:
        # profile.default_release_pos is a STATIC config value for the legacy
        # kinematic-release mode; opt_pose's actual release point comes from
        # real FK on the table's az~=0 posture and can differ substantially
        # (verified: 0.55,0.00 configured vs 0.696,-0.054 real -- a 0.15m gap,
        # huge relative to this throw's ~12cm range). Anchor target sampling
        # on the REAL release point instead.
        import pybullet as _p
        import pybullet_data as _pd
        _az0_entry = min(opt_posture_table, key=lambda e: abs(e["azimuth_deg"]))
        _cid = _p.connect(_p.DIRECT)
        _arm_tmp = _p.loadURDF(_pd.getDataPath() + "/" + profile.urdf_rel_path,
                               useFixedBase=True, physicsClientId=_cid)
        for _j in range(7):
            _p.resetJointState(_arm_tmp, _j, _az0_entry["q"][_j], physicsClientId=_cid)
        _real_pos = _p.getLinkState(_arm_tmp, profile.ee_link, computeForwardKinematics=True,
                                    physicsClientId=_cid)[4]
        _p.disconnect(_cid)
        release_xy = np.array(_real_pos[:2], dtype=float)
        # RELEASE_POS (3D) feeds the GP particle model's belief about where
        # the ball launches from (release_position= into MC_PILOT, and
        # initial_state below) -- NOT just release_xy used for target
        # sampling. Previously only release_xy got the real-FK fix; for the
        # old ~12cm-range throw the resulting z/x gap (0.55,0,0.45 configured
        # vs ~0.70,-0.05,0.72 real) was small enough to not matter. For the
        # overhead throw it's 0.75m in x and 0.55m in z -- the particle
        # rollout was training against a completely fictional launch point,
        # so the policy learned to compensate for THAT point and then
        # systematically undershot every real target (measured: 39.7cm mean
        # error, near-max speed commanded regardless of target distance).
        # Both must be anchored on the same real point.
        RELEASE_POS = np.array(_real_pos, dtype=float)
        print(f"opt_pose table: anchoring target sampling AND particle release "
             f"position on REAL release_pos={RELEASE_POS} "
             f"(profile default was {profile.default_release_pos})")
    if args.flight_targets:
        # lm/lM are on-axis landing distances (release y = 0, so on-axis
        # flight = distance - release_x); convert to a flight annulus.
        f_lo = lm - release_xy[0]
        f_hi = lM - release_xy[0]
        if not (0.0 < f_lo < f_hi):
            raise ValueError(
                f"flight_targets: invalid flight range [{f_lo:.3f}, {f_hi:.3f}] "
                f"from lm={lm}, lM={lM}, release_x={release_xy[0]}"
            )

        def sample_target():
            flight = np.random.uniform(f_lo, f_hi)
            beta = np.random.uniform(-gM, gM)
            return release_xy + flight * np.array([np.cos(beta), np.sin(beta)])
    else:
        def sample_target():
            dist = np.random.uniform(lm, lM)
            angle = np.random.uniform(-gM, gM)
            return np.array([dist * np.cos(angle), dist * np.sin(angle)])

    throwing_system = PyBulletThrowingSystem(
        mass=args.ball_mass,
        radius=args.ball_radius,
        launch_angle_deg=opt_launch_deg if args.opt_pose is not None else 35.0,
        arm_noise=None,
        t_w=T_W,
        t_r=T_R,
        robot_name=profile.name,
        target_height=args.target_height,
        opt_posture=opt_posture,
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
    if args.residual_dynamics:
        # Residual-physics at the MODEL level: GP learns delta_v - gravity, not full delta_v.
        f_model_learning = ML.Ballistic_SemiParametric_Model_learning_RBF

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
        # RBF centers must cover the flight-annulus target domain, which is a
        # much smaller region around the release point than the polar wedge.
        centers_init = np.column_stack(
            [
                np.random.uniform(release_xy[0], release_xy[0] + f_hi, Nb),
                np.random.uniform(-f_hi * np.sin(gM), f_hi * np.sin(gM), Nb),
            ]
        )
    else:
        centers_init = np.column_stack(
            [
                np.random.uniform(lm * np.cos(-gM), lM, Nb),
                np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
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

    if args.residual_physics:
        # Start at the analytical baseline: near-zero residual weights so the initial
        # policy output == v_hat, then trials refine only the residual.
        control_policy_par["weight_init"] = 1e-2 * (np.random.rand(1, Nb) - 0.5)
        control_policy_par["release_pos"] = RELEASE_POS
        control_policy_par["launch_angle_deg"] = 35.0
        control_policy_par["target_height"] = float(args.target_height)
        control_policy_par["delta_max_frac"] = float(args.delta_max_frac)
        f_control_policy = Policy.Residual_Throwing_Policy

    policy_reinit_dict = {
        "lenghtscales_par": lengthscales_init,
        "centers_par": np.array([1.0, 1.0]),
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

    results_root = args.results_root or default_results_root(profile.name, args.target_height)
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
        "lengthscales_init": list(lengthscales_init),
        "results_root": results_root,
        "ball_mass": float(args.ball_mass),
        "ball_radius": float(args.ball_radius),
        "residual_physics": bool(args.residual_physics),
        "residual_dynamics": bool(args.residual_dynamics),
        "launch_angle_deg": 35.0,
        "delta_max_frac": float(args.delta_max_frac) if args.residual_physics else None,
        "target_height": float(args.target_height),
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
