"""
MC-PILOT PyBullet noise study runner.

Purpose:
- Run a single training/evaluation job for one noise type and one noise level.
- Save logs in a structured folder so downstream scripts can build tables/plots.

Supported noise types:
- slip        : VelocitySlipNoise (multiplicative + Gaussian)
- saltpepper  : SaltAndPepperVelocityNoise (impulsive outliers)

Examples:
  python test_mc_pilot_pb_noise_study.py -noise_type slip -alpha 0.20 -sigma 0.04 -noise_aware 1
  python test_mc_pilot_pb_noise_study.py -noise_type saltpepper -p_spike 0.15 -spike_scale 0.30 -noise_aware 0
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
import simulation_class.model as model_module
from robot_arm.noise_models import SaltAndPepperVelocityNoise, VelocitySlipNoise


class NoisyThrowingSystem(model_module.ThrowingSystem):
    """
    Numpy-ballistics backend with release-velocity noise injection.

    This mirrors the one-shot speed->velocity flow from ThrowingSystem but applies
    arm_noise at release before free-flight integration.
    """

    def __init__(self, *args, arm_noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm_noise = arm_noise

    def rollout(self, s0, policy, T, dt, noise):
        state_dim = len(s0)
        release_pos = np.array(s0[0:3], dtype=float)
        target_xy = np.array(s0[6:8], dtype=float)

        u0 = np.array(policy(s0, 0.0)).flatten()
        speed = float(u0[0])
        release_vel = self._speed_to_velocity(speed, release_pos, target_xy)

        if self.arm_noise is not None:
            release_vel = self.arm_noise.pybullet_release_vel(release_vel, release_vel)

        pos_traj, vel_traj = self._simulate(release_pos, release_vel, T, dt)
        n = len(pos_traj)

        target_col = np.tile(target_xy, (n, 1))
        clean_ball = np.hstack([pos_traj, vel_traj])
        clean_states = np.hstack([clean_ball, target_col])

        noise_arr = np.ones(state_dim) * noise if np.isscalar(noise) else np.array(noise)
        noisy_states = clean_states + np.random.randn(n, state_dim) * noise_arr

        inputs = np.zeros((n, 1))
        inputs[0, 0] = speed
        return noisy_states, inputs, clean_states


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("mc-pilot-pybullet noise study")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-num_trials", type=int, default=6)
parser.add_argument("-noise_type", type=str, default="slip", choices=["slip", "saltpepper"])
parser.add_argument("-noise_aware", type=int, default=1,
                    help="1=inject noise in apply_policy, 0=naive baseline")
parser.add_argument("-backend", type=str, default="auto", choices=["auto", "pybullet", "numpy"],
                    help="simulation backend: auto tries pybullet first, then numpy fallback")

# slip params
parser.add_argument("-alpha", type=float, default=0.20,
                    help="VelocitySlip alpha (fractional speed loss)")
parser.add_argument("-sigma", type=float, default=0.04,
                    help="VelocitySlip additive Gaussian std (m/s)")

# salt-and-pepper params
parser.add_argument("-p_spike", type=float, default=0.10,
                    help="Salt-and-Pepper per-component impulse probability")
parser.add_argument("-spike_scale", type=float, default=0.30,
                    help="Salt-and-Pepper impulse magnitude (m/s)")
parser.add_argument("-sp_sigma", type=float, default=0.0,
                    help="Optional Gaussian background std for Salt-and-Pepper")

# training/sim params
parser.add_argument("-Nexp", type=int, default=8)
parser.add_argument("-Nopt", type=int, default=500)
parser.add_argument("-M", type=int, default=400)
parser.add_argument("-Nb", type=int, default=250)
parser.add_argument("-uM", type=float, default=3.0)
parser.add_argument("-uMin", type=float, default=1.4)
parser.add_argument("-Ts", type=float, default=0.02)
parser.add_argument("-T", type=float, default=0.60)
parser.add_argument("-lc", type=float, default=0.5)
parser.add_argument("-lm", type=float, default=0.6)
parser.add_argument("-lM", type=float, default=1.1)
parser.add_argument("-results_root", type=str, default="results_mc_pilot_pb_noise_study")
args = parser.parse_known_args()[0]

seed = args.seed
num_trials = args.num_trials
noise_aware = bool(args.noise_aware)

np.random.seed(seed)
torch.manual_seed(seed)

dtype = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

Nexp = args.Nexp
Nopt = args.Nopt
M = args.M
Nb = args.Nb
uM = args.uM
uMin = args.uMin
Ts = args.Ts
T = args.T
lc = args.lc
lm = args.lm
lM = args.lM
gM = np.pi / 6

STATE_DIM = 8
INPUT_DIM = 1
BALL_DIM = 6
TARGET_DIM = 2

RELEASE_POS = np.array([0.0, 0.0, 0.5])


def sample_target():
    dist = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])


# ---------------------------------------------------------------------------
# Noise setup
# ---------------------------------------------------------------------------
if args.noise_type == "slip":
    arm_noise = VelocitySlipNoise(alpha=args.alpha, sigma=args.sigma, seed=seed + 100)
    noise_name = "VelocitySlipNoise"
    noise_tag = f"slip_alpha_{args.alpha:.3f}_sigma_{args.sigma:.3f}".replace(".", "p")
elif args.noise_type == "saltpepper":
    arm_noise = SaltAndPepperVelocityNoise(
        p_spike=args.p_spike,
        spike_scale=args.spike_scale,
        sigma=args.sp_sigma,
        seed=seed + 100,
    )
    noise_name = "SaltAndPepperVelocityNoise"
    noise_tag = (
        f"saltpepper_p_{args.p_spike:.3f}_scale_{args.spike_scale:.3f}_sigma_{args.sp_sigma:.3f}"
    ).replace(".", "p")
else:
    raise ValueError(f"Unsupported noise_type: {args.noise_type}")

backend_used = None
if args.backend in ["auto", "pybullet"]:
    try:
        from simulation_class.model_pybullet import PyBulletThrowingSystem

        throwing_system = PyBulletThrowingSystem(
            mass=0.0577,
            radius=0.0327,
            launch_angle_deg=35.0,
            arm_noise=arm_noise,
        )
        backend_used = "pybullet"
    except Exception as e:
        if args.backend == "pybullet":
            raise
        print(f"[WARN] PyBullet unavailable, falling back to numpy backend. Reason: {e}")

if backend_used is None:
    throwing_system = NoisyThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
        arm_noise=arm_noise,
    )
    backend_used = "numpy"


# ---------------------------------------------------------------------------
# GP model learning
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Exploration + control policy
# ---------------------------------------------------------------------------
rand_exploration_policy_par = {
    "full_state_dim": STATE_DIM,
    "u_max": uM,
    "u_min": uMin,
    "n_strata": Nexp,
    "dtype": dtype,
    "device": device,
}
f_rand_exploration_policy = Policy.Stratified_Throwing_Exploration

centers_init = np.column_stack([
    np.random.uniform(lm * np.cos(-gM), lM, Nb),
    np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
])
weight_init = uM * (np.random.rand(1, Nb) - 0.5)
lengthscales_init = np.array([0.08, 0.08])

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
    "centers_par": np.array([1.0, 1.0]),
    "weight_par": uM,
}


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------
cost_function_par = {
    "position_indices": [0, 1],
    "target_indices": [6, 7],
    "lengthscale": lc,
    "dtype": dtype,
    "device": device,
}
f_cost_function = Cost_function.Throwing_Cost


# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
aware_tag = "aware" if noise_aware else "naive"
log_path = os.path.join(args.results_root, noise_tag + f"_{aware_tag}", str(seed))
os.makedirs(log_path, exist_ok=True)


# ---------------------------------------------------------------------------
# Build MC_PILOT object
# ---------------------------------------------------------------------------
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
    arm_noise=arm_noise if noise_aware else None,
)


# ---------------------------------------------------------------------------
# Optimization options
# ---------------------------------------------------------------------------
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
policy_optimization_dict["num_min_diff_cost"] = 300
policy_optimization_dict["min_step"] = min(300, Nopt)
policy_optimization_dict["lr_min"] = 0.0025
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict

centre_target = np.array([lm + (lM - lm) / 2, 0.0])
initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre_target])
initial_state_var = np.concatenate([
    1e-4 * np.ones(6),
    (0.5 * (lM - lm)) ** 2 * np.ones(2),
])

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


# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
config_log = {
    "seed": seed,
    "config": "PB_noise_study",
    "simulator": backend_used,
    "backend": backend_used,
    "arm_noise": noise_name,
    "noise_type": args.noise_type,
    "noise_aware": noise_aware,
    "alpha": args.alpha,
    "sigma": args.sigma,
    "p_spike": args.p_spike,
    "spike_scale": args.spike_scale,
    "sp_sigma": args.sp_sigma,
    "release_pos": RELEASE_POS.tolist(),
    "num_trials": num_trials,
    "Nexp": Nexp,
    "Nopt": Nopt,
    "M": M,
    "Nb": Nb,
    "uM": uM,
    "uMin": uMin,
    "Ts": Ts,
    "T": T,
    "lc": lc,
    "lm": lm,
    "lM": lM,
    "gM": float(gM),
    "lengthscales_init": list(lengthscales_init),
}
pkl.dump(config_log, open(os.path.join(log_path, "config_log.pkl"), "wb"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("\nRunning PB noise study")
print(f"  noise_type   : {args.noise_type}")
print(f"  noise_aware  : {noise_aware}")
print(f"  seed         : {seed}")
print(f"  num_trials   : {num_trials}")
print(f"  log_path     : {log_path}")

cost_trial_list, particles_states_list, particles_inputs_list = mc_pilot_obj.reinforce(
    **reinforce_param_dict
)

print("\nTraining complete.")
print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")
