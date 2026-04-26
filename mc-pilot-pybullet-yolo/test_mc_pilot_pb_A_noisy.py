"""
MC-PILOT Config PB-A-Noisy: PyBullet arm + VelocityBiasNoise — the core RL demo.
AR525 Group-3, IIT Mandi

This is the key experiment: the arm delivers v_cmd + dv at release, where
dv ~ N(0, sigma²·I₃).  The same distribution is injected in apply_policy so
the RL learns to be robust to this stochasticity.

Demonstrates that RL is genuinely necessary: without noise awareness (arm_noise=None
in MC_PILOT while keeping noise in PyBulletThrowingSystem), the policy ignores the
uncertainty and hit rate degrades.  With noise awareness (same instance in both),
the policy learns a robust speed mapping that accounts for velocity scatter.

Usage:
  python test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10
  python test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10 -sigma 0.10
  python test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10 -sigma 0.20
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
import simulation_class.model as model_module
from simulation_class.model_pybullet import PyBulletThrowingSystem
from robot_arm.noise_models import VelocityBiasNoise

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("mc-pilot-pybullet Config A with arm noise")
p.add_argument("-seed",       type=int,   default=1)
p.add_argument("-num_trials", type=int,   default=10)
p.add_argument("-sigma",      type=float, default=0.15,
               help="std dev of release velocity noise (m/s)")
p.add_argument("-noise_aware", type=int, default=1,
               help="1=inject noise in apply_policy (robust), 0=only in data collection")
locals().update(vars(p.parse_known_args()[0]))

torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Parameters  (same as PB-A, plus sigma)
# ---------------------------------------------------------------------------
Nexp   = 5
Nopt   = 1500
M      = 400
Nb     = 250
uM     = 2.5
uMin   = 1.4
Ts     = 0.02
T      = 0.60
lc     = 0.5
lm     = 0.6    # calibrated: v=1.4 reaches 0.58m, so 0.6m is safely above uMin floor
lM     = 1.1    # calibrated: v=2.5 reaches 1.14m, so 1.1m is reachable with margin
gM     = np.pi / 6

STATE_DIM  = 8
INPUT_DIM  = 1
BALL_DIM   = 6
TARGET_DIM = 2

RELEASE_POS = np.array([0.0, 0.0, 0.5])

# ---------------------------------------------------------------------------
# Target sampler
# ---------------------------------------------------------------------------
def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# ---------------------------------------------------------------------------
# Noise model — shared between simulator and MC_PILOT
# ---------------------------------------------------------------------------
arm_noise = VelocityBiasNoise(sigma=sigma, seed=seed + 100)

# ---------------------------------------------------------------------------
# PyBullet throwing system (with noise)
# ---------------------------------------------------------------------------
throwing_system = PyBulletThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=35.0,
    arm_noise=arm_noise,
)

# ---------------------------------------------------------------------------
# GP model learning
# ---------------------------------------------------------------------------
num_gp       = 3
gp_input_dim = BALL_DIM

init_dict_RBF = {}
init_dict_RBF["active_dims"]            = np.arange(0, gp_input_dim)
init_dict_RBF["lengthscales_init"]      = np.ones(gp_input_dim)
init_dict_RBF["flg_train_lengthscales"] = True
init_dict_RBF["lambda_init"]            = np.ones(1)
init_dict_RBF["flg_train_lambda"]       = False
init_dict_RBF["sigma_n_init"]           = 1 * np.ones(1)
init_dict_RBF["flg_train_sigma_n"]      = True
init_dict_RBF["sigma_n_num"]            = None
init_dict_RBF["dtype"]                  = dtype
init_dict_RBF["device"]                 = device

model_learning_par = {}
model_learning_par["num_gp"]          = num_gp
model_learning_par["T_sampling"]      = Ts
model_learning_par["approximation_mode"] = "SOD"
model_learning_par["approximation_dict"] = {
    "SOD_threshold_mode": "relative",
    "SOD_threshold": 0.5,
    "flg_SOD_permutation": False,
}
model_learning_par["init_dict_list"]  = [init_dict_RBF] * num_gp
model_learning_par["dtype"]           = dtype
model_learning_par["device"]          = device

f_model_learning = ML.Ballistic_Model_learning_RBF

# ---------------------------------------------------------------------------
# Exploration policy — stratified over [uMin, uM]
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

# ---------------------------------------------------------------------------
# Control policy
# ---------------------------------------------------------------------------
centers_init = np.column_stack([
    np.random.uniform(lm * np.cos(-gM), lM, Nb),
    np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
])
weight_init       = uM * (np.random.rand(1, Nb) - 0.5)
lengthscales_init = np.array([0.08, 0.08])

control_policy_par = {
    "full_state_dim": STATE_DIM,
    "target_dim":     TARGET_DIM,
    "num_basis":      Nb,
    "u_max":          uM,
    "lengthscales_init": lengthscales_init,
    "centers_init":   centers_init,
    "weight_init":    weight_init,
    "flg_drop":       True,
    "dtype":          dtype,
    "device":         device,
}
f_control_policy = Policy.Throwing_Policy

policy_reinit_dict = {
    "lenghtscales_par": lengthscales_init,
    "centers_par":      np.array([1.0, 1.0]),
    "weight_par":       uM,
}

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------
cost_function_par = {
    "position_indices": [0, 1],
    "target_indices":   [6, 7],
    "lengthscale":      lc,
    "dtype":            dtype,
    "device":           device,
}
f_cost_function = Cost_function.Throwing_Cost

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
sigma_str = f"{sigma:.2f}".replace(".", "p")
log_path  = f"results_mc_pilot_pb_A_noisy/sigma_{sigma_str}_{('aware' if noise_aware else 'naive')}/{seed}"
os.makedirs(log_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Build MC_PILOT object
# noise_aware=1: inject noise in apply_policy (robust RL)
# noise_aware=0: only inject in data collection (naive baseline — should do worse)
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
# MC-PILCO options
# ---------------------------------------------------------------------------
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"]   = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"]     = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"]       = 1001
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

policy_optimization_dict = {}
policy_optimization_dict["num_particles"]    = M
n_list = Nexp + num_trials
policy_optimization_dict["opt_steps_list"]   = [Nopt] * n_list
policy_optimization_dict["lr_list"]          = [0.01] * n_list
policy_optimization_dict["f_optimizer"]      = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["num_step_print"]   = 100
policy_optimization_dict["p_dropout_list"]   = [0.25] * n_list
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
policy_optimization_dict["alpha_diff_cost"]  = 0.99
policy_optimization_dict["min_diff_cost"]    = 0.02
policy_optimization_dict["num_min_diff_cost"]= 400
policy_optimization_dict["min_step"]         = 400
policy_optimization_dict["lr_min"]           = 0.0025
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict

centre_target = np.array([lm + (lM - lm) / 2, 0.0])
initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre_target])
initial_state_var = np.concatenate([
    1e-4 * np.ones(6),
    (0.5 * (lM - lm)) ** 2 * np.ones(2),
])

reinforce_param_dict = {
    "initial_state":     initial_state,
    "initial_state_var": initial_state_var,
    "T_exploration":     T,
    "T_control":         T,
    "num_trials":        num_trials,
    "num_explorations":  Nexp,
    "model_optimization_opt_list": model_optimization_opt_list,
    "policy_optimization_dict":    policy_optimization_dict,
}

# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
config_log = {
    "seed": seed, "config": "PB_A_noisy",
    "simulator": "PyBullet", "arm_noise": "VelocityBiasNoise",
    "sigma": sigma, "noise_aware": bool(noise_aware),
    "release_pos": RELEASE_POS.tolist(),
    "num_trials": num_trials,
    "Nexp": Nexp, "Nopt": Nopt, "M": M, "Nb": Nb,
    "uM": uM, "uMin": uMin, "Ts": Ts, "T": T, "lc": lc,
    "lm": lm, "lM": lM, "gM": float(gM),
    "lengthscales_init": list(lengthscales_init),
}
pkl.dump(config_log, open(log_path + "/config_log.pkl", "wb"))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"\nPB-A Noisy: sigma={sigma}, noise_aware={bool(noise_aware)}, seed={seed}")
cost_trial_list, particles_states_list, particles_inputs_list = mc_pilot_obj.reinforce(
    **reinforce_param_dict
)

print("\n\nTraining complete.")
print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")
