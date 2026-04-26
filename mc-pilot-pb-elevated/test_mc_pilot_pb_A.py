"""
MC-PILOT Config PB-A: PyBullet arm, zero arm noise — baseline parity test.
AR525 Group-3, IIT Mandi

Purpose: verify that swapping the numpy simulator for the PyBullet arm gives
comparable hit rates to Config A from mc-pilot/.  Ball velocity at release is
set explicitly to v_cmd (no noise), so the physics match the numpy sim exactly.

Key parameter choices:
  - KUKA iiwa7 limits max EE speed to ~2.5 m/s → uM=2.5 (vs paper's 3.5)
  - lm=0.6, lM=1.1: PyBullet calibrated (1.4→0.58m, 2.5→1.14m; old lm=0.5 was unreachable)
  - target range = 0.5m → lengthscales_init=[0.08, 0.08]  (E-Strat5 rule: 0.15*range)
  - lc=0.5, Nexp=5: same rationale as Config A from mc-pilot/
  - RELEASE_POS=[0,0,0.5]: IK places iiwa7 EE at approx (0.024, 0, 0.495)

Success criterion: hit rate ≥ 4/5 within 5 trials (comparable to mc-pilot Config A).
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("mc-pilot-pybullet Config A, zero noise")
p.add_argument("-seed", type=int, default=1)
p.add_argument("-num_trials", type=int, default=10)
locals().update(vars(p.parse_known_args()[0]))

torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Nexp   = 5        # exploration throws (5 strata cover [1.4, 2.5] m/s)
Nopt   = 1500
M      = 400
Nb     = 250
uM     = 2.5      # iiwa7 max achievable EE speed
uMin   = 1.4      # floor: speed=1.4 reaches 0.58m (below lm=0.6, gives GP coverage beneath target floor)
Ts     = 0.02
T      = 0.60     # ball lands ~0.52s at uM=2.5 from z=0.5m
lc     = 0.5
lm     = 0.6      # calibrated: v=1.4→0.58m, so 0.6m is safely above uMin floor
lM     = 1.1      # calibrated: v=2.5→1.14m, so 1.1m is reachable with margin
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
# PyBullet throwing system (zero noise)
# ---------------------------------------------------------------------------
throwing_system = PyBulletThrowingSystem(
    mass=0.0577,
    radius=0.0327,
    launch_angle_deg=35.0,
    arm_noise=None,     # zero noise — baseline
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
# Control policy — lengthscales scaled to 0.15 × target range (E-Strat5 rule)
# ---------------------------------------------------------------------------
centers_init = np.column_stack([
    np.random.uniform(lm * np.cos(-gM), lM, Nb),
    np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
])
weight_init       = uM * (np.random.rand(1, Nb) - 0.5)
lengthscales_init = np.array([0.08, 0.08])   # 0.15 × (lM - lm) = 0.15 × 0.5m

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
log_path = "results_mc_pilot_pb_A/" + str(seed)
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
    arm_noise=None,
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
    "seed": seed, "config": "PB_A_zero_noise",
    "simulator": "PyBullet", "arm_noise": "None",
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
cost_trial_list, particles_states_list, particles_inputs_list = mc_pilot_obj.reinforce(
    **reinforce_param_dict
)

print("\n\nTraining complete.")
print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")
