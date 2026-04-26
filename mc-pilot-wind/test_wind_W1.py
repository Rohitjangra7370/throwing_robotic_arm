"""
MC-PILOT Wind Experiment W1: Constant Wind

Systematic comparison of throwing accuracy under steady crosswind.
Tests whether the GP can implicitly absorb a constant wind bias.

Variants (set via -wind_speed and -wind_aware):
  W1-calm:     wind=[0,0,0]        blind   → baseline reference
  W1-light:    wind=[0.3,0,0]      blind   → mild deflection
  W1-moderate: wind=[0.7,0,0]      blind   → noticeable deflection
  W1-strong:   wind=[1.0,0,0]      blind   → significant deflection
  W1-aware:    wind=[0.7,0,0]      aware   → GP+policy see wind

Usage:
  python test_wind_W1.py -seed 1 -num_trials 15 -wind_speed 0.0
  python test_wind_W1.py -seed 1 -num_trials 15 -wind_speed 0.7
  python test_wind_W1.py -seed 1 -num_trials 15 -wind_speed 0.7 -wind_aware 1
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import model_learning.Model_learning as ML
import model_learning.Model_learning_wind as ML_wind
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO_module
import policy_learning.Policy as Policy
from simulation_class.wind_models import ConstantWind, WindModel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("MC-PILOT Wind W1 — Constant Wind")
p.add_argument("-seed",        type=int,   default=1)
p.add_argument("-num_trials",  type=int,   default=15)
p.add_argument("-wind_speed",  type=float, default=0.0,
               help="crosswind speed in +x direction (m/s)")
p.add_argument("-wind_aware",  type=int,   default=0,
               help="1=10-D state with wind input, 0=8-D blind")
p.add_argument("--pybullet",   action="store_true",
               help="Run physics in PyBullet instead of numpy")
p.add_argument("--resume",     action="store_true",
               help="Resume from previous log in log_path")
args = p.parse_known_args()[0]
seed = args.seed
num_trials = args.num_trials
wind_speed = args.wind_speed
wind_aware = bool(args.wind_aware)

torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Parameters (matching baseline Config A)
# ---------------------------------------------------------------------------
Nexp   = 10 if wind_aware else 5  # aware needs more initial data
Nopt   = 1500
M      = 400
Nb     = 250
uM     = 2.5
Ts     = 0.02
T      = 0.70
lc     = 0.5
lm     = 0.75
lM     = 1.75
gM     = np.pi / 6

if wind_aware:
    STATE_DIM = 10  # [x,y,z, vx,vy,vz, Px,Py, wx,wy]
else:
    STATE_DIM = 8   # [x,y,z, vx,vy,vz, Px,Py]
INPUT_DIM  = 1
BALL_DIM   = 6
TARGET_DIM = 2
WIND_DIM   = 2

RELEASE_POS = np.array([0.0, 0.0, 0.5])

# ---------------------------------------------------------------------------
# Target sampler
# ---------------------------------------------------------------------------
def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# ---------------------------------------------------------------------------
# Wind model
# ---------------------------------------------------------------------------
if wind_speed > 0:
    wind_model = ConstantWind(velocity=[wind_speed, 0.0, 0.0])
else:
    wind_model = WindModel()  # zero wind

# Wind sampler for apply_policy (wind-aware mode)
def wind_sampler():
    """Returns constant wind for all particles."""
    return np.array([wind_speed, 0.0])

# ---------------------------------------------------------------------------
# Throwing system
# ---------------------------------------------------------------------------
if args.pybullet:
    import sys
    sys.path.append("../mc-pilot-pybullet")
    from simulation_class.model_pybullet import PyBulletThrowingSystem
    throwing_system = PyBulletThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
        wind_model=wind_model,
        wind_aware=wind_aware,
        gui_mode=False,
    )
else:
    from simulation_class.model_wind import WindThrowingSystem
    throwing_system = WindThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
        wind_model=wind_model,
        wind_aware=wind_aware,
    )

# ---------------------------------------------------------------------------
# GP model learning
# ---------------------------------------------------------------------------
if wind_aware:
    num_gp = 3
    gp_input_dim = BALL_DIM + WIND_DIM  # 8
    f_model_learning = ML_wind.WindAware_Ballistic_Model_learning_RBF
else:
    num_gp = 3
    gp_input_dim = BALL_DIM  # 6
    f_model_learning = ML.Ballistic_Model_learning_RBF

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
model_learning_par["num_gp"]             = num_gp
model_learning_par["T_sampling"]         = Ts
model_learning_par["approximation_mode"] = "SOD"
model_learning_par["approximation_dict"] = {
    "SOD_threshold_mode": "relative",
    "SOD_threshold": 0.5,
    "flg_SOD_permutation": False,
}
model_learning_par["init_dict_list"] = [init_dict_RBF] * num_gp
model_learning_par["dtype"]          = dtype
model_learning_par["device"]         = device

# ---------------------------------------------------------------------------
# Exploration policy
# ---------------------------------------------------------------------------
rand_exploration_policy_par = {
    "full_state_dim": STATE_DIM,
    "u_max": uM,
    "n_strata": Nexp,
    "dtype": dtype,
    "device": device,
}
f_rand_exploration_policy = Policy.Stratified_Throwing_Exploration

# ---------------------------------------------------------------------------
# Control policy
# ---------------------------------------------------------------------------
if wind_aware:
    # 4-D RBF: [Px, Py, wx, wy]
    rbf_input_dim = TARGET_DIM + WIND_DIM
    centers_init = np.column_stack([
        np.random.uniform(lm * np.cos(-gM), lM, Nb),
        np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
        np.random.uniform(wind_speed - 0.1, wind_speed + 0.1, Nb),  # wind wx centers
        np.random.uniform(-0.1, 0.1, Nb),  # wind wy centers
    ])
    weight_init = uM * (np.random.rand(1, Nb) - 0.5)
    lengthscales_init = np.array([0.08, 0.08, 0.3, 0.3])

    control_policy_par = {
        "full_state_dim": STATE_DIM,
        "target_dim": TARGET_DIM,
        "wind_dim": WIND_DIM,
        "num_basis": Nb,
        "u_max": uM,
        "lengthscales_init": lengthscales_init,
        "centers_init": centers_init,
        "weight_init": weight_init,
        "flg_drop": True,
        "dtype": dtype,
        "device": device,
    }
    f_control_policy = Policy.WindAware_Throwing_Policy

    policy_reinit_dict = {
        "lenghtscales_par": lengthscales_init,
        "centers_par": np.array([1.0, 1.0, 0.5, 0.5]),
        "weight_par": uM,
    }
else:
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
    "target_indices":   [6, 7],
    "lengthscale": lc,
    "dtype": dtype,
    "device": device,
}
f_cost_function = Cost_function.Throwing_Cost

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
ws_tag = f"w{wind_speed:.1f}".replace(".", "p")
mode_tag = "aware" if wind_aware else "blind"
log_path = f"results_wind_W1/{ws_tag}_{mode_tag}/{seed}"
os.makedirs(log_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Build MC_PILOT_Wind object
# ---------------------------------------------------------------------------
mc_pilot_obj = MC_PILCO_module.MC_PILOT_Wind(
    target_sampler=sample_target,
    release_position=RELEASE_POS,
    throwing_system=throwing_system,
    wind_model=wind_model,
    wind_aware=wind_aware,
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
    wind_sampler=wind_sampler if wind_aware else None,
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
if wind_aware:
    initial_state = np.concatenate([initial_state, [wind_speed, 0.0]])
initial_state_var = np.concatenate([
    1e-4 * np.ones(6),
    (0.5 * (lM - lm)) ** 2 * np.ones(2),
])
if wind_aware:
    initial_state_var = np.concatenate([initial_state_var, [0.01, 0.01]])

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
    "seed": seed, "config": "W1_constant_wind",
    "wind_speed": wind_speed, "wind_aware": wind_aware,
    "wind_model": wind_model.describe(),
    "num_trials": num_trials,
    "Nexp": Nexp, "Nopt": Nopt, "M": M, "Nb": Nb,
    "uM": uM, "Ts": Ts, "T": T, "lc": lc,
    "lm": lm, "lM": lM, "gM": float(gM),
}
pkl.dump(config_log, open(log_path + "/config_log.pkl", "wb"))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"\nW1 Constant Wind: speed={wind_speed} m/s, aware={wind_aware}, seed={seed}")

if args.resume:
    log_file_path = os.path.join(log_path, "log.pkl")
    if os.path.exists(log_file_path):
        import pickle
        log_dict = pickle.load(open(log_file_path, "rb"))
        num_completed = len(log_dict.get("cost_trial_list", []))
        if num_completed > 0:
            print(f"\nResuming from trial {num_completed} (Control trial index).")
            mc_pilot_obj.load_model_from_log(
                num_trial=num_completed, 
                num_explorations=Nexp, 
                folder=log_path + "/"
            )
            reinforce_param_dict["loaded_model"] = True
            reinforce_param_dict["num_trials"] = max(0, num_trials - num_completed)
        else:
            print("\nNo completed trials found to resume from. Starting fresh.")
    else:
        print("\nNo log file found to resume from. Starting fresh.")

cost_trial_list, particles_states_list, particles_inputs_list = mc_pilot_obj.reinforce(
    **reinforce_param_dict
)

print("\n\nTraining complete.")
print(f"Final trial cost: {cost_trial_list[-1][-1]:.4f}")
