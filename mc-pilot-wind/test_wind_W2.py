"""
MC-PILOT Wind Experiment W2: Random Gusts

Piecewise-constant wind that changes every T_gust seconds.
This is the hardest case for wind-blind GP: each throw sees different
wind, so identical ball states produce different Δv — the GP must
explain this as noise unless given wind input.

Variants (set via -wind_aware):
  W2-blind:  gusts, 8-D state  → GP sees unexplained variance
  W2-aware:  gusts, 10-D state → GP conditions on (wx, wy)

Usage:
  python test_wind_W2.py -seed 1 -num_trials 15 -w_max 0.5
  python test_wind_W2.py -seed 1 -num_trials 15 -w_max 0.5 -wind_aware 1
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
from simulation_class.wind_models import GustWind

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("MC-PILOT Wind W2 — Random Gusts")
p.add_argument("-seed",        type=int,   default=1)
p.add_argument("-num_trials",  type=int,   default=15)
p.add_argument("-w_max",       type=float, default=0.5,
               help="max gust speed per component (m/s)")
p.add_argument("-T_gust",      type=float, default=0.15,
               help="gust change interval (s)")
p.add_argument("-wind_aware",  type=int,   default=0)
p.add_argument("--pybullet",   action="store_true")
p.add_argument("--resume",     action="store_true",
               help="Resume from previous log in log_path")
args = p.parse_known_args()[0]
seed = args.seed
num_trials = args.num_trials
w_max = args.w_max
T_gust = args.T_gust
wind_aware = bool(args.wind_aware)

torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Parameters (Config A baseline)
# ---------------------------------------------------------------------------
Nexp = 10 if wind_aware else 5  # aware needs more initial data
Nopt = 1500; M = 400; Nb = 250; uM = 2.5
Ts = 0.02; T = 0.70; lc = 0.5; lm = 0.75; lM = 1.75; gM = np.pi / 6

STATE_DIM  = 10 if wind_aware else 8
INPUT_DIM  = 1; BALL_DIM = 6; TARGET_DIM = 2; WIND_DIM = 2
RELEASE_POS = np.array([0.0, 0.0, 0.5])

def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# ---------------------------------------------------------------------------
# Wind model
# ---------------------------------------------------------------------------
wind_model = GustWind(w_max=w_max, T_gust=T_gust, seed=seed + 200)

# Wind sampler for particles: each particle gets a random gust sample
_gust_rng = np.random.default_rng(seed + 300)
def wind_sampler():
    return np.array([
        _gust_rng.uniform(-w_max, w_max),
        _gust_rng.uniform(-w_max, w_max),
    ])

# ---------------------------------------------------------------------------
# System, GP, policy setup (same structure as W1)
# ---------------------------------------------------------------------------
if args.pybullet:
    import sys
    sys.path.append("../mc-pilot-pybullet")
    from simulation_class.model_pybullet import PyBulletThrowingSystem
    throwing_system = PyBulletThrowingSystem(
        mass=0.0577, radius=0.0327, launch_angle_deg=35.0,
        wind_model=wind_model, wind_aware=wind_aware,
        gui_mode=False
    )
else:
    from simulation_class.model_wind import WindThrowingSystem
    throwing_system = WindThrowingSystem(
        mass=0.0577, radius=0.0327, launch_angle_deg=35.0,
        wind_model=wind_model, wind_aware=wind_aware,
    )

if wind_aware:
    gp_input_dim = BALL_DIM + WIND_DIM
    f_model_learning = ML_wind.WindAware_Ballistic_Model_learning_RBF
else:
    gp_input_dim = BALL_DIM
    f_model_learning = ML.Ballistic_Model_learning_RBF

num_gp = 3
init_dict_RBF = {
    "active_dims": np.arange(0, gp_input_dim),
    "lengthscales_init": np.ones(gp_input_dim),
    "flg_train_lengthscales": True,
    "lambda_init": np.ones(1), "flg_train_lambda": False,
    "sigma_n_init": np.ones(1), "flg_train_sigma_n": True,
    "sigma_n_num": None, "dtype": dtype, "device": device,
}
model_learning_par = {
    "num_gp": num_gp, "T_sampling": Ts,
    "approximation_mode": "SOD",
    "approximation_dict": {"SOD_threshold_mode": "relative",
                           "SOD_threshold": 0.5, "flg_SOD_permutation": False},
    "init_dict_list": [init_dict_RBF] * num_gp,
    "dtype": dtype, "device": device,
}

rand_exploration_policy_par = {
    "full_state_dim": STATE_DIM, "u_max": uM,
    "n_strata": Nexp, "dtype": dtype, "device": device,
}
f_rand_exploration_policy = Policy.Stratified_Throwing_Exploration

if wind_aware:
    centers_init = np.column_stack([
        np.random.uniform(lm * np.cos(-gM), lM, Nb),
        np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
        np.random.uniform(-w_max, w_max, Nb),
        np.random.uniform(-w_max, w_max, Nb),
    ])
    lw = 0.15 * (2 * w_max)
    lengthscales_init = np.array([0.08, 0.08, lw, lw])
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "wind_dim": WIND_DIM, "num_basis": Nb, "u_max": uM,
        "lengthscales_init": lengthscales_init,
        "centers_init": centers_init,
        "weight_init": uM * (np.random.rand(1, Nb) - 0.5),
        "flg_drop": True, "dtype": dtype, "device": device,
    }
    f_control_policy = Policy.WindAware_Throwing_Policy
    policy_reinit_dict = {"lenghtscales_par": lengthscales_init,
                          "centers_par": np.array([1.0, 1.0, 0.5, 0.5]),
                          "weight_par": uM}
else:
    centers_init = np.column_stack([
        np.random.uniform(lm * np.cos(-gM), lM, Nb),
        np.random.uniform(lm * np.sin(-gM), lM * np.sin(gM), Nb),
    ])
    lengthscales_init = np.array([0.08, 0.08])
    control_policy_par = {
        "full_state_dim": STATE_DIM, "target_dim": TARGET_DIM,
        "num_basis": Nb, "u_max": uM,
        "lengthscales_init": lengthscales_init,
        "centers_init": centers_init,
        "weight_init": uM * (np.random.rand(1, Nb) - 0.5),
        "flg_drop": True, "dtype": dtype, "device": device,
    }
    f_control_policy = Policy.Throwing_Policy
    policy_reinit_dict = {"lenghtscales_par": lengthscales_init,
                          "centers_par": np.array([1.0, 1.0]),
                          "weight_par": uM}

cost_function_par = {
    "position_indices": [0, 1], "target_indices": [6, 7],
    "lengthscale": lc, "dtype": dtype, "device": device,
}
f_cost_function = Cost_function.Throwing_Cost

# ---------------------------------------------------------------------------
# Log, build, run
# ---------------------------------------------------------------------------
mode_tag = "aware" if wind_aware else "blind"
log_path = f"results_wind_W2/wmax{w_max:.1f}_{mode_tag}/{seed}"
os.makedirs(log_path, exist_ok=True)

mc_pilot_obj = MC_PILCO_module.MC_PILOT_Wind(
    target_sampler=sample_target, release_position=RELEASE_POS,
    throwing_system=throwing_system, wind_model=wind_model,
    wind_aware=wind_aware, T_sampling=Ts, state_dim=STATE_DIM,
    input_dim=INPUT_DIM, f_model_learning=f_model_learning,
    model_learning_par=model_learning_par,
    f_rand_exploration_policy=f_rand_exploration_policy,
    rand_exploration_policy_par=rand_exploration_policy_par,
    f_control_policy=f_control_policy,
    control_policy_par=control_policy_par,
    f_cost_function=f_cost_function,
    cost_function_par=cost_function_par,
    std_meas_noise=1e-3 * np.ones(STATE_DIM),
    log_path=log_path, dtype=dtype, device=device,
    wind_sampler=wind_sampler if wind_aware else None,
)

model_optimization_opt_dict = {
    "f_optimizer": "lambda p : torch.optim.Adam(p, lr = 0.01)",
    "criterion": Likelihood.Marginal_log_likelihood,
    "N_epoch": 1001, "N_epoch_print": 500,
}
n_list = Nexp + num_trials
policy_optimization_dict = {
    "num_particles": M, "opt_steps_list": [Nopt] * n_list,
    "lr_list": [0.01] * n_list,
    "f_optimizer": "lambda p, lr : torch.optim.Adam(p, lr)",
    "num_step_print": 100, "p_dropout_list": [0.25] * n_list,
    "p_drop_reduction": 0.25 / 2, "alpha_diff_cost": 0.99,
    "min_diff_cost": 0.02, "num_min_diff_cost": 400,
    "min_step": 400, "lr_min": 0.0025,
    "policy_reinit_dict": policy_reinit_dict,
}

centre_target = np.array([lm + (lM - lm) / 2, 0.0])
initial_state = np.concatenate([RELEASE_POS, np.zeros(3), centre_target])
initial_state_var = np.concatenate([1e-4 * np.ones(6),
                                     (0.5 * (lM - lm)) ** 2 * np.ones(2)])
if wind_aware:
    initial_state = np.concatenate([initial_state, [0.0, 0.0]])
    initial_state_var = np.concatenate([initial_state_var, [w_max**2, w_max**2]])

config_log = {
    "seed": seed, "config": "W2_gusts", "w_max": w_max,
    "T_gust": T_gust, "wind_aware": wind_aware,
    "wind_model": wind_model.describe(), "num_trials": num_trials,
    "Nexp": Nexp, "Nopt": Nopt, "M": M, "Nb": Nb,
    "uM": uM, "Ts": Ts, "T": T, "lc": lc, "lm": lm, "lM": lM,
}
pkl.dump(config_log, open(log_path + "/config_log.pkl", "wb"))

print(f"\nW2 Gusts: w_max={w_max}, T_gust={T_gust}, aware={wind_aware}, seed={seed}")

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
            # We need to add loaded_model to the kwargs dict passed to reinforce
            # But the reinforce call below doesn't use a dict directly.
            pass
        else:
            print("\nNo completed trials found to resume from. Starting fresh.")
    else:
        print("\nNo log file found to resume from. Starting fresh.")

# To pass loaded_model to reinforce, we can just add it to kwargs.
reinforce_kwargs = dict(
    initial_state=initial_state, initial_state_var=initial_state_var,
    T_exploration=T, T_control=T, num_trials=num_trials,
    num_explorations=Nexp,
    model_optimization_opt_list=[model_optimization_opt_dict] * num_gp,
    policy_optimization_dict=policy_optimization_dict,
)

if args.resume and 'num_completed' in locals() and num_completed > 0:
    reinforce_kwargs["loaded_model"] = True
    reinforce_kwargs["num_trials"] = max(0, num_trials - num_completed)

cost_trial_list, _, _ = mc_pilot_obj.reinforce(**reinforce_kwargs)
print(f"\nTraining complete. Final cost: {cost_trial_list[-1][-1]:.4f}")
