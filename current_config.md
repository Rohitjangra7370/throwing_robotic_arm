# MC-PILOT Current Configuration
AR525 Group-3, IIT Mandi

---

## Algorithm Parameters

| Parameter | Our Value | Paper (Sim) | Notes |
|-----------|-----------|-------------|-------|
| `Nexp` | 5 | 5 | Exploration throws before first training |
| `Nopt` | 1500 | 1500 | Max policy optimisation steps per trial |
| `M` | 400 | 400 | Monte Carlo particles |
| `Nb` | 250 | 250 | RBF basis functions in policy |
| `uM` | 3.5 m/s | 3.5 m/s | Max release speed |
| `Ts` | 0.02 s | 0.01 s | Simulation timestep — doubled for ~3x speedup |
| `T` | 0.7 s | 1.0 s | Simulation horizon — trimmed (ball lands by ~0.58s) |
| `lc` | 0.5 m | 0.1 m | Cost lengthscale — widened; lc=0.1 causes gradient starvation from random exploration throws |
| `lm` | 0.75 m | 0.75 m | Min target distance |
| `lM` | 1.75 m | 1.75 m | Max target distance — reverted to paper's value (2.4 was beyond reachable range) |
| `γM` | π/6 rad | π/6 rad | Max lateral throw angle |

---

## Physical Setup

| Parameter | Value | Notes |
|-----------|-------|-------|
| Ball mass | 0.0577 kg | Tennis ball |
| Ball radius | 0.0327 m | Tennis ball |
| Launch angle α | 35° (fixed) | Elevation angle, same for all throws |
| Release height | 0.5 m | Fixed release point above ground |
| Release position | [0, 0, 0.5] | In world frame |
| Wind | None | No wind |

---

## GP Model (Ballistic_Model_learning_RBF)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of GPs | 3 | One per velocity dimension: Δvx, Δvy, Δvz |
| GP input | `[x, y, z, vx, vy, vz]` (6-D) | Ball state only — target dims stripped |
| GP output | `[Δvx, Δvy, Δvz]` | Velocity changes per timestep |
| Kernel | Squared-exponential (RBF) | |
| Lengthscales | Trainable, init = 1.0 (all 6 dims) | |
| Signal variance λ | Fixed at 1.0 | `flg_train_lambda = False` |
| Noise σ_n | Trainable, init = 1.0 | |
| Approximation | SOD (Subset of Data) | Threshold = 0.5 relative |
| GP optimiser | Adam, lr=0.01, 1001 epochs | Marginal log-likelihood |

State reconstruction (Eq. 18 from paper):
- `v_{t+1} = v_t + Δv`
- `p_{t+1} = p_t + Ts·v_t + (Ts/2)·Δv`
- Target dims [6:8] copied forward unchanged

---

## Policy (Throwing_Policy — RBF network)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input | Target `(Px, Py)` | Extracted from augmented state dims 6:8 |
| Output | Scalar release speed ∈ [0, uM] | |
| Squashing | `(uM/2)·(tanh(raw) + 1)` | Paper Eq. 21 |
| Basis functions | 250 RBFs | |
| Center init domain | `Px ∈ [0.65, 2.4]`, `Py ∈ [-0.375, 1.2]` | Approximates the actual target arc |
| Weight init range | `[-uM/2, uM/2]` = `[-1.75, 1.75]` | |
| Lengthscales | Trainable, init = [1.0, 1.0] | One per target dimension |
| Dropout | 25% during optimisation | Reduced by 12.5% each time LR drops |

---

## Policy Optimisation

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimiser | Adam | |
| Learning rate | 0.01 (init) | Halved when convergence criterion met |
| LR minimum | 0.0025 | |
| Early stopping criterion | `min_diff_cost=0.02`, `num_min_diff_cost=400`, `min_step=400` | Relaxed from cartpole defaults |
| EMA smoothing α | 0.99 | For cost-diff ratio |

---

## Exploration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Policy (baseline A) | `Random_Throwing_Exploration` | Uniform random speed ∈ [0, uM] at t=0 |
| Policy (elevated B–E) | `Stratified_Throwing_Exploration` | [0, uM] split into Nexp bands; throw i samples band i. Seed-robust. |
| Target sampling | Uniform over arc `[lm, lM] × [-γM, γM]` | Polar coordinates, converted to (Px, Py) |
| Data augmentation `Na` | 0 | No rotation augmentation (same as paper's simulation setting) |

**Why stratified for elevated configs:** At higher release heights, the speed-to-range mapping is more spread out — unlucky random seeds can draw all low speeds, leaving the GP blind to high-speed dynamics and causing the policy to saturate at max speed (0/10 hits with seed=1 on Config B). Stratified exploration guarantees full coverage in 5 throws regardless of seed.

---

## Key Implementation Differences from Paper

1. **`lc = 0.5m` vs paper's `0.1m`** — required because at lc=0.1, all early training errors (0.3–1.5m) saturate the cost to ~1.0, killing gradients. Root cause may be that our GP (with Ts=0.02, ~140 training points) is less accurate than the paper's (~495 points at Ts=0.01).

2. **`Ts = 0.02s` vs `0.01s`** — doubled for ~3x speedup. Ball physics smooth enough at this resolution. Confirmed no accuracy regression in our experiments.

3. **`T = 0.7s` vs `1.0s`** — trimmed because ball always lands by ~0.58s. No accuracy cost.

4. **Random exploration vs baseline policy** — paper Algorithm 1 uses Eq. 13 baseline for exploration. We tested this and it FAILED (hit rate 50% → 20%). Diverse random throws give better GP coverage than near-optimal similar parabolas.

5. **Particle landing freeze** — paper Section 4.2.2. We implemented this. Once z ≤ 0, particle state is frozen for remaining timesteps.

6. **No delay modelling** — paper Section 5 models gripper release delay `t_d ~ U(a, a+b)`. We simulate ideal release (no delay). Not applicable for pure simulation.

---

## Current Results (seed=1, 5 trials, lM=1.75m)

| Throw | Phase | Error (m) | Hit (<0.1m) | Target dist |
|-------|-------|-----------|-------------|-------------|
| 1–5 | Explore | 0.02–1.09m | — | random |
| 6 | Trial 1 | **0.085m** | HIT | 1.66m |
| 7 | Trial 2 | **0.033m** | HIT | 1.60m |
| 8 | Trial 3 | **0.023m** | HIT | 1.18m |
| 9 | Trial 4 | **0.017m** | HIT | 0.79m |
| 10 | Trial 5 | **0.008m** | HIT | 1.10m |

**5/5 hits (100%).** Cost: 0.74 → 0.009 (Trial 1), 0.001 (Trial 5). Policy converged by Trial 5.

---

## Noise Sweep Snapshot (April 26, 2026)

Backend used for this snapshot: `numpy` (PyBullet DLL blocked by host application-control policy).

Sweep outputs:
- `mc-pilot-pybullet/results_mc_pilot_pb_noise_sweep/noise_sweep_summary.md`
- `mc-pilot-pybullet/results_mc_pilot_pb_noise_sweep/noise_sweep_slip.png`
- `mc-pilot-pybullet/results_mc_pilot_pb_noise_sweep/noise_sweep_saltpepper.png`

| Noise Type | Level | Aware Hit Rate (policy throws) | Naive Hit Rate (policy throws) | Aware Mean Error | Naive Mean Error |
|-----------|-------|---------------------------------|---------------------------------|------------------|------------------|
| Slip (`sigma=0.04`) | `alpha=0.10` | 100% (4/4) | 50% (2/4) | 0.0287 m | 0.0979 m |
| Slip (`sigma=0.04`) | `alpha=0.20` | 100% (4/4) | 0% (0/4) | 0.0219 m | 0.1881 m |
| Salt-and-Pepper (`spike_scale=0.30`) | `p_spike=0.05` | 100% (4/4) | 75% (3/4) | 0.0292 m | 0.0439 m |
| Salt-and-Pepper (`spike_scale=0.30`) | `p_spike=0.15` | 50% (2/4) | 50% (2/4) | 0.0914 m | 0.0768 m |

Interpretation:
- Slip noise shows strong aware-vs-naive separation that widens with noise level (`alpha`), consistent with speed-dependent undershoot compensation.
- Salt-and-Pepper noise becomes outlier-dominated at higher `p_spike`; current policy settings show reduced separation at `p_spike=0.15`.
