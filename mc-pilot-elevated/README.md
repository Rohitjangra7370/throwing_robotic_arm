# mc-pilot-elevated — Study 1: Elevated Release Heights (NumPy)

Extends `mc-pilot/` to different release heights (z = 1.0, 1.5, 2.0 m) and a narrow-range ground config (z = 0.0 m, target range 0.35 m). Pure NumPy sim — no PyBullet.

**Research question:** Does the same MC-PILOT framework work under different launcher geometries? What breaks, and why?

---

## Entry points

```bash
# Recommended: stratified exploration (seed-robust)
python test_mc_pilot_b_strat.py -seed 1 -num_trials 10   # z = 1.0 m
python test_mc_pilot_c_strat.py -seed 1 -num_trials 10   # z = 1.5 m
python test_mc_pilot_d_strat.py -seed 1 -num_trials 10   # z = 2.0 m
python test_mc_pilot_e_strat5.py -seed 1 -num_trials 10  # z = 0.0 m, narrow range

# Failure baseline: random exploration (shows seed-dependence)
python test_mc_pilot_b.py -seed 1 -num_trials 10   # 0/10 at seed=1
```

Results saved to `results_mc_pilot_<config>/`. E.g. `results_mc_pilot_b_strat/<seed>/`.

---

## Results

| Config | z (m) | Max range | Hit rate | Exploration |
|--------|-------|-----------|----------|-------------|
| B-Strat | 1.0 | 1.90 m | **10/10** | Stratified |
| C-Strat | 1.5 | 2.15 m | **10/10** | Stratified |
| D-Strat | 2.0 | 2.35 m | **10/10** | Stratified |
| B-Random | 1.0 | 1.90 m | 0/10 (seed=1), 10/10 (seed=2) | Random |
| E-Strat5 | 0.0 | 0.35 m | **10/10** | Stratified + ℓs = 0.15 m |

---

## Two convergence rules discovered

**Rule 1 — Stratified Exploration**

Random exploration can draw all low-speed throws (seed=1, Config B → 0/10), leaving the GP with no data on high-speed dynamics. Fix: divide [0, uM] into Nexp equal bands; one throw per band. Guarantees full speed coverage regardless of seed.

**Rule 2 — RBF Lengthscale Must Match Target Range**

Policy sensitivity between two targets separated by distance d: `S = exp(−d²/(2ℓs²))`. At ℓs = 1.0 m and range = 0.35 m: `S = 0.941` — the policy is nearly constant for all targets.

Rule: `ℓs ≈ 0.15 × target_range`. Config E failed 4 iterations at ℓs = 1.0 m; changing to ℓs = 0.15 m → 10/10 immediately.

---

## File guide

- `test_mc_pilot_b_strat.py` / `_c_strat.py` / `_d_strat.py` / `_e_strat5.py` — **use these**
- `test_mc_pilot_b.py` / `_c.py` / `_d.py` / `_e.py` — random-exploration variants (show failure mode)
- `test_mc_pilot_e_strat.py` through `_e_strat4.py` — intermediate Config E debugging attempts (documented in `change_history.md`)
- `apply_mcpilco_policy.py`, `log_plot_cartpole.py`, `test_mcpilco_cartpole*.py` — upstream MC-PILCO cartpole boilerplate, not used here
