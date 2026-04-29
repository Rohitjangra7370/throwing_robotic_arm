# mc-pilot-wind — Study 4: Wind Conditions

Extends the NumPy ballistic simulator with a time-varying wind force term. Compares two architectures:

- **Blind:** standard 8-D state, 2-D policy (Px, Py) — treats wind as implicit dynamics bias
- **Wind-aware:** 10-D state (adds wx, wy), 4-D policy (Px, Py, wx, wy) — explicit wind conditioning

Three wind models tested:

| ID | Model | Parameters |
|----|-------|-----------|
| W1 | Constant | 0 / 2.5 / 5.0 / 8.0 m/s headwind |
| W2 | Random gusts | Poisson arrivals, peak 4.0 m/s |
| W3 | OU turbulence | σ = 4.0 m/s, mean = 0.3 m/s, α = 0.7 |

---

## New components (not in mc-pilot/)

| File | What it does |
|------|-------------|
| `simulation_class/wind_models.py` | `ConstantWind`, `GustWind`, `TurbulentWind` — common `WindModel` interface |
| `simulation_class/model_wind.py` | `WindThrowingSystem` — evaluates wind at each Euler step |
| `model_learning/Model_learning_wind.py` | `WindAware_Ballistic_Model_learning_RBF` — 8-D GP input `[ball(6), wind(2)]` |
| `policy_learning/Policy.py` | `Throwing_Policy` + `WindAware_Throwing_Policy` (4-D RBF) |
| `policy_learning/MC_PILCO.py` | `MC_PILOT_Wind` — wind-aware data collection and particle propagation |

---

## Entry points

```bash
python test_wind_W1.py              # constant wind (all speeds + aware comparison)
python test_wind_W2.py              # random gusts
python test_wind_W3.py              # OU turbulence

# Run all 9 configurations sequentially (15 trials each, ~15–20 h total):
python run_all_wind_experiments.py --num_trials 15

# Post-hoc analysis and comparison table:
python analyze_wind_results.py
```

Results saved to `results_wind_W1/`, `results_wind_W2/`, `results_wind_W3/`. Summary: `final_analysis.txt`.

---

## Key finding: blind GP consistently wins

| Config | Blind hit rate | Aware hit rate |
|--------|---------------|---------------|
| W1-moderate (5.0 m/s) | 53% | 47% |
| W2 (gusts 4.0 m/s) | 53% | 40% |
| W3 (turbulence) | **60%** | 47% |

**Why aware fails:** Increasing policy dimension from 2-D to 4-D with 250 RBF centers leaves ~4 centers per dimension instead of ~15. The GP is too sparse to generalise with only 15 training trials. Explicit wind conditioning needs ~40+ trials to overcome this curse of dimensionality.

**Practical conclusion:** For constant or slowly varying wind, treat wind as an implicit dynamics bias absorbed by the GP. Use a wind-aware model only when significantly more training data is available.
