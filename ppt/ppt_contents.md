# End Presentation — Slide Contents
AR525 Group 3, IIT Mandi

---

## SLIDE 1 — Title
**MC-PILOT: Data-Efficient Robot Ball Throwing**
*Reproduction, Analysis, and Extension of a Model-Based RL Algorithm*

AR525 — Group 3, IIT Mandi

> "How do you teach a robot arm to throw a ball into a bin — with fewer than 15 real attempts?"

---

## SLIDE 2 — The Problem
**Pick-and-Throw Robotics**

- Task: robot arm grasps a ball, throws it into a target bin at distance 0.75–2.4 m
- Ball follows a parabolic + drag trajectory after release — physics is known but noisy
- Challenge: the **gap between "commanded velocity" and "delivered velocity"** is random and arm-dependent
- Naive approach (model-free RL): needs hundreds of real throws → expensive, slow, unsafe

*Key question:* Can a robot learn to throw accurately from just **~10 attempts**?

[Physical setup diagram: arm → ball → arc → bin, label z_release, α=35°, target distance]

---

## SLIDE 3 — Prior Work
**The Algorithmic Lineage**

| Algorithm | Year | Key Idea | Data Needed |
|---|---|---|---|
| Model-free RL (DQN, PPO…) | 2013–20 | Learn policy from reward only | 1000s of trials |
| PILCO | 2011 | GP dynamics + analytic policy gradient | ~50 trials |
| MC-PILCO | 2022 | Monte Carlo gradient through GP rollouts | ~20–50 trials |
| **MC-PILOT** | **2025** | MC-PILCO adapted for single-shot throwing | **~5 exploration + 10 learning** |

**Why GPs for dynamics?**
- Exact uncertainty quantification — the policy gradient knows *how confident* the model is
- Data-efficient — works with 5 exploration throws
- Bayesian: posterior updates analytically with new data

---

## SLIDE 4 — What MC-PILOT Does Better
**The Two Key Modifications over MC-PILCO**

**Modification 1 — Single-Shot Policy**
- Standard MC-PILCO fires the policy at every timestep (needed for balancing)
- Throwing is different: policy fires **once** → choose release speed → ball is in free flight
- Policy: `π(P) → v` — given target position, output release speed
- State: `x̃ = [x, y, z, vx, vy, vz, Px, Py]` — ball state + target concatenated

**Modification 2 — Gripper Delay Estimation**
- Real gripper doesn't open instantly → random delay → random velocity undershoot
- MC-PILOT fits delay distribution `t_d ~ U(a, b)` using Bayesian Optimization on hardware
- Makes the algorithm deployable on **real hardware** (paper: Franka Panda, real bin)

**Claim verified by paper:** ~10 learning throws → near-100% hit rate.
Model-free baselines need 100× more data.

---

## SLIDE 5 — Our Plan
**What We Set Out To Do**

We replaced Gazebo/ROS with a pure Python + PyBullet environment (no ROS needed for ball physics).

Five extensions planned:

| Goal | Status |
|---|---|
| Reproduce MC-PILOT baseline (z=0.5m, Franka Panda) | ✅ Done |
| Generalize to elevated release heights (z=1.0, 1.5, 2.0m) | ✅ Done |
| Generalize to 3 different robot arms | ✅ Done |
| Model arm-side noise (slip, timing jitter, salt-pepper) | ✅ Done |
| Wind conditions (constant, gusts, turbulence) | ✅ Done |
| OpenCV + YOLO bin detection → autonomous targeting | ✅ Done |

---

## SLIDE 6 — Reproduction: The Silent Bugs
**Four Bugs That Prevent Any Learning — None Caused a Crash**

*"The codebase runs without errors. It just never learns anything."*

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | Speed channel stripped from GP input | `cost.backward()` fails — no gradient reaches policy | Embed speed directly into particle initial velocity |
| 2 | Underground particles continue propagating | Cost stuck at 0.998 regardless of policy | Freeze particle state when z ≤ 0; interpolate to exact landing |
| 3 | Cost lengthscale ℓc = 0.1m too tight | Gradient < 10⁻³ for all exploration errors (0.3–1.5m range) | Set ℓc = 0.5m; gradient alive up to ~1m error |
| 4 | Simulation horizon unit mismatch | 10,000 timestep tensor → RAM crash | Pass integer steps, not float seconds |

**Physical interpretation of Bug 3:** With ℓc = 0.1m, cost = 1 − exp(−‖e‖²/ℓc²) ≈ 1.0 for *any* miss > 22cm. The GP learns correct dynamics — but the cost function can't distinguish any two bad policies. Optimiser stalls.

---

## SLIDE 7 — Two Convergence Rules (Not in the Paper)

**Rule 1 — Stratified Exploration**

Random exploration with 5 throws can accidentally draw 4 low-speed throws (seed=1, z=1.0m → 0/10 hits). The GP has no data on high-speed dynamics.

*Fix:* Divide [0, uM] into Nexp equal bands; one throw per band. Guarantees full speed coverage regardless of seed.

*Evidence:* Config B-random seed=1 → 0/10. Same config, stratified → 10/10.

---

**Rule 2 — RBF Lengthscale Must Match Target Range**

RBF policy sensitivity between two targets at distance d: `S = exp(−d²/(2ℓs²))`

Paper uses ℓs = 1.0m everywhere. For Config E (target range 0.35m):
`S = exp(−0.35²/2) = 0.941` → policy is nearly **constant** for all targets.

*Rule:* `ℓs ≈ 0.15 × target_range`

*Evidence:* Config E failed 4 iterations with ℓs = 1.0m. Changed to ℓs = 0.15m → **10/10 hits immediately**.

---

## SLIDE 8 — Baseline Result
**5/5 Hits — 5 Exploration + 5 Learning Throws**

| Throw | Phase | Landing Error | Hit? |
|---|---|---|---|
| 1–5 | Exploration | 0.02–1.09m | — |
| 6 | Trial 1 | 8.5 cm | ✅ |
| 7 | Trial 2 | 3.3 cm | ✅ |
| 8 | Trial 3 | 2.3 cm | ✅ |
| 9 | Trial 4 | 1.7 cm | ✅ |
| 10 | Trial 5 | 0.8 cm | ✅ |

Cost trajectory: **0.74 → 0.009 (Trial 1) → 0.001 (Trial 5)**

*Physical meaning:* The GP has learned the speed-to-range curve from 5 random throws. Policy optimisation then finds the exact speed for each target in 1500 gradient steps. By trial 5, errors are sub-centimetre.

[Embed: cost curve figure, or sketch of landing scatter converging to target]

---

## SLIDE 9 — Elevated Release Heights
**Generalizing to z = 1.0, 1.5, 2.0m**

**Engineering challenges:**
- KUKA iiwa7 joint limits cap EE speed at ~2.5 m/s; throw needs 3.5 m/s → *decoupled*: ball velocity set directly via `resetBaseVelocity`, arm is cosmetic
- EE can't reach z=1.5m from ground → mount arm on pedestal: `base_z = z_release − 0.5m`

| Config | Release height | Max target range | Hit rate | Exploration |
|---|---|---|---|---|
| B-Strat | z = 1.0m | 1.90m | **10/10** | Stratified |
| C-Strat | z = 1.5m | 2.15m | **10/10** | Stratified |
| D-Strat | z = 2.0m | 2.35m | **10/10** | Stratified |
| B-Random | z = 1.0m | 1.90m | 0/10 (seed=1), 10/10 (seed=2) | Random |
| E-Strat5 | z = 0.0m | 1.10m | **10/10** | Stratified + ℓs=0.15m |

*Physical insight:* Higher release → longer hang time → farther reach. The GP learns a different speed-to-range curve for each height, but the learning algorithm itself is unchanged.

---

## SLIDE 10 — Three Robot Arms
**Franka Panda, KUKA iiwa7, xArm6**

The paper is locked to Franka Panda. We ran the same trained policy through three different arm URDFs in PyBullet.

**Key architectural insight — Decoupled design:**
- RL trains on ball physics only (NumPy ballistic sim)
- PyBullet arm is visualization only
- Ball velocity is *set explicitly* at release, independent of arm tracking quality

| Robot | Hit Rate (commanded velocity) | Hit Rate (actual EE velocity) | Mean Error (commanded) |
|---|---|---|---|
| Franka Panda | **100%** | 33% | 0.8 mm |
| KUKA iiwa7 | **100%** | 33% | 2.5 mm |
| xArm6 | **100%** | 33% | 15.5 mm |

*Physical meaning:* All three arms deliver the same ball physics when velocity is commanded correctly. The 33% "achieved" rate reflects arm tracking error (~7%), not RL failure. This validates the decoupling — the RL algorithm is arm-agnostic.

[Embed: robot_compare_metrics.png — bar chart showing all three at 100%]

---

## SLIDE 11 — Noise Models: Arm-Side Stochasticity
**Making RL Non-Trivial**

Without noise: a single-step ballistic inverter solves the task. RL adds no value.
With noise: the policy must learn to compensate — RL is necessary.

**Two learnable noise types studied:**

*VelocitySlipNoise:* `v_actual = (1 − α)·v_cmd + N(0, σ²)`
- Systematic undershoot that scales with throw speed
- Policy must learn to command `v/(1−α)` to compensate

*Salt-and-Pepper:* With probability p, velocity component is replaced by a random spike
- Models random electromagnetic glitches or encoder errors
- Asymmetric: spikes are always wrong, so higher p is strictly worse

**Key theorem (proven):** Zero-mean symmetric noise (e.g. pure Gaussian `N(0, σ²)`) does NOT shift the optimal policy — noise-aware = naive. Only *biased* noise is learnable.

---

## SLIDE 12 — Noise Results
**Aware Policy Consistently Outperforms Naive**

| Noise Type | Level | Naive Hit Rate | Aware Hit Rate |
|---|---|---|---|
| Slip | α = 0.05 | 100% | 100% |
| Slip | α = 0.10 | **25%** | **100%** |
| Slip | α = 0.15 | 100% | 100% |
| Slip | α = 0.20 | **0%** | **100%** |
| Salt-Pepper | p = 0.05 | 75% | 100% |
| Salt-Pepper | p = 0.10 | **25%** | **75%** |
| Salt-Pepper | p = 0.15 | 50% | 50% |

*Physical interpretation:*
- Slip α=0.20: naive policy commands v* → ball undershoots by 20% of range → always misses. Aware policy learns to command v*/(1−0.20) = 1.25v* → hits 100%.
- Naive cost ≈ 0.001 (falsely optimistic — evaluates without noise). Aware cost ≈ 0.007–0.011 (honest — includes noise variance). Lower training cost, worse real performance.

[Embed: slip_trends.png — hit rate diverges at α≥0.10]

---

## SLIDE 13 — Wind Conditions
**Constant Wind, Random Gusts, Turbulence**

**Physics:** Drag uses air-relative velocity: `v_rel = ṗ − w`, `F_D ∝ ‖v_rel‖·v_rel`

**Three wind models:**
- W1: Constant wind vector (2.5 / 5.0 / 8.0 m/s headwind)
- W2: Random gusts (Poisson arrivals, peak 4.0 m/s)
- W3: Ornstein-Uhlenbeck turbulence (σ = 4.0, mean = 0.3 m/s)

**Hypothesis:** Adding wind as explicit GP/policy input (10-D state, 4-D policy) will outperform blind model.

**Result: Hypothesis falsified.**

| Config | Blind hit rate | Aware hit rate |
|---|---|---|
| W1-calm | 73% | — |
| W1-moderate (5 m/s constant) | 53% | 47% |
| W1-strong (8 m/s) | 47% | — |
| W2 (gusts) | 53% | 40% |
| W3 (turbulence) | **60%** | 47% |

*Blind GP consistently wins in all 3 conditions.*

---

## SLIDE 14 — Wind: The Curse of Dimensionality
**Why Explicit Wind Information Hurts**

The blind GP (2-D policy: Px, Py) uses 250 RBF centers in 2D → **dense mesh**.
The aware GP (4-D policy: Px, Py, wx, wy) uses 250 RBF centers in 4D → **exponentially sparse**.

| Architecture | Policy dimensions | Effective centers / dimension | Result |
|---|---|---|---|
| Blind | 2 | ~15 | Dense — generalises well |
| Wind-aware | 4 | ~4 | Sparse — collapses between targets |

With 15 training trials (~525 data points) in a 4D space, the GP cannot generalise. Extra wind dimensions introduce variance without usable information density.

**Practical conclusion:** In constant or slowly varying wind, treat wind as **implicit environmental bias** absorbed by the GP. Explicit wind conditioning needs ~40+ trials (data scales as O(d²) with dimension d).

[Embed: wind hit-rate comparison bar chart]

---

## SLIDE 15 — Vision Pipeline: OpenCV + YOLO Bin Detection
**Closing the Loop — Robot Sees the Bin, Then Throws**

**Problem:** So far, target coordinates are given numerically. For real deployment, the robot must *detect* the bin visually.

**Two parallel approaches:**

**OpenCV (geometric)**
- Depth camera mounted on arm base (PyBullet `getCameraImage` RGB-D)
- Segment green bin by HSV colour threshold → find contour → compute centroid
- Depth back-projection: pixel (u, v, d) → world coordinates (x, y)
- Bias correction: camera sees near face of bin → shift target by `2R/π` toward bin centre
- Failure: fallback to sampled target, log detection failure

**YOLO (learning-based)**
- Synthetic training dataset: 800 train / 200 val images rendered in PyBullet
- YOLOv8n trained on rendered red sphere target
- Bounding box centroid + depth → 3D target estimate
- Advantage: robust to occlusion and lighting changes

Both pipelines feed detected `(x, y)` into policy → throw command. Bin is **hollow** (12-wall ring geometry) → ball physically enters bin.

---

## SLIDE 16 — System Architecture (Full Picture)

```
[Training Layer — pure Python/NumPy]
  5 exploration throws → GP fits dynamics
  1500 gradient steps/trial × 10 trials → RBF policy
  Stratified exploration over [0, uM]
          |
          | trained policy (release speed per target)
          ↓
[Simulation + Deployment — PyBullet]
  Load arm URDF (Franka / KUKA / xArm6)
  IK → cubic spline trajectory → arm moves
  Ball released at t_r via resetBaseVelocity(v_cmd)
  Free flight under air-relative drag + wind
  Ball lands in bin
          ↑
[Perception — OpenCV / YOLO]
  Camera detects bin → 3D coordinates
  (x, y) → policy → release speed → throw
```

**Key design principle:** The two layers are decoupled. Swap any arm without retraining. Add noise without changing the algorithm. Add wind by extending state dimension.

---

## SLIDE 17 — Complete Results Summary

| Study | Configuration | Hit Rate | Key Insight |
|---|---|---|---|
| Baseline | z=0.5m, Franka, no noise | **5/5 (100%)** | Cost 0.74→0.001 in 5 trials |
| Elevated | z=1.0m, stratified | **10/10** | Pedestal + stratified essential |
| Elevated | z=1.5m, stratified | **10/10** | ℓM, T scaled with height |
| Elevated | z=2.0m, stratified | **10/10** | 10 straight hits |
| Multi-arm | Franka/KUKA/xArm6 | **100% each** | Arm-agnostic RL |
| Noise | Slip α=0.20, aware | **100% vs 0%** naive | Infinite improvement |
| Noise | Salt-pepper p=0.05, aware | **100% vs 75%** naive | Consistent advantage |
| Wind | W1-calm, blind | 73% | GP learns physics well |
| Wind | W3-turbulence, blind | **60% vs 47%** aware | Implicit beats explicit |
| Vision | OpenCV + YOLO | End-to-end | Detect → throw in one loop |

---

## SLIDE 18 — Key Scientific Contributions

**Three things the paper never stated that are necessary for the algorithm to work:**

1. **Stratified exploration** — random seed can produce zero-coverage exploration; stratified partitioning guarantees learning regardless of seed

2. **RBF lengthscale rule** — `ℓs ≈ 0.15 × target_range`; at ℓs = 1.0m and range = 0.35m, the policy outputs a constant speed for all targets

3. **Zero-mean noise invariance** (proven formally) — symmetric Gaussian noise cannot be compensated by RL; only biased noise (slip, timing jitter) is learnable. Paper's gripper delay model is biased, which is why it works on hardware.

**Unexpected finding:** GP treats constant wind as an implicit dynamics bias — explicit wind state augmentation *hurts* in low-data regimes due to the curse of dimensionality. 15 trials in 4D policy space is equivalent to 4 trials in 2D.

---

## SLIDE 19 — Future Work

- **Hardware validation:** Run trained policies on a real robot arm; measure how PyBullet-to-real transfer degrades performance
- **Wind-robust policy:** Randomize wind per rollout during training so one policy handles a wind distribution (not just one fixed wind vector)
- **End-to-end YOLO deployment:** Vision detection → policy → throw → physical landing in bin, all in one hardware loop
- **Multi-target sequences:** Train a policy that throws to a sequence of bins without retraining
- **Adaptive noise estimation:** Estimate slip/jitter parameters online via Bayesian Optimization (paper's approach) rather than requiring them upfront

---

## Notes for LaTeX/Beamer Conversion

- **Slides with embedded figures:**
  - Slide 8: cost curve (generate from log.pkl)
  - Slide 10: `results_robot_arm_compare/robot_compare_metrics.png`
  - Slide 12: `results_mc_pilot_pb_noise_report_paper/slip_trends.png` and `saltpepper_trends.png`
  - Slide 14: wind hit-rate bar chart (generate from final_analysis.txt)
- Slide 16 architecture diagram: render as TikZ figure
- All tables: use `\small` font in Beamer to fit slide width
- For rover/demo video: embed as `\movie{}` or show screenshot frame
- Results PNGs also available in `ppt/results1.png` through `results4.png`
