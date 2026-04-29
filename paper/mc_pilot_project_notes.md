# AR525: MC-PILOT Project — Architecture & Planning Notes
**IIT Mandi, Group-3**
_Summary of design decisions and understanding built up prior to implementation._

---

## 1. What is the Paper?

**MC-PILOT** (Turcato et al., 2025) — arXiv:2502.05595
- Full title: "Data efficient Robotic Object Throwing with Model-Based Reinforcement Learning"
- Task: Pick-and-Throw (PnT) — a robot arm grasps an object and throws it into a target bin
- Hardware: Franka Emika Panda (7-DOF arm)
- Key claim: near-100% accuracy with only ~10 exploration throws, vs. hundreds for model-free methods

---

## 2. The Algorithm Stack (Simple Terms)

### MC-PILCO (parent algorithm, `merlresearch/MC-PILCO` on GitHub)
The upstream algorithm MC-PILOT is built on. Designed for general model-based RL on dynamical systems.

Core components:
- **Gaussian Process Regression (GPR)** — learns a stochastic model of system dynamics from data
- **Monte Carlo policy gradient** — optimizes the policy by simulating particles through the GP model
- **RBF policy network** — squashed radial basis function network as the policy
- **Reparameterization trick + Adam** — for differentiating through stochastic operations
- **Dropout** — for exploration during policy optimization

### MC-PILOT = MC-PILCO + two PnT-specific modifications

**Modification 1: Single-shot policy + augmented state**
- In MC-PILCO, the policy fires at every timestep (makes sense for cartpole balancing)
- In throwing, the policy fires **once at t'=0** (choosing release velocity), then the ball is in free flight
- The target position P is concatenated into the state: `x̃ = [x, P]`
- Policy becomes `π(P) → v` — given target position, output release velocity
- This is the **minimum viable adaptation** to make MC-PILCO into a throwing algorithm at all

**Modification 2: Gripper delay estimation**
- On real hardware, the gripper doesn't open instantly — there's an unknown, slightly random delay
- MC-PILCO ignored this; MC-PILOT estimates the delay distribution `td ~ U(a, a+b)` using Bayesian Optimization
- The BO loop fits the delay parameters by minimizing the difference between simulated and actual landing positions
- This is what makes MC-PILOT work on **real hardware**
- In simulation, the delay is known/controllable — so you can skip this module for sim-only work

---

## 3. Tech Stack

### Core Algorithm (pure Python, no ROS)
- **Python 3.9** (recommended — mujoco_py has issues on 3.11+)
- **PyTorch** — all ML (policy network, Monte Carlo gradient, Adam optimizer)
- **GPR / gpr_lib** — custom GP regression library bundled with MC-PILCO (MIT licensed)
- **bayesian-optimization** (pip) — for the delay estimation module
- **NumPy / SciPy** — numerical math
- **Matplotlib** — plotting results

### Simulation Environment (paper uses this, we are NOT using it)
- **ROS 1 Noetic** — middleware for commanding the robot
- **Gazebo** — robot + physics simulator
- **MoveIt!** — motion planning for the arm
- **ros_control** — joint trajectory controller
- **franka_ros** — Franka-specific ROS packages

### What we are using instead
See Section 6.

---

## 4. What Gazebo Actually Is (and Why We Don't Need It)

Gazebo is a **robot simulator** — like a physics-accurate video game engine for robots. In MC-PILOT it was used to:
1. Simulate the Franka arm executing throwing trajectories
2. Simulate the ball's free-flight and landing
3. Collect training data for the GP model

**The key insight:** The GP model only learns **free-flight ball dynamics** after the gripper opens. It has no knowledge of the robot arm — the arm just sets initial conditions (release position and velocity). After release, it's just a ball flying under gravity and drag.

This means Gazebo's contribution is almost entirely the arm simulation. For the ball physics, the paper even implements the drag equation manually in Python (Equation 35) rather than trusting Gazebo's built-in physics.

**Conclusion:** For our project, a numpy ballistic simulator gives the GP exactly the same data it would get from Gazebo, for a ball. No ROS, no Docker, full control.

---

## 5. Our Proposed Explorations (from slides)

### Exploration 1: Elevated Targets
- **Problem:** MC-PILOT fixes all targets on a single horizontal ground plane (zP = const), with hardcoded release angle α ≈ 0
- **Proposed change:** Expand policy to `π(P) → [zP, α, v]` — jointly optimize release height, throw angle, and velocity
- **Modified cost:** Extend Σc to penalize lateral and vertical error
- **Key question:** Where is the feasibility boundary within joint limits?

### Exploration 2: Wind Modelling
- **Problem:** MC-PILOT only models calm still air drag — no lateral airflow
- **Proposed change:** Add a wind force term to the drag equation
- **Three scenarios:** constant wind, random gusts, turbulence (Gaussian noise)
- **Key question:** Can the GP implicitly absorb wind effects, or does wind state need to be an explicit input?

### Important note on ordering
Both explorations sit on top of the **Modification 1** from MC-PILOT (single-shot policy + augmented state). That is the baseline you must implement first. Neither exploration touches the delay estimation module (Modification 2), which can be skipped for simulation work.

---

## 6. Our Architecture (No ROS, No Gazebo)

### Two-layer design

```
┌─────────────────────────────────────────┐
│           ALGORITHM LAYER               │
│  (pure Python/numpy — the real work)    │
│                                         │
│  - GP dynamics model (from MC-PILCO)    │
│  - Single-shot policy π(P) → v         │
│  - Monte Carlo policy optimization      │
│  - Ballistic sim for data collection    │
│  - Wind extension (Exploration 2)       │
│  - Elevated targets (Exploration 1)     │
└──────────────────┬──────────────────────┘
                   │ trained policy outputs
                   │ release velocity + position
┌──────────────────▼──────────────────────┐
│         VISUALIZATION LAYER             │
│  (PyBullet or MuJoCo — demo only)       │
│                                         │
│  - Load any 6/7-DOF arm URDF            │
│  - Animate throw using policy output    │
│  - Show ball flying, landing in bin     │
│  - Doesn't need to be Franka-specific   │
└─────────────────────────────────────────┘
```

The two layers are **almost completely decoupled**. Train and validate in numpy, then plug the trained policy into PyBullet purely for rendering a demo.

### Physics environment (replacing Gazebo)

```python
def throw(release_velocity, release_position, target, wind=None, delay=0.0):
    # Step forward ballistic equations with drag (Equation 35 from paper)
    # Optional: add delay to initial state (shift release point along trajectory)
    # Optional: add wind force vector w each timestep
    # Terminate when z <= zP (supports elevated targets, not just z=0)
    # Return: full trajectory + landing position
```

This is the only "environment" the GP needs.

### Visualization options (pick one)

**PyBullet** (recommended for ease)
```bash
pip install pybullet
```
- Franka Panda URDF included out of the box
- 3D viewer in a window
- Pure Python, no ROS
- Enough for a visual demo

**MuJoCo** (better physics, slightly more setup)
```bash
pip install mujoco
```
- Free since DeepMind acquisition
- Already used by MC-PILCO examples, so integration path is known
- Better physics quality, faster simulation

**Matplotlib 3D** (simplest, no arm)
- Animate ball trajectories in 3D
- Good for showing wind effects and elevated target curves visually
- No robot arm visible — weaker demo but zero overhead

---

## 7. Implementation Order

```
Week 1-2: Core algorithm
  ├── Clone merlresearch/MC-PILCO
  ├── Understand GP model and rollout structure
  ├── Implement numpy ballistic environment
  │     - ballistic equations + drag (Eq. 35)
  │     - configurable release position/velocity
  │     - landing detection at arbitrary zP
  ├── Implement single-shot policy structure (Eq. 14, 15)
  ├── Implement augmented state π(P) → v (Eq. 21)
  └── Implement cost function (Eq. 16, 17)

Week 3: Validate baseline
  ├── Run against paper's Table 1 hyperparameters
  ├── Reproduce ~100% accuracy on ground-level targets
  └── Generate accuracy vs. training throws plot (Fig. 6 equivalent)

Week 4: Exploration 1 — Elevated Targets
  ├── Expand policy output to [zP, α, v]
  ├── Modify cost function Σc
  └── Map feasibility boundary vs. target height

Week 5: Exploration 2 — Wind
  ├── Add wind force term to ballistic environment
  ├── Test constant wind, gusts, turbulence
  └── Accuracy vs. wind speed curves

Week 6: Visualization + writeup
  ├── PyBullet/MuJoCo demo using trained policy
  └── Final report and presentation
```

---

## 8. Key Parameters from the Paper (Table 1)

| Parameter | Simulation value | Meaning |
|-----------|-----------------|---------|
| Nexp | 5 | Exploration throws before first training |
| Na | 0 | Data augmentation rotations per trajectory |
| Nopt | 1500 | Policy optimization steps per trial |
| M | 400 | Monte Carlo particles per optimization step |
| Nb | 250 | Number of RBF basis functions in policy |
| uM | 3.5 m/s | Maximum release velocity |
| Ts | 0.01 s | Simulation timestep |
| T | 1.0 s | Simulation horizon |
| ℓc | 0.1 m | Cost function length scale (= success radius) |
| ℓm | 0.75 m | Minimum target distance |
| ℓM | 2.4 m | Maximum target distance |
| γM | π/6 rad | Maximum lateral throw angle |

---

## 9. Key Equations to Implement

**State:** `x = [p, ṗ]` — 3D position and velocity of ball center of mass

**Augmented state:** `x̃ = [x, P]` — append target position

**Policy (RBF network, Eq. 21):**
```
π_θ(P) = (uM/2) * (tanh(Σ wi * exp(-||ai - P||² / Σπ)) + 1)
```

**Cost function (Eq. 16):**
```
c(x̃_T) = 1 - exp(-||p_T - P||²_Σc)
Σc = diag(1/ℓc, 1/ℓc, 0)   ← ground plane only
Σc = diag(0, 1/ℓc, 1/ℓc)   ← elevated targets (our modification)
```

**Objective (Eq. 17):**
```
J(θ) = E[c(x̃_T)]   ← expectation of cost at landing
```

**Drag equation (Eq. 35):**
```
FD = -0.5 * ρ * CD(ṗ) * A * ||ṗ|| * ṗ
```
Where CD depends on Reynolds number Re = ||ṗ|| * 2r / ν

**Wind extension (our addition):**
```
F_total = FD + F_wind
F_wind = 0.5 * ρ * CD * A * ||w||  * w   (constant wind vector w)
```

---

## 10. Relevant Repos and Links

- **MC-PILCO (base repo):** https://github.com/merlresearch/MC-PILCO
- **MC-PILOT paper:** https://arxiv.org/abs/2502.05595
- **bayesian-optimization:** https://github.com/bayesian-optimization/BayesianOptimization
- **PyBullet:** `pip install pybullet`
- **MuJoCo:** `pip install mujoco`
- **Demo video (paper):** https://youtu.be/0e8IWstunsc

---

## 11. Environment Setup Notes (from actual setup)

### Compatibility fixes applied to MC-PILCO
- **numpy pinned to `<2`** (`numpy==1.26.4`) — NumPy 2.x breaks scipy's `odeint` when the ODE function returns a list containing a mix of scalars and numpy arrays.
- **`simulation_class/model.py` patched** — `Random_exploration.forward()` returns shape `(1, input_dim)` tensor; after `np.array()` this becomes a `(1,1)` array passed as odeint args. Fixed by `.flatten()` + `.item()` for single-input systems (both `Model` and `PMS_Model` rollout methods).
- **Python 3.10.14 via pyenv + venv** — system Python is 3.14 (too new); pyenv used to install 3.10.14. Conda not accessible on host (exists only in container overlay).

### GPU does not accelerate MC-PILCO's policy optimisation
The Monte Carlo simulation is a **sequential Python for-loop over timesteps** (`for t in range(1, control_horizon)` where `control_horizon = int(T_control / T_sampling) = 60`). With 400 particles and 60 serialised GP prediction calls per optimisation step, tensor sizes are too small to saturate the GPU and Python loop overhead dominates. Switching `device` to `cuda:0` gives no measurable speedup on this codebase.

**Implication for our implementation:** Our ballistic simulator can vectorise all 400 particles across the full trajectory without a Python timestep loop (pure tensor ops). That is where GPU will actually pay off — design the simulator accordingly.

---

## 12. What to Tell Your Evaluators

> "We implement MC-PILOT's core algorithm — derived from MC-PILCO — in a pure Python simulation environment, replacing Gazebo/ROS with a numpy ballistic simulator. This is consistent with the paper's own simulation methodology, where the delay distribution is injected as a known parameter rather than estimated from hardware. Our contributions focus on two algorithmic extensions: (1) elevated target support via joint policy optimization over release height, angle, and velocity, and (2) wind robustness analysis across constant, gust, and turbulence regimes. We demonstrate the trained policies visually using PyBullet."
