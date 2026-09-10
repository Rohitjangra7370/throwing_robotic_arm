# Comparison: This Project vs. the Original MC-PILOT Paper

**Original paper:** Turcato, Giacomuzzo, Terrean, Allegro, Carli, Dalla Libera —
*"Data-efficient Robotic Object Throwing with Model-Based Reinforcement
Learning"* (arXiv:2502.05595, Feb 2025). Real robot: Franka Emika Panda + Franka
Hand w/ custom 3D-printed fingertips. `MC_PILOT_ORIGINAL_PAPER.pdf` in this repo.

**This repo:** fork of `github.com/dnfy502/ar525_project`. Two phases:
- **Pre-fork (2026-03-25 → 2026-04-29, AR525 Group 3):** 5 studies, done and
  frozen — baseline, elevated release, PyBullet arm physics, wind, vision.
- **Post-fork (2026-07-02 → 2026-08-10, this progress report, 66 commits):**
  reliability fixes, real torque-controlled release, multi-arm generalization,
  overhead-throw redesign, height adaptation, and real Gen3 hardware bring-up.

Purpose of this doc: a plain list of where the current codebase does something
different from — or goes beyond — the original paper, and where it honestly
still falls short of it. Written for ICRA 2027 framing.

---

## 1. What the original paper actually did (baseline to compare against)

| Aspect | Original paper |
|---|---|
| Platform | Single real robot: Franka Emika Panda (7-DOF), Franka Hand gripper, custom prosthetic fingertips |
| Release pose | **Fixed analytically** (Eq. 31) — only the base yaw (aim) and a single scalar release *speed* are learned/commanded. The swing motion itself is hand-engineered, not searched |
| Release velocity | Real, physically produced by the arm's joint velocity controller (`libfranka`, 1 kHz) — a genuine physical throw |
| Release synchronization | Models gripper-open delay as a **learned probability distribution** `t_d ~ U(â, â+b̂)`, estimated via Bayesian optimization, propagated into the particle-based policy optimization |
| Targets | Mostly ground Cartesian points; one final demo retargets to a bin (z shifted) by re-optimizing the policy only, "~15 min on an RTX 3060 laptop," no quantitative height-vs-error curve |
| Objects | 5 real objects: rubber ball, tennis ball, cube, cylinder, hammer — real material/shape variety |
| Trials | `N_exp=10`, `N_a=2` real trials to convergence; sim ablations use 10 random seeds (`N_exp=5, N_a=0`) |
| Real-world accuracy | Test-target landing error roughly 5–15 cm median (Figs. 11/14/15/16 boxplots), ~100% hit rate at 0.1 m target radius with 3 objects in the final bin demo |
| Safety/feasibility analysis | Not discussed — real controller effort limits are assumed handled by the manufacturer's controller |
| Arms tested | One (Panda) |
| Wind / disturbance | Not studied |
| Vision | Camera network + AprilTags for target/object tracking only, not a learned detector |

---

## 2. Extensions from the pre-fork studies (AR525 Group 3, inherited — cite as prior work, not ours)

These are already differences from the paper, done before this project's own
work started:

- **Elevated release heights** (`mc-pilot-elevated/`) — paper doesn't vary the
  *release* height; this sweeps z_release = 1.0/1.5/2.0 m.
- **A full rigid-body arm physics simulator** (`mc-pilot-pybullet/`, Study 2) —
  the paper's simulated ablations use ideal ballistic dynamics with no arm; this
  puts a real URDF arm (KUKA iiwa7 initially) in PyBullet.
- **Wind robustness study** (`mc-pilot-wind/`) — constant wind, gusts, OU
  turbulence, blind-vs-wind-aware GP comparison. Not present in the paper at
  all. (Caveat, per this repo's own review: underpowered at 15 trials/seed —
  usable as motivation, not as a strong result.)
- **Vision-based target/bin detection** (`mc-pilot-pybullet-yolo/`) — HSV
  segmentation + YOLOv8 feeding the target estimate, vs. the paper's AprilTag
  markers for the same purpose. A genuine step toward a less-instrumented
  setup.

---

## 3. What this project's own work (post-fork) does differently from the paper

### 3.1 The release pose is *learned/searched*, not fixed by hand
The paper's biggest simplification: the throwing motion itself is an
engineered fixed configuration (Eq. 31) with only aim and speed adapted. This
project replaced that with a **direction-constrained LP search over the actual
release state** (joint angles + velocities at release) under the arm's real
torque and velocity limits, generalized across an azimuth wedge via a rotation
table, with axis-perpendicular joints only carrying throw velocity (roll/twist
frozen to avoid corkscrew motion). This is a materially larger scope of what
the "throw" contains as a learned/optimized object, not just its speed.

### 3.2 Whole-trajectory safety analysis (absent in the paper)
The paper does not discuss torque/velocity feasibility of the throwing motion
beyond "the manufacturer's controller handles it." This project built and
validated feasibility checks across **all three phases** — windup, throw, and
follow-through — after finding follow-through was silently violating torque
(3.2×) and velocity (1.9×) limits with zero checking. This produced a
measured, honest safety ceiling (0.83 m reach on the Gen3, inside its 0.87 m
kinematic reach) rather than an assumed one.

### 3.3 Reliability was audited with multiple seeds; the paper's real results are single-run
The paper's only multi-seed statistics are in **simulation** (10 seeds, ideal
ballistic dynamics, no arm). Its real-Panda results are a single run per
object. This project re-ran the baseline across 5 seeds and found the
originally reported "5/5 hits" was **single-seed luck** (true convergence rate
1–2/5 seeds); root-caused it to unstratified exploration + an unscaled RBF
lengthscale, fixed both, and re-verified 5/5 seeds, 250 fresh throws, 100% hit
rate, 1.86 cm mean error. This is a reliability standard the paper itself
doesn't apply to its hardware numbers.

### 3.4 Real torque-controlled release vs. assigned velocity
Earlier in this project's own arm-physics track, the ball's release velocity
was being *assigned* (`resetBaseVelocity`) rather than produced by the arm —
the same category of shortcut the paper's fixed-pose approach avoids by using
a real robot, but which this project's PyBullet track had reintroduced in
simulation. This was found and fixed: computed-torque control (gravity
compensation + Jacobian-transpose payload feedforward, since PyBullet's own
inverse dynamics can't see a separately-modeled gripped ball) now produces a
release velocity that is *measured* off the simulated physics for every
reported number, never assigned. Three further systematic biases (off-axis
target-sampling geometry, particle-model start-point bias, a zero-amplitude
windup bug) were found and fixed getting this to work — the paper doesn't
encounter these because its release pose is fixed and doesn't do release-pose
search.

### 3.5 Height generalization: continuous single policy vs. one discrete demo
The paper's Sec. 6.4 claim (no new trials needed to adapt to a new target
height) is demonstrated **once**, qualitatively, changing ground targets to a
single bin height. This project (a) reproduced that exact claim quantitatively
on the real-dynamics overhead throw (2.95/3.41/3.80 cm error at 3 heights, zero
new trials, matching/beating a full retrain), and (b) went further: trained a
**single policy over a continuous height range** (target = `(Px, Py, h)`,
8-D→9-D state expansion) that hits 100% on 100 fresh targets at randomly
sampled heights never seen in training, with no error trend across the range —
the paper never trains one policy across a height *range*, only re-optimizes
per discrete height.

### 3.6 Multi-arm generalization (paper: one robot)
The paper's real and simulated results are Panda-only. This project trained
the same fixed pipeline on four different arms (KUKA iiwa7, Franka Panda,
xArm6, Kinova Gen3), all converging to 1.4–2.7 cm mean error, showing the
release-pose-search fixes aren't arm-specific patches.

### 3.7 A drag-regime finding the paper doesn't report
The paper compares MC-PILOT to an analytical ballistic baseline only in the
low-drag regime it tested (rubber ball, tennis ball, etc. — Fig. 9 shows drag
is a small correction there). This project explicitly swept drag from <1% to
~19% of gravity and found a **crossover**: the analytical baseline actually
*beats* MC-PILOT at low drag, and MC-PILOT wins 3–7× only once drag is
significant. This reframes *when* learning is worth it — a sharper, more
falsifiable claim than the paper makes.

### 3.8 Delay handling: simpler in one respect, more direct in another
The paper's real contribution around release delay is a full **learned
probability distribution** `t_d ~ U(â, â+b̂)` fit via Bayesian optimization and
propagated stochastically through policy optimization. This project's real
Gen3 work instead **directly measured** gripper latency at 1 kHz over UDP
(67.9 ± 6.4 ms) and compensates it as a wall-clock lead time in the executor.
This is honestly a step back in generality (a point measurement vs. a fitted
distribution) but a step forward in grounding (measured on the actual gripper
hardware at high rate, not inferred from throw outcomes) — worth stating
explicitly rather than silently, since a reviewer familiar with the paper will
ask about it.

### 3.9 Real hardware bring-up beyond what the paper reports diagnosing
The paper doesn't narrate any hardware bring-up difficulties — it presents a
working real system. This project's Gen3 bring-up log (Section 10 of the
progress report) surfaced and fixed real issues the paper either didn't have
(different arm/SDK) or didn't report: a joint-angle wraparound convention bug,
a two-master safety hazard observed live, a control-loop rate correction (40
Hz actual ceiling vs. an assumed 1 kHz), a precheck that was silently running
without gravity compensation, a 3 kg phantom URDF mass, and soft-vs-hard
joint-limit confusion. Each is fixed and measured, not assumed away — this is
useful methods material even though it isn't a performance number.

---

## 4. Honest gaps — where the paper is still ahead of this project

Do not oversell these in the write-up; a reviewer who has read the original
paper will check exactly these:

1. **No real ball has been thrown yet on the Gen3, and no real landing has
   been measured.** The paper's headline results are real, physical, measured
   throws on Panda. This project's Kinova Gen3 numbers are, without exception,
   simulation results — read-only paths, gripper actuation, and full-speed
   joint-speed streaming have been validated live, but never with a ball in
   the gripper. This is stated explicitly in the progress report and must stay
   that way in the paper draft.
2. **No real, physical object-material variety.** The paper tests 5 real
   objects (rubber ball, tennis ball, cube, cylinder, hammer) with real
   friction/inertia/shape effects. This project's object generalization
   (payload mass/size sweep) is simulation-only.
3. **Delay is a point estimate here, a fitted distribution there** (see §3.8)
   — a genuine capability gap, not just a framing difference, if the paper's
   reviewers probe robustness to delay *uncertainty* rather than delay *value*.
4. **Single real robot family validated physically (paper); multi-arm claim
   here is simulation-only.** The 4-arm generalization result (§3.6) has not
   been run on any second piece of real hardware.
5. **The paper's real results already include multi-object, multi-trial
   real accuracy numbers with boxplots.** This project's equivalent statistics
   (multi-seed, 250-throw evaluation) are all simulation; real-hardware
   statistics do not exist yet.

---

## 5. One-paragraph framing suggestion

The honest contribution story for ICRA 2027 is **not** "we beat the paper's
accuracy" (sim-vs-sim comparisons aren't fair, and no real Gen3 throw exists
yet to compare against the paper's real Panda numbers). It is: (a) a
reliability audit that shows the original single-seed/single-run reporting
style hides real fragility, with root causes and fixes; (b) replacing the
paper's fixed, hand-engineered release pose with a searched, torque/velocity-
feasible release pose validated across the whole trajectory, which is what
made a genuinely different (weaker-actuator, 9 Nm wrist) arm than the paper's
Panda throwable at all; (c) a continuous height-generalized policy and a
drag-crossover result that go past what the paper demonstrated; and (d) a
transparent hardware bring-up log up to — but not yet including — a live
thrown ball. The paper is stronger than this project on real-world validation
today; this project is stronger on methodology rigor, release-pose learning,
and generalization breadth. Getting one real loaded throw on the Gen3 is what
would flip the comparison from "complementary" to "supersedes."

---

*Sources: `PROGRESS_REPORT.md` (this repo, 2026-08-10, commit `fb3d05e`),
`status_update/HANDOFF.md`, `status_update/PROGRESS_AND_PLAN.md`, `CLAUDE.md`,
and `MC_PILOT_ORIGINAL_PAPER.pdf` (arXiv:2502.05595v1) — page references: setup
p.10–11, real results p.12–17, Sec. 6.4 discussion p.16.*
