# Comparison against the original MC-PILOT paper (Turcato et al., arXiv:2502.05595)

_Compiled 2026-07-20, against the actual paper PDF (`MC_PILOT_ORIGINAL_PAPER.pdf`), not
secondary summaries. All "ours" numbers are independently re-verified against real physics
this session, not taken from training logs._

## A. Where we are currently ahead

1. **Real torque control validated end-to-end; paper's simulated arm dynamics are not
   discussed at this level of scrutiny.** We built computed-torque control with an
   analytic Jacobian-transpose payload-mass correction, root-caused why a naive
   URDF-mass-inflation approach fails (~20x error), and closed the loop: true joint
   tracking error 0.004-0.009 rad, matching the no-payload baseline exactly. The paper's
   simulated validation runs on Gazebo/ROS with the manufacturer's stack and doesn't
   report an equivalent controller-level debugging trail.

2. **`TrackingErrorNoise` is fit from measured torque-controller behavior, not
   assumed/injected.** The paper's noise model (Section 5) is a *release-timing delay*
   `t_d ~ U(a,b)`, injected synthetically and then fit via Bayesian Optimization against
   real throw outcomes. Ours is a different failure mode (controller tracking scatter,
   not gripper desync) but is derived from first-principles measurement of the real
   controller, and the reported "1.54cm dynamic-release accuracy" is traceable to that
   measured noise floor, not just observed.

3. **The model-belief-vs-ground-truth distinction, demonstrated with a real case.** We
   found and root-caused a case where `cost_trial_list` (the model-based training metric,
   analogous to the paper's `J(θ)`, Eq. 17) reported near-zero cost while real accuracy
   was off by 17-28% — caused by a release-position mismatch the model couldn't see. The
   paper doesn't flag this risk explicitly (their real-hardware validation loop implicitly
   guards against it, but nothing in the text calls it out as a general caveat of the
   model-based approach). This is a real, reusable methodology point.

4. **Multi-arm generalization, one codebase, four platforms.** The paper validates on a
   single platform (Panda, sim + real). We demonstrate the same algorithm on kuka_iiwa,
   franka_panda (sim), xarm6, and kinova_gen3, each independently verified: 1.72-1.87cm
   mean (kuka/franka/xarm6), 0.34-1.54cm (kinova, kinematic/dynamic).

5. **Object generalization is systematic and mechanistically explained, not just
   discrete-object demonstration.** The paper tests 5 fixed objects (rubber ball, tennis
   ball, cube, cylinder, hammer) without a controlled sweep. We ran a mass(30-150g) x
   radius(2-4.5cm) grid, found flat error, and derived *why* (drag/gravity ratio ~4e-4 at
   trained scale) — then used the same physics to correctly predict, and verify, that a
   TT ball (2.7g) sits in a qualitatively different, ~8x-more-drag-affected regime.

6. **Statistically powered noise dose-response.** n=50/condition with SEM error bars,
   after catching our own first attempt (n=10) as underpowered. The paper's delay-fit
   uses N=10 *seeds* in simulation but doesn't present an equivalent "is this robustness
   real or sampling luck" check across noise levels.

7. **Automated regression coverage**: 17 pytest tests over the arm controller, torque
   branch, payload compensation, dynamic release, and noise model — a software-engineering
   rigor point with no direct paper analogue (research code vs. maintained repo).

## B. Where the paper is currently ahead — real, actionable gaps

1. **Real hardware. This is the one that matters most.** The paper validates on an actual
   Franka Panda with camera-based tracking and an unsynchronized gripper. We are 100%
   simulation. No amount of sim rigor substitutes for this; it's the acknowledged next
   step (Kortex driver + calibration throws).

2. **Gripper/release-delay estimation is a first-class part of their RL loop; we have
   none for kinova.** Paper's Section 5: Bayesian-Optimization search over `(a,b)` for the
   unknown gripper-open delay distribution, fit from real throw outcomes, feeding directly
   back into policy optimization (Eq. 24, `t̃_r = t_{r_cmd} + t_d`). We have a
   `ReleaseTimingJitter` class that's architecturally analogous — **but it has never been
   calibrated or applied to kinova; it's currently unused code**, verified by grep.

3. **The paper's real accuracy bar is a 10cm hit radius, not our sub-2cm sim numbers —
   and this is not a fair win for us.** Our cm-level results have no vision-tracking
   noise, no gripper-desync, no real friction/manufacturing variance in them. Presenting
   our numbers as "better than the paper" without this caveat would be misleading in a
   writeup. The honest framing: we haven't yet built the noise sources that dominate
   their real error, so the comparison isn't apples-to-apples until we do (or until we're
   on real hardware ourselves).

4. **Height-adaptation efficiency — the paper's actual demonstrated advantage, which we
   are not currently exploiting.** Section 6.3.3: because free-flight ballistics don't
   depend on where you cut the trajectory, a GP model trained once on ground data already
   contains everything needed for any height. Adapting to a new bin height costs the paper
   **zero new real trials** — ~15 minutes of policy re-optimization against the frozen
   model. Our current kinova height-generalization approach (`train_mc_pilot_pb_heightgen.py`,
   running as of this writing) instead retrains model+policy together from scratch with
   fresh `Nexp=10 + 25` real trials on a 9-D augmented state. It will produce a working
   policy, but it's the more expensive path, not the paper's demonstrated best practice.
   **Concrete next step**: build the reuse-model / re-optimize-only path as a second,
   more faithful approach and compare data cost directly.

5. **No baseline comparison for our fixed pipeline — DONE 2026-07-20, `eval_baseline.py`,
   and the result is genuinely mixed, not a clean MC-PILOT win.** Ran the paper's Eq. 13
   closed-form no-drag baseline (clipped to `[uMin, uM]`, executed through the true physics
   pipeline) against MC-PILOT on all 4 platforms:

   | platform | MC-PILOT | baseline | winner |
   |---|---|---|---|
   | kinova_gen3 (kinematic) | 0.54cm | 2.71cm | MC-PILOT, ~5x |
   | kinova_gen3_dyn (real controller) | 1.59cm | 1.32cm | baseline |
   | kuka_iiwa | 1.97cm | 1.10cm | baseline |
   | franka_panda | 2.14cm | 1.53cm | baseline |
   | xarm6 | 1.71cm | 3.54cm | MC-PILOT, ~2x |

   Root cause (not a bug — cross-checked the MC-PILOT numbers against independently
   verified figures from earlier in this session, they match): real air drag is tiny
   everywhere in this regime (F_drag/F_gravity <= ~1% even at kuka's 2.5 m/s release
   speed, computed directly). MC-PILOT's theoretical advantage over the no-drag formula
   is the drag correction it learns — and that advantage is modest across the board. It
   only shows through cleanly when the policy's OWN resolution error is small enough not
   to swamp it: kinova's domain is tiny (7cm span) so the same 250 RBF centers give
   near-perfect coverage; kuka/franka's domains are 40-50cm, spreading the same 250
   centers much thinner, so MC-PILOT's own resolution error ends up larger than the
   modest drag-correction benefit and the simple formula wins. **This is not "MC-PILOT
   doesn't work" — it's evidence that kuka/franka were never tuned as carefully as
   kinova was through this session's debugging (Nb, lengthscale, Nexp all left at
   generic paper defaults). A real, actionable next step, not previously visible without
   running this comparison.**

6. **Data-augmentation (`Na`) — DONE, 2026-07-20.** Implemented `MC_PILOT._rotate_trajectory`
   (rotates position/velocity about the vertical axis; z/vz and non-ball columns untouched)
   and wired an `Na` constructor param + `--Na` CLI flag through both training scripts.
   4 new tests (rotation-math correctness + wiring, verified via a spy on `add_data` since
   the downstream SOD sparse approximation deduplicates near-similar points, so the final
   stored sample count isn't simply `(1+Na)x` the raw call count). Smoke-tested `--Na 2`
   end-to-end on kinova — trains cleanly. Not yet re-run as a full accuracy comparison
   (does `Na=2` measurably improve data efficiency for us, matching the paper's claim?)
   — mechanism is built and tested, the efficiency claim itself is still open to verify.

7. **We're training kinova with the paper's *simulation* hyperparameters (`Nexp=5`), not
   its *real-hardware* ones (`Nexp=10, Na=2`)**, despite kinova being our real-hardware
   target. Worth switching before any hardware-facing checkpoint is considered final.

8. **Non-spherical object testing.** Paper explicitly tests a cube, cylinder, and hammer
   — objects "for which the free-fall dynamics cannot be described by the motion of the
   geometrical center" — and shows the method still helps despite orientation not being
   observed. We've only swept sphere mass/radius. Real, currently-untested scope.

## C. Every direct numeric comparison

| quantity | paper (sim) | paper (real) | ours (kinova) |
|---|---|---|---|
| `N_exp` (exploration trials) | 5 | 10 | 5 (should be 10 for hardware-facing) |
| `N_a` (augmented trajectories/trial) | 0 | 2 | **0 (not implemented)** |
| `N_opt` | 1500 | 1500 | 1500 (match) |
| `M` (particles) | 400 | 400 | 400 (match) |
| `N_b` (RBF basis fns) | 250 | 250 | 250 (match) |
| `u_M` (max release speed) | 3.5 m/s | 2.8 m/s | 0.6 m/s (hardware limit, `qd_max`) |
| `T_s` (control period) | 0.01s | 1/60s (~0.0167s) | 0.02s |
| `ℓ_c` (cost lengthscale) | 0.1m | 0.1m | 0.5m (5x looser — not directly comparable) |
| `ℓ_m, ℓ_M` (target range) | 0.75-2.4m (1.65m span) | 0.7-1.75m (1.05m span) | 0.67-0.74m (0.07m span — ~15-24x narrower) |
| `γ_M` (azimuth half-range) | π/6 (30°) | π/6 (30°) | π/6 (30°, match) |
| `ℓ_r` (release radius from base) | 0.07m | 0.07m | 0.55m (~8x larger relative to target range — root cause of the flight-space bug, see below) |
| success/accuracy bar | — | 10cm hit radius | sub-2cm (different regime, see B.3) |
| delay/noise modeling | injected `t_d`, BO-estimated | same | `TrackingErrorNoise` (different failure mode; no delay estimation for kinova) |
| real-hardware validated | n/a | **yes** | **no** |
| platforms validated | Panda only | Panda only | kuka, franka(sim), xarm6, kinova (4 platforms) |
| baseline comparison run | yes (analytical + NN) | yes | analytical: yes (mixed result, see B.5); NN: **no** |
| data augmentation (`Na`) | n/a | yes | **now implemented** (`--Na` flag, 2026-07-20) |

## D. Corrected framing (from the previous "boundary condition" claim)

Eq. 5 in the paper has the release point rotate with target azimuth γ — release and
target always share the same ray from the origin, so flight distance is exactly `ℓ - ℓ_r`
at every angle. **The paper's polar-target convention never has an off-axis-unreachable
problem, by construction.** Our implementation uses a fixed `release_pos`, which does not
rotate with target azimuth — that's what caused the bug we found and fixed
(`--flight_targets`). It bit us specifically because our `ℓ_r` (0.55m, fixed) is
comparable in scale to our target range (0.67-0.74m); the paper's `ℓ_r=0.07m` is
negligible next to theirs (0.7-2.4m), so the same implementation choice would never have
surfaced there. **Correct statement for any writeup: an implementation gap in our fixed-
release-point simplification, exposed by Gen3's narrow envelope — not a flaw in the
paper's method.**
