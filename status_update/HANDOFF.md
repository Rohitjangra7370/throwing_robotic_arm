# Session Handoff — MC-PILOT Throwing Arm

_Last updated: 2026-07-20 (supersedes the 2026-07-17 handoff below this point).
Covers everything since email_update2.md (sent ~17 Jul) — a full "velocity-from-dynamics"
study plus a paper reality-check, run across a single long session with heavy autonomous
debugging. Read `paper/change_history.md` "Exploration 6" and "Exploration 7" and
`paper/paper_comparison.md` for the full technical detail behind every claim below._

## Who / what / goal

Rohit (2nd-year BTech, civil major) — intern in FDP Lab under Deepak Raina (mentor) +
prof Dharmendra Sharma. **Lab hardware target: Kinova Gen3 7-DOF.** End goal: first-author
publication (arXiv → ReScience C → ICRA 2027, deadline ~Sept 15 2026) + hardware deployment.

## The one-paragraph version

The arm's release was cosmetic (ball velocity assigned directly, `resetBaseVelocity`) in
every study before this one. Built real torque control with gravity compensation and an
analytic payload-mass correction. Debugging that control loop surfaced **four independent,
root-caused systematic biases** — a test-timing artifact, an unmodeled-payload feedforward
gap, a target-domain reachability bug, and (the deepest one) a release-position mismatch
that had been causing the trained policy to systematically overshoot every target by
17-28%. Fixing all four collapsed kinematic-mode accuracy to **0.34cm mean** (best in the
project) and gave a *derivable* (not just measured) **1.54cm mean** for the real,
physically-grounded hardware configuration. Then read the actual paper PDF end-to-end,
built a rigorous comparison (some wins, some real gaps, one claim of ours corrected), and
started closing gaps: data augmentation, an analytical-baseline comparison (mixed, honest
result), height-generalization for kinova, and — triggered by the user watching a video and
correctly saying "that doesn't look like a throw" — found and fixed a genuine zero-amplitude
windup bug that's been in every kinova demo this project has ever made.

## State of results (all independently re-verified against real physics with fresh RNG
seeds, not just training logs — see "the model-belief trap" below for why that distinction
matters)

| Result | Status |
|---|---|
| Torque control (Gen3) | real, physically-derived release; joint tracking 0.004-0.009 rad |
| Kinova kinematic-trained → kinematic release | **0.34cm mean / 1.00cm max** (5 seeds × 30) |
| Kinova dynamic-trained → dynamic release (hardware config) | **1.54cm mean / 3.23cm max** (5 seeds × 30) — *derivable* from measured controller noise, not just observed |
| Multi-arm (kuka/franka/xarm6), release-fixed | 1.72-1.87cm mean (3 seeds × 20 each) |
| Kinova height-generalization (H_MAX=0.10m) | 0.34-0.39cm (kinematic) / 1.83-1.88cm (dynamic), **flat across all height bands** |
| Analytical baseline (paper Eq. 13) vs MC-PILOT | **mixed** — MC-PILOT wins on kinova-kinematic (~5x) and xarm6 (~2x); baseline wins on kinova-dynamic/kuka/franka (see below) |
| Data augmentation (`Na`, paper's rotation trick) | implemented + tested; whether it improves *our* data efficiency is still unverified |
| Windup motion (visual + physical) | fixed a real 0deg-swing bug; costs ~1.1cm of accuracy, not yet recovered |

## The four root-caused biases (velocity-from-dynamics study, chronological)

1. **Test-timing artifact**: the tracking-error gate sampled the arm's state *after*
   `p.stepSimulation()` but compared it to the setpoint computed *before* that step — bakes
   in a `|qd|·dt` term that isn't real error. Fixed by sampling before the step (commit
   `38b1655`).
2. **Speed-envelope miscalibration**: `kinova_gen3`'s declared `speed_bounds=(0.3, 1.0)` was
   measured on-axis only; the real zero-clipping ceiling across the full ±30° wedge is
   **u=0.61 m/s**. Recalibrated to `(0.3, 0.6)`, target range `(0.67, 0.74)` (commit
   `276c9bf`).
3. **Unreachable polar target wedge**: targets sampled as (distance-from-origin, angle)
   ignore the release-point offset; off-axis cells needed up to 3x more flight than
   on-axis at the same nominal distance — at the recalibrated speed ceiling, nothing
   beyond ~15° was reachable at all. Fixed with `--flight_targets` (flight-distance
   annulus around the release point, not the origin) (commit `632a7ce`). **Corrected
   framing (commit `dd2b578`, after reading the actual paper)**: this is NOT a flaw in the
   paper's own convention — their release point rotates with target azimuth (Eq. 5), so
   flight distance is always `ℓ-ℓr` by construction. It's a gap in our fixed-release-point
   implementation, exposed because our `ℓr` (0.55m) is comparable to our target range
   (0.67-0.74m) while the paper's `ℓr` (0.07m) is negligible next to theirs (0.7-2.4m).
4. **The deepest one — release-position mismatch (commit `86164d5`)**: `cost_trial_list`
   ("Final trial cost") is computed by simulating particles through the *learned GP
   model*, never against real physics — a genuinely useful methodology finding on its own.
   Oracle-bisection against true physics showed the trained policy commanding **17-28%
   excess speed on 12/12 targets**, unchanged by 2.5x more training trials (ruled out
   data-starvation). Root cause: particles started at the *nominal* release position, but
   reality launches ~4-5cm elsewhere (safe-release teleport + tracking residual). Fix:
   propagate particles from the *empirically observed* mean release position. Kinematic
   accuracy collapsed from ~2.5cm to 0.34cm — the biggest single improvement of the
   session. Also fully explains an earlier, wrongly-celebrated finding ("dynamic beats
   kinematic") as two opposite biases cancelling, not a real effect.

## Paper reality check (`paper/paper_comparison.md`, read the actual PDF, not summaries)

**Where we're ahead**: real validated torque control + payload compensation (paper's sim
doesn't debug at this depth), the model-belief-vs-ground-truth methodology finding,
4-platform generalization (paper: 1 platform), systematic + mechanistically-explained
object sweep, statistically-powered noise dose-response, 21-test regression suite.

**Real gaps, not spin**: no real hardware (the big one — everything above is sim-only);
gripper-delay estimation doesn't exist for kinova (`ReleaseTimingJitter` class defined,
never applied — confirmed by grep); our sub-2cm accuracy is not comparable to the paper's
**10cm real-hardware hit-radius** bar without the vision/gripper-desync noise that
dominates their real error; height-adaptation currently retrains from scratch instead of
the paper's demonstrated zero-new-trials reuse-model trick (Sec 6.3.3); `Nexp=5` used for
kinova matches the paper's *simulation* setting, not its *real-hardware* one (`Nexp=10,
Na=2`) despite kinova being the real-hardware target; no non-spherical object testing.

**Analytical-baseline comparison result (genuinely mixed, not a clean win — done this
session, `eval_baseline.py`)**: MC-PILOT beats the paper's Eq. 13 closed-form baseline on
kinova-kinematic (0.54cm vs 2.71cm) and xarm6 (1.71cm vs 3.54cm), but the baseline wins on
kinova-dynamic (1.32cm vs 1.59cm), kuka (1.10cm vs 1.97cm), and franka (1.53cm vs 2.14cm).
Traced to real drag being tiny everywhere (≤1% of gravity even at kuka's 2.5 m/s) — MC-
PILOT's advantage over the no-drag formula is inherently modest, and only shows through
when policy resolution (250 RBF centers) is tight enough not to swamp it. Kinova's domain
is 7cm (tight coverage); kuka/franka's are 40-50cm (same 250 centers, thin coverage).
**Actionable**: kuka/franka were never hyperparameter-tuned the way kinova was forced to
be — real headroom there, untested.

## The windup bug (found 2026-07-20, triggered by watching the video)

User watched a dynamic-throw video and correctly said the arm "isn't even trying to
throw." Verified: `q_release` computed by `plan_throw`'s IK came out **exactly equal** to
`q_neutral` for kinova (diff = 0.0 rad exactly; kuka swings 44°, xarm6 38°, franka 8.6° for
comparison) — because `default_release_pos` was defined as precisely where the neutral
pose's own forward kinematics already sits. The windup formula collapses to zero for any
multiplier when the thing it's scaling is already zero. This bug has been in every kinova
video/demo the whole project has produced, not something new.

**Fix** (commit `7d8ffa4`): new `windup_delta` profile field gives kinova an explicit
cocked-back pose, independent of `q_release`/`q_neutral` (doesn't touch either — both are
load-bearing for the calibration above). First attempt (0.5/0.6 rad in 0.4s) demanded 134%
of `qd_max` — torque-infeasible, caught correctly, since the windup phase had no torque-
feasibility check before this (only the throw phase did; added one, mirroring it). Sized
down properly using the rest-to-rest cubic's known peak-velocity formula
(`1.5×delta/duration`) instead of guessing again — 0.2/0.25 rad (54% of `qd_max`), verified
torque-feasible and visually confirmed (frame extraction shows genuine rise-back-then-
forward motion).

**Real cost, not hidden**: kinematic mode unaffected (windup is cosmetic there). Dynamic-
trained (hardware config) regresses **1.54cm → 2.68cm mean** (5 seeds × 15 throws,
consistent, not noise). Root cause understood but not fixed: joint-level tracking stays
within gate, but the windup phase now has genuinely nonzero acceleration at its endpoint,
creating a discontinuity at the windup/throw handoff that the Jacobian-transpose payload
correction is more sensitive to than plain joint-angle tracking shows.

**Important physics point established in this same discussion**: a bigger windup CANNOT
make the arm throw farther — release speed is `qd_release = pinv(Jacobian) @ v_cmd`,
evaluated purely at the release configuration, with zero dependence on trajectory history.
The 0.6 m/s ceiling is a joint-*velocity* limit (`qd_max`), not a torque/momentum one, so
no amount of run-up increases it (unlike a human throw, where muscle force over distance
is the bottleneck). The only real lever is *accuracy*: smoothing the windup→throw
transition (matching acceleration across the boundary, not just position/velocity) should
recover some or all of the lost 1.1cm while keeping the real motion. **Not yet
implemented — proposed, agreed as the next step, not started.**

## Deliverables

- `paper/change_history.md` — "Exploration 6" and "Exploration 7" have the full technical
  narrative, every number, every commit reference.
- `paper/paper_comparison.md` — the full paper comparison (Section A/B/C/D as described
  above), parameter-by-parameter table included.
- `status_update/email_update3.md` — **STALE, needs a rewrite before sending.** Drafted
  mid-session (~commit `42cd3f2`) — covers the four-bias narrative and the 5-seed kinova
  results, but predates: multi-arm regeneration, height-generalization, the full paper
  comparison, `Na`/baseline implementation, and the windup fix. Also still has the
  incorrect "boundary condition of the paper's convention" framing that was corrected
  later (see item 3 above) — needs that specific paragraph rewritten before sending.
- `status_update/vids/mc_pilot_kinova_dynamic_throws.mp4` — dynamic (torque-controlled)
  throw video, regenerated with the windup fix; shows the real backswing-then-throw
  motion. `make_dynamic_video.py` regenerates it (uses the existing `frame_hook` +
  TinyRenderer method, same as every prior project video).
- Checkpoints (real, committed to results dirs, not scratch): `results_mc_pilot_pb_A_
  kinova_gen3/{1..5}`, `..._kinova_gen3_dyn/{1..5}`, `..._kinova_gen3_hgen/{1,2,3}`,
  `..._kinova_gen3_dyn_hgen/1`, `..._kuka_iiwa_flight/{1,2,3}`, `..._franka_panda_flight/
  {1,2,3}`, `..._xarm6_flight/{1,2,3}`. Superseded generations kept for before/after
  evidence: `..._uncalibrated/`, `..._releasebias/`.
- 21 pytest tests, `mc-pilot-pybullet/tests/` — first regression suite in the repo.

## New scripts this session

`measure_tracking_error.py`, `eval_sim2sim_gap.py`, `validate_dynamics.py`,
`eval_generalization.py`, `eval_noise_stress.py`, `eval_heightgen.py`, `eval_baseline.py`,
`make_dynamic_video.py` — all in `mc-pilot-pybullet/`. All real-physics evaluators (never
trust `cost_trial_list` alone — see the model-belief trap below).

## Gotchas learned this session

- **The model-belief trap**: `cost_trial_list` / "Final trial cost" is computed via
  particle simulation through the *learned GP model*. It can be near-zero while real
  accuracy is off by 17-28%. Never report training cost as an accuracy claim — always
  verify against the true rollout pipeline (`PyBulletThrowingSystem.rollout`), and prefer
  fresh, previously-unused RNG seeds when re-checking a number, not the same ones that
  produced it.
- PyBullet's `F.dropout(x, p=0.0)` is an exact no-op (rules out a dropout-scaling
  hypothesis quickly if it ever comes up again).
- Torque saturation and insufficient-gain-stiffness look similar at first (both show large
  tracking error) but behave oppositely under a gain sweep: insufficient stiffness
  improves with higher gains, saturation gets *worse* (bang-bang oscillation). If
  increasing kp/kd makes things worse, check torque headroom before tuning further.
  (Directly caused the windup-gain-tuning detour this session.)
- A rest-to-rest cubic's peak velocity is exactly `1.5×displacement/duration` and peak
  acceleration is `6×displacement/duration²` — use this to size any new windup/trajectory
  segment against `qd_max`/`tau_max` *before* running it, not after debugging a failure.
- PyBullet's stdout warnings (`b3Warning[...]`) don't end in a newline, so they can merge
  with the next `print()` call and get silently eaten by a `grep -v` filter tuned to
  remove them. If a print statement seems to have "vanished," check for this before
  assuming a crash.
- GPU (CUDA) is slower than CPU for this workload (215.6s vs 156.6s / 3 kinova trials) —
  small tensors, PyBullet itself is CPU-only regardless. Confirmed again this session;
  don't re-litigate it.

## Agreed next milestone / open items, roughly in priority order

1. **Smooth the windup→throw transition** (acceleration-continuous handoff) — agreed next
   step with the user, not started. Should recover some/all of the 1.1cm windup-fix cost.
2. **Rewrite `email_update3.md`** — stale, missing ~60% of this session's work, has one
   known-wrong paragraph.
3. Kuka/franka hyperparameter re-tuning (Nb/lengthscale/Nexp) + re-run the baseline
   comparison — direct test of whether MC-PILOT can be made to beat the analytical formula
   there too (currently it doesn't).
4. Gripper-delay estimation for kinova (paper Sec 5, Bayesian Optimization) — no real
   hardware yet, so build a sim-validated version first.
5. Reuse-model height-adaptation (paper's actual zero-new-trials trick, Sec 6.3.3) instead
   of the current full-retrain approach.
6. `Nexp=5 → 10` retrain for the hardware-facing kinova config (matches the paper's real-
   hardware setting, not its sim one).
7. Verify whether `Na` (now implemented) actually improves *our* data efficiency — the
   mechanism is built and tested, the claim itself is still open.
8. Non-spherical object testing (cube/cylinder-like, paper tests these).
9. Real hardware: Kortex driver + ~10 calibration throws, once the above sim-side items
   are in better shape.

---

# Superseded handoff (2026-07-17 evening) — kept for historical reference only

Everything below this line predates email_update2.md and the entire velocity-from-dynamics
study above. Left in place for continuity; do not treat any number below as current.

## State of results (all validated, all in status_update/)

| Result | Status |
|---|---|
| NumPy baseline seed study | random explor: 1/5 seeds converge → stratified: 3/5 → + lengthscale fix (ℓs=0.15×range): **5/5** |
| PyBullet release-collision bug | found (arm strikes released ball → GP data poisoned → lengthscale collapse 250→8.4), fixed, validated **5 seeds 50/50** |
| Proper eval (KUKA, ground) | 250 fresh targets: **100% <5cm, mean 1.86cm, max 3.81cm** |
| Variable basket height (new capability) | per-height policies h=0.25 (10/10), h=0.45 (9/10) |
| **Height-generalized single policy** | target=(Px,Py,h), 9-D state, 25 trials: **100/100 fresh throws at random h∈[0,0.45], mean 2.0cm** |
| Kinova Gen3 sim | profile + envelope measured (≤1.0 m/s, targets 0.67–0.87m); 2 seeds 9/10, converge 2–3cm |

## Gotchas learned (2026-07-17 session)

- PyBullet reuses client id 0 after disconnect → per-world init logic must reset manually
- OOM kills (no traceback) when Chrome eats RAM — free up before big batches; runs died twice
- Video: GUI/Xvfb capture path stalls; snapshot method (TinyRenderer) is fast and reliable
- Eval must use the true pipeline (PyBulletThrowingSystem.rollout) — hand-rolled demo re-implementations
  had ~10cm systematic discrepancy
- Prof cares about seed methodology (consecutive-seeds question) — protocol answers in meeting_prep.md
- Physical bucket collision walls deflect near-horizontal approaches — buckets in videos are visual-only
