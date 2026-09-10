# PyBullet study — bug found, root-caused, fixed, validated (for FDP lab meeting)

## Summary
The PyBullet (KUKA iiwa7) training pipeline had a physics bug that silently degraded
the learned model as training progressed. Found it, fixed it (2-line change), and
validated across 5 seeds: **50/50 hits, mean landing error ~2 cm, zero instability.**

## The bug
At the moment of ball release, the code re-enabled ball–arm collision while the ball
still overlapped the swinging arm. The arm's follow-through struck the just-released
ball, injecting contact impulses into the first flight transitions of the training
data (measured: vertical velocity changes up to 8x gravity per timestep, plus forward
"push" spikes — all located at z ≈ 0.5 m, the release height, on the fastest throws).

## Why it degraded training gradually (the interesting part)
With few contaminated points, the GP's likelihood optimization treats them as noise
(sigma_n ≈ 0.10, long lengthscales — early trials looked perfect). As contaminated
points accumulated over trials, the optimizer began *fitting* them instead: the Δvz
GP's lengthscale collapsed 250 → 8.4 and its noise estimate 0.10 → 0.006, producing
an overfit model whose rollout error exploded from ~1e-5 to 0.27. The policy then
optimized against a hallucinating model → hit rate fell to 40–50% with 20–40 cm misses.
Same GP-degeneration family as the sigma=0 lengthscale collapse documented earlier in
paper/change_history.md.

## The fix
Keep ball–arm collision disabled after release in the training rollout
(simulation_class/model_pybullet.py) and the GUI demo (demo_pybullet_gui.py).
Justified: this codebase already treats the arm as cosmetic at release — ball velocity
is set explicitly via resetBaseVelocity — so the follow-through strike was pure artifact.
Franka/xArm6 profiles already avoided it via use_safe_release=True; KUKA (the default)
did not.

## Validation (5 seeds, 10 policy trials each, seeds fixed a priori)
| Seed | Before fix | After fix |
|---|---|---|
| 1 | 4/10 hits, cost drifting up to 0.36 | 10/10, cost ~0.001 flat |
| 2 | 5/10 hits, cost spiked to 0.47 | 10/10, cost ~0.001 flat |
| 3–5 | (not run before fix) | 10/10 each |

Data sanity check after fix: GP training targets now physically clean —
Δvx/Δvy within ±0.004 (drag only; was ±0.45), Δvz in [−0.29, −0.12] bracketing the
−0.196 gravity step (was −1.63).

## Files
- fig4_pb_cost_before_after.png — per-trial cost, seeds 1–2, before vs after
- fig5_pb_hitrate_before_after.png — hit rates, all seeds
- mc_pilot_pb_A_seed1_demo.gif — demo recorded from the exact training pipeline
  (3/3 hits, 1.8–3.1 cm errors, red disc = target)
- Old (buggy) results preserved in mc-pilot-pybullet/results_mc_pilot_pb_A_releasebug/
  for the before/after comparison.
