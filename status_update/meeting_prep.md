# Meeting Prep — MC-PILOT Throwing Arm (FDP Lab)

## 1. The one-paragraph summary (open with this)

We stress-tested the MC-PILOT reproduction across random seeds and found the headline
results don't replicate: the NumPy baseline fails on 3 of 5 seeds, and the PyBullet arm
destabilizes mid-training on both seeds tried. We root-caused both failures (an
exploration/lengthscale mismatch in the baseline; a release-instant arm-ball collision
bug in PyBullet), fixed both, and validated with a proper protocol: 5 seeds x 50 fresh
targets = 250 evaluation throws, 100% hit rate, mean landing error 1.86 cm, worst
throw 3.81 cm.

## 2. How good is this — honest calibration

**Strong:**
- 250/250 hits on fresh (unseen) targets, all within 3.81 cm, at both the 10 cm and
  5 cm thresholds. Zero variance across seeds (all 5 between 1.8-2.0 cm mean).
- More importantly: the *reliability analysis* is the real contribution. Nobody
  publishes "we ran it 5 times and 3 failed" — that finding plus two closed-form,
  validated fixes is what's new. The 100% itself matches what the original paper
  claims in sim; our value-add is showing WHEN it breaks and WHY.

**Be careful not to oversell:**
- Evaluation is in the same deterministic simulator the policy trained in, with
  noise=0. It proves the learning pipeline is correct and stable — it does NOT prove
  robustness to noise or hardware transfer. If asked "would this work on a real arm?"
  the honest answer: unknown; that's exactly the gap the noise-injection studies
  (velocity slip, timing jitter) were built to bridge, and hardware validation
  remains future work.
- Launch angle fixed at 35 deg; policy learns speed only.
- These are Config A results (ground targets, z=0.5 m release). Elevated configs
  B/C/D still carry the untuned lengthscale default — same fix should apply, not yet run.

## 3. What we used (technical stack)

| Layer | What |
|---|---|
| Algorithm | MC-PILOT (Turcato et al., arXiv:2502.05595), built on MC-PILCO (Amadio et al., IEEE T-RO 2022) |
| Dynamics model | 3 independent sparse GPs (squared-exponential kernel, subset-of-data approx), input = 6-D ball state, output = per-step velocity corrections |
| Policy | RBF network, 250 basis functions, target (Px,Py) -> release speed, tanh-squashed to [0, uM]; fired once per throw (single-shot) |
| Policy gradient | Monte Carlo: 400 particles through the GP posterior via reparameterization trick, Adam, up to 1500 steps/trial |
| Simulator | PyBullet, KUKA iiwa7 (URDF), IK + cubic 3-phase joint trajectory (neutral->windup->release), ball velocity set at release via resetBaseVelocity, manual drag force (Eq. 35) |
| Software | Python 3.10, PyTorch 2.9 (CPU — GPU benchmarked at only ~15% gain, tensors too small), NumPy, Matplotlib |
| Compute | Laptop; 5 seeds trained in parallel (~15 min total per condition) |
| Protocol | Seeds 1-5 fixed a priori; evaluation = 50 fresh targets/seed, identical target set across seeds, hit = <10 cm |

## 4. The three results (know these cold)

### A. NumPy baseline fragility -> fixed (figs 1-3)
- Reported "5/5 hits" was single-seed luck: random exploration converges on 2/5 seeds.
- Fix 1 (stratified exploration): 3/5. Fix 1+2 (policy lengthscale = 0.15 x target
  range instead of default 1.0): **5/5 seeds converge**, mean errors 1-3 cm.
- Mechanism: default lengthscale 1.0 makes the policy nearly blind to which target it
  got (RBF activation ~0.94 across the whole target range -> throws max speed at
  everything, converges on lucky seeds only).

### B. PyBullet destabilization -> root-caused and fixed (figs 4-5, meeting_notes_pybullet_fix.md)
- Both seeds: perfect first ~5 trials, then cost blows up 10-30x.
- Diagnosis chain: GP rollout error exploded 1e-5 -> 0.27 -> scanned training data ->
  all impossible points at z=0.5 (release), signature = arm follow-through STRIKING
  the just-released ball (collision re-enabled at release; KUKA profile lacked the
  safe-release guard Franka/xArm6 had).
- Gradual failure mechanism (the interesting part): GP first absorbs contact points
  as noise (sigma=0.10), then as they accumulate MLE starts fitting them —
  lengthscale collapses 250 -> 8.4, noise 0.10 -> 0.006, model overfits and
  hallucinates. Same degeneration family as the sigma=0 collapse already in
  paper/change_history.md.
- Fix: keep ball-arm collision disabled post-release (arm is already cosmetic by
  design — velocity set explicitly). 2-line change + comment.
- Data verification after fix: delta-v targets now pure physics (dvz brackets the
  -0.196 gravity step; was -1.63).

### C. Proper evaluation (fig 6, eval_matrix.md, video)
- 5 seeds x 50 fresh targets = 250 throws: **100% hit rate (<10 cm AND <5 cm),
  mean 1.86 cm, median 1.77 cm, P95 3.17 cm, max 3.81 cm.**
- Video: mc_pilot_pb_A_demo.mp4 — 6/6 hits, recorded from the exact training
  pipeline (not a reimplementation), HUD + target disc + ball trail.

## 5. Numbers to memorize
- 2/5 -> 3/5 -> 5/5 (baseline seeds converging: random -> stratified -> +lengthscale)
- ls = 0.15 x target_range (the lengthscale rule)
- 250/250, 1.86 cm mean, 3.81 cm max (evaluation)
- GP degeneration: lengthscale 250 -> 8.4, noise 0.10 -> 0.006
- Contact contamination: |dvz| up to 8x gravity per step at release
- ~10 trials per training run (5 exploration + policy trials); ~15 min for 5 seeds parallel

## 6. Likely questions + answers

**Q: Why consecutive seeds 1-5?**
Seed value has no meaning — it initializes the RNG; consecutive integers are the field
convention (Henderson et al., AAAI 2018). What matters: fixed before running, all
reported including failures. Happy to switch to documented random draws going forward.

**Q: Is 100% believable?**
Yes, because it's a deterministic simulator the model learned near-perfectly — the
original paper reports the same in sim. The novelty isn't the 100%; it's the seed
fragility we exposed and fixed. Real-arm performance would be lower; that's the
noise/hardware gap, which is future work.

**Q: What's actually new vs. the paper?**
(1) Multi-seed reliability analysis (paper reports single runs); (2) the two
convergence rules with mechanism explanations; (3) the contact-contamination ->
GP-degeneration failure mode, diagnosed end-to-end with data; (4) honest evaluation
protocol on fresh targets.

**Q: Did you just tune hyperparameters until it worked?**
No — each change is a diagnosed root cause with a mechanism, a prediction, and a
validation across all 5 seeds. E.g., the lengthscale rule predicts sensitivity
S = exp(-d^2/2ls^2); we verified failing configs had S~0.94 and fixed ones ~0.07.

**Q: Next steps?**
(1) Port the lengthscale fix to elevated configs B/C/D + multi-seed (same protocol);
(2) noise-robustness evaluation (slip/jitter at eval time); (3) write up as
replication+extension paper — ReScience C (rolling) and/or ICRA 2027 (deadline
Sept 15, 2026); arXiv preprint first. (4) Discuss authorship: I do experiments+writing
as first author, you and the original group as co-authors.

## 7. Files to bring (all in status_update/)
- [ ] mc_pilot_pb_A_demo.mp4 — THE video (6/6 hits, HUD, 17 s)
- [ ] eval_matrix.md / .csv — the 250-throw results table
- [ ] fig6_eval_error_distribution.png — error boxplot, all seeds
- [ ] fig4_pb_cost_before_after.png + fig5_pb_hitrate_before_after.png — PyBullet bug story
- [ ] fig1/2/3 — NumPy baseline fragility story
- [ ] meeting_notes_pybullet_fix.md — full bug writeup
- [ ] status_email.md — what was already sent to Deepak/Dharmendra
- Laptop: repo at ~/trade/research/throwing_robotic_arm, demo rerunnable via
  `cd mc-pilot-pybullet && python3 demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1`
  (GUI mode, now also collision-fixed)

## 8. Uncommitted changes (mention if asked about repo state)
- simulation_class/model_pybullet.py — release-collision fix + optional frame_hook
- demo_pybullet_gui.py — same collision fix
- simulation_class/wind_models.py — repaired corrupted symlink (was blocking all
  PyBullet training)
- mc-pilot-elevated/test_mc_pilot_a_strat.py, test_mc_pilot_a_strat_ls.py — new
  baseline-geometry configs used for the seed study
- New results dirs: results_mc_pilot_a_strat*, results_mc_pilot_pb_A (fixed),
  results_mc_pilot_pb_A_releasebug (preserved buggy runs for before/after)
