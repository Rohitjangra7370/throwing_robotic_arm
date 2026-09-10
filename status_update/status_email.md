Subject: MC-PILOT baseline — found a reliability issue, fix confirmed across 5 seeds

Hi [Prof / Mentor name],

Quick update on the throwing-arm project. Short version: the original baseline result
("5/5 hits in 5 trials") only held for one lucky random seed. I re-ran it across 5 seeds,
found it fails on 3 of them, tracked down why, and fixed it — now all 5 seeds converge
cleanly.

**What we found**
Running the published baseline config on 5 different random seeds (not just the one
originally reported) showed hit rates of 60%, 20%, 80%, 10%, 10% — i.e. it reliably works
on only 1–2 seeds out of 5. Root cause: two hyperparameters that the project's own later
experiments (on a different config) had already identified as failure points, but which
were never applied back to the main baseline:
1. Random exploration throws can leave the model blind to part of the speed range.
2. The policy network's "sensitivity to target position" parameter (lengthscale) was left
   at a default that's too coarse for this target geometry, so the policy can't reliably
   tell nearby targets apart and just throws near max speed.

**What we fixed and observed**
Applying both fixes (stratified exploration + a target-range-scaled lengthscale) and
re-running all 5 seeds: all 5 now converge to >=90% hit rate (four hit 100%) with landing
error under 2–3 cm on average, including the two seeds that failed under every previous
configuration. Fig. 3 shows one of those seeds directly — same seed, same target sequence,
before vs. after.

**Attached**
- fig1_hitrate_by_seed.png — hit rate per seed, all three conditions side by side
- fig2_seeds_converged.png — reliability summary (seeds converged out of 5)
- fig3_seed2_before_after.png — one previously-failing seed, fixed

**Next step**
I'd like to fold this into a short writeup (results + the two identified failure modes) —
happy to talk through authorship/scope when you have a few minutes.

Thanks,
[Your name]
