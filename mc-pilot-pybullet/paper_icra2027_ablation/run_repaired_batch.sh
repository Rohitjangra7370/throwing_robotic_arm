#!/usr/bin/env bash
# Regenerate the paper's feasibility ablation under corrected inertials
# (bodyless links given an explicit zero mass instead of PyBullet's 1 kg).
# Only the Gen3 family is affected -- the Panda has no bodyless links, so its
# repaired run must reproduce its published numbers exactly, which is the
# control for this whole exercise.
cd "$(dirname "$0")/.."
printf '%s\n' \
  franka_panda_dyn \
  kinova_gen3_dyn_tau0.50 \
  kinova_gen3_dyn_tau0.75 \
  kinova_gen3_dyn_tau1.00 \
  kinova_gen3_dyn_tau1.50 \
  kinova_gen3_dyn_tau2.25 \
  kinova_gen3_dyn_tau3.33 \
| xargs -P 4 -I{} sh -c '/usr/bin/python3 paper_ablation_feasibility.py {} \
    paper_icra2027_ablation/sweep/{}_repaired.json --repair_inertials \
    > paper_icra2027_ablation/sweep/{}_repaired.log 2>&1'
echo BATCH_COMPLETE
