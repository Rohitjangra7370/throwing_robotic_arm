# Overhead Beyond-Reach Throw — Design (2026-07-22)

## Requirement (from scratch)
A throw only counts if the ball lands **beyond the reachable workspace** —
otherwise the EE can just visit the bin and "throwing" proves nothing
(TossingBot's stated purpose: throwing extends range beyond the workspace).
Gen3 floor-level touch reach: ~0.86 m horizontal. Must run on real hardware:
joint velocity <= qd_max, full-trajectory torque <= 80% tau_max (incl. ball
wrench + Coriolis), joint limits, no base collision.

## Physics budget (measured, this repo, 2026-07-22)
Release-state-first optimization (the literature-standard method: optimize
(q, qd) at release for landing distance under ONLY hardware constraints —
no cosmetic posture filters). Result over 63k feasible states:
- **v_release = 1.89 m/s** (3x the old pipeline's 0.66; binding constraint is
  qd_max, torque only at 34%)
- optimal posture: **fully stretched overhead**, release at z=1.106 m nearly
  above the base axis, launch elevation ~10 deg (height does the range work)
- landing ~0.95-0.98 m from base -> **targets 0.75-0.92 m band** (upper half
  strictly beyond floor-touch reach)

This is the trebuchet/overarm solution — high release, flat angle, whip.

## Architecture
1. **Release-state search** (`find_throw_pose.py --mode overhead`): sagittal
   plane (roll joints = 0), in-plane LP for max speed (the swing plane is
   slightly tilted off x-z by URDF y-offsets — aim the plane, don't fight
   it), filters: joint limits, qd_max, 80% torque incl. ball at release
   velocity, windup-within-limits, windup-path torque. NO elevation caps or
   reach floors.
2. **Rotation-built table**: one verified state rotated across +-33 deg
   (base-rotation invariance verified 1e-6; sign: q[0] -= az). Uniform by
   construction. Entries carry exact `v_dir` (J0 @ qd0 rotated per entry) so
   runtime needs NO LP re-solve — qd scales linearly with commanded speed.
3. **Aiming**: release point ~3 cm from base axis -> release-point azimuth
   offset (the -51.7 deg disaster class) is eliminated by geometry; turret
   correction retained (now ~1-2 deg, exact).
4. **Trajectory**: existing `plan_throw(monotonic_windup=True)` — cock-back,
   linear velocity ramp to release, follow-through decel; torque
   time-scaling loop already validates the whole stroke.
5. **Learning**: MC-PILOT unchanged (paper requirements): policy(target) ->
   u in [1.0, 1.85], GP flight model, 10 trials, flight-annulus targets
   lm=0.75 lM=0.92, +-30 deg azimuth, lengthscale_xy ~0.026.

## Validation gates
- Aim probe: lateral miss ~0 at test targets before training.
- Whole-trajectory |qd| <= qd_max and |tau| <= 0.8 tau_max.
- Real-physics landing error < 10 cm mean after 10 trials.
- Frames visually show windup -> whip -> free flight to a bin the arm
  cannot touch.

## References
- Asgari & Nikoobin 2021 (throw-able workspace outside reachable workspace)
- Monastirsky et al. (release-state optimization = upper bound)
- Zeng et al., TossingBot, T-RO 2020 (fixed release state, velocity control)
- Turcato et al., MC-PILOT, arXiv 2502.05595 (Eq. 5 rotating release)
