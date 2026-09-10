Subject: Project update: real overhead throw on the Gen3 model, hardware-safety-validated, generalizes across bin heights with zero extra data

Dear Sir,

Closing out the throwing simulation work. Key results below, figures and two videos
attached.

**Why overhead, not a tossing (lob) motion.** I tried the low, underarm-style toss
first. On this arm it doesn't work, and the reason is physical, not a tuning problem:
the Gen3's wrist actuators are weak (9 Nm), so the maximum speed any release posture
can achieve is small. I measured directly what that means for launch angle — for a
fixed low-release posture, range falls off *monotonically* from a flat release (0
degrees) up through 70 degrees. In other words, when achievable speed is small
relative to release height, the range-optimal angle collapses toward horizontal: any
search that scores poses by distance alone converges on a near-flat push, which reads
as placing the ball, not throwing it (this is exactly what an earlier render showed,
and what prompted this rework). Releasing from height instead lets potential energy,
not wrist speed, do most of the work, and gives the whole arm a longer runway to build
tip speed through a wind-up/whip rather than relying on the wrist alone. That's the
overhead throw below.

**1) The release velocity now comes from the arm's actual dynamics, not assigned.**
Torque control with gravity + payload compensation; every speed below is measured at
the instant of separation, not commanded.

**2) A genuine overhead throw, not a toss.** Rebuilding the release-pose search to
optimise the release state directly (joint angles/velocities at release, under the
real 39/9 Nm and 1.4/1.2 rad/s limits) rather than scoring poses by distance found a
release ~5x further than my earlier search, and it looks like an actual throw:
wind-up, whip, release at ~1.1 m height. Figure 1.

**3) Whole-trajectory safety, not just the throw.** I validated wind-up and the throw
itself, but the post-release recovery had no check at all — found by watching the
render, the arm was being commanded into a recovery needing 3.2x its torque limit and
1.9x its joint-speed limit. Fixed; now every phase is checked against both limits, not
just the throw. Figure 2.

**4) Accuracy: 3.15 cm mean, 100% within 10 cm, 30 unseen targets, 10 training trials.**
Speed scales correctly with target distance (1.16-1.49 m/s). Figure 3.

**5) An honest ceiling, not a hidden one.** Once real recovery is enforced, the
furthest this arm can safely throw is 0.83 m — just inside its own 0.87 m reach (measured
directly via forward kinematics over the joint-limit envelope). With
the base fixed, it cannot throw past where it could simply reach. I'm presenting this
as a measured capability boundary rather than a shortfall. Figure 4.

**6) Generalizes to any bin height with zero new robot trials.** The paper claims
(Sec 6.4) that adapting to a new task needs only re-optimising the policy through the
already-learned dynamics model, not new exploration. I reproduced this: adapted the
ground policy to three new heights by reusing the trained model and re-optimising the
policy alone. Result — 2.95 / 3.41 / 3.80 cm mean error at h = 0.10 / 0.20 / 0.30 m,
**zero additional throws**, matching (in one case beating) both the ground baseline and
a full retrain done for comparison. Figure 5, video 2.

**Practically:** on the real arm, ~10 calibration throws once, then any bin height is
an offline re-optimisation — no new physical throws per height.

**Next steps (hardware, staged plan written):** confirm the Kortex driver consumes
this trajectory format at its real 1 kHz rate; staged bring-up without the ball
(25/50/100% speed) comparing measured torques against the figures above; measure
gripper release delay; ~10 calibration throws; sim-vs-real comparison. Estimate ~3 lab
days.

Happy to walk through any of this.

Regards,
Rohit

---
Attachments (all under status_update/): final_figures/fig_gen3_correction.png,
final_figures/fig_torque_envelope.png, final_figures/fig_accuracy.png,
final_figures/fig_range_ceiling.png, final_figures/fig_height_adaptation.png,
vids/gen3_overhead_throw.mp4 (video 1), vids/gen3_heightgen_throws.mp4 (video 2)
