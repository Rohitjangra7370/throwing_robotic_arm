Subject: Update: real arm dynamics for the throw, three systematic biases found and fixed, and a derivable accuracy floor

Dear Sir,

This update covers the velocity-from-dynamics work I proposed in my last mail — the
release velocity now comes from the arm's tracked motion under torque control, not from
directly assigning the ball's velocity. Getting there surfaced three systematic biases
(one of which touches the original paper's own experimental convention), and I want to
report those as plainly as the results, because finding them is most of the value.

1) Torque control with a real release — the arm is no longer cosmetic
I implemented computed-torque control for the Gen3 sim (gravity compensation via inverse
dynamics, plus an analytic Jacobian-transpose term for the ball's mass, which PyBullet's
inverse dynamics cannot see because the gripped ball is a separate body). The ball is now
released by removing the grip constraint and keeping whatever velocity the physics gave
it. Joint tracking error is 0.004-0.009 rad across the whole speed/angle envelope with
the wrist torques (9 Nm) respected. This is the first physically real release in the
project's lineage — everything before assigned the ball's velocity directly.

2) A boundary condition of the paper's target convention (candidate paper contribution)
The paper samples targets by (distance-from-origin, angle). But the ball flies from the
release point, which is offset from the origin — so an off-axis target at the same
"distance" needs up to ~3x more flight distance than an on-axis one. With the Gen3's
honest speed ceiling (0.61 m/s off-axis, not the ~1.0 I reported earlier — that number
was an on-axis measurement), nothing beyond ~15 degrees azimuth was reachable at all:
the policy saturated at maximum speed and training sat at a cost floor no amount of
trials could fix. The paper's Panda never hits this because its speed headroom covers
its whole wedge. I fixed it by sampling targets in flight-space (an annulus around the
release point); the same pipeline then trains cleanly on four different arms (KUKA,
Franka, xArm6, Kinova — 1.4-2.7 cm mean errors, multi-seed).

3) The training loss is a belief, not a measurement — and what that hid
MC-PILCO's reported trial cost is computed by simulating particles through the learned
GP model, so it measures how well the policy satisfies the model's belief, never real
accuracy. Comparing the trained policy against the true-physics optimal speed (bisection
per target) showed it commanding 17-28% excess speed on every target, unchanged with
2.5x more training trials. The root cause: the particle simulation started the ball at
the nominal release position, but reality launches it ~4 cm downrange (a safe-release
offset inherited from the collision fix, plus imperfect tracking of the release pose).
A start-point error is a landing bias the GP cannot learn away, because it only models
velocity dynamics — so the policy compensated by over-throwing, while the model believed
it was on target. The fix is to propagate particles from the empirical mean release
position observed in the collected trials — data-driven, works for any arm or release
convention.

4) Results after the fix
- Kinematic release (idealised): 0.52 cm mean / 1.16 cm max on 30 fresh targets — the
  best accuracy in the project so far, and the systematic overshoot is gone (was 12/12
  targets over; now centred).
- Training directly on the dynamic (torque) release — the hardware configuration:
  1.67 cm mean / 3.17 cm max. This number is derivable, not just observed: the measured
  per-throw tracking scatter (0.06 m/s) times the flight-per-speed slope (0.29 m per
  m/s) predicts a ~1.4-1.8 cm floor. So 1.67 cm is the irreducible scatter of a 50 Hz
  controller, not a modelling residual — and the real Gen3's 1 kHz Kortex loop should
  sit below it.
- An earlier apparent result ("dynamic release lands better than the idealised one")
  turned out to be two opposite biases partially cancelling; after the fix each release
  mode calibrates cleanly to its own physics, which we confirmed in both directions.

5) Smaller items
- Object generalization: landing error is flat (1.4 cm) across 30-150 g and 2-4.5 cm
  ball sizes without retraining — at these speeds drag is negligible and the controller
  compensates the payload mass it measures. This would not hold at higher speeds.
- A noise dose-response study (n=50/condition) confirms the noise taxonomy: zero-mean
  noise degrades accuracy monotonically and is uncompensatable, exactly as the
  framework predicts.
- The repo now has its first regression test suite (17 tests covering the controller,
  planner, release, and noise-model fitting).
- Multi-seed re-runs of everything with the fixed pipeline are in progress; the numbers
  above are what I will finalize across 5 seeds before writing anything into the draft.

Next steps: multi-seed confirmation, then the Kortex driver skeleton and the ~10
calibration throws protocol for the real arm, using the measured release position and
tracking-scatter machinery above (which is exactly what hardware calibration needs).

Attachments: sim2sim comparison plot, tracking-error sweep, noise dose-response plot.

Regards,
Rohit
