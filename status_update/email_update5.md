Subject: Update: the release velocity now comes from the arm's dynamics — the open problem from my last mail is solved

Dear Sir,

In my last mail I wrote that the genuinely hard half of this problem was generating the
ball's velocity from the arm's actual dynamics rather than assigning it, and that I
intended to take that up next. That is now done, and the arm performs a real overhead
throw. Below is what it took, including two things I got wrong along the way and a
safety issue I think is the most useful finding of the lot.

**1) The release velocity is now measured, not assigned**

The Gen3 model runs under computed-torque control with gravity and payload compensation.
The ball is carried by the arm and released with whatever momentum the tracked motion
actually gave it — there is no `resetBaseVelocity` anywhere in this path. Every speed
quoted in this mail is measured at the instant of separation, and differs from the
commanded value, which is itself the evidence that it is real tracking rather than
assignment.

Joint tracking error is 0.004–0.009 rad through the throw. Getting there required adding
the payload's generalised force analytically (J^T m (a - g)) — PyBullet's inverse dynamics
only knows the arm's own URDF masses, and the gripped ball is a separate body, so without
that term the controller was systematically under-torquing.

**2) Three systematic biases I had to find first**

These are worth recording because two of them are properties of the method, not typos.

*(a) A boundary condition of the paper's target convention — I think this is a candidate
contribution.* The paper samples targets by (distance-from-origin, angle), but the ball
flies from the release point, which is offset from the origin. An off-axis target at the
same nominal "distance" therefore needs up to ~3x more flight than an on-axis one. With
the Gen3's honest speed ceiling, nothing beyond about 15 degrees azimuth was reachable at
all: the policy saturated at maximum speed and training sat on a cost floor that no number
of trials could fix. The paper's Panda never encounters this because its speed headroom
covers its whole wedge. Fixed by sampling targets in flight-space — an annulus around the
release point — after which the same pipeline trains cleanly on four arms (KUKA, Franka,
xArm6, Kinova; 1.4–2.7 cm mean error, multi-seed).

I should also correct a number from my last mail: I quoted ~1.0 m/s for the Gen3's
end-effector speed, but that was an on-axis measurement. The honest figure across the
full wedge was 0.61 m/s.

*(b) The training loss is a belief, not a measurement.* MC-PILCO's reported trial cost is
computed by simulating particles through the *learned GP model*, so it measures how well
the policy satisfies the model's own belief — never real accuracy. Checking the trained
policy against the true-physics optimal speed (bisection per target) showed it commanding
17–28% excess speed on every target, and this did not improve with 2.5x more trials. Cause:
the particles started at the nominal release position while reality launches a few cm
downrange, and a start-point error is a landing bias the GP structurally cannot learn away,
since it only models velocity dynamics. Fixed by propagating particles from the empirical
mean release position observed in the collected trials.

*(c) The pose search was optimising the wrong thing.* Scoring candidate release poses by
throwing distance alone quietly collapses, on a torque-limited arm, onto a near-horizontal
release — technically a launch, but in practice the arm was setting the ball down rather
than throwing it. Replacing that with optimisation of the *release state* (joint angles and
joint velocities at the release instant) under the real actuator limits — the standard
formulation in the throwing literature — changes the outcome substantially on the same arm
with the same 39/9 Nm limits:

[figure: fig_gen3_correction.png]

The resulting motion is a genuine overhead throw: wind-up, whip through, release at about
1.10 m height roughly above the arm's own base.

**3) Accuracy: 3.2 cm mean over 30 unseen targets, all within 10 cm, from 10 trials**

[figure: fig_accuracy.png]

Trained with MC-PILOT exactly as in the paper (10 trials), then evaluated on 30 freshly
sampled targets using an unused random seed. Mean landing error 3.2 cm, worst 6.7 cm, 100%
inside 10 cm. Release speed varies with target distance (1.16–1.49 m/s), so the policy is
genuinely modulating speed rather than saturating.

Aiming is exact by construction rather than learned: the release posture is searched once
in the arm's vertical plane and then rotated about the base to aim, so the commanded
velocity vector points at the target to numerical precision at every azimuth across the
±33 degree wedge.

**4) A safety gap that I think is the most useful finding here**

While checking the motion I found the arm was being commanded into a post-release
deceleration requiring up to **3.2x its torque limit** and, separately, **1.9x its joint
velocity limit**. The throw itself had been validated; the recovery after the ball leaves
the hand had never been checked. On the real arm this is not cosmetic — it is a trajectory
the controller cannot execute and would fault on.

There were two independent causes: the follow-through phase had no feasibility check of its
own (wind-up and throw both had one), and the simulation stopped commanding the arm at the
moment of release, so it was free-falling under gravity instead of tracking the planned
recovery. Both are fixed, and validation now covers the entire motion — wind-up, throw and
follow-through — sampled continuously against both torque and joint-velocity limits:

[figure: fig_torque_envelope.png]

Peak 0.94 of the torque limit and 0.93 of the velocity limit, everywhere, including after
release. I dwell on this because throwing work generally validates the throw and stops
there; on a torque-limited arm the *recovery* turns out to be the binding constraint, and
I think that belongs in the writeup.

**5) The honest ceiling: safe recovery costs range**

[figure: fig_range_ceiling.png]

Once safe recovery is enforced, the furthest this arm can throw drops to 0.83 m — just
inside its own 0.86 m reach. With the base fixed, the Gen3 cannot throw to a point it could
not also have reached by extending. Faster releases exist (1.93 m/s, landing past 1.0 m) but
only if one ignores whether the arm can stop safely afterwards, which I do not think we
should.

So on our hardware the contribution is the precision and data-efficiency of a physically
real learned release, plus a quantified statement of where the safety boundary sits — rather
than reach extension, which needs a faster arm. I would present that boundary as a
measurement, not as a shortcoming of the method.

**6) When is learning actually worth it, versus a closed-form formula?**

I compared MC-PILOT against the paper's analytical (no-drag) release formula. With a normal
ball, aerodynamic drag is under 1% of gravity, so the closed form is already near-optimal
and learning adds little — on some arms the formula is slightly better. With a light,
high-drag object (~19% drag) the closed form degrades to 4–6 cm error while MC-PILOT holds
0.6–1.7 cm, a 3–7x gain. Learning earns its value precisely where the analytical model
fails, and I think this crossover is a clean result for the writeup.

**7) Where this leaves the hardware plan**

The simulation side is complete, and what runs in simulation is what would run on the Kortex
driver — torque control, gravity and payload compensation, measured release. I have also
written the hardware control layer: a staged, safety-gated throw executor that is dry-run by
default, speed-limited, and aborts on any joint-limit violation; the dry-run stage passes on
the trained policy.

Remaining, in order: (i) confirm the driver consumes the new trajectory format, and that
setpoints are evaluated at the Gen3's 1 kHz rate rather than replayed at the simulation's
50 Hz; (ii) staged bring-up without the ball at 25%, 50% and 100% speed, comparing measured
Kortex torques against the predictions in the figure above; (iii) measure the gripper's
opening delay, which at 1.5 m/s costs about 3 cm of accuracy per 20 ms — this is exactly the
paper's t_d ~ U(a,b) term; (iv) about ten calibration throws, which double as MC-PILOT's real
training data; then the sim-versus-real comparison.

I would estimate roughly three lab days for that sequence, and have written it up as a staged
plan with safety gates between the steps.

Happy to walk through any of this whenever convenient.

Regards,
Rohit

---
Attachments: fig_gen3_correction.png, fig_accuracy.png, fig_torque_envelope.png,
fig_range_ceiling.png
