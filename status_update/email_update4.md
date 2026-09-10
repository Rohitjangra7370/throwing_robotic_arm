Subject: Update: real arm dynamics, an accuracy floor, and an honest note on the Gen3 as a throwing platform

Dear Sir,

A consolidated update on the throwing project since my last mail (heights + Gen3 sim),
including one thing about the Gen3 specifically that I want to state plainly.

1) Real release from the arm's dynamics (not assigned)
The release velocity now comes from the arm under computed-torque control (gravity +
payload compensation), instead of directly setting the ball's velocity. Getting there
surfaced three systematic biases I found and fixed: an unreachable off-axis target region
(fixed by sampling in flight-space), a "training cost is the model's belief, not real
accuracy" trap (the policy was over-throwing 17-28%; fixed by starting the prediction from
the measured release position), and a windup-pose bug (the Gen3 was not visibly swinging).

2) Results (5 seeds x 30 fresh targets, real-physics evaluation)
- Idealised (kinematic) release: 0.34 cm mean, worst 1.00 cm.
- Dynamic (torque) release, the hardware configuration: 1.54 cm mean, worst 3.23 cm.
  This floor is derivable from the controller's measured tracking scatter, so the real
  Gen3's 1 kHz loop should sit below it.
The same pipeline trains cleanly on four arms (KUKA / Franka / xArm6 / Kinova), 1.4-2.7 cm.

3) A candid note on the Gen3 as a throwing platform
Throwing only extends a robot's reach when the arm is fast. The Panda, KUKA and xArm6
reach 2.0-2.5 m/s at the end-effector and throw 0.6-1.1 m, so their results genuinely
demonstrate throwing. The Gen3, by contrast, tops out near 0.6 m/s (its wrist actuators
are only 9 Nm), which limits its throw to roughly 15 cm - so on the Gen3 the ball barely
leaves the hand and the "hits" look almost trivial. This is a property of the hardware,
not the method (for reference, TossingBot used a fast industrial UR5, which is exactly
why their throws travel a metre).

So for our lab arm I would frame the contribution not as distance but as the precision and
data-efficiency of a physically-real learned release - which we can still test meaningfully
on the real Gen3: a handful of calibration throws, characterising the gripper's opening
delay, and a sim-vs-real accuracy comparison. The "long throw" story, if we want it, lives
in simulation on the fast arms (and would only need real hardware if the lab ever had a
faster arm - a used xArm6 is about a quarter the cost of a UR5, but this is optional, not
required).

4) When is learning actually worth it vs a closed-form formula?
I compared MC-PILOT against the paper's analytical (no-drag) baseline. With a normal ball,
drag is under 1% of gravity, so the closed-form is already near-optimal and learning adds
little. With a light, high-drag object (~19% drag) the closed-form breaks to 4-6 cm error
while MC-PILOT stays at 0.6-1.7 cm - a 3-7x gain. So learning earns its value precisely
where the analytical model fails; I think this crossover is a clean result for the writeup.

5) Toward hardware
I wrote the Kortex control layer for the real Gen3 - a staged, safety-gated throw executor
(dry-run by default, speed-limited, aborts on any joint-limit violation); the dry-run stage
already passes on the trained policy. Next is bringing it up on the arm in stages and
measuring the gripper's release delay.

Happy to walk through any of this.

Regards,
Rohit
