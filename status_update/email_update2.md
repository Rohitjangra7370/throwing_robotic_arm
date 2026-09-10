Subject: Update: variable basket heights, one generalized policy, and Kinova Gen3 in sim

Dear Sir,

Three updates since my last mail, plus one realisation about the simulation that I think
is important to state clearly.

1) Variable basket heights
I extended the simulator so the landing plane is a parameter instead of fixed ground:
the trajectory now terminates at the basket's height plane, and the policy-optimisation
particles freeze at the same plane so training and reality agree. Trained separate
policies at h = 0.25 m and h = 0.45 m (on pedestals): 10/10 and 9/10 hits respectively,
2-4 cm errors. Video attached (mc_pilot_heights.mp4).

2) One generalized policy across all heights
Then instead of one policy per height, I trained a single policy that takes basket
height as an input: target becomes (Px, Py, h), state 8-D -> 9-D, with the height
lengthscale set by the same 0.15 x range rule. Because the extra input dimension
thins the RBF coverage (the same effect we saw in the wind study), I raised the budget
to 10 exploration throws + 25 trials. Result: on 100 fresh targets at continuously
random heights (0 to 0.45 m, values never seen in training), the single policy scores
100% hits with mean error 2.0 cm and worst 5.0 cm - and the error shows no trend with
height (fig11 attached). Video: mc_pilot_one_policy_all_heights.mp4 - same policy,
bucket climbing five heights.

3) Kinova Gen3 in simulation (our lab arm)
Since our goal is the lab's Gen3 7-DOF, I integrated it into the simulator using
Kinova's official URDF with the real joint velocity limits (1.396 / 1.222 rad/s).
Two integration issues had to be fixed on the way: the URDF's unbounded continuous
joints broke the controller's limit handling (release pose came out wrong), and
position control could not drive the arm (wrist actuators are only 9 Nm), so it uses
kinematic trajectory mode like the xArm profile. Measured envelope: the Gen3 reaches
only ~1.0 m/s end-effector speed on the throw direction, giving a narrow reachable
band of 0.67-0.87 m - which is exactly the narrow-range geometry where the lengthscale
rule is critical. Training on the Gen3 model: 9/10 hits on both seeds, converging to
2-3 cm (fig12). Video: mc_pilot_kinova_throws.mp4.

The realisation I want to state plainly
While studying why the arm's motion looks similar across throws, I confirmed that the
code I received never actually used the arm's dynamics for the throw: the ball's
release velocity is assigned directly (resetBaseVelocity) and the arm's motion is
purely cosmetic - a documented design decision of the original project, made because
PyBullet's arm tracking is poor (~7% EE error, oscillations). Even this cosmetic layer
had small errors: the release-collision bug I fixed earlier was exactly a violation of
this contract (the cosmetic arm was striking the ball it had already released). The
Kinova sim I built is still kinematic at release, but unlike the earlier arms it is
constrained by the real Gen3's specifications - official URDF geometry and true joint
velocity limits - so its commanded speeds and reachable band are physically achievable
on our actual hardware. Two consequences worth being upfront about:
- In the noise-free sim, the task could in principle be solved by curve inversion;
  the learning method earns its value when the release is imperfect and unknown -
  our noise studies show 100% (learning) vs 0% (inversion) at 20% velocity slip.
- On the real Gen3 the assignment gets replaced by measurement: we command a speed,
  measure the actual release velocity from the arm's kinematic feedback, and let the
  GP learn reality - plus the gripper's opening delay, which is exactly the paper's
  t_d ~ U(a,b) problem and is what the noise-injection framework was built to absorb.

The open problem I want to take up next
The deeper issue behind the "cosmetic arm" is that generating the ball's velocity from
the arm's actual dynamics is the genuinely hard half of this problem, and no part of
the codebase attempts it. A throw requires the end-effector to reach a specific
position, with a specific velocity, in a specific direction, at a specific instant -
and position control only tracks the first of these (measured here: a 2.0 m/s command
produced actual EE velocities oscillating between 0.1 and 6.6 m/s). Achievable speed
also depends on arm posture through the Jacobian (EE velocity = J(q)*q_dot), so a fast
throw needs a trajectory where all seven joints contribute constructively at release
while each stays within its own velocity limit - a constrained trajectory-optimisation
problem, not a single IK call. Release timing compounds it: at 2 m/s, one 0.02 s
timestep is 4 cm of hand motion. This is also why the original authors sidestepped it
the same way even on real hardware - they command, measure the actual release
velocity, and let the GP learn around the arm's imperfection.

I plan to continue working on exactly this: velocity-space trajectory optimisation and
torque-level control (with gravity compensation) on the Gen3 model, so the release
velocity comes from the arm's tracked motion rather than assignment - and then
measuring the resulting tracking-error distribution and feeding it into the noise
models. That would turn the simulator from "assumes a perfect release" into "predicts
what the real Gen3's release will actually do", which is the bridge we need before
hardware.

Other improvements queued behind it
- Train with injected release noise (slip + timing jitter distributions) on the Gen3
  profile so the policy arrives on hardware already robust to the real gripper.
- Learn launch angle jointly with speed (currently fixed 35 deg) - would widen the
  Gen3's narrow reachable band.
- Add approach angle to the cost: I found that for baskets near the release height
  the ball arrives almost horizontally and would clip a real basket's rim - the
  current cost only knows the landing point, not the entry angle.
- Hardware steps: Kortex driver, ~10 calibration throws to characterise the gripper
  delay, then the MC-PILOT loop on the real arm - the 10-trial data efficiency means
  this is one lab session.

Attached: fig11_hgen_error_vs_height.png, fig12_kinova_convergence.png, and the three
videos above.

Regards,
Rohit
