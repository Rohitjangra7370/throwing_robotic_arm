# Hardware Execution Plan — Overhead Throw on Real Gen3 (2026-07-22)

> State at plan time: sim stack DONE and verified (velocity from real torque
> dynamics, whole trajectory within 39/9 Nm + 1.396/1.222 rad/s, 3.0cm mean /
> 100% hit<10cm over 10 trials, 53/53 tests). Everything UNCOMMITTED on branch
> `kinetic-chain-throw-pose`. Honest scope: safe landing 0.60–0.83m from base
> (within 0.86m reach — the safety-vs-range trade-off is the quantified
> finding, do not overclaim "beyond reach").

## Phase 0 — Freeze the code (today, ~10 min)

- [ ] Commit sim work, staged individually (never `git add -A`):
  `find_throw_pose.py`, `simulation_class/model_pybullet.py`,
  `train_mc_pilot_pb_arm.py`, `robot_arm/arm_controller.py`,
  `tests/test_throw_pose_search.py`, `tests/test_follow_through_feasible.py`,
  `throw_pose_table.npy`, `results_kinetic_chain_gen3/`,
  `docs/superpowers/specs/2026-07-22-overhead-throw-design.md`, this plan.
- [ ] Update `paper/results_ledger.md` + `status_update/HANDOFF.md` with this
  session's chain (6 real bugs, final numbers, safety-ceiling finding).
- [ ] Multi-seed the sim result (seeds 2,3 background overnight) — presentation
  needs "3.0±x cm across N seeds", not one seed. `--seed 2/3`, same flags:
  `--opt_pose throw_pose_table.npy --flight_targets --lm 0.60 --lM 0.80
  --uM 1.60 --uMin 1.18 --lengthscale_xy 0.03`.

## Phase 1 — Hardware-integration check, NO ARM MOTION (0.5 day)

- [ ] Read `robot_arm/kinova_hardware.py` + `run_hardware_throw.py`: do they
  consume `plan_throw` coeffs (windup/throw/follow cubics + stagger) or the
  OLD pre-table path? Port to the new path if stale. The interface that must
  survive: per-tick joint setpoints (q, qd) at Kortex rate from
  `arm.get_setpoint(coeffs, t)` — Gen3 low-level joint control is 1kHz vs
  sim's 50Hz, so setpoints must be evaluated at 1ms, NOT replayed at 20ms.
- [ ] Dry-run stage 0 (no power to arm / arm e-stopped): full pipeline prints
  trajectory, timings, limits check passes on the real table entry.
- [ ] Workspace survey at the lab bench: overhead clearance ≥ 1.25m above
  base (release z=1.10 + ball + margin), floor wedge ±33°, 0.5–0.9m from
  base clear of obstacles/people. Mark landing zone.

## Phase 2 — Staged bring-up on the arm, NO BALL (0.5 day)

Safety gates between every stage; e-stop within reach; nobody in the wedge.

- [ ] Stage 1: full trajectory at 0.25× time scale (all durations ×4), no
  ball. Verify joint tracking error < 0.05 rad everywhere; log Kortex torque
  readings, compare against sim predictions (expect < 80% of 39/9 Nm).
- [ ] Stage 2: 0.5×, then 1.0× speed, still no ball. Same gates. If any
  torque reading exceeds 90% of limit at 1.0×: STOP, reduce uM, re-derive
  table ceiling — sim/real dynamics gap goes in the paper either way.
- [ ] Stage 3: with ball, gripper CLOSED throughout (no release). Confirms
  payload compensation on real dynamics.

## Phase 3 — Calibration throws (0.5 day, the paper's ~10 real trials)

- [ ] Gripper-latency estimation first (paper Sec. 5): command release at
  known trajectory time, high-speed phone video (240fps is enough) of actual
  ball separation, 5 throws → mean delay. At 1.5 m/s, 20ms = 3cm shift.
  Compensate by advancing `release_step` by the measured delay.
- [ ] 10 calibration throws at fixed speeds (u = 1.2, 1.35, 1.5 × few each),
  measure landing (tape grid or top-down phone video). This IS the MC-PILOT
  exploration data — feed real (u, landing) pairs into the GP exactly as sim
  trials are fed.
- [ ] Sanity gate: real landing vs sim prediction at same u. If gap > 15cm,
  stop and diagnose (likely latency or real qd/torque saturation) before
  training on it.

## Phase 4 — MC-PILOT loop on hardware (0.5–1 day)

- [ ] Train GP+policy on the real trials (existing `MC_PILCO` loop, Nexp=5
  real exploration + policy updates, ~10 trials total per paper protocol;
  consider Nexp=10, Na=2 = the paper's REAL-system setting).
- [ ] Evaluate: 10–15 held-out targets across the wedge, measure landing
  error + hit rate into a real bin. This is THE headline number.
- [ ] Record everything: video of throws, per-throw (target, u, landing),
  Kortex torque logs — paper figures + sim-vs-real section.

## Phase 5 — Present (1 day)

- [ ] Results package: sim numbers (multi-seed), real numbers, sim-vs-real
  gap, the safety-ceiling finding (0.83m safe vs 0.86m reach — quantified
  range/safety trade-off), the 6-bug validation story as the methods
  narrative (full-trajectory validation incl. follow-through is the novelty
  over "endpoint-checked" throwing).
- [ ] Update `paper/` draft + `results_ledger.md` with final tables.

## Hard rules carried from this session

- Velocity claims: MEASURED at detach only, never commanded values.
- Any new "it works" claim: verify frames/video AND numbers, both.
- Torque/velocity feasibility: whole trajectory, all three phases, never
  endpoint-only.
- Landing distance: `hypot(actual_landing_xy)`, never release-radius + range.
- If a real-arm reading contradicts sim: stop, measure, put the gap in the
  paper — do not tune it away silently.
