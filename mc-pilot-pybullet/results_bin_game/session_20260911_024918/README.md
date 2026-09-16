# Closed-loop bin-aiming session — 2026-09-11, 02:57–03:23

Kinova Gen3 7-DOF throwing a tennis ball at a movable target marked with a single
ArUco tag. The operator moved the bin, clicked **Aim at bin**, and threw. Repeated
22 times.

**This is the first closed-loop hardware accuracy result for this rig.**

---

## Headline

**22 throws attempted. 14 measured. Mean error 1.9 cm from the aim point.**

Never quote the 14 without the 22. The 8 excluded throws are excluded by the
*measurement system*, not because the throw failed — see [Exclusions](#exclusions),
which is the part of this document that matters most.

| | |
|---|---|
| throws attempted | 22 |
| throws measured | 14 |
| error vs aim point | **mean 1.9 cm · median 1.8 cm · max 3.7 cm · sd 1.0 cm** |
| within 3 cm / 5 cm | 12/14 · 14/14 |
| per-landing σ (propagated fit covariance) | 2.5 – 6.7 mm |
| commanded release speed | 1.42 – 1.66 m/s |
| aim points spanned | x 1.11–1.28 m, y −0.50 to +0.36 m |

Two by-products of the session, both unplanned:

- **End-to-end system repeatability 7.6 mm.** Throws 16 and 17 are the same
  unmoved marker, read twice and thrown twice. Aim + throw + measurement compounded.
- **Marker re-read repeatability 1.4 mm and 1.2 mm** (two pairs, bin unmoved),
  against 1.6–1.9 mm predicted from a synthetic sweep *before* the session ran.

---

## What was actually measured

`error` is the distance between **where the marker was seen** and **where the ball
was seen to land** — both in the base frame, both through the same camera and the
same extrinsic. The extrinsic's own error therefore cancels out of this number; it
is a genuine miss distance, not a comparison against a model's own prediction.

It is *not* a measure of MC-PILOT's accuracy. The trained policy still plans as
though it throws 45 cm shorter than it does; the aim comes from inverting a fitted
system-identification model on top of it (tool offset 0.22 m, release-speed excess
+0.335 m/s). This is open-loop pre-compensation, and the distinction must survive
into any write-up.

---

## Exclusions

8 of 22 produced no landing. **None of them was a failed throw** — the operator
observed every throw land on the marker. All 8 were refused by `measure_landing`:

| reason | n |
|---|---|
| ball still rising in its last detected frame (bounce arc, flight not recoverable) | 5 |
| no ballistic arc found | 2 |
| no left/right candidates paired at all (ball never entered view) | 1 |

**Known bias in the survivors.** The refused throws skew to negative y:
aim y **−0.23 ± 0.22** for refused against **+0.05 ± 0.29** for measured. That side
of the workspace is under-sampled. It is still represented among the measured
throws down to y = −0.50.

**No sign of accuracy-correlated selection.** Four throws were measured under the
original analysis; the other ten were recovered by the pipeline fixes below. The
recovered set averages **2.1 cm** against the original set's **1.7 cm** — slightly
worse, no new outliers, max unchanged. Had the fixes been manufacturing agreement,
the recovered throws would cluster more tightly on the marker, not less.

---

## Analysis-pipeline caveat — read before publishing

The measurement pipeline was **revised after these recordings were inspected**, and
after the operator reported that all throws had landed on the marker. Measured
throws went 4 → 6 → 14 across three fixes. That ordering is a real risk of
confirmation bias and is disclosed here rather than buried.

What limits the risk:

- Fix 1 (**flight arc, not largest arc**) was made *before* the operator's report,
  and is justified a priori: the ball is in flight before it bounces.
- Fix 2 (**area sparing**) rests on a measurement taken from the images before any
  landing was computed — the static source measures 24–54 px of area, the ball
  crossing it 104–168 px — and can only ever *spare* a candidate, never reject more.
- Fix 3 (**drop cap → σ gate**) was preceded by an explicit check that the cap was
  not what rejected the known-bad recordings, at 0.30, 0.60 and 2.00 m.
- Every change is pinned by regression tests against three real recordings known
  *not* to be throws (hand-carried ball, stationary ball); all remain refused.
- A held-out consistency check: throw 10, accepted by the strict path, is
  reproduced by the loosened path to within **2 mm**.

**What would remove the concern entirely:** one fresh session analysed with the
pipeline frozen as it now stands. That is the standard remedy and it costs one
run day. Until then, these 14 are a strong result obtained with a pipeline tuned
on the same data.

---

## Provenance

| | |
|---|---|
| arm | Kinova Gen3 7-DOF, torque mode (`kinova_gen3_dyn`), base plate 0.433 m |
| ball | tennis, 57.7 g, 65.4 mm |
| checkpoint | `results_kinetic_chain_gen3_tcp/1` |
| pose table | `throw_pose_table_tcp.npy` (`tool_offset` 0.12, `floor_z` −0.433) |
| aim model | additive: tool offset 0.22 m, release speed +0.335 m/s |
| wrist roll offset | 90° |
| speed scale | 1.00 throughout |
| camera | overhead Intel RealSense D435i, dual IR 848×480 @ 90 fps, exposure 4000 µs, emitter on |
| extrinsic | `calib/T_B_C_20260911_025300.npz`, written 02:53 — REPEAT gate 2.3 cm / 0.71°, FLOOR −1.6 cm, PnP-vs-depth +0.7 cm |
| git base | `0f11e2d`, plus `analysis_pipeline.patch` in this directory |
| test suite | 368 passing at time of writing |

The camera mount moved 19.4 cm between the 00:03 and 02:53 calibrations. That is
**before** this session, which used the 02:53 extrinsic throughout.

## Files

| file | what |
|---|---|
| `results.json` | per-throw records + summary block |
| `results.csv` | the same, flat, for plotting |
| `analysis_pipeline.patch` | exact diff of `measure_landing.py`, `perception/ball_track.py`, `perception/trajectory.py` against `0f11e2d` — the pipeline that produced these numbers |
| `recording_sha256_prefix.json` | SHA-256 prefix of all 22 raw dual-IR recordings, so the record is verifiable against the data |

Raw recordings are `throws/throw_20260911_024918_*.npz` (~106 MB each, untracked).
Regenerate this result with:

```bash
python3 compile_bin_game.py --since 2026-09-11T02:50 \
    --out results_bin_game/session_20260911_024918/results.json
```
