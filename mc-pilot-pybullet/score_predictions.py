"""
Score a committed `predict_landing.py` prediction file against what was thrown.

    python3 score_predictions.py predictions_20260911_014500.json

Matches each prediction to the session-log record with the same commanded
target, thrown AFTER the predictions were written -- a throw logged before the
prediction file existed cannot score it, and is skipped rather than quietly
counted. Refused measurements are reported, never dropped: a target that could
not be measured is missing evidence, not a pass.

WHAT THE ANSWER MEANS
---------------------
The corrections in `predict_landing.py` were fitted on 13 throws spanning only
1.40-1.54 m/s of commanded release speed. This scores them on new throws over a
much wider band, so read the RESIDUAL VS SPEED table, not just the mean:

  * flat residual, ~2 cm            -> the two-parameter model generalises.
                                       The retrain can be planned on it, and
                                       ~2.6 cm real accuracy is a forecast.
  * residual growing with speed     -> the x1.11 is not a constant gain. It is
                                       a speed-dependent effect wearing a gain
                                       as a disguise, and a re-searched pose
                                       table (which lands at a different speed)
                                       would inherit the wrong value.
  * residual jumps only off-axis    -> an azimuth/base-rotation term, not a
                                       release-speed one.
  * controls miss their old landing -> something drifted between sessions
                                       (extrinsic, mount, ball, tool). Nothing
                                       else in the run is interpretable until
                                       that is explained.
"""

import argparse
import json
import sys

import numpy as np


def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions")
    ap.add_argument("--log", default="hardware_session_log.jsonl")
    ap.add_argument("--tol_m", type=float, default=1e-3,
                    help="how closely a logged target must match a predicted "
                         "one to be considered the same throw")
    args = ap.parse_args()

    with open(args.predictions) as f:
        pred = json.load(f)
    written = pred["written"]
    log = [r for r in load_log(args.log) if r.get("timestamp", "") >= written]
    if not log:
        print(f"no throws logged after {written} -- nothing to score yet.")
        return 1

    print(f"predictions written {written}   "
          f"tool_offset {pred['tool_offset_m']:.2f} m, gain x{pred['release_speed_gain']:.2f}")
    print(f"{len(log)} throws logged since.\n")
    print("  commanded target    cmd v   predicted           measured            error   ")
    print("  " + "-" * 82)

    errs, speeds, lat_errs, rng_errs, missing = [], [], [], [], []
    for pr in pred["predictions"]:
        tx, ty = pr["target"]
        hits = [r for r in log
                if abs(r["target"][0] - tx) < args.tol_m
                and abs(r["target"][1] - ty) < args.tol_m]
        if not hits:
            missing.append((tx, ty, "not thrown"))
            continue
        for r in hits:
            if not r.get("landing_xy"):
                reason = (r.get("refusal_reason") or "refused")[:58]
                print(f"  ({tx:+.3f},{ty:+.3f})       {pr['commanded_speed']:.3f}   "
                      f"({pr['predicted_landing'][0]:+.3f},{pr['predicted_landing'][1]:+.3f})"
                      f"     REFUSED: {reason}")
                missing.append((tx, ty, reason))
                continue
            m = np.array(r["landing_xy"], float)
            q = np.array(pr["predicted_landing"], float)
            rel = np.array(pr["release_pos_corrected"], float)[:2]
            d = q - rel
            u = d / np.linalg.norm(d)
            n = np.array([-u[1], u[0]])
            dm = m - rel
            rng = float(dm @ u - np.linalg.norm(d))
            lat = float(dm @ n)
            e = float(np.linalg.norm(m - q))
            errs.append(e); speeds.append(pr["commanded_speed"])
            rng_errs.append(rng); lat_errs.append(lat)
            print(f"  ({tx:+.3f},{ty:+.3f})       {pr['commanded_speed']:.3f}   "
                  f"({q[0]:+.3f},{q[1]:+.3f})   ({m[0]:+.3f},{m[1]:+.3f})   "
                  f"{e * 100:5.1f} cm   (range {rng * 100:+.1f}, lat {lat * 100:+.1f})")

    if not errs:
        print("\nno prediction was both thrown AND measured -- nothing to conclude.")
        return 1

    errs = np.array(errs); speeds = np.array(speeds)
    rng_errs = np.array(rng_errs); lat_errs = np.array(lat_errs)
    tol = 2 * pred["prediction_sigma_m"]
    print(f"\n  scored {errs.size} throws"
          + (f", {len(missing)} unscored ({sum(1 for m in missing if m[2] != 'not thrown')} refused)"
             if missing else ""))
    print(f"  |error|        mean {errs.mean() * 100:.1f} cm   max {errs.max() * 100:.1f} cm   "
          f"({np.count_nonzero(errs <= tol)}/{errs.size} inside the "
          f"{tol * 100:.1f} cm 2-sigma band)")
    print(f"  range residual mean {rng_errs.mean() * 100:+.1f} cm   sd {rng_errs.std() * 100:.1f} cm")
    print(f"  lat   residual mean {lat_errs.mean() * 100:+.1f} cm   sd {lat_errs.std() * 100:.1f} cm")

    # The whole point of the wide speed spread: is the residual flat in speed?
    if errs.size >= 4 and speeds.ptp() > 0.2:
        A = np.stack([np.ones_like(speeds), speeds], 1)
        c, *_ = np.linalg.lstsq(A, rng_errs, rcond=None)
        r = float(np.corrcoef(speeds, rng_errs)[0, 1])
        print(f"\n  RANGE RESIDUAL VS COMMANDED SPEED over {speeds.min():.2f}-"
              f"{speeds.max():.2f} m/s:")
        print(f"    residual(cm) = {c[0] * 100:+.1f} {c[1] * 100:+.1f} * speed   r = {r:+.2f}")
        span = abs(c[1]) * speeds.ptp() * 100
        if abs(r) < 0.5 or span < 2.0:
            print(f"    -> FLAT ({span:.1f} cm across the whole speed range). The gain is a "
                  f"constant;\n       the model generalises out of sample.")
        else:
            print(f"    -> SLOPED ({span:.1f} cm across the speed range). The x"
                  f"{pred['release_speed_gain']:.2f} is NOT a constant gain --\n"
                  f"       do not carry it into a re-searched pose table, which throws at a "
                  f"different speed.")
    else:
        print("\n  too few throws or too narrow a speed spread to test speed dependence.")

    if missing:
        print("\n  unscored:")
        for tx, ty, why in missing:
            print(f"    ({tx:+.3f},{ty:+.3f})  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
