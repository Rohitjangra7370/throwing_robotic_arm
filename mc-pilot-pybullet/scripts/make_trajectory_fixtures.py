"""
Regenerate tests/fixtures/obs_real_*.npz from the real dual-IR recordings.

The recordings in `throws/` are ~106 MB each and untracked; the observation
rows `build_observations` extracts from them are a couple of KB. Committing
the rows keeps the regression permanent without committing the video -- the
same reasoning as "every recording is a permanent regression fixture", applied
to the part that fits in git.

Run this only when the DETECTOR or the PAIRING changes (`ball_track.py`,
`stereo.pair_candidates`) -- the fixtures are their output. A change to
`trajectory.py` or `measure_landing.py` must be tested against the fixtures as
they are; regenerating them for a fitter change would erase the evidence.

THE SOURCE RECORDINGS FOR ALL THREE ARE GONE (2026-09-11). Until that day the
capture filename was `throw_<index>.npz` and the index restarted at 0 every
session, so later sessions overwrote earlier ones -- `throws/throw_001.npz`,
`002` and `006` now hold completely different throws. That is exactly why
extracting the rows into a committed fixture was worth doing, and it is why
`--check` distinguishes SOURCE REPLACED (the file on disk is a different
recording; the fixture is still the right data and must NOT be regenerated)
from DRIFT (same recording, different detector output -- a real change worth
looking at). The naming bug itself is fixed in
`hardware_session.capture_filename`.

    python3 scripts/make_trajectory_fixtures.py            # needs throws/ present
    python3 scripts/make_trajectory_fixtures.py --check    # report drift, write nothing
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_landing import build_observations, load_extrinsic  # noqa: E402
from perception.ir_capture import load_recording                # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures")

# `lost` marks a source whose recording no longer exists on disk -- overwritten
# by the pre-2026-09-11 filename collision. It is asserted from direct evidence
# (every file in throws/ now has a 2026-09-11 mtime, and the session log shows
# throws/throw_000.npz alone written ten times across five dates), not inferred
# from a mismatch, because one of the three carries no metadata to mismatch on.
SOURCES = [
    ("throws/throw_001.npz", "obs_real_throw.npz", True,
     "2026-09-10 run-day throw_001, real Gen3 throw. The ball enters the "
     "overhead FOV at t = 1.013 s, is tracked to first floor contact at "
     "~1.24 s, then bounces and stays in view to the end of the 1.447 s "
     "window. 16 of 37 rows fit the flight arc at 0.81 px -- the recording "
     "that showed ransac_track's global inlier fraction was unreachable."),
    ("throws/throw_002.npz", "obs_real_not_a_throw.npz", True,
     "2026-08-26 throw_002. NOT a throw: a hand-carried or rolling ball (see "
     "CLAUDE.md, 2026-08-31 -- fitting acceleration freely gives 0.16-2.54 "
     "m/s^2 against 9.81). Detection and triangulation are excellent and it "
     "fits a constrained g = 9.81 parabola to 0.95 px with a span-local "
     "inlier fraction of 1.00, which is why the unobserved-drop gate has to "
     "exist."),
    ("throws/throw_006.npz", "obs_real_stationary_ball.npz", True,
     "2026-08-26 throw_006. NOT a throw: a ball sitting nearly still on the "
     "floor (0.807 -> 0.837 m in x over 0.156 s, z jittering +-3 cm on "
     "triangulation noise). Fitting acceleration freely gives |a| = 1.58 "
     "m/s^2 against 9.81, yet a CONSTRAINED g = 9.81 parabola fits it to "
     "0.55 px with a span-local inlier fraction of 0.83, DESCENDING, only "
     "0.22 m above the impact plane -- it passes the unobserved-drop cap by "
     "putting its apex inside the observed span. The recording that showed "
     "that cap needed the observed-drop floor alongside it."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extrinsic", default="calib/T_B_C.npz")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed fixtures and report, "
                         "writing nothing")
    args = ap.parse_args()

    R, t = load_extrinsic(args.extrinsic)
    rc = 0
    for src, name, lost, note in SOURCES:
        dst = os.path.join(FIXTURES, name)
        if lost:
            print(f"SOURCE LOST  {name}: {src} was overwritten before 2026-09-11 "
                  f"and the path now holds a different throw. This fixture is the "
                  f"surviving copy -- it cannot be regenerated and must not be.")
            rc = max(rc, 1)
            continue
        if not os.path.exists(src):
            print(f"SKIP {name}: {src} is not on this machine")
            rc = max(rc, 1)
            continue
        rec = load_recording(src)
        if args.check:
            z = np.load(dst)
            stored_meta = json.loads(str(z["meta"]))
            live_meta = rec.get("meta", {})
            if isinstance(live_meta, np.ndarray):
                live_meta = json.loads(str(live_meta))
            if stored_meta and live_meta and stored_meta != live_meta:
                print(f"SOURCE REPLACED {name}: {src} is no longer the recording "
                      f"this fixture came from. Do NOT regenerate -- the fixture "
                      f"is the surviving copy of that throw.")
                rc = max(rc, 1)
                continue
            obs, _ = build_observations(rec)
            old = np.asarray(z["obs"], float)
            same = old.shape == obs.shape and np.allclose(old, obs, atol=1e-9)
            print(f"{'OK  ' if same else 'DRIFT'} {name}: committed {old.shape}, "
                  f"regenerated {obs.shape}")
            rc = max(rc, 0 if same else 2)
            continue
        obs, _ = build_observations(rec)
        os.makedirs(FIXTURES, exist_ok=True)
        np.savez_compressed(dst, obs=obs, R=R, t=t, source=src, note=note,
                            meta=json.dumps(rec.get("meta", {}), default=str))
        print(f"wrote {dst}  {obs.shape}  "
              f"{os.path.getsize(dst) / 1024:.1f} KB")
    return rc


if __name__ == "__main__":
    sys.exit(main())
