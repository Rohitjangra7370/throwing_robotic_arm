"""
Render the feasibility-ablation JSONs as the paper's Table I, one column per
run, plus the raw-vs-repaired delta.

`repair_inertials` is the condition under test: with it False (the historical
default) PyBullet gives every bodyless URDF link 1 kg, which on the Gen3 is
+3.00 kg / +46% hung off the wrist and on the UR7e +5.00 kg / +23%, while the
KUKA and Panda have no bodyless links at all and are unaffected either way.
Every stage of this cascade is a torque gate, so the condition moves all of it.

    usage:  python3 summarize.py [dir]
"""
import glob
import json
import os
import sys


ROWS = [
    ("Release-instant-feasible", lambda s: s["n_release_instant_ok"], None),
    ("Also full-traj.-feasible", lambda s: s["n_full_ok"], "n_release_instant_ok"),
    ("Rej. windup-path", lambda s: s["n_windup_path_fail"], "n_release_instant_ok"),
    ("Rej. throw-ramp", lambda s: s["n_ramp_fail"], "n_release_instant_ok"),
    ("Rej. follow-through", lambda s: s["n_follow_fail"], "n_release_instant_ok"),
]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        out[os.path.basename(f)[:-5]] = r
    return out


def main(d="."):
    runs = load(d)
    if not runs:
        print("no result JSONs yet"); return
    names = list(runs)
    w = max(26, max(len(n) for n in names) + 2)

    print(f"{'':28}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (28 + w * len(names)))
    for label, fn, denom_key in ROWS:
        cells = []
        for n in names:
            s = runs[n]["stats"]
            v = fn(s)
            if denom_key:
                den = s[denom_key]
                cells.append(f"{v:,} ({100*v/den:.1f}%)" if den else f"{v:,} (n/a)")
            else:
                cells.append(f"{v:,}")
        print(f"{label:28}" + "".join(f"{c:>{w}}" for c in cells))

    for label, key in (("Best range, instant-only", "best_release_instant_only"),
                       ("Best range, full check", "best_full_trajectory")):
        cells = []
        for n in names:
            b = runs[n].get(key)
            cells.append("none" if not b
                         else f"{b['land']:.3f} m @ {b['speed']:.2f} m/s, {b['elev_deg']}deg")
        print(f"{label:28}" + "".join(f"{c:>{w}}" for c in cells))

    print(f"\n{'repair_inertials':28}" +
          "".join(f"{str(runs[n]['repair_inertials']):>{w}}" for n in names))
    print(f"{'pitch joints':28}" +
          "".join(f"{str(runs[n].get('pitch_joints','-')):>{w}}" for n in names))
    print(f"{'runtime (min)':28}" +
          "".join(f"{runs[n]['elapsed_s'] / 60:>{w}.1f}" for n in names))

    # raw -> repaired deltas, per arm
    for arm in sorted({runs[n]["robot"] for n in names}):
        raw = next((runs[n] for n in names
                    if runs[n]["robot"] == arm and not runs[n]["repair_inertials"]), None)
        rep = next((runs[n] for n in names
                    if runs[n]["robot"] == arm and runs[n]["repair_inertials"]), None)
        if not (raw and rep):
            continue
        a, b = raw["stats"], rep["stats"]
        sa = 100 * a["n_full_ok"] / max(a["n_release_instant_ok"], 1)
        sb = 100 * b["n_full_ok"] / max(b["n_release_instant_ok"], 1)
        print(f"\n{arm}: phantom mass removed ->")
        print(f"    static-feasible postures {a['n_static_feasible']:,} -> {b['n_static_feasible']:,}")
        print(f"    release-instant pool     {a['n_release_instant_ok']:,} -> {b['n_release_instant_ok']:,}")
        print(f"    survival after cascade   {sa:.1f}% -> {sb:.1f}%  ({sb-sa:+.1f} pts)")
        ra = raw["best_full_trajectory"]["land"] if raw["best_full_trajectory"] else 0.0
        rb = rep["best_full_trajectory"]["land"] if rep["best_full_trajectory"] else 0.0
        print(f"    best safe-throw range    {ra:.3f} m -> {rb:.3f} m  ({rb-ra:+.3f} m)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
