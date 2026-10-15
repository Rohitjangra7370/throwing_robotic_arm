#!/usr/bin/env python3
"""Recompute the headline statistics of a bin-aiming session from its results.csv.

Standard library only, so it runs without the simulation or perception stack.

    python3 scripts/summarize_bin_game.py \
        results_bin_game/session_20260911_024918/results.csv

`error_m` in the CSV is the distance between where the floor marker was seen and
where the ball was seen to land, both in the base frame through the same camera
and extrinsic. Rows with measured == 0 were refused by the landing-measurement
pipeline (not necessarily failed throws) and carry no error; they are counted but
never averaged. Always quote the measured count together with the attempted count.
"""
import argparse
import csv
import re
import statistics


def summarize(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    measured = [r for r in rows if r["measured"] == "1"]
    errs_cm = [float(r["error_m"]) * 100.0 for r in measured]
    if not errs_cm:
        raise SystemExit(f"{path}: no measured throws")
    refusals = {}
    for r in rows:
        if r["measured"] != "1":
            # numbers differ per throw ("+2.19 m/s", "t = 1.424 s"); mask them so
            # throws refused for the same reason group together
            key = re.sub(r"[+-]?\d+(\.\d+)?", "#", r.get("refusal") or "(no reason recorded)")[:90]
            refusals[key] = refusals.get(key, 0) + 1
    return {
        "attempted": len(rows),
        "measured": len(measured),
        "mean_cm": statistics.mean(errs_cm),
        "median_cm": statistics.median(errs_cm),
        "max_cm": max(errs_cm),
        "sd_cm": statistics.stdev(errs_cm) if len(errs_cm) > 1 else 0.0,
        "within_3cm": sum(e <= 3.0 for e in errs_cm),
        "within_5cm": sum(e <= 5.0 for e in errs_cm),
        "refusals": refusals,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv_path")
    s = summarize(ap.parse_args().csv_path)
    n = s["measured"]
    print(f"throws attempted : {s['attempted']}")
    print(f"throws measured  : {n}")
    print(f"error vs aim     : mean {s['mean_cm']:.2f} cm | median {s['median_cm']:.2f} cm | "
          f"max {s['max_cm']:.2f} cm | sd {s['sd_cm']:.2f} cm")
    print(f"within 3 / 5 cm  : {s['within_3cm']}/{n} | {s['within_5cm']}/{n}")
    for reason, count in sorted(s["refusals"].items(), key=lambda kv: -kv[1]):
        print(f"refused ({count}): {reason}")


if __name__ == "__main__":
    main()
