"""
Settle the two questions spec section 9 deliberately left open: emitter on or
off, and what exposure.

Neither is answerable from a datasheet. The IR projector's dot pattern is static
on the floor, so background subtraction removes it -- and on the BALL it adds
texture, which helps. But it can also saturate. Short exposure cuts motion blur
(2 ms is 2.2 px at the 5.8 m/s impact speed) but may leave the scene too dark
indoors with the emitter off. Ten minutes of A/B beats any amount of reasoning.

Wave the ball through the frame during each capture.

    python3 tune_ir_exposure.py
    python3 tune_ir_exposure.py --exposures 1000 2000 4000 8000
"""
import argparse

import numpy as np

from perception.ball_track import detect_candidates, median_background
from perception.ir_capture import IRRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposures", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()

    print(f"{'emitter':>8} {'exp_us':>7} {'fps':>6} {'mean_lvl':>9} "
          f"{'sat%':>6} {'frames_with_ball':>17} {'med_area':>9} {'med_circ':>9}")
    for emitter in (True, False):
        for exp in args.exposures:
            with IRRecorder(exposure_us=exp, emitter=emitter) as r:
                rec = r.record(args.seconds)
            ir1 = rec["ir1"]
            bg = median_background(ir1)
            areas, circs, hits = [], [], 0
            for f in ir1:
                cands = detect_candidates(f, bg)
                if cands:
                    hits += 1
                    best = max(cands, key=lambda c: c.circularity)
                    areas.append(best.area_px); circs.append(best.circularity)
            sat = 100.0 * float(np.mean(ir1 >= 250))
            print(f"{str(emitter):>8} {exp:>7} {rec['meta']['achieved_fps']:>6.1f} "
                  f"{ir1.mean():>9.1f} {sat:>6.2f} {hits:>10d}/{len(ir1):<6d} "
                  f"{np.median(areas) if areas else float('nan'):>9.1f} "
                  f"{np.median(circs) if circs else float('nan'):>9.2f}")

    print("\nPick the row with the most frames_with_ball at the SHORTEST exposure,")
    print("with sat% near zero. Record the choice in HARDWARE_RUNBOOK.md.")


if __name__ == "__main__":
    main()
