"""
The parts of the hardware throw session that are decisions or mathematics
rather than plumbing, kept free of Tk, PyBullet and pyrealsense2 so they can be
tested without a window, an arm, or a camera.

See docs/superpowers/specs/2026-08-31-hardware-session-design.md.
"""
from __future__ import annotations

import numpy as np

SCALE_LADDER = (0.15, 0.30, 0.60, 1.00)
TRAINED_BAND = ((0.68, 0.74), (-0.25, 0.25))


def propose_targets(n, band=TRAINED_BAND, seed=0):
    """
    `n` targets spread across the trained band.

    Stratified, not uniform-random: n random draws routinely cluster, and ten
    throws that land in the same place tell the GP almost nothing it does not
    already believe. y gets the stratification because it is the wide axis;
    x is jittered inside its much narrower range.
    """
    (x_lo, x_hi), (y_lo, y_hi) = band
    rng = np.random.default_rng(seed)
    edges = np.linspace(y_lo, y_hi, n + 1)
    ys = edges[:-1] + rng.uniform(0.0, 1.0, n) * np.diff(edges)
    xs = x_lo + rng.uniform(0.0, 1.0, n) * (x_hi - x_lo)
    order = rng.permutation(n)
    return np.stack([xs[order], ys[order]], axis=1)


def next_allowed_scale(logged_scales, ladder=SCALE_LADDER):
    """Highest rung reachable given the clean runs logged so far."""
    best = -1
    for s in logged_scales:
        for i, rung in enumerate(ladder):
            if abs(s - rung) < 1e-9:
                best = max(best, i)
    return ladder[min(best + 1, len(ladder) - 1)]


def scale_allowed(requested, logged_scales, ladder=SCALE_LADDER):
    """
    (ok, reason). The runbook's escalation ladder, enforced rather than advised:
    every rung needs a clean run at the rung below it first. Repeating a rung or
    dropping back down is always fine.
    """
    ceiling = next_allowed_scale(logged_scales, ladder)
    if requested <= ceiling + 1e-9:
        return True, ""
    return False, (f"speed_scale {requested:.2f} needs a clean logged run at "
                   f"{ceiling:.2f} first -- escalate one rung at a time")
