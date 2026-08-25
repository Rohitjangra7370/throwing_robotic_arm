"""
Ballistic trajectory fitting and the floor-impact solve.

WHAT THIS MEASURES, AND WHY IT IS NOT THE RESTING POSITION
-----------------------------------------------------------
`ball_detector.py` measures where the ball SETTLES. On a tile floor a ball
thrown at ~1.6 m/s bounces and rolls, so the settled position is not the
first-contact point -- and it is first contact that the policy's landing error
is defined against. The gap is a systematic, and nothing in the static pipeline
bounds it. Fitting the flight and solving for the floor crossing measures first
contact directly.

WHY NO DRAG TERM
----------------
Tennis ball, 57.7 g, 65.4 mm diameter (fixed by the trained checkpoints and
HARDWARE_RUNBOOK.md). Drag at 2 m/s is 0.5*1.2*0.5*3.36e-3*v^2 = 4.0e-3 N
against 0.566 N of weight -- 0.7%. A drag-free parabola is valid HERE.

It is NOT valid for the whiffle ball (0.004 kg / 0.06 m) of the drag-crossover
study, where drag is the entire point. If this is ever pointed at that ball,
`g` alone will not describe the motion and a drag term has to be added to
`ballistic_position` and to the fit residual together.

FRAME. Base frame throughout: base at z = 0, floor at z = -base_height =
-0.433, g = (0, 0, -9.81).
"""

from __future__ import annotations

import numpy as np

__all__ = ["G_BASE", "Z_FLOOR_BASE", "BALL_RADIUS", "ballistic_position",
           "solve_impact_time", "solve_impact", "fit_two_points"]

G_BASE = np.array([0.0, 0.0, -9.81])
Z_FLOOR_BASE = -0.433      # measured base plate; see CLAUDE.md's frame note
BALL_RADIUS = 0.0327       # tennis ball, matches the trained checkpoints


def ballistic_position(p0, v0, t, g=G_BASE):
    """p(t) = p0 + v0*t + 0.5*g*t^2. Scalar t -> (3,); array t -> (N, 3)."""
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    t = np.asarray(t, float)
    return p0 + v0 * t[..., None] + 0.5 * g * (t[..., None] ** 2)


def solve_impact_time(p0, v0, z_target, g=G_BASE):
    """
    Time of the DESCENDING crossing of the horizontal plane z = z_target.

    A trajectory that rises through the plane and falls back has TWO positive
    roots. The smaller one is the ball on its way up; reporting it would give a
    confident landing point for a ball that has not landed. Always the larger.

    Raises RuntimeError if the plane is never reached.
    """
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    a = 0.5 * float(g[2])
    b = float(v0[2])
    c = float(p0[2]) - float(z_target)
    if a == 0.0:
        raise RuntimeError("zero vertical acceleration -- not a ballistic arc")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise RuntimeError(
            f"trajectory never reaches z = {z_target:.4f} (discriminant "
            f"{disc:.4e}); apex is z = {float(p0[2]) - b * b / (4.0 * a):.4f}")
    root = np.sqrt(disc)
    # a < 0, so the LARGER time is (-b - root)/(2a). Both roots computed so the
    # failure message can say which ones existed.
    t_desc = (-b - root) / (2.0 * a)
    t_asc = (-b + root) / (2.0 * a)
    if t_desc <= 0.0:
        raise RuntimeError(
            f"no descending crossing at positive time (roots {t_asc:.4f}, "
            f"{t_desc:.4f} s) -- the ball is already past this plane")
    return float(t_desc)


def solve_impact(p0, v0, z_floor=Z_FLOOR_BASE, ball_radius=BALL_RADIUS, g=G_BASE):
    """
    First-contact point on the floor: returns (x, y, t).

    Solves for the ball's CENTRE reaching z = z_floor + ball_radius, because
    that is where the centre is at the instant the sphere touches down. For a
    sphere the centre's (x, y) is the contact (x, y). Identical convention to
    `ray_plane.ball_center_on_plane` -- the two must agree or comparing them
    means nothing.
    """
    t = solve_impact_time(p0, v0, float(z_floor) + float(ball_radius), g=g)
    p = ballistic_position(p0, v0, t, g=g)
    return float(p[0]), float(p[1]), float(t)


def fit_two_points(t_a, p_a, t_b, p_b, g=G_BASE, min_dt=0.02):
    """
    Exact (p0, v0) through two timed 3-D points. Six equations, six unknowns.

    This is RANSAC's minimal sample. `min_dt` rejects a degenerate pair: two
    nearly simultaneous points leave v0 = dp/dt numerically meaningless, and the
    resulting wild trajectory would be scored against the data as if it were a
    real hypothesis.
    """
    g = np.asarray(g, float)
    t_a, t_b = float(t_a), float(t_b)
    if abs(t_b - t_a) < min_dt:
        raise RuntimeError(f"sample points are separated by only "
                           f"{abs(t_b - t_a) * 1e3:.1f} ms (need >= "
                           f"{min_dt * 1e3:.0f} ms) -- degenerate")
    q_a = np.asarray(p_a, float) - 0.5 * g * t_a ** 2
    q_b = np.asarray(p_b, float) - 0.5 * g * t_b ** 2
    v0 = (q_b - q_a) / (t_b - t_a)
    p0 = q_a - v0 * t_a
    return p0, v0
