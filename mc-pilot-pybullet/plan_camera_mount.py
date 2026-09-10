"""Score candidate D435i mount poses against the landing zone and the hand-eye task.

Where the camera goes decides what the vision half of this experiment can measure, and
the decision is geometric, not a matter of taste. This scores candidates on the four
things that actually gate it.

  1. COVERAGE. The landing zone must be fully in frame with margin. Anything outside is
     unmeasurable, and you find out after throwing into it.

  2. RAY-PLANE ERROR AMPLIFICATION. We do not use depth for position (D435i depth is ~2%
     of range = 2-4 cm at our distances, the same size as the landing error being
     measured). We intersect the pixel ray with the known floor plane. A ray meeting the
     plane at grazing incidence turns a small pixel error into a large ground error:
     amplification = 1 / sin(incidence). At 45 deg that is 1.41x; at 20 deg it is 2.92x,
     which alone would exceed the 1 cm budget. This is the constraint that kills
     otherwise-sensible low side mounts.

  3. RESOLUTION. mm per pixel on the floor, and the ball's apparent diameter. A ball a
     few pixels across cannot be centroided to the accuracy we need.

  4. HAND-EYE FEASIBILITY. Extrinsic calibration needs the gripper marker VISIBLE across
     many arm poses -- not reachable. That distinction matters here because the landing
     zone at ~0.96 m from base is outside the arm's 0.902 m reach, so no procedure
     requiring the EE to touch the landing plane is possible at all.

FRAME. World = floor at z=0, arm base axis at x=y=0, base mounted at z=BASE_HEIGHT.
Camera poses are given in this frame.

    python3 plan_camera_mount.py
    python3 plan_camera_mount.py --pos 0.82 -1.20 1.30
"""
import argparse

import numpy as np

# Measured on the unit, 2026-08-12 (serial 349522070924, FW 5.17.0.10, USB 3.2).
# Colour stream is factory-rectified: all five distortion coefficients read 0.0, so
# ray-plane needs no undistortion and no intrinsic calibration. Only T_B_C is unknown.
FX, FY = 1366.19, 1365.22
PPX, PPY = 981.10, 556.04
W, H = 1920, 1080

BASE_HEIGHT = 0.433        # measured plate
RELEASE_Z = 1.137 + BASE_HEIGHT
R_MIN, R_MAX = 0.70, 0.94  # landing annulus, plate geometry
AZ_MAX_DEG = 33.0
BALL_D = 2 * 0.0327
MARKER_M = 0.080           # printed ArUco edge


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera->world rotation, OpenCV convention (+z forward, +x right, +y down)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(float(f @ up)) > 0.999:          # looking straight down: pick a stable right
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.column_stack([r, d, f])       # columns are camera axes in world


def project(pts, pos, R):
    """World points -> pixels + depth along the optical axis."""
    rel = (np.asarray(pts, float) - np.asarray(pos, float)) @ R   # world -> camera
    z = rel[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * rel[:, 0] / z + PPX
        v = FY * rel[:, 1] / z + PPY
    return u, v, z


def zone_points(n_r=6, n_a=15):
    r = np.linspace(R_MIN, R_MAX, n_r)
    a = np.radians(np.linspace(-AZ_MAX_DEG, AZ_MAX_DEG, n_a))
    rr, aa = np.meshgrid(r, a, indexing="ij")
    return np.stack([rr * np.cos(aa), rr * np.sin(aa), np.zeros_like(rr)], -1).reshape(-1, 3)


def workspace_points(n=400, seed=0):
    """Plausible EE positions for waving a gripper marker at the camera."""
    rng = np.random.default_rng(seed)
    rad = rng.uniform(0.25, 0.75, n)
    az = rng.uniform(-np.pi / 2, np.pi / 2, n)
    z = rng.uniform(BASE_HEIGHT + 0.05, BASE_HEIGHT + 0.75, n)
    return np.stack([rad * np.cos(az), rad * np.sin(az), z], -1)


def score(pos, target=None, label=""):
    pos = np.asarray(pos, float)
    if target is None:
        target = np.array([(R_MIN + R_MAX) / 2, 0.0, 0.0])
    R = look_at(pos, target)

    P = zone_points()
    u, v, z = project(P, pos, R)
    vis = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    rays = P - pos
    dist = np.linalg.norm(rays, axis=1)
    # incidence against the floor (normal = +z): sin(incidence) = |ray_z| / |ray|
    sin_inc = np.abs(rays[:, 2]) / dist
    inc_deg = np.degrees(np.arcsin(sin_inc))
    amp = 1.0 / sin_inc

    mm_px = 1000.0 * dist / FX
    ball_px = FX * BALL_D / dist

    Wp = workspace_points()
    uw, vw, zw = project(Wp, pos, R)
    wvis = (zw > 0) & (uw >= 0) & (uw < W) & (vw >= 0) & (vw < H)
    wdist = np.linalg.norm(Wp - pos, axis=1)[wvis]
    marker_px = FX * MARKER_M / wdist if wdist.size else np.array([0.0])

    print(f"\n=== {label or 'candidate'}  pos={np.round(pos,3)}  "
          f"look={np.round(target,3)}")
    print(f"  landing zone in frame : {100*vis.mean():5.1f} %"
          f"{'   <-- INCOMPLETE' if vis.mean() < 0.999 else ''}")
    if vis.any():
        print(f"  distance to zone      : {dist[vis].min():.2f} - {dist[vis].max():.2f} m")
        print(f"  incidence angle       : {inc_deg[vis].min():4.1f} - {inc_deg[vis].max():4.1f} deg")
        print(f"  ray-plane amplification: {amp[vis].min():.2f}x - {amp[vis].max():.2f}x"
              f"{'   <-- GRAZING' if amp[vis].max() > 2.0 else ''}")
        print(f"  ground resolution     : {mm_px[vis].min():.2f} - {mm_px[vis].max():.2f} mm/px")
        print(f"  ball diameter         : {ball_px[vis].min():.0f} - {ball_px[vis].max():.0f} px")
    print(f"  EE poses visible      : {100*wvis.mean():5.1f} %  "
          f"(hand-eye needs a good spread here)")
    if wvis.any():
        print(f"  80 mm marker          : {marker_px.min():.0f} - {marker_px.max():.0f} px"
              f"   ({'OK' if marker_px.min() > 60 else 'MARGINAL <60 px'})")
    # Is the mount inside the throw corridor? The ball leaves at RELEASE_Z heading out
    # along +x within the azimuth wedge and descends to the floor across the zone.
    horiz = float(np.hypot(pos[0], pos[1]))
    if horiz < R_MAX + 0.15 and 0.0 < pos[2] < RELEASE_Z + 0.2 \
       and abs(np.degrees(np.arctan2(pos[1], pos[0]))) < AZ_MAX_DEG + 10:
        print("  !! mount sits inside the throw corridor -- the ball can hit it")
    return vis.mean(), (amp[vis].max() if vis.any() else np.inf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=float, nargs=3, default=None,
                    help="score one custom mount position (world frame, floor at z=0)")
    ap.add_argument("--look", type=float, nargs=3, default=None)
    args = ap.parse_args()

    print(f"intrinsics: fx={FX} fy={FY} {W}x{H}  "
          f"FOV {2*np.degrees(np.arctan(W/(2*FX))):.1f} x "
          f"{2*np.degrees(np.arctan(H/(2*FY))):.1f} deg")
    print(f"landing zone: r={R_MIN}-{R_MAX} m, +-{AZ_MAX_DEG} deg, floor z=0")
    print(f"arm base at z={BASE_HEIGHT} m, release at z={RELEASE_Z:.3f} m")

    if args.pos is not None:
        score(args.pos, args.look, "custom")
        return

    for label, pos in [
        ("A  overhead, 2.2 m",        (0.82, 0.00, 2.20)),
        ("B  overhead, 1.6 m",        (0.82, 0.00, 1.60)),
        ("C  side-oblique, 1.3 m",    (0.82, -1.20, 1.30)),
        ("D  side-oblique, 1.8 m",    (0.82, -1.20, 1.80)),
        ("E  side, low 0.9 m",        (0.82, -1.40, 0.90)),
        ("F  behind arm, 1.8 m",      (-0.60, 0.00, 1.80)),
    ]:
        score(pos, None, label)


if __name__ == "__main__":
    main()
