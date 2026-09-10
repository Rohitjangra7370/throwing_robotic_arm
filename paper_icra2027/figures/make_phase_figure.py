#!/usr/bin/env python3
"""Render the throw-phase figure ("what the motion actually looks like").

Replaces the old ``throw_motion_ghost.png``, whose alpha-blended PyBullet
ghosts collapsed into an unreadable tangle over a checkerboard floor, with a
2-D projection into the swing plane: link polylines at the four planned phases
plus the ball's flight arc.  Everything drawn here is read back from the real
planner -- ``ArmController.plan_throw`` setpoints evaluated through PyBullet
forward kinematics -- so the figure cannot drift from the trajectory the
simulator and the hardware executor actually run.

Must be able to import the ``mc-pilot-pybullet`` package tree, which resolves
its own imports relative to that directory; this script therefore chdir's
there before importing, exactly as the study scripts do.

Usage:  python3 make_phase_figure.py [--outdir ../overleaf/figs]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
VARIANT = REPO / "mc-pilot-pybullet"

# --- print geometry / palette: kept identical to make_figures.py -----------
COL = 3.11
DPI = 400
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#595959"
INK = "#1a1a1a"

TABLE = "throw_pose_table_tcp.npy"
ROBOT = "kinova_gen3_dyn"
BASE_HEIGHT = 0.433
TOOL_OFFSET = (0.0, 0.0, 0.12)
BALL_MASS = 0.0577
BALL_RADIUS = 0.0327
TARGET_XY = (0.75, 0.0)


def _fk_points(p, arm_id, cid, q, joint_ids, ee_link):
    """World-frame positions of the base and every actuated joint, plus the
    end-effector -- i.e. the polyline a reader reads as "the arm"."""
    for local_i, jid in enumerate(joint_ids):
        p.resetJointState(arm_id, jid, targetValue=float(q[local_i]),
                          targetVelocity=0.0, physicsClientId=cid)
    pts = [np.array(p.getBasePositionAndOrientation(arm_id, physicsClientId=cid)[0])]
    for jid in list(joint_ids) + [ee_link]:
        ls = p.getLinkState(arm_id, jid, computeForwardKinematics=True,
                            physicsClientId=cid)
        pts.append(np.array(ls[4]))          # world link-frame origin
    return np.asarray(pts)


def _tcp(p, arm_id, cid, ee_link):
    ls = p.getLinkState(arm_id, ee_link, computeForwardKinematics=True,
                        physicsClientId=cid)
    pos, orn = np.array(ls[4]), ls[5]
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return pos + rot @ np.asarray(TOOL_OFFSET)


def collect():
    """Run the real planner once and return the phase polylines + ball arc."""
    os.chdir(VARIANT)
    sys.path.insert(0, str(VARIANT))
    import pybullet as p
    import pybullet_data
    from robot_arm.arm_controller import ArmController
    from robot_arm.robot_profiles import get_robot_profile
    from simulation_class.release_solver import OptimizedReleaseSolver

    profile = get_robot_profile(ROBOT)
    table = np.load(VARIANT / TABLE, allow_pickle=True)
    solver = OptimizedReleaseSolver(opt_posture_table=table,
                                    tool_offset=np.asarray(TOOL_OFFSET))

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    urdf = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
    arm = ArmController(cid, urdf, base_position=(0, 0, BASE_HEIGHT),
                        robot_name=ROBOT)

    # Release state for a mid-range target, straight ahead: the throw the
    # policy actually commands, not a hand-picked pose.
    speed = 1.45
    target_xy = np.array(TARGET_XY, dtype=float)
    v_hint = np.array([speed, 0.0, 0.0])
    release_pos, q_rel, qd_rel, v_cmd = solver.solve(arm, v_hint, target_xy)

    t_w, t_r, T = profile.timing
    coeffs, q_release, qd_release, v_planned = arm.plan_throw(
        v_cmd, release_pos, t_w, t_r, T,
        q_release_override=q_rel, qd_release_override=qd_rel,
        monotonic_windup=True,
    )
    t_r_actual = coeffs["t_r"]
    t_w_actual = coeffs["t_w"]

    jid, ee = profile.joint_ids, profile.ee_link
    phases = {}
    for name, t in (("neutral", 0.0),
                    ("windup", t_w_actual),
                    ("release", t_r_actual),
                    ("follow", min(coeffs["T"], t_r_actual + 0.45))):
        q_t, _, _ = arm.get_setpoint(coeffs, t, with_accel=True)[:3]
        phases[name] = _fk_points(p, arm._arm_id, cid, q_t, jid, ee)
        if name == "release":
            phases["release_tcp"] = _tcp(p, arm._arm_id, cid, ee)

    # Swing-plane sweep of the throwing hand: the continuous path between the
    # discrete phase poses, so the reader sees direction of travel.
    sweep = []
    for t in np.linspace(0.0, t_r_actual, 60):
        q_t, _, _ = arm.get_setpoint(coeffs, t, with_accel=True)[:3]
        _fk_points(p, arm._arm_id, cid, q_t, jid, ee)
        sweep.append(_tcp(p, arm._arm_id, cid, ee))
    sweep = np.asarray(sweep)

    # Ball flight: drag-free integration from the planned release state, cut at
    # the descending floor crossing (floor = world z = 0).
    v_rel = np.asarray(v_planned, dtype=float)
    x0 = np.asarray(phases["release_tcp"], dtype=float)
    ts = np.linspace(0.0, 2.0, 400)
    arc = x0[None, :] + v_rel[None, :] * ts[:, None]
    arc[:, 2] -= 0.5 * 9.81 * ts ** 2
    below = np.where(arc[:, 2] <= BALL_RADIUS)[0]
    if len(below):
        arc = arc[: below[0] + 1]

    p.disconnect(physicsClientId=cid)
    return phases, sweep, arc, float(np.linalg.norm(v_rel))


def plot(phases, sweep, arc, v_rel, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
    "axes.labelweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.5,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    # Project into the swing plane. The base joint is frozen at the target
    # azimuth throughout the throw, so the motion is planar by construction and
    # the radial coordinate loses nothing.
    def rz(pts):
        pts = np.atleast_2d(pts)
        sgn = np.sign(np.where(np.abs(pts[:, 0]) > 1e-9, pts[:, 0], 1.0))
        return np.hypot(pts[:, 0], pts[:, 1]) * sgn, pts[:, 2]

    ar, az = rz(arc)
    tr, tz = rz(phases["release_tcp"])
    base_z = float(phases["neutral"][0, 2])

    # Small multiples, one phase per panel. A single overlaid panel was tried
    # first and is what the figure this replaces already did wrong: four poses
    # of the same arm share a base and cross each other, so the reader cannot
    # tell which segment belongs to which phase no matter how they are coloured.
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.62), sharey=True)
    # Short titles by necessity: at column width the spelled-out
    # "(c) follow-through" ran into panel (b)'s title. The full phase names are
    # given in the caption instead.
    panels = [("windup", "(a) windup"),
              ("release", "(b) release"),
              ("follow", "(c) recovery")]

    for ax, (name, title) in zip(axes, panels):
        ax.axhline(0.0, color=GREY, linewidth=0.9, zorder=1)
        ax.add_patch(plt.Rectangle((-0.075, 0.0), 0.15, base_z,
                                   facecolor="#ececec", edgecolor=GREY,
                                   linewidth=0.5, zorder=1))
        # neutral pose behind every panel: the fixed reference the reader
        # measures each phase against
        rn, zn = rz(phases["neutral"])
        ax.plot(rn, zn, color="#cfcfcf", linewidth=1.1, zorder=2,
                solid_capstyle="round", solid_joinstyle="round")
        # hand path up to release, in every panel, so the swing direction is
        # legible from any single panel
        sr, sz = rz(sweep)
        ax.plot(sr, sz, color="#b4b4b4", linewidth=0.6, linestyle=(0, (2.2, 1.4)),
                zorder=2)

        r, z = rz(phases[name])
        ax.plot(r, z, color=INK, linewidth=1.9, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(r[1:], z[1:], "o", color=INK, markersize=1.7, zorder=4)

        if name == "windup":
            ax.plot([r[-1]], [z[-1]], "o", color=GREEN, markersize=3.4,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.5)
        elif name == "release":
            ax.plot([tr[0]], [tz[0]], "o", color=BLUE, markersize=3.6,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.5)
            d = arc[3] - arc[0]
            d = d / np.linalg.norm(d) * 0.26
            dr = np.hypot(d[0], d[1]) * np.sign(d[0] if abs(d[0]) > 1e-9 else 1.0)
            ax.annotate("", xy=(tr[0] + dr, tz[0] + d[2]), xytext=(tr[0], tz[0]),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=BLUE,
                                        mutation_scale=7), zorder=5)
        else:
            ax.plot(ar, az, color=BLUE, linewidth=1.3, zorder=3,
                    solid_capstyle="round")
            ax.plot([ar[-1]], [0.0], "v", color=BLUE, markersize=4.0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)

        # Phase name above its own panel -- the arm fills the panel interior,
        # so every in-plot position tried for this label sat on top of a link.
        ax.set_title(title, fontsize=6.9, fontweight="bold", color=INK,
                     pad=2.0)
        ax.set_xlim(-0.34, 0.74)
        ax.set_ylim(-0.04, 1.80)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0.0, 0.5])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, linestyle="-", alpha=0.6)
        ax.set_axisbelow(True)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    axes[0].set_ylabel("height above\nfloor (m)")
    axes[0].set_yticks([0.0, 0.5, 1.0, 1.5])
    axes[1].set_xlabel("distance from base axis (m)")
    fig.subplots_adjust(wspace=0.08)

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_throw_phases.{ext}")
    print(f"  wrote fig_throw_phases.pdf / .png  (|v_release| = {v_rel:.3f} m/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(HERE.parent / "overleaf" / "figs"))
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir).resolve()
    phases, sweep, arc, v_rel = collect()
    plot(phases, sweep, arc, v_rel, outdir)


if __name__ == "__main__":
    main()
