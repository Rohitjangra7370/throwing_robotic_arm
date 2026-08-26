"""
Annotated video + 3D plot from a recorded throw -- the visual half of the
pipeline nothing else in perception/ produces.

Every stage up to this point (detect_candidates, pair_candidates,
StereoRig.triangulate) can be right on synthetic data and silently wrong on
real IR frames -- a systematic exposure/background/threshold problem doesn't
raise, it just puts the circle somewhere that isn't the ball. This renders
exactly what the detector saw, frame by frame, so that question gets answered
by looking rather than trusting the numbers. Camera frame only -- no base
extrinsic is applied here, see stereo.py.
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perception.ball_track import detect_candidates, median_background
from perception.ray_plane import D435I_IR_848x480 as INTR
from perception.stereo import D435I_IR_BASELINE_M, StereoRig, pair_candidates

__all__ = ["render_annotated"]


def render_annotated(rec, out_video, out_plot, title=None, playback_fps=15):
    """
    Recording -> annotated side-by-side IR1/IR2 video + 3D triangulated-track plot.

    Returns {"n_frames", "n_paired"}. `playback_fps` below the recording's own
    fps slows playback down -- a real throw crosses the frame in well under a
    second at native speed, too fast to actually watch.
    """
    ir1, ir2, t = rec["ir1"], rec["ir2"], rec["t"]
    n = len(t)
    bg1 = median_background(ir1)
    bg2 = median_background(ir2)
    rig = StereoRig(INTR, D435I_IR_BASELINE_M)
    W, H = rec["meta"]["width"], rec["meta"]["height"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, playback_fps, (2 * W, H))

    trail = []
    pts3d, pts_t = [], []
    n_paired = 0

    for k in range(n):
        a, b = ir1[k], ir2[k]
        c1 = detect_candidates(a, bg1)
        c2 = detect_candidates(b, bg2)
        va = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        vb = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        for c in c1:
            cv2.circle(va, (int(round(c.u)), int(round(c.v))),
                       max(4, int(c.radius_px) + 3), (0, 165, 255), 1)
        for c in c2:
            cv2.circle(vb, (int(round(c.u)), int(round(c.v))),
                       max(4, int(c.radius_px) + 3), (0, 165, 255), 1)

        if c1 and c2:
            lt = [c.as_uv_area() for c in c1]
            rt = [c.as_uv_area() for c in c2]
            pairs = pair_candidates(lt, rt)
            if pairs:
                li, ri = pairs[0]
                u1, v1, _ = lt[li]
                u2, v2, _ = rt[ri]
                try:
                    p3 = rig.triangulate(u1, v1, u2, v2)
                except RuntimeError:
                    p3 = None
                if p3 is not None:
                    pts3d.append(p3)
                    pts_t.append(t[k])
                    n_paired += 1
                    trail.append((int(round(u1)), int(round(v1))))
                    cv2.circle(va, (int(round(u1)), int(round(v1))), 8, (0, 255, 0), 2)
                    cv2.circle(vb, (int(round(u2)), int(round(v2))), 8, (0, 255, 0), 2)
                    cv2.putText(va, f"z={p3[2]:.2f}m", (int(u1) + 12, int(v1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for i in range(1, len(trail)):
            cv2.line(va, trail[i - 1], trail[i], (0, 0, 255), 2)

        frame = np.concatenate([va, vb], axis=1)
        cv2.putText(frame, f"frame {k:03d}/{n}  t={t[k]:.3f}s  paired={n_paired}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (W, 0), (W, H), (80, 80, 80), 1)
        cv2.putText(frame, "IR1 (left)", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "IR2 (right)", (W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()

    pts3d = np.array(pts3d) if pts3d else np.zeros((0, 3))
    pts_t = np.array(pts_t)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts3d):
        sc = ax.scatter(pts3d[:, 0], -pts3d[:, 1], pts3d[:, 2], c=pts_t, cmap="viridis", s=25)
        ax.plot(pts3d[:, 0], -pts3d[:, 1], pts3d[:, 2], color="gray", alpha=0.4, linewidth=1)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label("t (s)")
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("-Y (up-ish, camera frame) (m)")
    ax.set_zlabel("Z forward / depth (m)")
    ax.set_title(f"{title or 'triangulated ball track'}\n"
                 f"{n_paired}/{n} frames paired, camera frame (no base extrinsic applied)")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    return {"n_frames": n, "n_paired": n_paired}
