"""
Load the current camera-to-base extrinsic and apply it -- the "auto tf" half
of the calibration story: nothing downstream should need to know a file path,
only whether a calibration exists yet.

WHY A CANONICAL PATH
---------------------
Every extrinsic this project has produced so far lived only in a one-off JSON
next to whichever script wrote it, or in agent memory -- never a file
anything else automatically picked up. That is the reason no landing point
has ever been reported in the base frame from a real recording, despite
measure_landing.py being able to do exactly that since it was written.
CANONICAL_PATH fixes one place `scripts/calibrate_marker_tf.py` writes to and
everything else reads from, so a fresh calibration is the only step needed to
make base-frame output start working everywhere `auto_to_base` is called.

Validation logic is deliberately re-implemented here rather than imported
from measure_landing.py: perception/ is a leaf package and importing a
root-level CLI script back into it is the wrong direction of dependency.
"""

from __future__ import annotations

import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mc-pilot-pybullet
CANONICAL_PATH = os.path.join(_ROOT, "calib", "T_B_C.npz")

__all__ = ["CANONICAL_PATH", "has_calibration", "load_extrinsic", "to_base", "auto_to_base"]


def has_calibration(path=None):
    return os.path.exists(path or CANONICAL_PATH)


def load_extrinsic(path=None):
    """
    .npz with `R` (3x3), `t` (3,) -> (R, t). Convention: p_base = R @ p_cam + t,
    the one every extrinsic in this project uses (matches measure_landing.py).
    """
    path = path or CANONICAL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no extrinsic calibration at {path} -- run "
            f"scripts/calibrate_marker_tf.py first")
    z = np.load(path, allow_pickle=False)
    R, t = np.asarray(z["R"], float), np.asarray(z["t"], float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"expected R (3,3) and t (3,), got {R.shape} and {t.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthonormal -- this is not a rotation")
    return R, t


def to_base(points_cam, R, t):
    """
    Camera-frame point(s) -> base-frame point(s). (3,) in -> (3,) out;
    (N, 3) in -> (N, 3) out.
    """
    p = np.asarray(points_cam, float)
    return p @ R.T + t


def auto_to_base(points_cam, path=None):
    """Load the canonical (or given) calibration and transform in one call."""
    R, t = load_extrinsic(path)
    return to_base(points_cam, R, t)
