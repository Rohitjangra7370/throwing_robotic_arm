"""
Minimal local wind models for mc-pilot-pybullet.

This repo previously pointed to an external `mc-pilot-wind` checkout. For the
current project we only need a calm-wind default so PyBullet training and
evaluation can run standalone.
"""

from __future__ import annotations

import numpy as np


class WindModel:
    """Calm-wind fallback compatible with the PyBullet throwing system."""

    def __init__(self, wind_xyz=(0.0, 0.0, 0.0)):
        self._wind = np.array(wind_xyz, dtype=float)

    def reset(self):
        return None

    def __call__(self, _t):
        return self._wind.copy()
