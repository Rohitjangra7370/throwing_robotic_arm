"""
Wind models for mc-pilot-wind.

Three plug-in wind scenarios for studying robustness of the MC-PILOT
throwing policy under realistic airflow disturbances.

All models implement the WindModel interface:
    __call__(t)  -> np.ndarray (3,)   wind velocity at time t
    reset()                           reinitialize for new episode
    describe()   -> str               human-readable description

Wind convention: [wx, wy, wz] in world frame (m/s).
Positive wx = wind blowing in +x direction (tail/headwind depending on throw).
Positive wy = crosswind to the right.
wz is always 0 (no vertical wind in indoor/warehouse scenarios).
"""

import numpy as np


class WindModel:
    """Base class — zero wind (calm). Subclass to override."""

    def __call__(self, t):
        """Return wind velocity [wx, wy, wz] at time t (seconds)."""
        return np.zeros(3)

    def reset(self):
        """Reset internal state for a new episode/throw."""
        pass

    def describe(self):
        """Human-readable description of this wind model."""
        return "Calm (no wind)"


class ConstantWind(WindModel):
    """
    Steady-state wind — constant velocity throughout each throw and
    across all throws.

    Models a persistent environmental wind (e.g. ventilation draft,
    open warehouse door).

    Parameters
    ----------
    velocity : array-like (3,)
        Wind velocity [wx, wy, wz] in m/s.  Typical values:
        - Light:    [0.3, 0, 0] — barely perceptible, ~2cm deflection
        - Moderate: [0.7, 0, 0] — noticeable, ~5cm deflection
        - Strong:   [1.0, 0, 0] — significant, ~8cm deflection
    """

    def __init__(self, velocity):
        self.velocity = np.array(velocity, dtype=float)

    def __call__(self, t):
        return self.velocity.copy()

    def describe(self):
        return f"Constant wind: [{self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}] m/s"


class GustWind(WindModel):
    """
    Random gusts — piecewise-constant wind that changes direction and
    magnitude every T_gust seconds.

    Models intermittent disturbances: doors opening/closing, passing
    vehicles, thermal updrafts.  The wind is constant within each
    T_gust window but different between windows, creating step changes
    mid-flight.

    Parameters
    ----------
    w_max : float
        Maximum wind speed per component (m/s).  Each component is
        sampled uniformly from [-w_max, +w_max] at each gust change.
    T_gust : float
        Duration of each constant-wind segment (seconds).  Should be
        shorter than ball flight time (~0.5s) to create mid-flight changes.
        Recommended: 0.10–0.20 s.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, w_max, T_gust=0.15, seed=None):
        self.w_max = w_max
        self.T_gust = T_gust
        self.rng = np.random.default_rng(seed)
        self._current = np.zeros(3)
        self._last_change = -1.0  # force immediate sample on first call

    def __call__(self, t):
        # Check if we've entered a new gust segment
        segment = int(t / self.T_gust)
        last_segment = int(self._last_change / self.T_gust) if self._last_change >= 0 else -1
        if segment != last_segment:
            # Sample new gust: 2D only (wz = 0)
            self._current = np.array([
                self.rng.uniform(-self.w_max, self.w_max),
                self.rng.uniform(-self.w_max, self.w_max),
                0.0,
            ])
            self._last_change = t
        return self._current.copy()

    def reset(self):
        """Reset for new episode — next call will sample fresh gust."""
        self._last_change = -1.0

    def describe(self):
        return f"Gust wind: w_max={self.w_max:.2f} m/s, T_gust={self.T_gust:.2f}s"


class TurbulentWind(WindModel):
    """
    Turbulence — continuous Gaussian noise with optional temporal
    correlation (exponential smoothing).

    Models persistent low-level atmospheric turbulence overlaid on a
    mean wind vector.  The temporal correlation prevents unrealistic
    frame-to-frame jumps.

    Wind at time t:
        raw(t)    = w_mean + N(0, sigma^2 * I_2)     [per-component]
        w(t)      = alpha * w(t-1) + (1 - alpha) * raw(t)   [smoothed]

    alpha=0: pure white noise (no correlation between timesteps)
    alpha=0.9: strongly correlated (slowly varying turbulence)

    Parameters
    ----------
    w_mean : array-like (3,)
        Mean wind velocity [wx, wy, wz].  This is the DC component
        that the policy could potentially learn to compensate.
    sigma : float
        Standard deviation of per-component noise (m/s).
    alpha : float in [0, 1)
        Exponential smoothing coefficient.  Higher = smoother.
        Recommended: 0.5–0.8 for realistic turbulence.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, w_mean, sigma, alpha=0.7, seed=None):
        self.w_mean = np.array(w_mean, dtype=float)
        self.sigma = sigma
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self._current = self.w_mean.copy()

    def __call__(self, t):
        # Sample raw noise (2D only, wz = 0)
        noise = np.array([
            self.rng.normal(0, self.sigma),
            self.rng.normal(0, self.sigma),
            0.0,
        ])
        raw = self.w_mean + noise
        # Exponential smoothing
        self._current = self.alpha * self._current + (1.0 - self.alpha) * raw
        return self._current.copy()

    def reset(self):
        """Reset smoothing state for new episode."""
        self._current = self.w_mean.copy()

    def describe(self):
        return (f"Turbulent wind: mean=[{self.w_mean[0]:.2f}, {self.w_mean[1]:.2f}], "
                f"sigma={self.sigma:.2f}, alpha={self.alpha:.2f}")
