"""
Plug-in arm noise models for mc-pilot-pybullet.

All classes implement the ArmNoise interface with two key methods:

  pybullet_release_vel(v_cmd, ee_vel) -> np.array (3,)
      What velocity to set on the ball at release inside PyBullet.
      Called once per real throw.

  perturb_numpy(v3d_np, n) -> (scale, additive)
      For use in MC_PILOT.apply_policy().
      v3d_np : (n, 3) numpy array of commanded velocities per particle
      Returns:
        scale    : (n,) array — multiplicative factor (gradient flows through v3d)
        additive : (n, 3) array — additive offset (detached noise)
      In apply_policy: v3d_actual = scale[:, None] * v3d + additive

  sample_release_offset() -> int
      Extra PyBullet steps to wait before releasing (timing jitter only).

Available classes:
  VelocityBiasNoise  — zero-mean Gaussian on release velocity (unlearnable, good baseline)
  VelocitySlipNoise  — speed-dependent loss (learnable: policy must scale up all throws)
    SaltAndPepperVelocityNoise — sparse impulse outliers on release velocity
  ReleaseTimingJitter — gripper opens t_d ~ U(a,b) seconds late (learnable: policy must
                        compensate for arm deceleration during the delay; matches paper §5)
"""

import numpy as np


class ArmNoise:
    """Base class — zero noise (identity). Subclass to override."""

    def pybullet_release_vel(self, v_cmd, ee_vel):
        """Velocity to set on ball at release in PyBullet. Default: v_cmd exactly."""
        return np.array(v_cmd, dtype=float)

    def perturb_numpy(self, v3d_np, n):
        """
        Returns (scale, additive) for apply_policy particle perturbation.
        v3d_np : (n, 3) numpy — commanded velocities (used for state-dependent noise)
        scale  : (n,) numpy — multiplicative factor on v3d tensor (gradient-safe)
        additive : (n, 3) numpy — additive noise (detached constant)
        """
        return np.ones(n), np.zeros((n, 3))

    def sample_release_offset(self):
        """Number of extra sim steps before release (timing jitter). Returns int."""
        return 0


class VelocityBiasNoise(ArmNoise):
    """
    Zero-mean Gaussian noise on the ball's release velocity.

    v_actual = v_cmd + dv,  dv ~ N(0, sigma^2 * I_3)

    This is unlearnable: zero-mean noise cannot be compensated by changing the
    policy output.  The noise-aware policy is identical to the noiseless policy;
    only the cost calibration (honest vs optimistic uncertainty) differs.

    Use for baseline comparison only.  For a demo where RL has something to
    learn, use VelocitySlipNoise or ReleaseTimingJitter.

    Parameters
    ----------
    sigma : float — std dev of per-component velocity noise (m/s)
    seed  : optional int — for reproducibility
    """

    def __init__(self, sigma, seed=None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def pybullet_release_vel(self, v_cmd, ee_vel):
        dv = self.rng.normal(0.0, self.sigma, 3)
        return np.array(v_cmd) + dv

    def perturb_numpy(self, v3d_np, n):
        additive = self.rng.normal(0.0, self.sigma, (n, 3))
        return np.ones(n), additive


class VelocitySlipNoise(ArmNoise):
    """
    Speed-dependent gripper slip: faster throws lose a larger fraction of speed.

    v_actual = (1 - alpha) * v_cmd + N(0, sigma^2 * I_3)

    This IS learnable: the optimal policy must output v_needed / (1 - alpha) for
    every target, a multiplicative correction that scales with throw distance.
    A naive policy fitted to ideal ballistics will systematically undershoot,
    more so at longer ranges.

    Parameters
    ----------
    alpha : float — fraction of v_cmd lost at release (0.10–0.15 recommended)
    sigma : float — additive scatter std dev (m/s); 0.03–0.05 recommended
    seed  : optional int
    """

    def __init__(self, alpha, sigma=0.04, seed=None):
        self.alpha = alpha
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def pybullet_release_vel(self, v_cmd, ee_vel):
        dv = self.rng.normal(0.0, self.sigma, 3)
        return (1.0 - self.alpha) * np.array(v_cmd) + dv

    def perturb_numpy(self, v3d_np, n):
        scale    = (1.0 - self.alpha) * np.ones(n)
        additive = self.rng.normal(0.0, self.sigma, (n, 3))
        return scale, additive


class SaltAndPepperVelocityNoise(ArmNoise):
    """
    Salt-and-Pepper release noise: sparse positive/negative impulse outliers.

    v_actual = v_cmd + eps, where each component eps_j is:
      - +spike_scale with probability p_spike/2   (salt)
      - -spike_scale with probability p_spike/2   (pepper)
      - 0               with probability 1-p_spike
    plus optional Gaussian background N(0, sigma^2).

    This noise type models intermittent actuator glitches or release disturbances
    that are not present at every throw. Increasing p_spike raises outlier rate,
    while increasing spike_scale raises outlier severity.

    Parameters
    ----------
    p_spike     : float in [0, 1] — probability of an impulse on each velocity component
    spike_scale : float — absolute magnitude of salt/pepper impulse (m/s)
    sigma       : float — optional Gaussian background std dev (m/s)
    seed        : optional int
    """

    def __init__(self, p_spike=0.10, spike_scale=0.30, sigma=0.0, seed=None):
        if p_spike < 0.0 or p_spike > 1.0:
            raise ValueError(f"Need 0 <= p_spike <= 1, got {p_spike}")
        if spike_scale < 0.0:
            raise ValueError(f"Need spike_scale >= 0, got {spike_scale}")
        if sigma < 0.0:
            raise ValueError(f"Need sigma >= 0, got {sigma}")

        self.p_spike = float(p_spike)
        self.spike_scale = float(spike_scale)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def _sample_additive(self, shape):
        additive = np.zeros(shape, dtype=float)
        if self.sigma > 0.0:
            additive += self.rng.normal(0.0, self.sigma, size=shape)

        mask = self.rng.random(size=shape) < self.p_spike
        signs = self.rng.choice(np.array([-1.0, 1.0]), size=shape)
        additive += self.spike_scale * signs * mask
        return additive

    def pybullet_release_vel(self, v_cmd, ee_vel):
        return np.array(v_cmd, dtype=float) + self._sample_additive((3,))

    def perturb_numpy(self, v3d_np, n):
        return np.ones(n), self._sample_additive((n, 3))


class ReleaseTimingJitter(ArmNoise):
    """
    Gripper timing jitter: object releases t_d ~ U(a, b) seconds after command.

    Matches the paper's Section 5 model.  The arm is decelerating during [t_r, t_r+t_d],
    so the actual release velocity is lower than commanded.  The effect scales with
    throw speed (faster throws → arm decelerates more during delay → bigger undershoot).

    This IS learnable: the policy must output a higher speed to compensate for the
    mean velocity loss decel_rate * E[t_d].  The policy-level fix is to aim faster;
    the paper's complementary fix is to shift t_{r_cmd} earlier by the estimated mean.

    In PyBullet and apply_policy: the same velocity loss formula is applied:
        v_actual = v_cmd * (1 - decel_rate * t_d / ||v_cmd||)
    The same t_d sampled by sample_release_offset() is reused in pybullet_release_vel()
    so the simulator and the particle model see identical noise distributions.
    (Using the actual arm EE velocity is unreliable: PyBullet position control does
    not accurately track joint velocities, and joint limits cap achievable EE speed.)

    Parameters
    ----------
    a          : float — minimum delay (s).  a > 0 creates a systematic bias the
                 policy can learn to compensate.  Recommended: 0.02–0.04 s.
    b          : float — maximum delay (s).  Recommended: 0.06–0.10 s.
    decel_rate : float — EE deceleration (m/s^2) during the rest phase.
                 For iiwa7 rest phase 0.60→1.20 s with uM=2.5 m/s: ~4.0 m/s^2.
                 Calibrate by running a rollout and logging EE velocity over the
                 rest phase, or approximate as uM / (T_arm - t_r).
    dt         : float — sim timestep (s), same as Ts in the test script.
    seed       : optional int
    """

    def __init__(self, a, b, decel_rate, dt, seed=None):
        if a < 0 or b <= a:
            raise ValueError(f"Need 0 <= a < b, got a={a}, b={b}")
        self.a = a
        self.b = b
        self.decel_rate = decel_rate
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def sample_release_offset(self):
        self._last_t_d = self.rng.uniform(self.a, self.b)
        return max(1, round(self._last_t_d / self.dt))

    def pybullet_release_vel(self, v_cmd, ee_vel):
        # Apply the same decel_rate model as perturb_numpy so that the PyBullet
        # throw distribution matches what apply_policy simulates.
        # (Using actual ee_vel is unreliable: PyBullet position control doesn't
        # track joint velocities accurately, and joint limits cap the arm at ~25%
        # of commanded EE speed, making the actual ee_vel random and useless.)
        t_d = getattr(self, '_last_t_d', (self.a + self.b) / 2)
        v_cmd = np.array(v_cmd, dtype=float)
        v_mag = max(np.linalg.norm(v_cmd), 1e-6)
        scale = max(0.0, 1.0 - self.decel_rate * t_d / v_mag)
        return scale * v_cmd

    def perturb_numpy(self, v3d_np, n):
        t_d   = self.rng.uniform(self.a, self.b, n)          # (n,)
        v_mag = np.linalg.norm(v3d_np, axis=1)               # (n,)
        v_mag = np.maximum(v_mag, 1e-6)
        # Fractional velocity loss: decel_rate * t_d / ||v||
        scale = 1.0 - self.decel_rate * t_d / v_mag          # (n,)
        scale = np.clip(scale, 0.0, 1.0)
        return scale, np.zeros((n, 3))
