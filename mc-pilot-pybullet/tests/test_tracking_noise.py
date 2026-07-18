import numpy as np

from robot_arm.noise_models import TrackingErrorNoise


def _synthetic_npz(tmp_path, slip=0.10, vz_bias=-0.03, sigma=0.005, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.deg2rad(35.0)
    u = np.repeat(np.linspace(0.3, 1.0, 25), 9)
    ang = np.tile(np.deg2rad(np.linspace(-30, 30, 9)), 25)
    v_cmd = np.stack([
        u * np.cos(alpha) * np.cos(ang),
        u * np.cos(alpha) * np.sin(ang),
        u * np.sin(alpha),
    ], axis=1)
    # ground truth: lose `slip` fraction of speed along the command direction,
    # constant vertical bias, small isotropic scatter
    v_release = (1.0 - slip) * v_cmd
    v_release[:, 2] += vz_bias
    v_release += rng.normal(0.0, sigma, v_cmd.shape)
    path = str(tmp_path / "sweep.npz")
    np.savez(path, u_cmd=u, angle=ang, v_cmd=v_cmd,
             v_planned=v_cmd, v_release=v_release,
             release_pos_err=np.zeros_like(u), time_scale=np.ones_like(u),
             land_xy=np.zeros((len(u), 2)), flag=np.zeros_like(u))
    return path


def test_fit_recovers_speed_proportional_slip(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path), seed=1)
    # parallel component: dv_par = -slip * u  =>  a_par ~ -0.10, b_par ~ 0
    assert abs(noise.coef_a[0] - (-0.10)) < 0.02
    assert abs(noise.coef_b[0]) < 0.02
    # vertical: constant bias => a_z ~ 0, b_z ~ -0.03
    assert abs(noise.coef_b[2] - (-0.03)) < 0.02
    # residual scatter should be near sigma, far below the bias magnitudes
    assert np.all(np.sqrt(np.diag(noise.resid_cov)) < 0.02)


def test_release_vel_applies_bias_in_command_frame(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path, sigma=1e-6), seed=1)
    v_cmd = np.array([0.6, 0.2, 0.5])
    out = noise.pybullet_release_vel(v_cmd, ee_vel=None)
    u = np.linalg.norm(v_cmd)
    # bias along command direction ~ -slip*u; overall speed must shrink
    assert np.linalg.norm(out) < u
    assert abs((np.linalg.norm(out) - u) / u + 0.10) < 0.05


def test_perturb_numpy_shapes(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path), seed=1)
    v3d = np.tile(np.array([0.5, 0.0, 0.35]), (7, 1))
    scale, additive = noise.perturb_numpy(v3d, 7)
    assert scale.shape == (7,)
    assert additive.shape == (7, 3)
