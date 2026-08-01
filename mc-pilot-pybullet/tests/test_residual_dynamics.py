"""
Unit tests for Ballistic_SemiParametric_Model_learning_RBF (model-level residual
physics: GP learns delta_v - gravity, full prediction = gravity + GP residual).

Key properties:
  1. The analytical mean is gravity-only: m = [0, 0, -g*Ts].
  2. GP targets have gravity removed: residual_vz = full_delta_vz + g*Ts.
  3. With a ZERO GP residual (untrained / extrapolation), the one-step prediction
     reverts to GRAVITY (next vz = vz - g*Ts), NOT to "no change" as a zero-mean
     GP would. This is the whole point -- correct out-of-distribution physics.
"""

import numpy as np
import torch

import model_learning.Model_learning as ML


G = 9.81
TS = 0.02
GP_INPUT_DIM = 6


def _init_dict():
    return {
        "active_dims": np.arange(0, GP_INPUT_DIM),
        "lengthscales_init": np.ones(GP_INPUT_DIM),
        "flg_train_lengthscales": True,
        "lambda_init": np.ones(1),
        "flg_train_lambda": False,
        "sigma_n_init": np.ones(1),
        "flg_train_sigma_n": True,
        "sigma_n_num": None,
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }


def _model():
    return ML.Ballistic_SemiParametric_Model_learning_RBF(
        num_gp=3,
        init_dict_list=[_init_dict()] * 3,
        T_sampling=TS,
        g=G,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


def test_mean_is_gravity_only():
    m = _model()._mean_delta_v(4)
    assert m.shape == (4, 3)
    np.testing.assert_allclose(m[:, 0].numpy(), 0.0)
    np.testing.assert_allclose(m[:, 1].numpy(), 0.0)
    np.testing.assert_allclose(m[:, 2].numpy(), -G * TS)


def test_gp_targets_have_gravity_removed():
    mdl = _model()
    # two consecutive 8-D ball states; vz drops by exactly gravity + a small drag bit
    s0 = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 2.0, 0.7, 0.0])
    dvx, dvy = 0.01, -0.02
    dvz = -G * TS - 0.003          # gravity plus a small extra (residual)
    s1 = s0.copy()
    s1[3] += dvx; s1[4] += dvy; s1[5] += dvz
    states = torch.tensor(np.stack([s0, s1]), dtype=torch.float64)

    out = mdl.data_to_gp_output(states)   # list of 3 residual targets [1,1]
    np.testing.assert_allclose(out[0].item(), dvx, atol=1e-12)          # vx residual = full (mean 0)
    np.testing.assert_allclose(out[1].item(), dvy, atol=1e-12)          # vy residual = full (mean 0)
    np.testing.assert_allclose(out[2].item(), dvz + G * TS, atol=1e-12)  # vz residual = full + g*Ts (gravity removed)


def test_zero_residual_reverts_to_gravity():
    mdl = _model()
    vx0, vy0, vz0 = 1.0, 0.5, 2.0
    state = torch.tensor([[0.5, 0.0, 0.5, vx0, vy0, vz0, 0.7, 0.0]], dtype=torch.float64)
    zero = [torch.zeros(1, 1, dtype=torch.float64) for _ in range(3)]
    var = [1e-8 * torch.ones(1, 1, dtype=torch.float64) for _ in range(3)]

    nxt, _, _ = mdl.get_next_state_from_gp_output(state, None, zero, var, particle_pred=False)
    # vx, vy unchanged (zero mean + zero residual); vz drops by gravity
    np.testing.assert_allclose(nxt[0, 3].item(), vx0, atol=1e-9)
    np.testing.assert_allclose(nxt[0, 4].item(), vy0, atol=1e-9)
    np.testing.assert_allclose(nxt[0, 5].item(), vz0 - G * TS, atol=1e-9)
    # position integrates p + Ts*v + Ts/2*delta
    np.testing.assert_allclose(
        nxt[0, 2].item(), 0.5 + TS * vz0 + TS / 2 * (-G * TS), atol=1e-9
    )
    # target dims carried forward
    np.testing.assert_allclose(nxt[0, 6:8].numpy(), [0.7, 0.0], atol=1e-12)


def test_roundtrip_full_delta_preserved():
    """subtract-then-add must reconstruct the full delta_v for an arbitrary residual."""
    mdl = _model()
    state = torch.tensor([[0.5, 0.0, 0.5, 1.0, 0.0, 2.0, 0.7, 0.0]], dtype=torch.float64)
    residual = [torch.tensor([[0.004]], dtype=torch.float64),
                torch.tensor([[-0.001]], dtype=torch.float64),
                torch.tensor([[0.002]], dtype=torch.float64)]
    var = [1e-9 * torch.ones(1, 1, dtype=torch.float64) for _ in range(3)]
    nxt, _, _ = mdl.get_next_state_from_gp_output(state, None, residual, var, particle_pred=False)
    # full delta_vz = residual + (-g*Ts)
    np.testing.assert_allclose(nxt[0, 5].item(), 2.0 + 0.002 - G * TS, atol=1e-9)
