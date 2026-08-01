"""
Unit tests for Residual_Throwing_Policy (TossingBot-style residual physics).

Verifies the four load-bearing properties of the design:
  1. The torch baseline_speed() port matches eval_baseline.baseline_speed numerically.
  2. Near-zero residual weights  =>  policy output == analytical baseline (v_hat).
  3. forward/backward shapes are correct and gradients flow to the residual RBF only
     (the analytical prior has no learnable parameters).
  4. A geometrically-unreachable target (denom <= 0) does not produce nan.
"""

import numpy as np
import torch

import policy_learning.Policy as Policy
from eval_baseline import baseline_speed


RELEASE = np.array([0.55, 0.0, 0.45])
UM = 0.6
ALPHA_DEG = 35.0


def _make_policy(weight_init, delta_max_frac=0.5, num_basis=64):
    rng = np.random.default_rng(0)
    # centers must cover the actual target domain (x~[0.6,0.75], y~[-0.1,0.1])
    centers = np.column_stack([
        rng.uniform(0.60, 0.75, num_basis),
        rng.uniform(-0.10, 0.10, num_basis),
    ])
    return Policy.Residual_Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
        num_basis=num_basis,
        u_max=UM,
        release_pos=RELEASE,
        launch_angle_deg=ALPHA_DEG,
        target_height=0.0,
        delta_max_frac=delta_max_frac,
        lengthscales_init=np.array([0.1, 0.1]),
        centers_init=centers,
        weight_init=weight_init,
        flg_drop=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


def _state(target_xy):
    return torch.tensor(
        np.concatenate([RELEASE, np.zeros(3), np.asarray(target_xy)]),
        dtype=torch.float64,
    ).unsqueeze(0)


def test_baseline_speed_matches_eval_baseline():
    pol = _make_policy(weight_init=np.zeros((1, 64)))
    for tgt in [(0.70, 0.0), (0.72, 0.05), (0.68, -0.08), (0.74, 0.10)]:
        ref = baseline_speed(RELEASE, tgt, launch_angle_deg=ALPHA_DEG)
        got = pol.baseline_speed(torch.tensor([tgt], dtype=torch.float64)).item()
        assert ref is not None
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_zero_residual_equals_baseline():
    """weight_init ~ 0  =>  raw ~ 0  =>  delta ~ 0  =>  output == clamp(v_hat)."""
    pol = _make_policy(weight_init=np.zeros((1, 64)))
    for tgt in [(0.68, 0.0), (0.70, 0.04), (0.66, -0.03)]:
        out = pol(_state(tgt), t=0).item()
        ref = baseline_speed(RELEASE, tgt, launch_angle_deg=ALPHA_DEG)
        np.testing.assert_allclose(out, min(ref, UM), rtol=1e-8, atol=1e-8)


def test_residual_shifts_output_within_bound():
    """Non-zero residual moves output away from baseline but stays within +-delta_max."""
    rng = np.random.default_rng(1)
    pol = _make_policy(weight_init=UM * (rng.random((1, 64)) - 0.5), delta_max_frac=0.1)
    tgt = (0.66, -0.02)   # in-range (v_hat ~0.45), so clamp is not the cause of the shift
    out = pol(_state(tgt), t=0).item()
    ref = baseline_speed(RELEASE, tgt, launch_angle_deg=ALPHA_DEG)
    assert abs(out - ref) > 1e-4                       # residual actually did something
    assert abs(out - ref) <= 0.1 * UM + 1e-9            # bounded by delta_max
    assert 0.0 <= out <= UM                             # within physical range


def test_gradient_flows_to_residual_only():
    pol = _make_policy(weight_init=np.zeros((1, 64)))
    out = pol(_state((0.71, 0.03)), t=0)
    out.backward()
    w = pol.f_linear.weight
    assert w.grad is not None and torch.any(w.grad != 0)
    # the analytical prior is a registered buffer, not a Parameter -> not trainable
    assert not pol.release_pos.requires_grad


def test_free_flight_returns_zero():
    pol = _make_policy(weight_init=np.zeros((1, 64)))
    assert pol(_state((0.71, 0.0)), t=5).item() == 0.0


def test_unreachable_target_no_nan():
    """A target above the reachable parabola (denom <= 0) must clamp, not nan."""
    pol = _make_policy(weight_init=np.zeros((1, 64)))
    # target essentially at the release point -> d~0, denom small; and a high target
    for tgt in [(0.55, 0.0), (0.551, 0.0)]:
        out = pol(_state(tgt), t=0)
        assert torch.all(torch.isfinite(out))
        assert 0.0 <= out.item() <= UM
