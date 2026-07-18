import numpy as np

from measure_tracking_error import run_sweep


def test_quick_sweep_produces_records():
    u_grid = np.linspace(0.3, 1.0, 3)
    angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 2))
    rec = run_sweep(u_grid, angle_grid, robot_name="kinova_gen3_dyn")
    n = 3 * 2
    assert rec["u_cmd"].shape == (n,)
    assert rec["v_cmd"].shape == (n, 3)
    assert rec["v_release"].shape == (n, 3)
    assert rec["flag"].shape == (n,)
    assert np.all(np.isfinite(rec["v_release"]))
    # commanded speeds actually span the grid
    np.testing.assert_allclose(np.unique(rec["u_cmd"]), u_grid, atol=1e-12)
