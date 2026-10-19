import numpy as np
import pytest

torch = pytest.importorskip("torch")

from battery_pinn.interpolation import torch_interp


def _interp(x, xp, fp):
    out = torch_interp(torch.tensor(x, dtype=torch.float32),
                       torch.tensor(xp, dtype=torch.float32),
                       torch.tensor(fp, dtype=torch.float32))
    return out.numpy().flatten()


def test_matches_numpy_inside_range():
    xp = np.array([0.0, 0.2, 0.5, 1.0])
    fp = np.array([1.0, 3.0, 2.0, 5.0])
    x = np.linspace(0.0, 1.0, 41)
    np.testing.assert_allclose(_interp(x, xp, fp), np.interp(x, xp, fp), atol=1e-5)


def test_extrapolates_linearly_beyond_edges():
    xp = np.array([0.0, 1.0])
    fp = np.array([0.0, 2.0])
    np.testing.assert_allclose(_interp(np.array([-0.5, 1.5]), xp, fp), [-1.0, 3.0], atol=1e-5)


def test_output_is_column_vector():
    out = torch_interp(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert out.shape == (2, 1)


def _ulp_above(v):
    return float(np.nextafter(np.float32(v), np.float32(np.inf)))


def _extrapolate(x, anchor_x, anchor_y, neighbour_x, neighbour_y):
    return anchor_y + (neighbour_y - anchor_y) / (neighbour_x - anchor_x) * (x - anchor_x)


def test_near_duplicate_edge_breakpoints_do_not_blow_up_extrapolation():
    """Two cycles' first/last breakpoints one float32 ulp apart used to give slopes of ~1e6.

    The end value stays the anchor; the slope is taken to the nearest *distinct* breakpoint.
    """
    below, above = np.array([0.0, 0.019]), np.array([1.01, 1.2])

    # Lower edge: a one-ulp neighbour of the first breakpoint carrying a slightly different value.
    xp, fp = [0.02, _ulp_above(0.02), 0.5, 1.0], [2.0, 2.1, 3.0, 4.0]
    np.testing.assert_allclose(_interp(below, xp, fp), _extrapolate(below, 0.02, 2.0, 0.5, 3.0), rtol=1e-5)
    np.testing.assert_allclose(_interp(above, xp, fp), _extrapolate(above, 1.0, 4.0, 0.5, 3.0), rtol=1e-5)

    # Upper edge likewise.
    xp, fp = [0.02, 0.5, 1.0, _ulp_above(1.0)], [2.0, 3.0, 4.0, 4.2]
    np.testing.assert_allclose(_interp(below, xp, fp), _extrapolate(below, 0.02, 2.0, 0.5, 3.0), rtol=1e-5)
    np.testing.assert_allclose(_interp(above, xp, fp), _extrapolate(above, 1.0, 4.2, 0.5, 3.0), rtol=1e-5)

    # Exact duplicates at both ends.
    xp, fp = [0.02, 0.02, 0.5, 1.0, 1.0], [2.0, 2.1, 3.0, 4.0, 4.2]
    np.testing.assert_allclose(_interp(below, xp, fp), _extrapolate(below, 0.02, 2.0, 0.5, 3.0), rtol=1e-5)
    np.testing.assert_allclose(_interp(above, xp, fp), _extrapolate(above, 1.0, 4.2, 0.5, 3.0), rtol=1e-5)


def test_near_duplicate_edge_slope_was_the_old_failure_mode():
    """Regression guard with the magnitude of the original bug: the result at t=0 was ~ -1e6."""
    xp, fp = [0.02, _ulp_above(0.02), 0.5, 1.0], [2.0, 2.1, 3.0, 4.0]
    out = _interp(np.array([0.0]), xp, fp)
    assert 1.9 < out[0] < 2.0


def test_near_duplicate_breakpoints_keep_every_output_bounded():
    rng = np.random.default_rng(0)
    base = np.sort(rng.uniform(0.0, 1.0, 40))
    base[0], base[-1] = 0.0, 1.0
    # Pool two "cycles": the same grid plus a copy perturbed by one ulp, with noisy values.
    xp = np.sort(np.concatenate([base, np.array([_ulp_above(v) for v in base])]))
    fp = np.interp(xp, base, np.sin(3 * base)) + rng.normal(0, 0.05, len(xp))
    x = np.linspace(-0.1, 1.1, 501)
    out = _interp(x, xp, fp)
    assert np.isfinite(out).all()
    # Inside the range a linear interpolant never leaves the data range; outside it continues
    # the end segments, whose slopes are now O(1), so values stay within a few units.
    assert out[(x >= 0) & (x <= 1)].min() >= fp.min() - 1e-6
    assert out[(x >= 0) & (x <= 1)].max() <= fp.max() + 1e-6
    assert np.abs(out).max() < 5.0


def test_near_duplicate_interior_gap_returns_a_value_between_its_neighbours():
    xp = [0.0, 0.5, _ulp_above(0.5), 1.0]
    fp = [1.0, 2.0, 2.5, 3.0]
    out = _interp(np.array([0.25, 0.5, 0.75]), xp, fp)
    np.testing.assert_allclose(out[[0, 2]], [1.5, 2.75], atol=1e-6)  # distinct segments unaffected
    assert 2.0 <= out[1] <= 2.5


def test_all_breakpoints_coincident_gives_a_constant():
    out = _interp(np.array([-1.0, 0.5, 2.0]), [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out, 1.0)


def test_min_spacing_override_merges_close_breakpoints():
    xp, fp = [0.0, 0.001, 1.0], [0.0, 10.0, 1.0]
    # Default tolerance (1e-6 of the range): 0.001 is a real breakpoint, slope 1e4 at the edge.
    np.testing.assert_allclose(_interp(np.array([-0.001]), xp, fp), [-10.0], rtol=1e-4)
    # A coarser tolerance treats it as coincident with 0.0 and extrapolates to x=1.0 instead.
    out = torch_interp(torch.tensor([-0.001]), torch.tensor(xp), torch.tensor(fp), min_spacing=0.01)
    np.testing.assert_allclose(out.numpy().flatten(), [-0.001], atol=1e-6)


def test_gradient_flows_through_x():
    x = torch.tensor([0.3, 0.7, -0.5, 1.5], requires_grad=True)
    xp = torch.tensor([0.0, 0.5, _ulp_above(0.5), 1.0])
    fp = torch.tensor([0.0, 1.0, 1.2, 2.0])
    torch_interp(x, xp, fp).sum().backward()
    assert torch.isfinite(x.grad).all()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 1.6, 2.0, 1.6], rtol=1e-5)


def test_discharge_profile_with_pooled_near_duplicate_times_stays_finite():
    import pandas as pd

    from battery_pinn.interpolation import DischargeProfile

    t = np.linspace(0.0, 1.0, 50)
    pooled = np.sort(np.concatenate([t, [_ulp_above(v) for v in t[1:]]]))
    df_agg = pd.DataFrame({
        "Time_normalized": pooled,
        "Current_measured": -2.0 + 0.01 * np.sin(20 * pooled),
        "Voltage_measured": 4.0 - pooled + 0.01 * np.cos(20 * pooled),
        "SoC": 100 * (1 - pooled),
    })
    profile = DischargeProfile(df_agg, device="cpu")
    current, voltage, soc = profile.at(torch.tensor([[0.0], [1e-9], [0.5], [1.0]]))
    for out in (current, voltage, soc):
        assert out.shape == (4, 1) and torch.isfinite(out).all()
    np.testing.assert_allclose(soc.numpy().flatten(), [100.0, 100.0, 50.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(current.numpy().flatten(), -2.0, atol=0.02)
