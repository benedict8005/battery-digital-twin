"""PDE residual checked against fixed analytic temperature fields (no trained network).

The "network output" ``y = [T_delta, c]`` is an explicit function of the input columns
``[space, SoH, Re, Ambient, time]``, so every residual can be written down by hand.
A wrong sign, a missing ``TEMP_SCALE_C`` or ``t_max`` factor, or a unit slip in
``battery_pinn.physics.make_pde`` shows up as a failed closed-form comparison.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
dde = pytest.importorskip("deepxde")

from battery_pinn import config
from battery_pinn.interpolation import DischargeProfile
from battery_pinn.physics import make_pde

T_MAX = 3600.0
MASS = config.RHO * config.CP
AMBIENT_C = 25.0
LOG_A = 22.0
# Constant-OCV polynomial (degree 0): U_ocv = 3.0 V whatever the state of charge.
OCV_COEFFS = [3.0]


def _profile(current: float, voltage: float) -> DischargeProfile:
    df = pd.DataFrame(
        {
            "Time_normalized": np.linspace(0, 1, 20),
            "Current_measured": current,
            "Voltage_measured": voltage,
            "SoC": np.linspace(100, 0, 20),
        }
    )
    return DischargeProfile(df, device="cpu")


@pytest.fixture
def x():
    """Fixed collocation points; column 3 is the ambient temperature in Celsius."""
    pts = torch.tensor(
        [
            [0.1, 0.9, 0.05, AMBIENT_C, 0.15],
            [0.5, 0.8, 0.06, AMBIENT_C, 0.40],
            [0.7, 1.0, 0.04, AMBIENT_C, 0.65],
            [0.9, 0.7, 0.07, AMBIENT_C, 0.90],
        ]
    )
    return pts.requires_grad_(True)


def _zero(x):
    """A zero that still depends (quadratically) on every input, so autograd never prunes a column."""
    return 0.0 * (x**2).sum(dim=1, keepdim=True)


def _residual(x, field, *, current=0.0, voltage=3.0, **kwargs):
    dde.grad.clear()
    y = torch.cat(field(x), dim=1)
    pde = make_pde(_profile(current, voltage), OCV_COEFFS, torch.tensor(LOG_A), T_MAX, **kwargs)
    res_thermal, res_sei = pde(x, y)
    dde.grad.clear()
    return res_thermal.detach().numpy(), res_sei.detach().numpy()


def _rate(c, temp_c):
    """Closed-form SEI reaction rate A * c * exp(-Ea / (R T))."""
    return 10.0**LOG_A * c * np.exp(-config.EA_SEI / (config.R_GAS * (temp_c + 273.15)))


def test_heating_ramp_gives_time_derivative_with_scale_and_t_max(x):
    slope = 0.2  # T_delta = slope * t, in units of TEMP_SCALE_C per normalised time

    def field(x):
        return slope * x[:, 4:5] + _zero(x), _zero(x)

    res_thermal, res_sei = _residual(x, field)
    expected = slope * config.TEMP_SCALE_C / T_MAX  # K/s
    np.testing.assert_allclose(res_thermal, expected, rtol=1e-5)
    np.testing.assert_allclose(res_sei, 0.0, atol=1e-12)


def test_curvature_enters_with_negative_conduction_sign(x):
    curvature = 0.3  # T_delta = curvature * space**2  =>  d2T/dx2 = 2 * curvature * TEMP_SCALE_C

    def field(x):
        return curvature * x[:, 0:1] ** 2 + _zero(x), _zero(x)

    res_thermal, _ = _residual(x, field)
    expected = -config.K_COND * 2.0 * curvature * config.TEMP_SCALE_C / MASS
    np.testing.assert_allclose(res_thermal, expected, rtol=1e-5)
    assert np.all(res_thermal < 0)  # conduction out of a hot spot cools it: residual is negative


def test_joule_heating_is_current_times_overpotential_over_thermal_mass(x):
    def field(x):
        return _zero(x), _zero(x)

    res_thermal, _ = _residual(x, field, current=-2.0, voltage=3.5)
    expected = -abs(-2.0 * (3.5 - OCV_COEFFS[0])) / MASS  # |I (V - U)| = 1 W/m3
    np.testing.assert_allclose(res_thermal, expected, rtol=1e-5)


def test_joule_heating_ignores_the_sign_of_the_current(x):
    def field(x):
        return _zero(x), _zero(x)

    discharge, _ = _residual(x, field, current=-2.0, voltage=3.5)
    charge, _ = _residual(x, field, current=2.0, voltage=3.5)
    np.testing.assert_allclose(discharge, charge, rtol=1e-6)


def test_constant_concentration_residual_is_the_arrhenius_rate(x):
    c0 = 0.5

    def field(x):
        return _zero(x), c0 + _zero(x)

    res_thermal, res_sei = _residual(x, field)
    rate = _rate(c0, AMBIENT_C)
    np.testing.assert_allclose(res_sei, rate, rtol=1e-5)
    # The reaction heats the cell: it enters the thermal residual with a minus sign.
    np.testing.assert_allclose(res_thermal, -config.H_SEI * rate / MASS, rtol=1e-5)


def test_decaying_concentration_adds_time_derivative_over_t_max(x):
    def field(x):
        return _zero(x), 1.0 - x[:, 4:5] + _zero(x)

    _, res_sei = _residual(x, field)
    t = x.detach().numpy()[:, 4:5]
    expected = -1.0 / T_MAX + _rate(1.0 - t, AMBIENT_C)
    np.testing.assert_allclose(res_sei, expected, rtol=1e-5)


def test_rate_grows_with_temperature(x):
    def field_at(offset):
        def field(x):
            return offset + _zero(x), 0.5 + _zero(x)

        return field

    _, cold = _residual(x, field_at(0.0))
    _, warm = _residual(x, field_at(0.4))  # 20 K warmer
    np.testing.assert_allclose(warm, _rate(0.5, AMBIENT_C + 0.4 * config.TEMP_SCALE_C), rtol=1e-5)
    assert np.all(warm > cold)


def test_newton_cooling_adds_surface_loss_proportional_to_excess_temperature(x):
    excess = 0.3  # T - T_ambient = excess * TEMP_SCALE_C = 15 K

    def field(x):
        return excess + _zero(x), _zero(x)

    plain, _ = _residual(x, field)
    cooled, _ = _residual(x, field, newton_cooling=True)
    np.testing.assert_allclose(plain, 0.0, atol=1e-12)
    # sources = -h (T - T_amb), so the residual -sources / mass is positive for a warm cell.
    expected = config.NEWTON_COOLING_COEFF * excess * config.TEMP_SCALE_C / MASS
    np.testing.assert_allclose(cooled, expected, rtol=1e-5)
