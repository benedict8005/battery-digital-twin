import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
dde = pytest.importorskip("deepxde")

from battery_pinn import config
from battery_pinn.interpolation import DischargeProfile
from battery_pinn.ocv import ocv_from_soc_torch
from battery_pinn.physics import arrhenius_exponents, make_pde, recommended_log_a_sei


def synthetic_profile():
    return pd.DataFrame({
        "Time_normalized": np.linspace(0, 1, 20),
        "Current_measured": -2.0,
        "Voltage_measured": np.linspace(4.0, 3.0, 20),
        "SoC": np.linspace(100, 0, 20),
    })


def test_recommended_log_a_sei_makes_rate_order_one():
    log_a = recommended_log_a_sei()
    rate = 10 ** log_a * np.exp(arrhenius_exponents(config.REFERENCE_TEMP_K - 273.15))
    assert rate == pytest.approx(1.0, rel=1e-6)
    assert 22.0 < log_a < 24.0


def test_exponent_rises_with_temperature():
    exps = arrhenius_exponents([0, 25, 60, 150])
    assert np.all(np.diff(exps) > 0)


def test_old_clamp_saturated_but_new_clamp_does_not():
    exps = arrhenius_exponents([0, 4, 24, 43])
    assert np.all(exps < -50)                      # old lower bound would have clipped these
    assert np.all(exps > config.EXPONENT_CLAMP_MIN)  # the widened bound does not


def test_pde_returns_two_finite_residuals():
    df_agg = pd.DataFrame({
        "Time_normalized": np.linspace(0, 1, 20),
        "Current_measured": -2.0,
        "Voltage_measured": np.linspace(4.0, 3.0, 20),
        "SoC": np.linspace(100, 0, 20),
    })
    profile = DischargeProfile(df_agg, device=torch.device("cpu"))
    coeffs = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
    log_a = dde.Variable(recommended_log_a_sei())
    pde = make_pde(profile, coeffs, log_a, t_max_seconds=3000.0)

    net = dde.nn.FNN(config.NETWORK_LAYERS, "tanh", "Glorot normal")
    x = torch.rand(8, 5, requires_grad=True)
    y = net(x)
    res = pde(x, y)

    assert len(res) == 2
    for r in res:
        assert r.shape == (8, 1)
        assert torch.isfinite(r).all()


def _reference_residual(x, y, profile, ocv_coeffs, log_a_sei, t_max):
    """Independent transcription of the original (v10 notebook) residual, constants inlined."""
    T_delta, c = y[:, 0:1], y[:, 1:2]
    T = T_delta * 50.0 + x[:, 3:4] + 273.15
    T_safe = torch.clamp(T, 273.15, 600.0)
    c_safe = torch.clamp(c, 0.0, 1.0)
    dT_t = dde.grad.jacobian(y, x, i=0, j=4) / t_max
    dc_t = dde.grad.jacobian(y, x, i=1, j=4) / t_max
    dT_xx = dde.grad.hessian(y, x, i=0, j=0)
    I, V, soc = profile.at(x[:, 4:5])
    U = ocv_from_soc_torch(soc / 100.0, ocv_coeffs)
    rate = 10.0 ** log_a_sei * c_safe * torch.exp(torch.clamp(-1.35e5 / (8.314 * T_safe + 1e-6), -70, 50))
    res_T = dT_t * 50.0 - (1.0 * dT_xx * 50.0 + torch.abs(I * (V - U)) + 2.57e5 * rate) / (2000.0 * 1000.0 + 1e-6)
    return res_T, dc_t + rate


@pytest.fixture
def pde_setup():
    from battery_pinn.ocv import fit_ocv_coeffs

    df_agg = synthetic_profile()
    profile = DischargeProfile(df_agg, device="cpu")
    coeffs = fit_ocv_coeffs(df_agg)
    torch.manual_seed(0)
    x = torch.rand(64, 5, requires_grad=True)
    net = torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    return profile, coeffs, x, net


def test_default_options_reproduce_the_original_residual_exactly(pde_setup):
    profile, coeffs, x, net = pde_setup
    log_a = dde.Variable(20.0)
    y = net(x)
    dde.grad.clear()
    res = make_pde(profile, coeffs, log_a, 3600.0)(x, y)
    dde.grad.clear()
    ref = _reference_residual(x, net(x), profile, coeffs, log_a, 3600.0)
    for got, want in zip(res, ref):
        assert torch.allclose(got, want, atol=1e-7, rtol=1e-6)
    # Explicitly passing the defaults is the same thing.
    dde.grad.clear()
    res2 = make_pde(profile, coeffs, log_a, 3600.0, newton_cooling=False, thermal=None)(x, net(x))
    for a, b in zip(res, res2):
        assert torch.allclose(a, b)


def test_newton_cooling_adds_exactly_the_surface_loss_term(pde_setup):
    from battery_pinn import config

    profile, coeffs, x, net = pde_setup
    log_a = dde.Variable(20.0)
    dde.grad.clear()
    base = make_pde(profile, coeffs, log_a, 3600.0)(x, net(x))
    dde.grad.clear()
    newton = make_pde(profile, coeffs, log_a, 3600.0, newton_cooling=True)(x, net(x))
    y = net(x)
    delta_T = y[:, 0:1] * config.TEMP_SCALE_C            # T - T_ambient
    expected = base[0] + config.NEWTON_COOLING_COEFF * delta_T / (config.RHO * config.CP + 1e-6)
    assert torch.allclose(newton[0], expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(newton[1], base[1])            # SEI residual untouched


def test_learnable_thermal_constants_start_at_config_values_and_are_trainable(pde_setup):
    from battery_pinn import config
    from battery_pinn.physics import make_thermal_variables

    profile, coeffs, x, net = pde_setup
    thermal = make_thermal_variables()
    assert set(thermal) == {"log10_k_cond", "log10_thermal_mass", "log10_newton_coeff"}
    assert float(10 ** thermal["log10_k_cond"].detach()) == pytest.approx(config.K_COND)
    assert float(10 ** thermal["log10_thermal_mass"].detach()) == pytest.approx(config.RHO * config.CP)
    assert all(v.requires_grad for v in thermal.values())
    log_a = dde.Variable(20.0)
    dde.grad.clear()
    base = make_pde(profile, coeffs, log_a, 3600.0)(x, net(x))
    dde.grad.clear()
    learn = make_pde(profile, coeffs, log_a, 3600.0, thermal=thermal)(x, net(x))
    for a, b in zip(base, learn):                         # identical at the initial values ...
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    learn[0].pow(2).sum().backward()                     # ... and the residual depends on them
    assert thermal["log10_thermal_mass"].grad is not None and thermal["log10_thermal_mass"].grad.abs() > 0
    assert thermal["log10_k_cond"].grad is not None
    assert thermal["log10_newton_coeff"].grad is None or thermal["log10_newton_coeff"].grad == 0  # off without newton_cooling
